#include "vulkan_runtime.h"
#include "slopfab/tensor_convert.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace slopfab::krea_vulkan {
namespace {
struct Image {
  int h, w, c;
  mutable vk::DeviceTensor data;

  Image(Runtime& rt, int height, int width, int channels)
      : h(height), w(width), c(channels),
        data(rt.engine->allocate(h * w, c, ScalarType::kFloat32)) {
  }
};

void multiply(Runtime& rt, vk::DeviceTensor& x, vk::DeviceTensor& weight, vk::DeviceTensor& out,
              int rows, int columns, int inner, int input_offset = 0, int output_offset = 0,
              vk::DeviceTensor* bias = nullptr) {
  DenseGemmPlanDesc d;
  d.max_rows = rows;
  d.out_features = columns;
  d.in_features = inner;
  d.mode = DenseGemmMode::kFloat32;
  d.bias = bias ? DenseGemmBias::kFloat32 : DenseGemmBias::kNone;
  auto plan = vk::DenseGemmPlan::create(rt.engine->context, d);
  plan.record(rt.engine->batch(), x, weight, out, rows, input_offset, output_offset, bias);
}

Image conv(Runtime& rt, const Image& x, const std::string& p) {
  const auto& tensor = rt.checkpoint.at(p + ".weight");
  const auto& shape = tensor.shape;
  if ((shape.size() != 4 && shape.size() != 5) || shape[1] != x.c ||
      shape[shape.size() - 2] != shape.back() || (shape.back() != 1 && shape.back() != 3))
    throw std::invalid_argument("Qwen Vulkan VAE convolution shape: " + p);
  int co = int(shape[0]), kernel = int(shape.back()), columns = x.c * kernel * kernel;
  int temporal = shape.size() == 5 ? int(shape[2]) : 1;
  auto all = to_f32(tensor);
  std::vector<float> host(size_t(co) * columns);
  for (int o = 0; o < co; ++o)
    for (int i = 0; i < x.c; ++i)
      std::copy_n(all.data() + ((size_t(o) * x.c + i) * temporal + temporal - 1) * kernel * kernel,
                  kernel * kernel, host.data() + (size_t(o) * x.c + i) * kernel * kernel);
  auto weight = rt.engine->upload(host, co, columns, true);
  auto bias = rt.engine->upload(to_f32(rt.checkpoint.at(p + ".bias")), 1, co, true, true);
  Image y(rt, x.h, x.w, co);
  for (int first = 0; first < x.h * x.w; first += 2048) {
    int count = std::min(2048, x.h * x.w - first);
    if (kernel == 1)
      multiply(rt, x.data, weight, y.data, count, co, columns, first, first, &bias);
    else {
      auto patches = rt.engine->allocate(count, columns, ScalarType::kFloat32);
      uint32_t n = count * columns;
      auto& input = x.data;
      dispatch(*rt.engine, input, input, input, patches,
               {Patches, n, uint32_t(columns), uint32_t(first), uint32_t(x.h), uint32_t(x.w),
                uint32_t(kernel), uint32_t(x.c)},
               (n + 255) / 256);
      multiply(rt, patches, weight, y.data, count, co, columns, 0, first, &bias);
    }
  }
  return y;
}

Image normalize(Runtime& rt, const Image& x, const std::string& name, bool silu = true) {
  if (rt.checkpoint.at(name + ".gamma").numel() != x.c)
    throw std::invalid_argument("Qwen Vulkan RMS shape");
  Image y(rt, x.h, x.w, x.c);
  auto& w = rt.float_weight(name + ".gamma");
  auto& input = x.data;
  dispatch(*rt.engine, input, w, w, y.data,
           {ImageRms, uint32_t(x.h * x.w), uint32_t(x.c), uint32_t(silu)}, x.h * x.w);
  return y;
}

Image resnet(Runtime& rt, Image x, const std::string& p) {
  auto y = conv(rt, normalize(rt, x, p + ".residual.0"), p + ".residual.2");
  y = conv(rt, normalize(rt, y, p + ".residual.3"), p + ".residual.6");
  if (rt.checkpoint.find(p + ".shortcut.weight"))
    x = conv(rt, x, p + ".shortcut");
  uint32_t n = x.h * x.w * x.c;
  dispatch(*rt.engine, y.data, x.data, x.data, y.data, {ImageResidual, n}, (n + 255) / 256);
  return y;
}

Image upsample(Runtime& rt, const Image& x, const std::string& p) {
  Image y(rt, x.h * 2, x.w * 2, x.c);
  uint32_t n = y.h * y.w * y.c;
  auto& input = x.data;
  dispatch(*rt.engine, input, input, input, y.data, {Nearest, n, uint32_t(x.c), uint32_t(x.w)},
           (n + 255) / 256);
  return conv(rt, y, p + ".resample.1");
}

Image attention(Runtime& rt, Image x, const std::string& p) {
  auto qkv = conv(rt, normalize(rt, x, p + ".norm", false), p + ".to_qkv");
  Image q(rt, x.h, x.w, x.c), k(rt, x.h, x.w, x.c), v(rt, x.h, x.w, x.c);
  for (int i = 0; i < 3; ++i) {
    auto& dst = i == 0 ? q.data : (i == 1 ? k.data : v.data);
    uint32_t n = x.h * x.w * x.c;
    dispatch(*rt.engine, qkv.data, qkv.data, qkv.data, dst,
             {FloatColumns, n, uint32_t(x.c), uint32_t(3 * x.c), uint32_t(i * x.c)},
             (n + 255) / 256);
  }
  int pixels = x.h * x.w;
  auto vt = rt.engine->allocate(x.c, pixels, ScalarType::kFloat32);
  rt.engine->batch().transpose_2d(v.data, vt);
  Image y(rt, x.h, x.w, x.c);
  float scale = 1 / std::sqrt(float(x.c));
  uint32_t scale_bits;
  std::memcpy(&scale_bits, &scale, 4);
  for (int first = 0; first < pixels; first += 128) {
    int count = std::min(128, pixels - first);
    auto scores = rt.engine->allocate(count, pixels, ScalarType::kFloat32);
    multiply(rt, q.data, k.data, scores, count, pixels, x.c, first);
    dispatch(*rt.engine, scores, scores, scores, scores,
             {Softmax, uint32_t(count), uint32_t(pixels), scale_bits}, count);
    multiply(rt, scores, vt, y.data, count, x.c, pixels, 0, first);
  }
  y = conv(rt, y, p + ".proj");
  uint32_t n = pixels * x.c;
  dispatch(*rt.engine, y.data, x.data, x.data, y.data, {ImageResidual, n}, (n + 255) / 256);
  return y;
}
}

std::vector<float> decode_image(const std::string& path, const std::vector<float>& latent,
                                int height, int width) {
  if (height < 1 || width < 1 || latent.size() != size_t(height) * width * 64)
    throw std::invalid_argument("Krea 2 VAE requires [H/16,W/16,64] latents");
  const float mean[] = {-.7571f, -.7089f, -.9113f, .1075f,  -.1745f, .9653f,  -.1517f, 1.5508f,
                        .4134f,  -.0715f, .5517f,  -.3632f, -.1922f, -.9497f, .2503f,  -.2921f};
  const float stddev[] = {2.8184f, 1.4541f, 2.3275f, 2.6558f, 1.2196f, 1.7708f, 2.6052f, 2.0743f,
                          3.2687f, 2.1526f, 2.8652f, 1.5579f, 1.6382f, 1.1253f, 2.8251f, 1.9160f};
  Runtime rt(path);
  std::vector<float> unpacked(latent.size());
  for (int y = 0; y < height; ++y)
    for (int xx = 0; xx < width; ++xx)
      for (int c = 0; c < 16; ++c)
        for (int dy = 0; dy < 2; ++dy)
          for (int dx = 0; dx < 2; ++dx)
            unpacked[(size_t(y * 2 + dy) * width * 2 + xx * 2 + dx) * 16 + c] =
                latent[(size_t(y) * width + xx) * 64 + c * 4 + dy * 2 + dx] * stddev[c] + mean[c];
  Image x(rt, height * 2, width * 2, 16);
  x.data = rt.engine->upload(unpacked, x.h * x.w, x.c, true);
  x = conv(rt, x, "conv2");
  x = conv(rt, x, "decoder.conv1");
  x = resnet(rt, std::move(x), "decoder.middle.0");
  x = attention(rt, std::move(x), "decoder.middle.1");
  x = resnet(rt, std::move(x), "decoder.middle.2");
  for (int i = 0; i < 15; ++i) {
    const std::string p = "decoder.upsamples." + std::to_string(i);
    // Temporal upsampling is bypassed on the first frame, as in the upstream cache.
    if (i == 3 || i == 7 || i == 11)
      x = upsample(rt, x, p);
    else
      x = resnet(rt, std::move(x), p);
  }
  x = conv(rt, normalize(rt, x, "decoder.head.0"), "decoder.head.2");
  std::vector<float> pixels(size_t(x.h) * x.w * x.c), planar(pixels.size());
  rt.engine->flush();
  rt.engine->context.download(x.data, pixels.data(), pixels.size());
  for (size_t i = 0; i < pixels.size() / 3; ++i)
    for (int c = 0; c < 3; ++c) {
      float value = pixels[i * 3 + c];
      if (!std::isfinite(value))
        throw std::runtime_error("non-finite Krea 2 VAE output");
      planar[size_t(c) * pixels.size() / 3 + i] = std::clamp(value * .5f + .5f, 0.f, 1.f);
    }
  return planar;
}
}
