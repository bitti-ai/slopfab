#include "vulkan_runtime.h"
#include "slopfab/tensor_convert.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace slopfab::moss_vulkan {
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

std::string vae_name(Runtime& rt, std::string name) {
  if (!rt.checkpoint.find("decoder.mid.block_1.norm1.weight")) return name;
  const auto replace = [&](const std::string& from, const std::string& to) {
    auto at = name.find(from); if (at != std::string::npos) name.replace(at, from.size(), to);
  };
  if (name == "post_quant_conv") return "decoder.post_quant_conv";
  if (name == "quant_conv") return "encoder.quant_conv";
  replace(".mid_block.resnets.0", ".mid.block_1");
  replace(".mid_block.resnets.1", ".mid.block_2");
  replace(".mid_block.attentions.0", ".mid.attn_1");
  for (int i = 0; i < 4; ++i) {
    replace("decoder.up_blocks." + std::to_string(i), "decoder.up." + std::to_string(3-i));
    replace("encoder.down_blocks." + std::to_string(i), "encoder.down." + std::to_string(i));
  }
  replace(".resnets.", ".block."); replace(".upsamplers.0", ".upsample");
  replace(".downsamplers.0", ".downsample"); replace(".conv_norm_out", ".norm_out");
  replace(".conv_shortcut", ".nin_shortcut"); replace(".group_norm", ".norm");
  replace(".to_q", ".q"); replace(".to_k", ".k"); replace(".to_v", ".v"); replace(".to_out.0", ".proj_out");
  return name;
}
Image conv(Runtime& rt, const Image& x, const std::string& name, int stride = 1) {
  const auto p = vae_name(rt, name);
  const auto& tensor = rt.checkpoint.at(p + ".weight");
  const auto& shape = tensor.shape;
  if ((shape.size() != 4 && shape.size() != 2) || shape[1] != x.c ||
      (shape.size() == 4 && (shape[2] != shape[3] || (shape[2] != 1 && shape[2] != 3))))
    throw std::invalid_argument("FLUX Vulkan VAE convolution shape: " + p);
  int co = int(shape[0]), kernel = shape.size() == 4 ? int(shape[2]) : 1;
  int columns = x.c * kernel * kernel;
  auto weight = rt.engine->upload(to_f32(tensor), co, columns, true);
  auto bias = rt.engine->upload(to_f32(rt.checkpoint.at(p + ".bias")), 1, co, true, true);
  Image y(rt, x.h / stride, x.w / stride, co);
  int pixels = y.h * y.w;
  for (int first = 0; first < pixels; first += 2048) {
    int count = std::min(2048, pixels - first);
    if (kernel == 1 && stride == 1)
      multiply(rt, x.data, weight, y.data, count, co, columns, first, first, &bias);
    else {
      auto patches = rt.engine->allocate(count, columns, ScalarType::kFloat32);
      uint32_t n = count * columns;
      dispatch(*rt.engine, x.data, x.data, x.data, patches,
               {FluxPatches, n, uint32_t(columns), uint32_t(first), uint32_t(x.h), uint32_t(x.w),
                uint32_t(kernel), uint32_t(x.c), uint32_t(stride), uint32_t(y.w)}, (n + 255) / 256);
      multiply(rt, patches, weight, y.data, count, co, columns, 0, first, &bias);
    }
  }
  return y;
}
Image groupnorm(Runtime& rt, const Image& x, const std::string& name, bool silu = true) {
  const auto p = vae_name(rt, name);
  if (x.c % 32 || rt.checkpoint.at(p + ".weight").numel() != x.c ||
      rt.checkpoint.at(p + ".bias").numel() != x.c)
    throw std::invalid_argument("FLUX Vulkan VAE normalization shape: " + p);
  Image y(rt, x.h, x.w, x.c);
  auto& w = rt.float_weight(p + ".weight");
  auto& b = rt.float_weight(p + ".bias");
  dispatch(*rt.engine, x.data, w, b, y.data,
           {FluxGroupNorm, uint32_t(x.h * x.w), uint32_t(x.c), uint32_t(silu)}, 32);
  return y;
}
Image resnet(Runtime& rt, Image x, const std::string& p) {
  auto y = conv(rt, groupnorm(rt, x, p + ".norm1"), p + ".conv1");
  y = conv(rt, groupnorm(rt, y, p + ".norm2"), p + ".conv2");
  if (rt.checkpoint.find(vae_name(rt, p + ".conv_shortcut") + ".weight"))
    x = conv(rt, x, p + ".conv_shortcut");
  uint32_t n = x.h * x.w * x.c;
  dispatch(*rt.engine, y.data, x.data, x.data, y.data,
           {krea_vulkan::ImageResidual, n}, (n + 255) / 256);
  return y;
}
Image upsample(Runtime& rt, const Image& x, const std::string& p) {
  Image y(rt, x.h * 2, x.w * 2, x.c);
  uint32_t n = y.h * y.w * y.c;
  auto& input = x.data;
  dispatch(*rt.engine, input, input, input, y.data, {krea_vulkan::Nearest, n, uint32_t(x.c), uint32_t(x.w)},
           (n + 255) / 256);
  return conv(rt, y, p + ".conv");
}

Image attention(Runtime& rt, Image x, const std::string& p) {
  auto normalized = groupnorm(rt, x, p + ".group_norm", false);
  auto q = conv(rt, normalized, p + ".to_q"), k = conv(rt, normalized, p + ".to_k"),
       v = conv(rt, normalized, p + ".to_v");
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
             {krea_vulkan::Softmax, uint32_t(count), uint32_t(pixels), scale_bits}, count);
    multiply(rt, scores, vt, y.data, count, x.c, pixels, 0, first);
  }
  y = conv(rt, y, p + ".to_out.0");
  uint32_t n = pixels * x.c;
  dispatch(*rt.engine, y.data, x.data, x.data, y.data, {krea_vulkan::ImageResidual, n}, (n + 255) / 256);
  return y;
}
}

std::vector<float> encode_flux_image(const std::string& path, const std::vector<uint8_t>& pixels,
                                    int height, int width) {
  if (height < 16 || width < 16 || height % 16 || width % 16 || pixels.size() != size_t(height)*width*3)
    throw std::invalid_argument("FLUX.2 VAE encoding needs an RGB image on the 16-pixel lattice");
  Runtime rt(path);
  std::vector<float> input(pixels.size());
  for (size_t i=0;i<input.size();++i) input[i] = pixels[i] / 127.5f - 1;
  Image x(rt,height,width,3);
  x.data = rt.engine->upload(input, height * width, 3, true);
  x=conv(rt,x,"encoder.conv_in");
  for(int i=0;i<4;++i){const std::string p="encoder.down_blocks."+std::to_string(i);
    for(int j=0;j<2;++j)x=resnet(rt,std::move(x),p+".resnets."+std::to_string(j));
    if(i<3)x=conv(rt,x,p+".downsamplers.0.conv",2);
  }
  x=resnet(rt,std::move(x),"encoder.mid_block.resnets.0");
  x=attention(rt,std::move(x),"encoder.mid_block.attentions.0");
  x=resnet(rt,std::move(x),"encoder.mid_block.resnets.1");
  x=conv(rt,groupnorm(rt,x,"encoder.conv_norm_out"),"encoder.conv_out");
  x=conv(rt,x,"quant_conv");
  if(x.c!=64)throw std::invalid_argument("FLUX.2 encoder must output 32 Gaussian means and log variances");
  std::vector<float> moments(size_t(x.h) * x.w * x.c);
  rt.engine->flush();
  rt.engine->context.download(x.data, moments.data(), moments.size());
  auto mean=to_f32(rt.checkpoint.at("bn.running_mean")),var=to_f32(rt.checkpoint.at("bn.running_var"));
  if(mean.size()!=128||var.size()!=128)throw std::invalid_argument("FLUX.2 latent statistics shape");
  const int h=height/16,w=width/16;std::vector<float> packed(size_t(h)*w*128);
  for(int y=0;y<h;++y)for(int xx=0;xx<w;++xx)for(int c=0;c<32;++c)for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx){
    int channel=c*4+dy*2+dx;float v=moments[(size_t(y*2+dy)*w*2+xx*2+dx)*64+c];
    packed[(size_t(y)*w+xx)*128+channel]=(v-mean[channel])/std::sqrt(var[channel]+1e-4f);
  }
  return packed;
}

std::vector<float> decode_image(const std::string& path, const std::vector<float>& latent,
                                int height, int width) {
  Runtime rt(path);
  const auto mean = to_f32(rt.checkpoint.at("bn.running_mean"));
  const auto variance = to_f32(rt.checkpoint.at("bn.running_var"));
  if (mean.size() != 128 || variance.size() != 128 || latent.size() != size_t(height) * width * 128)
    throw std::invalid_argument("FLUX.2 VAE requires 128 packed latent channels");
  std::vector<float> unpacked(size_t(height) * width * 128);
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x)
      for (int c = 0; c < 32; ++c)
        for (int dy = 0; dy < 2; ++dy)
          for (int dx = 0; dx < 2; ++dx) {
            const int channel = c * 4 + dy * 2 + dx;
            const float value = latent[(size_t(y) * width + x) * 128 + channel] *
                                    std::sqrt(variance[channel] + 1e-4f) +
                                mean[channel];
            unpacked[(size_t(y * 2 + dy) * width * 2 + x * 2 + dx) * 32 + c] = value;
          }
  Image x(rt,height * 2, width * 2, 32);
  x.data = rt.engine->upload(unpacked, x.h * x.w, x.c, true);
  x = conv(rt, x, "post_quant_conv");
  x = conv(rt, x, "decoder.conv_in");
  x = resnet(rt, std::move(x), "decoder.mid_block.resnets.0");
  x = attention(rt, std::move(x), "decoder.mid_block.attentions.0");
  x = resnet(rt, std::move(x), "decoder.mid_block.resnets.1");
  for (int block = 0; block < 4; ++block) {
    const std::string p = "decoder.up_blocks." + std::to_string(block);
    for (int layer = 0; layer < 3; ++layer)
      x = resnet(rt, std::move(x), p + ".resnets." + std::to_string(layer));
    if (block < 3)
      x = upsample(rt, x, p + ".upsamplers.0");
  }
  x = conv(rt, groupnorm(rt, x, "decoder.conv_norm_out"), "decoder.conv_out");
  std::vector<float> pixels(size_t(x.h) * x.w * 3);
  rt.engine->flush();
  rt.engine->context.download(x.data, pixels.data(), pixels.size());
  // Shared delivery contract is planar RGB in [0,1].
  std::vector<float> planar(pixels.size());
  for (size_t i = 0; i < pixels.size() / 3; ++i)
    for (int c = 0; c < 3; ++c) {
      if (!std::isfinite(pixels[i * 3 + c]))
        throw std::runtime_error("non-finite MOSS VAE output");
      planar[size_t(c) * pixels.size() / 3 + i] =
          std::clamp(pixels[i * 3 + c] * .5f + .5f, 0.f, 1.f);
    }
  return planar;
}
} // namespace slopfab::moss_vulkan
