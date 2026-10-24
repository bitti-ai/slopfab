#include "runtime.cuh"
#include "slopfab/tensor_convert.h"
#include "slopfab/cuda/gemm.cuh"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace slopfab::moss {
namespace {
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
struct Image {
  int h, w, c;
  cuda::DeviceBuffer<float> data;

  Image(int height, int width, int channels)
      : h(height), w(width), c(channels), data(size_t(h) * w * c) {
  }
};

__global__ void im2col(const float* x, float* out, int h, int w, int ci, int kernel, int first,
                       int count, int stride, int out_w) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int columns = ci * kernel * kernel;
  if (index >= count * columns)
    return;
  const int r = first + index / columns, col = index % columns;
  const int pad = stride == 1 ? kernel / 2 : 0;
  const int y = r / out_w * stride + col / kernel % kernel - pad, xx = r % out_w * stride + col % kernel - pad;
  out[index] = (y >= 0 && y < h && xx >= 0 && xx < w)
                   ? x[(size_t(y) * w + xx) * ci + col / (kernel * kernel)]
                   : 0.f;
}

__global__ void add_bias(float* x, const float* bias, int n, int c) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    x[i] += bias[i % c];
}

Image conv(Runtime& rt, const Image& x, const std::string& name, int stride = 1) {
  const auto p = vae_name(rt, name);
  const auto& shape = rt.checkpoint.at(p + ".weight").shape;
  if (shape.size() != 4 || shape[1] != x.c || shape[2] != shape[3] ||
      (shape[2] != 1 && shape[2] != 3))
    throw std::invalid_argument("FLUX VAE convolution shape: " + p);
  const int co = int(shape[0]), kernel = int(shape[2]), columns = x.c * kernel * kernel;
  Image y(x.h / stride, x.w / stride, co);
  const float* weight = rt.float_weight(p + ".weight");
  const int tile = 2048, pixels = y.h * y.w;
  cuda::DeviceBuffer<float> patches(size_t(std::min(tile, pixels)) * columns);
  for (int first = 0; first < pixels; first += tile) {
    const int count = std::min(tile, pixels - first);
    const float* input = x.data.get() + size_t(first) * x.c;
    if (kernel != 1) {
      im2col<<<(count * columns + 255) / 256, 256>>>(x.data.get(), patches.get(), x.h, x.w, x.c,
                                                     kernel, first, count, stride, y.w);
      input = patches.get();
    }
    cuda::gemm_nt(rt.blas, input, weight, y.data.get() + size_t(first) * co, count, co, columns);
  }
  const int n = pixels * co;
  add_bias<<<(n + 255) / 256, 256>>>(y.data.get(), rt.float_weight(p + ".bias"), n, co);
  return y;
}

__global__ void group_stats(const float* x, float* stats, int pixels, int channels) {
  __shared__ float sum[256], square[256];
  const int group = blockIdx.x, width = channels / 32;
  float s = 0, ss = 0;
  for (int i = threadIdx.x; i < pixels * width; i += blockDim.x) {
    float a = x[size_t(i / width) * channels + group * width + i % width];
    s += a;
    ss += a * a;
  }
  sum[threadIdx.x] = s;
  square[threadIdx.x] = ss;
  __syncthreads();
  for (int d = 128; d; d /= 2) {
    if (threadIdx.x < d) {
      sum[threadIdx.x] += sum[threadIdx.x + d];
      square[threadIdx.x] += square[threadIdx.x + d];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    float mean = sum[0] / (pixels * width);
    stats[group * 2] = mean;
    stats[group * 2 + 1] = rsqrtf(fmaxf(square[0] / (pixels * width) - mean * mean, 0.f) + 1e-6f);
  }
}

__global__ void group_apply(const float* x, float* y, const float* stats, const float* weight,
                            const float* bias, int n, int channels, bool silu) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  const int c = i % channels, g = c / (channels / 32);
  float a = (x[i] - stats[g * 2]) * stats[g * 2 + 1] * weight[c] + bias[c];
  y[i] = silu ? a / (1.f + expf(-a)) : a;
}

Image groupnorm(Runtime& rt, const Image& x, const std::string& name, bool silu = true) {
  const auto p = vae_name(rt, name);
  if (x.c % 32 || rt.checkpoint.at(p + ".weight").numel() != x.c ||
      rt.checkpoint.at(p + ".bias").numel() != x.c)
    throw std::invalid_argument("FLUX VAE groupnorm shape: " + p);
  Image y(x.h, x.w, x.c);
  cuda::DeviceBuffer<float> stats(64);
  group_stats<<<32, 256>>>(x.data.get(), stats.get(), x.h * x.w, x.c);
  const int n = x.h * x.w * x.c;
  group_apply<<<(n + 255) / 256, 256>>>(x.data.get(), y.data.get(), stats.get(),
                                        rt.float_weight(p + ".weight"),
                                        rt.float_weight(p + ".bias"), n, x.c, silu);
  return y;
}

__global__ void add(float* x, const float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    x[i] += y[i];
}

Image resnet(Runtime& rt, Image x, const std::string& p) {
  auto y = conv(rt, groupnorm(rt, x, p + ".norm1"), p + ".conv1");
  y = conv(rt, groupnorm(rt, y, p + ".norm2"), p + ".conv2");
  if (rt.checkpoint.find(vae_name(rt, p + ".conv_shortcut") + ".weight"))
    x = conv(rt, x, p + ".conv_shortcut");
  const int n = x.h * x.w * x.c;
  add<<<(n + 255) / 256, 256>>>(y.data.get(), x.data.get(), n);
  return y;
}

__global__ void nearest(const float* x, float* y, int h, int w, int channels) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= h * w * 4 * channels)
    return;
  int r = i / channels, c = i % channels;
  y[i] = x[(size_t(r / (2 * w) / 2) * w + (r % (2 * w)) / 2) * channels + c];
}

Image upsample(Runtime& rt, const Image& x, const std::string& p) {
  Image y(x.h * 2, x.w * 2, x.c);
  const int n = y.h * y.w * y.c;
  nearest<<<(n + 255) / 256, 256>>>(x.data.get(), y.data.get(), x.h, x.w, x.c);
  return conv(rt, y, p + ".conv");
}

__global__ void softmax(float* scores, int width, float scale) {
  __shared__ float reduce[256];
  float* row = scores + size_t(blockIdx.x) * width;
  float maximum = -3.402823466e38f;
  for (int i = threadIdx.x; i < width; i += 256)
    maximum = fmaxf(maximum, row[i] * scale);
  reduce[threadIdx.x] = maximum;
  __syncthreads();
  for (int d = 128; d; d /= 2) {
    if (threadIdx.x < d)
      reduce[threadIdx.x] = fmaxf(reduce[threadIdx.x], reduce[threadIdx.x + d]);
    __syncthreads();
  }
  maximum = reduce[0];
  __syncthreads();
  float total = 0;
  for (int i = threadIdx.x; i < width; i += 256) {
    row[i] = expf(row[i] * scale - maximum);
    total += row[i];
  }
  reduce[threadIdx.x] = total;
  __syncthreads();
  for (int d = 128; d; d /= 2) {
    if (threadIdx.x < d)
      reduce[threadIdx.x] += reduce[threadIdx.x + d];
    __syncthreads();
  }
  total = reduce[0];
  for (int i = threadIdx.x; i < width; i += 256)
    row[i] /= total;
}

Image project(Runtime& rt, const Image& x, const std::string& name) {
  const auto p = vae_name(rt, name);
  const auto& shape = rt.checkpoint.at(p + ".weight").shape;
  if ((shape != std::vector<int64_t>{x.c, x.c} && shape != std::vector<int64_t>{x.c,x.c,1,1}) ||
      rt.checkpoint.at(p + ".bias").numel() != x.c)
    throw std::invalid_argument("FLUX VAE attention projection shape: " + p);
  Image y(x.h, x.w, x.c);
  cuda::gemm_nt(rt.blas, x.data.get(), rt.float_weight(p + ".weight"), y.data.get(), x.h * x.w, x.c,
                x.c);
  const int n = x.h * x.w * x.c;
  add_bias<<<(n + 255) / 256, 256>>>(y.data.get(), rt.float_weight(p + ".bias"), n, x.c);
  return y;
}

Image attention(Runtime& rt, Image x, const std::string& p) {
  auto normalized = groupnorm(rt, x, p + ".group_norm", false);
  auto q = project(rt, normalized, p + ".to_q"), k = project(rt, normalized, p + ".to_k"),
       v = project(rt, normalized, p + ".to_v");
  Image y(x.h, x.w, x.c);
  const int pixels = x.h * x.w, tile = 128;
  cuda::DeviceBuffer<float> scores(size_t(tile) * pixels);
  for (int first = 0; first < pixels; first += tile) {
    int count = std::min(tile, pixels - first);
    cuda::gemm_nt(rt.blas, q.data.get() + size_t(first) * x.c, k.data.get(), scores.get(), count,
                  pixels, x.c);
    softmax<<<count, 256>>>(scores.get(), pixels, 1.f / std::sqrt(float(x.c)));
    cuda::gemm_nn(rt.blas, scores.get(), v.data.get(), y.data.get() + size_t(first) * x.c, count,
                  x.c, pixels);
  }
  y = project(rt, y, p + ".to_out.0");
  int n = pixels * x.c;
  add<<<(n + 255) / 256, 256>>>(y.data.get(), x.data.get(), n);
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
  Image x(height,width,3);
  SLOPFAB_CUDA_CHECK(cudaMemcpy(x.data.get(),input.data(),input.size()*4,cudaMemcpyHostToDevice));
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
  std::vector<float> moments(x.data.size());
  SLOPFAB_CUDA_CHECK(cudaMemcpy(moments.data(),x.data.get(),moments.size()*4,cudaMemcpyDeviceToHost));
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
  Image x(height * 2, width * 2, 32);
  SLOPFAB_CUDA_CHECK(
      cudaMemcpy(x.data.get(), unpacked.data(), unpacked.size() * 4, cudaMemcpyHostToDevice));
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
  SLOPFAB_CUDA_CHECK(
      cudaMemcpy(pixels.data(), x.data.get(), pixels.size() * 4, cudaMemcpyDeviceToHost));
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
} // namespace slopfab::moss
