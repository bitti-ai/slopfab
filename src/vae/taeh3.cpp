#include "slopfab/vae/taeh3.h"
#include "slopfab/tensor_convert.h"
#include "../upscale/backend.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>

namespace slopfab::vae {
using namespace upscale_detail;
struct TaeH3::Impl {
  struct Conv { Buffer weight, bias; int input, output, kernel; };
  std::unique_ptr<Backend> backend;
  std::map<std::string, Conv> layers;

  Impl(const std::string& path, DeviceBackend device) {
    SafeTensors file;
    file.open(path);
    // Validate the exact base H3 architecture before allocating GPU buffers.
    struct Spec { std::string name; int in, out, kernel, stored; bool bias; };
    std::vector<Spec> specs = {{"1",24,256,3,256,true}};
    for (auto [start, channels] : {std::pair{3,256}, {9,128}, {15,64}}) {
      for (int i = start; i < start + 3; ++i)
        for (int c = 0; c < 3; ++c)
          specs.push_back({std::to_string(i)+".conv."+std::to_string(c*2),
                           c == 0 ? channels*2 : channels,channels,3,channels,true});
    }
    for (auto s : std::vector<Spec>{{"7.conv",256,256,1,256,false},
             {"8",256,128,3,128,false}, {"13.conv",128,128,1,256,false},
             {"14",128,64,3,64,false}, {"19.conv",64,64,1,128,false},
             {"20",64,64,3,64,false}, {"22",64,12,3,12,true}}) specs.push_back(s);
    for (const auto& s : specs) {
      const auto key = "decoder." + s.name;
      const auto& w = file.at(key + ".weight");
      if (w.shape != std::vector<int64_t>{s.stored,s.in,s.kernel,s.kernel} ||
          (w.dtype != DType::kF32 && w.dtype != DType::kF16 && w.dtype != DType::kBF16))
        throw std::invalid_argument("TAEH3: invalid decoder tensor " + key);
      if (s.bias && file.at(key + ".bias").shape != std::vector<int64_t>{s.out})
        throw std::invalid_argument("TAEH3: invalid bias " + key);
    }
    switch (device) {
#if SLOPFAB_WITH_CUDA
    case DeviceBackend::kCuda: backend = make_cuda_backend(); break;
#endif
#if SLOPFAB_WITH_VULKAN
    case DeviceBackend::kVulkan: backend = make_vulkan_backend(); break;
#endif
    default: throw std::invalid_argument("TAEH3: backend unavailable");
    }
    for (const auto& s : specs) {
      const auto key = "decoder." + s.name;
      auto w = to_f32(file.at(key + ".weight"));
      // Upstream patch_tgrow_layers keeps the last timestep when reducing time.
      w.erase(w.begin(), w.end() - size_t(s.out)*s.in*s.kernel*s.kernel);
      auto b = s.bias ? to_f32(file.at(key + ".bias")) : std::vector<float>(s.out, 0);
      for (const auto* values : {&w, &b})
        for (float v : *values)
          if (!std::isfinite(v)) throw std::invalid_argument("TAEH3: non-finite weight");
      layers.emplace(s.name, Conv{backend->upload(w), backend->upload(b),s.in,s.out,s.kernel});
    }
  }
  Buffer op(Operation op, const Buffer& x, const Buffer& y, int h, int w, int in, int out) {
    const auto n = uint32_t(h*w*out);
    auto result = backend->allocate(n);
    backend->run({op,uint32_t(h),uint32_t(w),uint32_t(in),uint32_t(out),n,0,1},x,y,y,result);
    return result;
  }
  Buffer conv(const Buffer& x, const std::string& name, int h, int w) {
    const auto& c = layers.at(name);
    const auto n = uint32_t(h*w*c.output);
    auto result = backend->allocate(n);
    // The shared 3-D kernel with K=1 is a 1x1 convolution.
    backend->run({c.kernel == 1 ? kConv3d : kConv,uint32_t(h),uint32_t(w),
                   uint32_t(c.input),uint32_t(c.output),n,0,0,1,uint32_t(c.kernel)},
                 x,c.weight,c.bias,result);
    return result;
  }
};
TaeH3::TaeH3(const std::string& path, DeviceBackend backend) : impl_(new Impl(path,backend)) {}
TaeH3::~TaeH3() = default;

std::vector<uint8_t> TaeH3::decode(const std::vector<float>& latents, int frames, int h, int w,
                                  const std::function<bool()>& cancelled) {
  if (frames < 1 || frames > 128 || h < 1 || w < 1 || h > 128 || w > 128 ||
      latents.size() != size_t(24)*frames*h*w)
    throw std::invalid_argument("TAEH3: preview dimensions exceed limits");
  auto& d = *impl_;
  std::vector<Buffer> memory(23);
  std::vector<uint8_t> rgb(size_t(frames)*h*w*12);
  const size_t spatial = size_t(h)*w;
  for (int t = 0; t < frames; ++t) {
    if (cancelled && cancelled()) return {};
    std::vector<float> input(spatial*24);
    for (size_t p = 0; p < spatial; ++p)
      for (int c = 0; c < 24; ++c)
        input[p*24+c] = 3*std::tanh(latents[(size_t(c)*frames+t)*spatial+p]/3);
    auto x = d.conv(d.backend->upload(input),"1",h,w);
    x = d.op(kRelu,x,x,h,w,256,256);
    for (auto [start,channels] : {std::pair{3,256},{9,128},{15,64}}) {
      for (int i = start; i < start+3; ++i) {
        if (cancelled && cancelled()) return {};
        if (!memory[i]) memory[i] = d.backend->upload(std::vector<float>(spatial*channels,0));
        auto branch = d.op(kConcat,x,memory[i],h,w,channels,channels*2);
        memory[i] = x;
        for (int c = 0; c < 3; ++c) {
          branch = d.conv(branch,std::to_string(i)+".conv."+std::to_string(c*2),h,w);
          if (c < 2) branch = d.op(kRelu,branch,branch,h,w,channels,channels);
        }
        x = d.op(kResidual,x,branch,h,w,channels,channels);
        x = d.op(kRelu,x,x,h,w,channels,channels);
      }
      x = d.conv(x,std::to_string(start+4)+".conv",h,w);
      x = d.conv(x,std::to_string(start+5),h,w);
    }
    x = d.op(kRelu,x,x,h,w,64,64);
    const auto pixels = d.backend->download(d.conv(x,"22",h,w),spatial*12);
    // PixelShuffle(2), HWC features -> packed RGB8. Preview playback clamps [0,1].
    for (int y = 0; y < h*2; ++y)
      for (int xx = 0; xx < w*2; ++xx)
        for (int c = 0; c < 3; ++c) {
          const float v = pixels[(size_t(y/2)*w+xx/2)*12+c*4+(y%2)*2+xx%2];
          rgb[((size_t(t)*h*2+y)*w*2+xx)*3+c] =
              std::isfinite(v) ? uint8_t(std::clamp(v,0.f,1.f)*255.f+.5f) : 0;
        }
  }
  return rgb;
}
}
