#pragma once
#include "slopfab/seedvr2.h"
#include "../moss/vulkan_runtime.h"

namespace slopfab::seedvr2_vulkan {
namespace vk = slopfab::vulkan;
using seedvr2::Window;
struct Tensor {
  int t = 1, h = 1, w = 1, c = 1;
  music_gpu::Tensor values;
  Tensor() = default;
  Tensor(int tt, int hh, int ww, int cc) : t(tt), h(hh), w(ww), c(cc), values(tt*hh*ww, cc) {}
  int rows() const { return t*h*w; }
  size_t size() const { return values.engine ? size_t(rows())*c : 0; }
  Tensor rows_view(int offset, int count) const;
};
enum Operation {
  Patches = 31, GroupNorm, Modulate, PackWindow, UnpackWindow, PoolText, Upsample, Bias
};
class Runtime : public moss_vulkan::Runtime {
public:
  using moss_vulkan::Runtime::Runtime;
  SafeTensors* file = nullptr;
  std::map<std::string, vk::DeviceTensor> weights;
  Tensor tensor(int t, int h, int w, int c) { return Tensor(t,h,w,c); }
  void clear(SafeTensors& f);
  void end_segment();
  vk::DeviceTensor& weight(const std::string&, int rows, int cols, bool vector = false);
  Tensor upload(const std::vector<float>&, int, int, int, int);
  std::vector<float> download(const Tensor&);
  void copy_rows(const Tensor&, Tensor&, int source, int destination, int count);
  Tensor linear(const Tensor&, const std::string&);
  Tensor conv(const Tensor&, const std::string&, bool down = false, int temporal_stride = 1);
  Tensor groupnorm(const Tensor&, const std::string&, bool silu = true);
  Tensor rms(const Tensor&, const std::string& = "");
  void activation(Tensor&, const Tensor* gate = nullptr);
  void add(Tensor&, const Tensor&);
  void modulate(Tensor&, const Tensor&, const std::string&, int layer, bool gate);
  void mlp(Tensor&, const Tensor&, const std::string&, const std::string&);
  Tensor upsample(const Tensor&, int temporal_ratio);
  Tensor attention(const Tensor&, const Tensor&, const Tensor&, int heads, int dim);
  void window_attention(const Tensor&, const Tensor&, Tensor&, Tensor&, const std::string&,
                        const std::string&, const std::string&, const std::vector<Window>&);
};
}
