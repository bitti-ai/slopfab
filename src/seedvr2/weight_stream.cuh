#pragma once
#include "slopfab/safetensors.h"
#include <cuda_bf16.h>
#include <memory>
#include <string>

namespace slopfab::seedvr2 {
// Two reusable DiT block slots. All consuming kernels must use the default
// stream; begin_block records their completion before a slot can be reused.
// SafeTensors must outlive the streamer. Methods run on its creating device.
class WeightStream {
public:
  struct View {
    __nv_bfloat16* data = nullptr;
    size_t count = 0;
  };

  static std::unique_ptr<WeightStream> create(SafeTensors& file, int device, size_t budget);
  ~WeightStream();
  WeightStream(const WeightStream&) = delete;
  WeightStream& operator=(const WeightStream&) = delete;
  void begin();
  void begin_block(int block);
  void finish();
  View find(const std::string& name) const;
  size_t device_bytes() const;

private:
  struct Impl;
  explicit WeightStream(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};
} // namespace slopfab::seedvr2
