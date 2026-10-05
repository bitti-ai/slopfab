#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace slopfab::seedvr2 {

struct Options {
  std::string transformer;
  std::string vae;
  int width = 1280;
  int height = 720;
  // Jointly restored frames. 4n+1, or 1 for independent images.
  int segment_frames = 5;
  // Spatial VAE tiles in output pixels; 0 disables tiling. Overlap is 64 pixels.
  int vae_tile = 256;
  uint64_t seed = 666;
  int device = 0;
  bool color_match = true;
};

struct Window {
  int t0, t1, y0, y1, x0, x1;
};

void validate(const Options& options);
std::vector<Window> attention_windows(int frames, int height, int width, bool shifted);

// CPU orchestration is deliberately independent of the CUDA model. At most
// one segment and one overlapping output frame are retained, regardless of
// clip duration. Frames are packed RGB float in [0,1], already at target size.
using Frame = std::vector<float>;
using ReadFrame = std::function<bool(Frame&)>;
using WriteFrame = std::function<void(const Frame&)>;
using RestoreSegment = std::function<std::vector<Frame>(const std::vector<Frame>&, uint64_t)>;
uint64_t stream(const Options&, const ReadFrame&, const WriteFrame&, const RestoreSegment&);

// Native CUDA inference; no Python/ComfyUI runtime. Weights are memory mapped
// and uploaded one block at a time. Instances are not thread safe.
class Restorer {
public:
  explicit Restorer(const Options&);
  ~Restorer();
  Restorer(const Restorer&) = delete;
  Restorer& operator=(const Restorer&) = delete;
  std::vector<Frame> restore(const std::vector<Frame>& frames, uint64_t first_frame = 0);
  std::function<void(const std::string&)> progress;
  std::function<bool()> cancelled;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace slopfab::seedvr2
