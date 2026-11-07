#pragma once
#include <functional>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace slopfab::syncnet {
struct Tensor {
  std::vector<int> shape;
  std::vector<float> data;
};

using Weights = std::map<std::string, Tensor>;
// Hash the same bounded byte buffer that supplies all tensor data. No pickle execution.
Weights load_weights(const std::string& path, bool detector = false);

struct Image {
  int h, w, c;
  std::vector<float> data; // HWC, including audio images and flattened temporal channels
  Image(int height, int width, int channels);
};

class Network {
public:
  Network(Weights weights, const std::string& backend, std::function<void()> check = {});
  ~Network();
  std::string backend() const;
  Image conv(const Image&, const std::string&, int stride = 1, int pad = 0, int dilation = 1,
             bool relu = false);
  void norm(Image&, const std::string&);
  void l2norm(Image&, const std::string&);
  static Image pool(const Image&, int kh, int kw, int sh, int sw, int pad = 0, bool ceil = false);
  std::vector<float> embedding(Image, bool audio);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

struct Box {
  float x1, y1, x2, y2, score;
};

float iou(const Box&, const Box&);
std::vector<Box> detect(Network&, const Image& rgb, float threshold = .9f);
// RGB input in 0..255. Out-of-image pixels use the upstream gray padding.
Image resize_crop(const Image&, float left, float top, float width, float height, int out_h,
                  int out_w);

struct Observation {
  int frame;
  Box box;
};

struct Track {
  int id;
  std::vector<Observation> observations;
};

class Tracker {
public:
  void update(int frame, const std::vector<Box>&, bool scene_cut = false);

  const std::vector<Track>& tracks() const {
    return tracks_;
  }

private:
  std::vector<Track> tracks_;
  std::vector<size_t> active_;
};

std::vector<Box> track_boxes(const Track&); // interpolate gaps, median smooth center/size
std::vector<float> mfcc(const std::vector<int16_t>& pcm); // time-major 13 coefficients

struct Score {
  int offset_frames;
  double distance, confidence, zero_shift_distance;
  bool search_boundary;
  std::vector<double> shift_distances;
};

Score score(const std::vector<std::vector<float>>& video,
            const std::vector<std::vector<float>>& audio, int shift);
} // namespace slopfab::syncnet
