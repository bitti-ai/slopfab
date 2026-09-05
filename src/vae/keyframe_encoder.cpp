#include "slopfab/vae/keyframe_encoder.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <array>

#include "slopfab/dtype.h"
#include "slopfab/nf4.h"

namespace slopfab::vae {
namespace {

class TorchMT19937 {
 public:
  explicit TorchMT19937(uint64_t seed) {
    state_[0] = static_cast<uint32_t>(seed);
    for (uint32_t i = 1; i < 624; ++i)
      state_[i] = 1812433253u * (state_[i - 1] ^ (state_[i - 1] >> 30)) + i;
  }
  uint32_t next() {
    if (left_-- == 1) twist();
    uint32_t y = state_[next_++];
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    return y ^ (y >> 18);
  }
 private:
  void twist() {
    for (int i = 0; i < 624; ++i) {
      const uint32_t mixed = (state_[i] & 0x80000000u) | (state_[(i + 1) % 624] & 0x7fffffffu);
      state_[i] = state_[(i + 397) % 624] ^ (mixed >> 1) ^
                  ((mixed & 1u) ? 0x9908b0dfu : 0u);
    }
    left_ = 624;
    next_ = 0;
  }
  std::array<uint32_t, 624> state_{};
  int left_ = 1;
  int next_ = 0;
};

void require_tensor(const SafeTensors& ckpt, const std::string& name,
                    std::initializer_list<int64_t> shape, EncoderWeightSummary* summary) {
  const TensorView& tensor = ckpt.at(name);
  const std::vector<int64_t> expected(shape);
  bool valid = false;
  if (is_nf4_weight(ckpt, name)) {
    valid = tensor.dtype == DType::kU8 &&
            read_nf4_state(ckpt, name, "keyframe encoder").shape == expected;
  } else {
    valid = (tensor.dtype == DType::kF16 || tensor.dtype == DType::kBF16) &&
            tensor.shape == expected;
  }
  if (!valid) {
    throw std::runtime_error("keyframe encoder: tensor '" + name + "' has wrong dtype or shape");
  }
  ++summary->tensors;
  summary->bytes += tensor.nbytes;
}

void require_affine(const SafeTensors& ckpt, const std::string& name, int channels,
                    EncoderWeightSummary* summary) {
  require_tensor(ckpt, name + ".weight", {channels}, summary);
  require_tensor(ckpt, name + ".bias", {channels}, summary);
}

void require_conv(const SafeTensors& ckpt, const std::string& name, int out_channels,
                  int in_channels, int kernel, EncoderWeightSummary* summary) {
  require_tensor(ckpt, name + ".weight",
                 {out_channels, in_channels, kernel, kernel, kernel}, summary);
  require_tensor(ckpt, name + ".bias", {out_channels}, summary);
}

}  // namespace

std::vector<float> prepare_keyframe_pixels(const RGBImage& image) {
  if (image.width <= 0 || image.height <= 0 ||
      image.pixels.size() != static_cast<size_t>(image.width) * image.height * 3) {
    throw std::runtime_error("keyframe encoder: invalid RGB image");
  }
  constexpr float mean[3] = {0.485f, 0.456f, 0.406f};
  constexpr float std_dev[3] = {0.229f, 0.224f, 0.225f};
  const size_t plane = static_cast<size_t>(image.width) * image.height;
  std::vector<float> out(3 * plane);
  for (size_t i = 0; i < plane; ++i) {
    for (int c = 0; c < 3; ++c) {
      out[static_cast<size_t>(c) * plane + i] =
          (static_cast<float>(image.pixels[3 * i + c]) / 255.0f - mean[c]) / std_dev[c];
    }
  }
  return out;
}

std::vector<float> sample_keyframe_latents(const float* moments, const float* normal,
                                           int height, int width,
                                           const std::vector<float>& latents_mean,
                                           const std::vector<float>& latents_std) {
  constexpr int channels = 24;
  if (!moments || !normal || height <= 0 || width <= 0 ||
      latents_mean.size() != channels || latents_std.size() != channels) {
    throw std::runtime_error("keyframe encoder: invalid posterior inputs");
  }
  const size_t voxels = static_cast<size_t>(height) * width;
  std::vector<float> out(static_cast<size_t>(channels) * voxels);
  for (int c = 0; c < channels; ++c) {
    if (!(latents_std[c] > 0.0f)) throw std::runtime_error("keyframe encoder: invalid latent std");
    for (size_t i = 0; i < voxels; ++i) {
      const size_t j = static_cast<size_t>(c) * voxels + i;
      // DiagonalGaussianDistribution clamps logvar to [-30,20]. The reference
      // then casts the sampled posterior through fp16 before normalization.
      const float logvar = std::clamp(moments[static_cast<size_t>(channels) * voxels + j],
                                      -30.0f, 20.0f);
      const float sampled = moments[j] + std::exp(0.5f * logvar) * normal[j];
      const float rounded = f16_to_f32(f32_to_f16(sampled));
      out[j] = (rounded - latents_mean[c]) / latents_std[c];
    }
  }
  return out;
}

EncoderWeightSummary validate_keyframe_encoder_weights(const SafeTensors& checkpoint) {
  if (!checkpoint.is_open()) throw std::runtime_error("keyframe encoder: checkpoint is not open");
  EncoderWeightSummary result;
  constexpr int channels[] = {128, 256, 256, 512, 512, 1024};
  constexpr int space_down[] = {2, 2, 2, 2, 1, 1};

  require_conv(checkpoint, "encoder.conv_in", 128, 3, 3, &result);
  int previous = 128;
  for (int level = 0; level < 6; ++level) {
    const int output = channels[level];
    for (int block = 0; block < 2; ++block) {
      const int input = block == 0 ? previous : output;
      const std::string prefix = "encoder.down." + std::to_string(level) + ".block." +
                                 std::to_string(block);
      require_affine(checkpoint, prefix + ".norm1", input, &result);
      require_conv(checkpoint, prefix + ".conv1", output, input, 3, &result);
      require_affine(checkpoint, prefix + ".norm2", output, &result);
      require_conv(checkpoint, prefix + ".conv2", output, output, 3, &result);
      if (input != output) require_conv(checkpoint, prefix + ".nin_shortcut", output, input, 1, &result);
    }
    if (space_down[level] == 2)
      require_conv(checkpoint, "encoder.down." + std::to_string(level) + ".downsample.conv",
                   output, output, 3, &result);
    previous = output;
  }
  require_affine(checkpoint, "encoder.norm_out", 1024, &result);
  require_conv(checkpoint, "encoder.conv_out", 48, 1024, 3, &result);
  require_conv(checkpoint, "quant_conv", 48, 48, 1, &result);
  return result;
}

std::vector<float> patchify_keyframe_latents(const float* latents, int height, int width) {
  if (!latents || height <= 0 || width <= 0 || (height & 1) || (width & 1))
    throw std::runtime_error("keyframe encoder: patchify requires positive even dimensions");
  constexpr int channels = 24;
  const int patch_h = height / 2;
  const int patch_w = width / 2;
  const size_t plane = static_cast<size_t>(height) * width;
  std::vector<float> rows(static_cast<size_t>(patch_h) * patch_w * channels * 4);
  for (int ph = 0; ph < patch_h; ++ph) {
    for (int pw = 0; pw < patch_w; ++pw) {
      const size_t row = static_cast<size_t>(ph) * patch_w + pw;
      for (int c = 0; c < channels; ++c) {
        for (int dy = 0; dy < 2; ++dy) {
          for (int dx = 0; dx < 2; ++dx) {
            const size_t column = static_cast<size_t>(c) * 4 + dy * 2 + dx;
            rows[row * channels * 4 + column] =
                latents[static_cast<size_t>(c) * plane + (ph * 2 + dy) * width + pw * 2 + dx];
          }
        }
      }
    }
  }
  return rows;
}

std::vector<float> torch_cpu_normal_seed42(size_t count) {
  if (count < 16) throw std::runtime_error("keyframe encoder: Torch normal field must have >=16 values");
  TorchMT19937 generator(42);
  std::vector<float> out(count);
  auto uniform = [&] { return static_cast<float>(generator.next() & 0xFFFFFFu) / 16777216.0f; };
  for (float& value : out) value = uniform();
  auto fill16 = [](float* data) {
    constexpr float two_pi = 6.2831853071795864769f;
    for (int j = 0; j < 8; ++j) {
      const float radius = std::sqrt(-2.0f * std::log(1.0f - data[j]));
      const float theta = two_pi * data[j + 8];
      data[j] = radius * std::cos(theta);
      data[j + 8] = radius * std::sin(theta);
    }
  };
  for (size_t i = 0; i + 15 < count; i += 16) fill16(out.data() + i);
  if (count % 16) {
    float* tail = out.data() + count - 16;
    for (int i = 0; i < 16; ++i) tail[i] = uniform();
    fill16(tail);
  }
  return out;
}

}  // namespace slopfab::vae
