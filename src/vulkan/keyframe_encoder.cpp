#include "slopfab/vulkan/keyframe_encoder.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

#include "slopfab/vae/keyframe_encoder.h"
#include "slopfab/vulkan/tensor.h"

namespace slopfab::vulkan {
namespace {

constexpr uint32_t kChannels[] = {128, 256, 256, 512, 512, 1024};
constexpr uint32_t kDownsample[] = {2, 2, 2, 2, 1, 1};
constexpr uint32_t kOperators = 72;

TensorLayout vector(uint64_t count) {
  return TensorLayout::contiguous(&count, 1);
}

struct WeightSpec {
  std::string name;
  std::vector<uint64_t> shape;
};

void append_conv(std::vector<WeightSpec>& out, const std::string& name,
                 uint32_t output, uint32_t input, uint32_t kernel) {
  out.push_back({name + ".weight", {output, input, kernel, kernel, kernel}});
  out.push_back({name + ".bias", {output}});
}
void append_norm(std::vector<WeightSpec>& out, const std::string& name,
                 uint32_t channels) {
  out.push_back({name + ".weight", {channels}});
  out.push_back({name + ".bias", {channels}});
}

std::vector<WeightSpec> weight_specs() {
  std::vector<WeightSpec> out;
  out.reserve(118);
  append_conv(out, "encoder.conv_in", 128, 3, 3);
  uint32_t previous = 128;
  for (uint32_t level = 0; level < 6; ++level) {
    const uint32_t output = kChannels[level];
    for (uint32_t block = 0; block < 2; ++block) {
      const uint32_t input = block == 0 ? previous : output;
      const std::string p = "encoder.down." + std::to_string(level) +
          ".block." + std::to_string(block);
      append_norm(out, p + ".norm1", input);
      append_conv(out, p + ".conv1", output, input, 3);
      append_norm(out, p + ".norm2", output);
      append_conv(out, p + ".conv2", output, output, 3);
      if (input != output)
        append_conv(out, p + ".nin_shortcut", output, input, 1);
    }
    if (kDownsample[level] == 2) {
      append_conv(out, "encoder.down." + std::to_string(level) +
                          ".downsample.conv", output, output, 3);
    }
    previous = output;
  }
  append_norm(out, "encoder.norm_out", 1024);
  append_conv(out, "encoder.conv_out", 48, 1024, 3);
  append_conv(out, "quant_conv", 48, 48, 1);
  if (out.size() != 118) {
    throw std::logic_error("Vulkan keyframe: internal weight manifest mismatch");
  }
  return out;
}

bool encoder_tensor_name(const std::string& name) {
  return name.rfind("encoder.", 0) == 0 || name == "quant_conv.weight" ||
      name == "quant_conv.bias";
}

void validate_archive(const SafeTensors& checkpoint,
                      const std::vector<WeightSpec>& specs) {
  vae::validate_keyframe_encoder_weights(checkpoint);
  std::set<std::string> expected;
  for (const WeightSpec& spec : specs) {
    expected.insert(spec.name);
    const TensorView& view = checkpoint.at(spec.name);
    std::vector<int64_t> shape;
    shape.reserve(spec.shape.size());
    for (uint64_t extent : spec.shape) {
      if (extent > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
        throw std::runtime_error("Vulkan keyframe: tensor shape overflow");
      shape.push_back(static_cast<int64_t>(extent));
    }
    uint64_t elements = 1;
    for (uint64_t extent : spec.shape) {
      if (extent == 0 || elements > std::numeric_limits<uint64_t>::max() / extent)
        throw std::runtime_error("Vulkan keyframe: tensor shape overflow");
      elements *= extent;
    }
    if (view.dtype != DType::kF16 || view.shape != shape ||
        elements > std::numeric_limits<size_t>::max() / 2 ||
        view.nbytes != static_cast<size_t>(elements * 2)) {
      throw std::runtime_error(
          "Vulkan keyframe: invalid exact fp16 tensor '" + spec.name + "'");
    }
  }
  size_t observed = 0;
  for (const auto& entry : checkpoint.tensors()) {
    if (!encoder_tensor_name(entry.first)) continue;
    ++observed;
    if (expected.count(entry.first) == 0) {
      throw std::runtime_error(
          "Vulkan keyframe: unexpected encoder tensor '" + entry.first + "'");
    }
  }
  if (observed != specs.size()) {
    throw std::runtime_error("Vulkan keyframe: incomplete exact encoder manifest");
  }
}

}  // namespace

struct KeyframeEncoder::Impl {
  struct Pair { DeviceTensor weight, bias; };
  struct Shape {
    uint32_t height = 0, width = 0;
    uint64_t capacity = 0;
    DeviceTensor input, moments;
    DeviceTensor arena[3];
  };

  TensorContext context;
  std::map<std::string, Pair> weights;
  std::unique_ptr<Shape> shape;
  bool is_loaded = false;
  KeyframeEncoderStats statistics;

  explicit Impl(const Device& device) : context(device, [] {
    TensorContextOptions options;
    options.max_batch_operators = 128;
    return options;
  }()) {
    context.require_exact_fp32_vae_normalization();
    context.require_exact_vae_pointwise();
  }

  Pair& pair(const std::string& name) { return weights.at(name); }

  Shape& prepare_shape(uint32_t height, uint32_t width) {
    if (shape && shape->height == height && shape->width == width) return *shape;
    const uint64_t plane = static_cast<uint64_t>(height) * width;
    if (height == 0 || width == 0 || height % 16 || width % 16 ||
        plane > std::numeric_limits<uint64_t>::max() / 128 ||
        128 * plane > std::numeric_limits<uint32_t>::max()) {
      throw std::invalid_argument(
          "Vulkan keyframe: dimensions exceed exact flat-arena contract");
    }
    // Never overlap two three-arena shapes. This is deliberately a bounded
    // one-shape cache: reference encode completes synchronously, so no command
    // can still reference the old shape here.
    shape.reset();
    context.collect();
    auto next = std::make_unique<Shape>();
    next->height = height;
    next->width = width;
    next->capacity = 128 * plane;
    next->input = context.allocate(vector(3 * plane));
    const uint64_t moments = 48 * (height / 16ull) * (width / 16ull);
    next->moments = context.allocate(vector(moments));
    for (DeviceTensor& arena : next->arena)
      arena = context.allocate(vector(next->capacity));
    shape = std::move(next);
    return *shape;
  }

  void conv(TensorBatch& batch, DeviceTensor& input, DeviceTensor& output,
            const std::string& name, uint32_t cin, uint32_t cout,
            uint32_t height, uint32_t width, uint32_t kernel,
            uint32_t stride = 1, bool asymmetric = false) {
    Pair& value = pair(name);
    batch.keyframe_conv3d_f16(input, value.weight, value.bias, output,
                              cin, cout, height, width, kernel, stride,
                              !asymmetric, asymmetric);
  }
  void norm(TensorBatch& batch, DeviceTensor& input, DeviceTensor& output,
            const std::string& name, uint32_t channels, uint32_t height,
            uint32_t width) {
    Pair& value = pair(name);
    batch.keyframe_group_norm_silu_f16_affine(
        input, value.weight, value.bias, output, channels, height, width,
        32, 1.0e-6f);
  }
};

KeyframeEncoder::KeyframeEncoder() = default;
KeyframeEncoder::~KeyframeEncoder() = default;
KeyframeEncoder::KeyframeEncoder(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
KeyframeEncoder::KeyframeEncoder(KeyframeEncoder&&) noexcept = default;
KeyframeEncoder& KeyframeEncoder::operator=(KeyframeEncoder&&) noexcept = default;

KeyframeEncoder KeyframeEncoder::create(const Device& device) {
  return KeyframeEncoder(std::make_unique<Impl>(device));
}

void KeyframeEncoder::load(const SafeTensors& checkpoint) {
  if (!impl_) throw std::logic_error("Vulkan keyframe: empty encoder");
  if (impl_->is_loaded) {
    throw std::logic_error(
        "Vulkan keyframe: unload before replacing active checkpoint weights");
  }
  const std::vector<WeightSpec> specs = weight_specs();
  validate_archive(checkpoint, specs);
  std::map<std::string, Impl::Pair> next;
  uint64_t persistent = 0;
  for (size_t at = 0; at < specs.size(); at += 2) {
    const WeightSpec& matrix_spec = specs[at];
    const WeightSpec& bias_spec = specs[at + 1];
    const size_t suffix = matrix_spec.name.rfind(".weight");
    if (suffix == std::string::npos ||
        bias_spec.name != matrix_spec.name.substr(0, suffix) + ".bias") {
      throw std::logic_error("Vulkan keyframe: invalid paired manifest");
    }
    TensorLayout matrix_layout = TensorLayout::contiguous(
        matrix_spec.shape.data(), static_cast<uint32_t>(matrix_spec.shape.size()));
    TensorLayout bias_layout = TensorLayout::contiguous(
        bias_spec.shape.data(), static_cast<uint32_t>(bias_spec.shape.size()));
    Impl::Pair pair;
    pair.weight = impl_->context.allocate(matrix_layout, ScalarType::kFloat16);
    pair.bias = impl_->context.allocate(bias_layout, ScalarType::kFloat16);
    const TensorView& matrix_view = checkpoint.at(matrix_spec.name);
    const TensorView& bias_view = checkpoint.at(bias_spec.name);
    impl_->context.upload_transient_bytes(
        pair.weight, matrix_view.data, matrix_view.nbytes);
    impl_->context.upload_transient_bytes(pair.bias, bias_view.data, bias_view.nbytes);
    persistent += matrix_view.nbytes + bias_view.nbytes;
    next.emplace(matrix_spec.name.substr(0, suffix), std::move(pair));
  }
  impl_->weights = std::move(next);
  impl_->shape.reset();
  impl_->is_loaded = true;
  impl_->statistics = {};
  impl_->statistics.persistent_bytes = persistent;
}

void KeyframeEncoder::unload() noexcept {
  if (!impl_) return;
  impl_->shape.reset();
  impl_->weights.clear();
  impl_->is_loaded = false;
  impl_->statistics = {};
  try { impl_->context.collect(); } catch (...) {}
}

bool KeyframeEncoder::loaded() const noexcept {
  return impl_ && impl_->is_loaded;
}

std::vector<float> KeyframeEncoder::encode_moments(
    const float* pixels, int height, int width) {
  if (!impl_ || !impl_->is_loaded || pixels == nullptr || height <= 0 ||
      width <= 0) {
    throw std::invalid_argument("Vulkan keyframe: invalid encode request");
  }
  Impl::Shape& shape = impl_->prepare_shape(
      static_cast<uint32_t>(height), static_cast<uint32_t>(width));
  const uint64_t plane = static_cast<uint64_t>(height) * width;
  impl_->context.upload_transient(shape.input, pixels, 3 * plane);
  const auto start = std::chrono::steady_clock::now();
  TensorBatch batch = impl_->context.begin_batch();
  batch.require_operator_capacity(kOperators);
  uint32_t h = static_cast<uint32_t>(height);
  uint32_t w = static_cast<uint32_t>(width);
  uint32_t current = 128;
  uint32_t state = 0;
  impl_->conv(batch, shape.input, shape.arena[state], "encoder.conv_in",
              3, 128, h, w, 3);
  for (uint32_t level = 0; level < 6; ++level) {
    const uint32_t output = kChannels[level];
    for (uint32_t block = 0; block < 2; ++block) {
      const uint32_t input_channels = current;
      const std::string p = "encoder.down." + std::to_string(level) +
          ".block." + std::to_string(block);
      const uint32_t first = (state + 1) % 3;
      const uint32_t second = (state + 2) % 3;
      if (input_channels != output) {
        impl_->conv(batch, shape.arena[state], shape.arena[first],
                    p + ".nin_shortcut", input_channels, output, h, w, 1);
        impl_->norm(batch, shape.arena[state], shape.arena[second],
                    p + ".norm1", input_channels, h, w);
        impl_->conv(batch, shape.arena[second], shape.arena[state],
                    p + ".conv1", input_channels, output, h, w, 3);
        impl_->norm(batch, shape.arena[state], shape.arena[second],
                    p + ".norm2", output, h, w);
        impl_->conv(batch, shape.arena[second], shape.arena[state],
                    p + ".conv2", output, output, h, w, 3);
        batch.keyframe_add_f32(shape.arena[first], shape.arena[state],
                               shape.arena[second],
                               static_cast<uint64_t>(output) * h * w);
        state = second;
      } else {
        impl_->norm(batch, shape.arena[state], shape.arena[first],
                    p + ".norm1", input_channels, h, w);
        impl_->conv(batch, shape.arena[first], shape.arena[second],
                    p + ".conv1", input_channels, output, h, w, 3);
        impl_->norm(batch, shape.arena[second], shape.arena[first],
                    p + ".norm2", output, h, w);
        impl_->conv(batch, shape.arena[first], shape.arena[second],
                    p + ".conv2", output, output, h, w, 3);
        batch.keyframe_add_f32(shape.arena[state], shape.arena[second],
                               shape.arena[first],
                               static_cast<uint64_t>(output) * h * w);
        state = first;
      }
      current = output;
    }
    if (kDownsample[level] == 2) {
      const uint32_t destination = (state + 1) % 3;
      impl_->conv(batch, shape.arena[state], shape.arena[destination],
                  "encoder.down." + std::to_string(level) + ".downsample.conv",
                  current, current, h, w, 3, 2, true);
      state = destination;
      h /= 2;
      w /= 2;
    }
  }
  uint32_t destination = (state + 1) % 3;
  impl_->norm(batch, shape.arena[state], shape.arena[destination],
              "encoder.norm_out", 1024, h, w);
  state = destination;
  destination = (state + 1) % 3;
  impl_->conv(batch, shape.arena[state], shape.arena[destination],
              "encoder.conv_out", 1024, 48, h, w, 3);
  state = destination;
  destination = (state + 1) % 3;
  impl_->conv(batch, shape.arena[state], shape.arena[destination],
              "quant_conv", 48, 48, h, w, 1);
  state = destination;
  batch.audio_copy_prefix(shape.arena[state], shape.moments,
                          static_cast<uint64_t>(48) * h * w);
  batch.submit().wait();
  std::vector<float> result(static_cast<size_t>(48) * h * w);
  impl_->context.download(shape.moments, result.data(), result.size());
  impl_->context.collect();
  const auto stop = std::chrono::steady_clock::now();
  impl_->statistics.last_encode_seconds =
      std::chrono::duration<double>(stop - start).count();
  impl_->statistics.activation_bytes =
      (3 * shape.capacity + 3 * plane + result.size()) * sizeof(float);
  impl_->statistics.allocator_peak_used_bytes =
      std::max(impl_->statistics.allocator_peak_used_bytes,
               impl_->context.pooled_used_bytes());
  impl_->statistics.allocator_used_bytes = impl_->context.pooled_used_bytes();
  impl_->statistics.allocator_reserved_bytes = impl_->context.reserved_bytes();
  impl_->statistics.descriptor_set_allocations =
      impl_->context.descriptor_set_allocations();
  impl_->statistics.operators = kOperators;
  return result;
}

std::vector<float> KeyframeEncoder::encode_condition_rows(
    const RGBImage& image, const float* normal,
    const std::vector<float>& latents_mean,
    const std::vector<float>& latents_std) {
  if (!normal) throw std::invalid_argument("Vulkan keyframe: missing posterior normal");
  const std::vector<float> pixels = vae::prepare_keyframe_pixels(image);
  const std::vector<float> moments =
      encode_moments(pixels.data(), image.height, image.width);
  const int latent_h = image.height / 16;
  const int latent_w = image.width / 16;
  const std::vector<float> latents = vae::sample_keyframe_latents(
      moments.data(), normal, latent_h, latent_w, latents_mean, latents_std);
  return vae::patchify_keyframe_latents(latents.data(), latent_h, latent_w);
}

std::vector<float> KeyframeEncoder::encode_reference_image(
    const RGBImage& image, const std::vector<float>& latents_mean,
    const std::vector<float>& latents_std) {
  if (image.height <= 0 || image.width <= 0 || image.height % 16 ||
      image.width % 16) {
    throw std::invalid_argument(
        "Vulkan keyframe: reference dimensions must be positive multiples of 16");
  }
  const size_t count = static_cast<size_t>(24) * (image.height / 16) *
      (image.width / 16);
  const std::vector<float> normal = vae::torch_cpu_normal_seed42(count);
  return encode_condition_rows(image, normal.data(), latents_mean, latents_std);
}

const KeyframeEncoderStats& KeyframeEncoder::stats() const noexcept {
  static const KeyframeEncoderStats empty;
  return impl_ ? impl_->statistics : empty;
}

}  // namespace slopfab::vulkan
