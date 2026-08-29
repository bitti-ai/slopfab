#include "vidfab/vulkan/audio_decoder.h"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "vidfab/vae/audio_primitives.h"
#include "vidfab/vulkan/tensor.h"

namespace vidfab::vulkan {
namespace {

constexpr uint32_t kGraphOperators = 497;
constexpr uint32_t kGraphCapacity = 512;
constexpr int kStereo = 2;
constexpr int kDilations[3] = {1, 3, 5};

TensorLayout layout(std::initializer_list<uint64_t> extents) {
  std::vector<uint64_t> shape(extents);
  return TensorLayout::contiguous(shape.data(),
                                  static_cast<uint32_t>(shape.size()));
}

int padding(int kernel, int dilation) {
  return (kernel * dilation - dilation) / 2;
}

void validate_config(const vae::AudioVAEConfig& config) {
  if (config.latent_channels <= 0 || config.latent_dim <= 0 ||
      config.decoder_dim <= 0 || config.output_channels != kStereo ||
      config.sample_rate <= 0 || config.decoder_rates.size() != 7 ||
      config.decoder_kernel_sizes.size() != config.decoder_rates.size() ||
      config.resblock_kernel_sizes != std::vector<int>({3, 7, 11})) {
    throw std::invalid_argument("Vulkan audio VAE: unsupported configuration");
  }
  int channels = config.decoder_dim;
  for (size_t i = 0; i < config.decoder_rates.size(); ++i) {
    const int rate = config.decoder_rates[i];
    const int expected_kernel = 2 * rate - (rate % 2);
    if (rate <= 0 || config.decoder_kernel_sizes[i] != expected_kernel ||
        (channels & 1) != 0) {
      throw std::invalid_argument("Vulkan audio VAE: invalid decoder stages");
    }
    channels /= 2;
  }
  if (channels != 8 || config.total_upsample() != 800) {
    throw std::invalid_argument("Vulkan audio VAE: unsupported output geometry");
  }
}

uint64_t bytes(const DeviceTensor& tensor) {
  return tensor ? tensor.layout().bytes(tensor.type()) : 0;
}

struct Conv {
  DeviceTensor weight;
  DeviceTensor bias;
  uint32_t out_channels = 0;
  uint32_t in_channels = 0;
  uint32_t kernel = 0;
};

struct Activation {
  DeviceTensor log_alpha, log_beta, up_filter, down_filter;
  uint32_t channels = 0;
};

struct AmpBlock {
  uint32_t channels = 0;
  uint32_t kernel = 0;
  std::array<Conv, 3> convs1;
  std::array<Conv, 3> convs2;
  std::array<Activation, 6> acts;
};

struct Stage {
  Conv up;
  uint32_t rate = 0;
  uint32_t kernel = 0;
  std::array<AmpBlock, 3> blocks;
};

struct Weights {
  Conv dec_in_proj;
  Conv conv_pre;
  std::vector<Stage> stages;
  Activation activation_post;
  Conv conv_post;
  uint64_t bytes = 0;
  uint32_t tensors = 0;
};

std::string indexed(const std::string& prefix, int index,
                    const std::string& suffix) {
  return prefix + std::to_string(index) + suffix;
}

}  // namespace

struct AudioDecoder::Impl {
  vae::AudioVAEConfig config;
  TensorContext context;
  Weights weights;
  std::vector<float> mean;
  std::vector<float> std_dev;
  bool loaded = false;

  std::array<DeviceTensor, 6> arena;
  DeviceTensor aa_scratch;
  uint64_t arena_elements = 0;
  uint64_t last_transient_bytes = 0;

  explicit Impl(const Device& device)
      : context(device, [] {
          TensorContextOptions options;
          options.max_batch_operators = kGraphCapacity;
          return options;
        }()) {
    context.require_exact_audio_vae_primitives();
  }

  DeviceTensor upload(const std::vector<float>& host,
                      std::initializer_list<uint64_t> extents) {
    DeviceTensor result = context.allocate(layout(extents));
    context.upload(result, host.data(), host.size());
    return result;
  }

  Conv load_conv(const SafeTensors& checkpoint, const std::string& name,
                 uint32_t out_channels, uint32_t in_channels, uint32_t kernel,
                 bool transpose, bool require_bias, Weights& destination) {
    const std::vector<int64_t> shape = transpose
        ? std::vector<int64_t>{in_channels, out_channels, kernel}
        : std::vector<int64_t>{out_channels, in_channels, kernel};
    vae::AudioConvWeights host = vae::load_audio_conv_weights(
        checkpoint, name, shape, require_bias ? out_channels : 0, require_bias);
    Conv result;
    result.out_channels = out_channels;
    result.in_channels = in_channels;
    result.kernel = kernel;
    result.weight = transpose
        ? upload(host.weight, {in_channels, out_channels, kernel})
        : upload(host.weight, {out_channels, in_channels, kernel});
    destination.bytes += host.weight.size() * sizeof(float);
    destination.tensors += host.folded_weight_norm ? 2u : 1u;
    if (!host.bias.empty()) {
      result.bias = upload(host.bias, {out_channels});
      destination.bytes += host.bias.size() * sizeof(float);
      ++destination.tensors;
    }
    return result;
  }

  Activation load_activation(const SafeTensors& checkpoint,
                             const std::string& prefix, uint32_t channels,
                             Weights& destination) {
    Activation result;
    result.channels = channels;
    auto load = [&](const std::string& name,
                    const std::vector<int64_t>& shape,
                    std::initializer_list<uint64_t> extents) {
      std::vector<float> host = vae::load_audio_f32_tensor(
          checkpoint, prefix + name, shape);
      destination.bytes += host.size() * sizeof(float);
      ++destination.tensors;
      return upload(host, extents);
    };
    result.log_alpha = load("act.alpha", {channels}, {channels});
    result.log_beta = load("act.beta", {channels}, {channels});
    result.up_filter = load("upsample.filter", {1, 1, 12}, {12});
    result.down_filter = load("downsample.lowpass.filter", {1, 1, 12}, {12});
    return result;
  }

  void ensure_arena(int num_latents) {
    uint64_t length = static_cast<uint64_t>(num_latents);
    uint64_t channels = static_cast<uint64_t>(config.decoder_dim);
    uint64_t widest = static_cast<uint64_t>(config.latent_dim) * length;
    widest = std::max(widest, channels * length);
    for (const Stage& stage : weights.stages) {
      if (length > std::numeric_limits<uint64_t>::max() / stage.rate)
        throw std::overflow_error("Vulkan audio VAE: length overflow");
      length *= stage.rate;
      channels /= 2;
      widest = std::max(widest, channels * length);
    }
    if (widest > std::numeric_limits<uint64_t>::max() / kStereo)
      throw std::overflow_error("Vulkan audio VAE: arena overflow");
    widest *= kStereo;
    if (widest > std::numeric_limits<uint32_t>::max())
      throw std::out_of_range("Vulkan audio VAE: activation exceeds exact indexing");
    if (widest <= arena_elements) return;
    std::array<DeviceTensor, 6> replacement;
    for (DeviceTensor& tensor : replacement)
      tensor = context.allocate(layout({widest}));
    DeviceTensor replacement_scratch = context.allocate(layout({widest * 2}));
    arena = std::move(replacement);
    aa_scratch = std::move(replacement_scratch);
    arena_elements = widest;
  }

  void record_conv(TensorBatch& batch, Conv& conv, DeviceTensor& input,
                   DeviceTensor& output, uint32_t batch_size,
                   uint32_t length_in, uint32_t length_out, uint32_t pad,
                   uint32_t dilation) {
    vae::AudioConv1DDesc desc{batch_size, conv.in_channels,
        conv.out_channels, length_in, length_out, conv.kernel, pad, dilation};
    batch.audio_conv1d(input, conv.weight, conv.bias ? &conv.bias : nullptr,
                       output, desc);
  }

  void record_activation(TensorBatch& batch, Activation& activation,
                         DeviceTensor& input, DeviceTensor& output,
                         uint32_t batch_size, uint32_t length) {
    batch.audio_aa_upsample_snake(
        input, activation.up_filter, activation.log_alpha,
        activation.log_beta, aa_scratch, batch_size, activation.channels,
        length);
    batch.audio_aa_downsample(
        aa_scratch, activation.down_filter, output, batch_size,
        activation.channels, length * 2, length);
  }

  void record_block(TensorBatch& batch, AmpBlock& block, DeviceTensor& x,
                    uint32_t batch_size, uint32_t length, DeviceTensor& t1,
                    DeviceTensor& t2) {
    const uint64_t count = static_cast<uint64_t>(batch_size) * block.channels * length;
    for (int index = 0; index < 3; ++index) {
      const uint32_t dilation = kDilations[index];
      record_activation(batch, block.acts[2 * index], x, t1, batch_size, length);
      record_conv(batch, block.convs1[index], t1, t2, batch_size, length,
                  length, padding(block.kernel, dilation), dilation);
      record_activation(batch, block.acts[2 * index + 1], t2, t1,
                        batch_size, length);
      record_conv(batch, block.convs2[index], t1, t2, batch_size, length,
                  length, padding(block.kernel, 1), 1);
      batch.audio_add_inplace(x, t2, count);
    }
  }

  uint64_t activation_bytes() const noexcept {
    return arena_elements * sizeof(float) * 8;
  }
};

AudioDecoder::AudioDecoder() = default;
AudioDecoder::~AudioDecoder() = default;
AudioDecoder::AudioDecoder(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
AudioDecoder::AudioDecoder(AudioDecoder&&) noexcept = default;
AudioDecoder& AudioDecoder::operator=(AudioDecoder&&) noexcept = default;

AudioDecoder AudioDecoder::create(const Device& device) {
  return AudioDecoder(std::make_unique<Impl>(device));
}

void AudioDecoder::load(const SafeTensors& checkpoint,
                        const vae::AudioVAEConfig& config) {
  if (!impl_) throw std::logic_error("Vulkan audio VAE: empty decoder");
  if (!checkpoint.is_open())
    throw std::invalid_argument("Vulkan audio VAE: checkpoint is not open");
  validate_config(config);
  Impl& d = *impl_;
  d.loaded = false;
  Weights next;
  next.dec_in_proj = d.load_conv(checkpoint, "dec_in_proj", config.latent_dim,
                                 config.latent_channels, 1, false, true, next);
  next.conv_pre = d.load_conv(checkpoint, "decoder.conv_pre", config.decoder_dim,
                              config.latent_dim, 7, false, true, next);
  uint32_t channels = static_cast<uint32_t>(config.decoder_dim);
  next.stages.reserve(config.decoder_rates.size());
  for (size_t stage_index = 0; stage_index < config.decoder_rates.size(); ++stage_index) {
    Stage stage;
    stage.rate = static_cast<uint32_t>(config.decoder_rates[stage_index]);
    stage.kernel = static_cast<uint32_t>(config.decoder_kernel_sizes[stage_index]);
    const uint32_t output_channels = channels / 2;
    const std::string up = "decoder.ups." + std::to_string(stage_index) + ".0";
    stage.up = d.load_conv(checkpoint, up, output_channels, channels,
                           stage.kernel, true, true, next);
    for (int block_index = 0; block_index < 3; ++block_index) {
      const int flat_index = static_cast<int>(stage_index) * 3 + block_index;
      const std::string prefix =
          "decoder.resblocks." + std::to_string(flat_index) + ".";
      AmpBlock& block = stage.blocks[block_index];
      block.channels = output_channels;
      block.kernel = static_cast<uint32_t>(config.resblock_kernel_sizes[block_index]);
      for (int dilation_index = 0; dilation_index < 3; ++dilation_index) {
        block.convs1[dilation_index] = d.load_conv(
            checkpoint, indexed(prefix + "convs1.", dilation_index, ""),
            output_channels, output_channels, block.kernel, false, true, next);
        block.convs2[dilation_index] = d.load_conv(
            checkpoint, indexed(prefix + "convs2.", dilation_index, ""),
            output_channels, output_channels, block.kernel, false, true, next);
      }
      for (int activation_index = 0; activation_index < 6; ++activation_index) {
        block.acts[activation_index] = d.load_activation(
            checkpoint,
            indexed(prefix + "activations.", activation_index, "."),
            output_channels, next);
      }
    }
    next.stages.push_back(std::move(stage));
    channels = output_channels;
  }
  next.activation_post = d.load_activation(
      checkpoint, "decoder.activation_post.", channels, next);
  if (checkpoint.find("decoder.conv_post.bias") != nullptr) {
    throw std::runtime_error(
        "Vulkan audio VAE: unexpected decoder.conv_post.bias");
  }
  next.conv_post = d.load_conv(checkpoint, "decoder.conv_post", 1, channels,
                               7, false, false, next);
  if (next.tensors != 779)
    throw std::runtime_error("Vulkan audio VAE: decoder tensor count is not 779");

  std::vector<float> next_mean = vae::load_audio_f32_tensor(
      checkpoint, "latents_mean", {config.latent_channels});
  std::vector<float> next_std = vae::load_audio_f32_tensor(
      checkpoint, "latents_std", {config.latent_channels});
  d.config = config;
  d.weights = std::move(next);
  d.mean = std::move(next_mean);
  d.std_dev = std::move(next_std);
  d.loaded = true;
}

void AudioDecoder::unload() {
  if (!impl_) return;
  impl_->loaded = false;
  impl_->weights = Weights{};
  impl_->mean.clear();
  impl_->std_dev.clear();
  impl_->arena = {};
  impl_->aa_scratch = {};
  impl_->arena_elements = 0;
  impl_->last_transient_bytes = 0;
}

const vae::AudioVAEConfig& AudioDecoder::config() const {
  if (!impl_) throw std::logic_error("Vulkan audio VAE: empty decoder");
  return impl_->config;
}
const std::vector<float>& AudioDecoder::latents_mean() const {
  if (!impl_) throw std::logic_error("Vulkan audio VAE: empty decoder");
  return impl_->mean;
}
const std::vector<float>& AudioDecoder::latents_std() const {
  if (!impl_) throw std::logic_error("Vulkan audio VAE: empty decoder");
  return impl_->std_dev;
}

vae::DecodedAudio AudioDecoder::decode(const float* latents, int num_latents) {
  if (!impl_ || !impl_->loaded)
    throw std::logic_error("Vulkan audio VAE: decode before load");
  if (latents == nullptr || num_latents <= 0)
    throw std::invalid_argument("Vulkan audio VAE: invalid latent input");
  Impl& d = *impl_;
  d.ensure_arena(num_latents);
  const uint32_t batch_size = static_cast<uint32_t>(d.config.output_channels);
  const uint32_t latent_length = static_cast<uint32_t>(num_latents);
  const uint64_t latent_count = static_cast<uint64_t>(batch_size) *
      d.config.latent_channels * latent_length;
  DeviceTensor input = d.context.allocate(layout(
      {batch_size, static_cast<uint32_t>(d.config.latent_channels), latent_length}));
  const uint32_t output_length = latent_length *
      static_cast<uint32_t>(d.config.total_upsample());
  DeviceTensor interleaved = d.context.allocate(
      layout({output_length, batch_size}));
  d.last_transient_bytes = (latent_count +
      static_cast<uint64_t>(output_length) * batch_size) * sizeof(float);
  d.context.upload(input, latents, latent_count);

  TensorBatch commands = d.context.begin_batch();
  DeviceTensor* a = &d.arena[0];
  DeviceTensor* b = &d.arena[1];
  DeviceTensor* accumulator = &d.arena[2];
  DeviceTensor* work = &d.arena[3];
  DeviceTensor* t1 = &d.arena[4];
  DeviceTensor* t2 = &d.arena[5];
  d.record_conv(commands, d.weights.dec_in_proj, input, *a, batch_size,
                latent_length, latent_length, 0, 1);
  d.record_conv(commands, d.weights.conv_pre, *a, *b, batch_size,
                latent_length, latent_length, 3, 1);
  DeviceTensor* current = b;
  DeviceTensor* spare = a;
  uint32_t length = latent_length;
  for (Stage& stage : d.weights.stages) {
    const uint32_t length_out = length * stage.rate;
    vae::AudioConvTranspose1DDesc desc{
        batch_size, stage.up.in_channels, stage.up.out_channels, length,
        length_out, stage.kernel, stage.rate,
        (stage.kernel - stage.rate) / 2};
    commands.audio_conv_transpose1d(
        *current, stage.up.weight, &stage.up.bias, *spare, desc);
    std::swap(current, spare);
    length = length_out;
    const uint64_t count = static_cast<uint64_t>(batch_size) *
        stage.up.out_channels * length;
    for (int block_index = 0; block_index < 3; ++block_index) {
      DeviceTensor* destination = block_index == 0 ? accumulator : work;
      commands.audio_copy_prefix(*current, *destination, count);
      d.record_block(commands, stage.blocks[block_index], *destination,
                     batch_size, length, *t1, *t2);
      if (block_index != 0)
        commands.audio_add_inplace(*accumulator, *work, count);
    }
    commands.audio_scale_inplace(*accumulator, 1.0f / 3.0f, count);
    DeviceTensor* next_spare = current;
    current = accumulator;
    accumulator = next_spare;
  }
  d.record_activation(commands, d.weights.activation_post, *current, *spare,
                      batch_size, length);
  d.record_conv(commands, d.weights.conv_post, *spare, *current, batch_size,
                length, length, 3, 1);
  const uint64_t samples = static_cast<uint64_t>(batch_size) * length;
  commands.audio_clamp_inplace(*current, -1.0f, 1.0f, samples);
  commands.audio_interleave(*current, interleaved, batch_size, length);
  if (commands.remaining_operator_capacity() != kGraphCapacity - kGraphOperators)
    throw std::logic_error("Vulkan audio VAE: graph operator count drift");
  commands.submit().wait();

  vae::DecodedAudio result;
  result.channels = batch_size;
  result.sample_rate = d.config.sample_rate;
  result.samples.resize(samples);
  d.context.download(interleaved, result.samples.data(), samples);
  if (length != latent_length * static_cast<uint32_t>(d.config.total_upsample()))
    throw std::logic_error("Vulkan audio VAE: output length drift");
  return result;
}

uint64_t AudioDecoder::weight_bytes() const noexcept {
  return impl_ ? impl_->weights.bytes : 0;
}
uint64_t AudioDecoder::peak_device_bytes() const noexcept {
  return impl_ ? impl_->weights.bytes + impl_->activation_bytes() +
      impl_->last_transient_bytes : 0;
}
uint64_t AudioDecoder::allocator_used_bytes() const noexcept {
  return impl_ ? impl_->context.pooled_used_bytes() : 0;
}
uint64_t AudioDecoder::allocator_reserved_bytes() const noexcept {
  return impl_ ? impl_->context.reserved_bytes() : 0;
}
uint64_t AudioDecoder::descriptor_set_allocations() const noexcept {
  return impl_ ? impl_->context.descriptor_set_allocations() : 0;
}
uint32_t AudioDecoder::recorded_operators() const noexcept {
  return kGraphOperators;
}

}  // namespace vidfab::vulkan
