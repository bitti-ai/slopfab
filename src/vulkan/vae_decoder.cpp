#include "slopfab/vulkan/vae_decoder.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "slopfab/tensor_convert.h"
#include "slopfab/vae/vit_block.h"
#include "slopfab/w4a8.h"
#include "slopfab/vulkan/gemm.h"
#include "slopfab/vulkan/tensor.h"
#include "slopfab/vulkan/vae_vit_block.h"

namespace slopfab::vulkan {
namespace {

TensorLayout matrix(uint64_t rows, uint64_t columns) {
  const uint64_t shape[] = {rows, columns};
  return TensorLayout::contiguous(shape, 2);
}

TensorLayout vector(uint64_t count) {
  return TensorLayout::contiguous(&count, 1);
}

uint64_t tensor_bytes(const DeviceTensor& tensor) {
  return tensor ? tensor.layout().bytes(tensor.type()) : 0;
}

constexpr uint32_t kDecoderFixedOperators = 15;
constexpr uint32_t kBlockOperators = 20;
constexpr uint32_t kMaxBatchOperators = 4096;

uint32_t decoder_operator_count(const vae::ViTConfig& config) {
  if (config.transformer_mode != vae::ViTTransformerMode::kExact) {
    throw std::invalid_argument("Vulkan video VAE: transformer_mode must be kExact");
  }
  if (config.num_layers <= 0) {
    throw std::invalid_argument("Vulkan video VAE: layer count must be positive");
  }
  const uint64_t count =
      kDecoderFixedOperators + static_cast<uint64_t>(kBlockOperators) * config.num_layers;
  if (count > kMaxBatchOperators) {
    throw std::invalid_argument("Vulkan video VAE: one document exceeds transaction capacity");
  }
  return static_cast<uint32_t>(count);
}

std::vector<float> load_vector(const SafeTensors& checkpoint, const std::string& name,
                               uint64_t count) {
  const TensorView& tensor = checkpoint.at(name);
  if (tensor.numel() != static_cast<int64_t>(count) ||
      (tensor.dtype != DType::kF16 && tensor.dtype != DType::kF32)) {
    throw std::runtime_error("Vulkan video VAE: invalid vector '" + name + "'");
  }
  return to_f32(tensor);
}

std::vector<uint16_t> load_matrix(const SafeTensors& checkpoint, const std::string& name,
                                  uint64_t rows, uint64_t columns, bool flattened = false) {
  const TensorView& tensor = checkpoint.at(name);
  const bool matrix_shape = tensor.shape.size() == 2 &&
                            tensor.shape[0] == static_cast<int64_t>(rows) &&
                            tensor.shape[1] == static_cast<int64_t>(columns);
  if ((!matrix_shape && (!flattened || tensor.numel() != static_cast<int64_t>(rows * columns))) ||
      (tensor.dtype != DType::kF16 && tensor.dtype != DType::kF32)) {
    throw std::runtime_error("Vulkan video VAE: invalid fp16/fp32 matrix '" + name + "'");
  }
  std::vector<uint16_t> result(static_cast<size_t>(rows * columns));
  if (tensor.dtype == DType::kF16) {
    std::memcpy(result.data(), tensor.data, tensor.nbytes);
  } else {
    // Comfy INT8 VAEs retain their unquantized endpoint matrices as FP32.
    // Narrow once to the same canonical FP16 contract as native FP16 files.
    const auto values = to_f32(tensor);
    for (size_t i = 0; i < values.size(); ++i)
      result[i] = f32_to_f16(values[i]);
  }
  for (uint16_t& word : result) {
    if ((word & 0x7c00u) == 0 && (word & 0x03ffu) != 0)
      word &= 0x8000u;
  }
  return result;
}

void upload_vector(TensorContext& context, DeviceTensor& destination,
                   const std::vector<float>& values) {
  context.upload(destination, values.data(), values.size());
}

void upload_matrix(TensorContext& context, DeviceTensor& destination,
                   const std::vector<uint16_t>& values) {
  context.upload_bytes(destination, values.data(), values.size() * 2);
}

} // namespace

struct VideoVaeDecoder::Impl {
  static constexpr uint32_t kShapeCache = 2;

  vae::ViTConfig config;
  uint32_t operators_per_document = 0;
  TensorContext context;
  ExactViTBlockGraph graph;
  bool loaded = false;
  uint64_t common_weight_bytes = 0;

  DeviceTensor post_weight, post_bias, embed_weight, embed_bias, register_tokens, zero_token,
      norm_weight, norm_bias, proj_weight, proj_bias;
  DeviceTensor denorm_input, denorm_mean, denorm_std, denorm_output;
  uint64_t denorm_capacity = 0;

  struct Document {
    DeviceTensor latent, patch, quantized, tokens, patch_tokens, normed, projected, pixels;
  };

  struct ShapeSlot {
    uint32_t time = 0, height = 0, width = 0, patches = 0, sequence = 0;
    uint64_t stamp = 0;
    DeviceTensor cosine, sine, register_indices, zero_index, patch_indices;
    PreparedF16Activation post_prepared, embed_prepared, proj_prepared;
    DenseGemmPlan post_plan, embed_plan, proj_plan;
    std::vector<Document> documents;
  };

  std::vector<std::unique_ptr<ShapeSlot>> shapes;
  uint64_t shape_clock = 0;

  Impl(const Device& device, const vae::ViTConfig& cfg)
      : config(cfg), operators_per_document(decoder_operator_count(cfg)),
        context(device,
                [] {
                  TensorContextOptions options;
                  options.pipeline_sets = TensorPipelineSet::kCore | TensorPipelineSet::kVideo |
                                          TensorPipelineSet::kBlockedAttention;
                  options.max_batch_operators = kMaxBatchOperators;
                  return options;
                }()),
        graph(ExactViTBlockGraph::create(
            context,
            [&] {
              vae::ViTBlockConfig block;
              block.sequence = static_cast<uint32_t>(cfg.num_suffix);
              block.num_patches = 0;
              block.dim = static_cast<uint32_t>(cfg.dim);
              block.heads = static_cast<uint32_t>(cfg.heads);
              block.head_dim = static_cast<uint32_t>(cfg.head_dim);
              block.ffn_inner = static_cast<uint32_t>(cfg.ffn_inner);
              block.rope_dim = static_cast<uint32_t>(cfg.rope_dim);
              block.epsilon = cfg.eps;
              return block;
            }(),
            static_cast<uint32_t>(cfg.num_layers))) {
    if (cfg.num_layers <= 0 || cfg.dim <= 0 || cfg.in_channels <= 0 || cfg.num_register <= 0 ||
        cfg.num_suffix != cfg.num_register + 1 || cfg.patch_dim() <= 0) {
      throw std::invalid_argument("Vulkan video VAE: invalid configuration");
    }
    context.require_exact_fp32_vae_normalization();
    context.require_exact_vae_pointwise();
    context.require_exact_blocked_attention();
  }

  ShapeSlot& select_shape(uint32_t time, uint32_t height, uint32_t width) {
    for (const std::unique_ptr<ShapeSlot>& owned : shapes) {
      ShapeSlot& slot = *owned;
      if (slot.time == time && slot.height == height && slot.width == width) {
        slot.stamp = ++shape_clock;
        graph.prepare_shape(slot.sequence, slot.patches);
        return slot;
      }
    }
    if (shapes.size() == kShapeCache) {
      const auto oldest = std::min_element(
          shapes.begin(), shapes.end(),
          [](const std::unique_ptr<ShapeSlot>& a, const std::unique_ptr<ShapeSlot>& b) {
            return a->stamp < b->stamp;
          });
      shapes.erase(oldest);
    }
    const uint64_t patches64 = static_cast<uint64_t>(time) * height * width;
    if (patches64 == 0 || patches64 + config.num_suffix > UINT32_MAX)
      throw std::overflow_error("Vulkan video VAE: window shape overflow");
    auto slot = std::make_unique<ShapeSlot>();
    slot->time = time;
    slot->height = height;
    slot->width = width;
    slot->patches = static_cast<uint32_t>(patches64);
    slot->sequence = slot->patches + static_cast<uint32_t>(config.num_suffix);
    slot->stamp = ++shape_clock;
    slot->cosine = context.allocate(matrix(slot->sequence, config.rope_dim));
    slot->sine = context.allocate(matrix(slot->sequence, config.rope_dim));
    slot->register_indices = context.allocate(vector(config.num_register), ScalarType::kInt32);
    slot->zero_index = context.allocate(vector(1), ScalarType::kInt32);
    slot->patch_indices = context.allocate(vector(slot->patches), ScalarType::kInt32);
    const vae::ViTRopeTables rope =
        vae::build_vit_rope_tables(time, height, width, static_cast<uint32_t>(config.num_suffix),
                                   static_cast<uint32_t>(config.rope_dim), config.rope_theta);
    context.upload(slot->cosine, rope.cosine.data(), rope.cosine.size());
    context.upload(slot->sine, rope.sine.data(), rope.sine.size());
    std::vector<int32_t> registers(config.num_register);
    for (int i = 0; i < config.num_register; ++i)
      registers[static_cast<size_t>(i)] = static_cast<int32_t>(slot->patches + i);
    const int32_t zero = static_cast<int32_t>(slot->patches + config.num_register);
    std::vector<int32_t> patches(slot->patches);
    for (uint32_t i = 0; i < slot->patches; ++i)
      patches[i] = static_cast<int32_t>(i);
    context.upload_bytes(slot->register_indices, registers.data(),
                         registers.size() * sizeof(int32_t));
    context.upload_bytes(slot->zero_index, &zero, sizeof(zero));
    context.upload_bytes(slot->patch_indices, patches.data(), patches.size() * sizeof(int32_t));
    slot->post_prepared = PreparedF16Activation::create(context, slot->patches,
                                                        static_cast<uint32_t>(config.in_channels));
    slot->embed_prepared = PreparedF16Activation::create(context, slot->patches,
                                                         static_cast<uint32_t>(config.in_channels));
    slot->proj_prepared =
        PreparedF16Activation::create(context, slot->patches, static_cast<uint32_t>(config.dim));
    slot->post_plan =
        DenseGemmPlan::create(context, {slot->patches, static_cast<uint32_t>(config.in_channels),
                                        static_cast<uint32_t>(config.in_channels),
                                        DenseGemmMode::kFloat16Vae, DenseGemmBias::kNone, true});
    slot->embed_plan =
        DenseGemmPlan::create(context, {slot->patches, static_cast<uint32_t>(config.dim),
                                        static_cast<uint32_t>(config.in_channels),
                                        DenseGemmMode::kFloat16Vae, DenseGemmBias::kNone, true});
    slot->proj_plan =
        DenseGemmPlan::create(context, {slot->patches, static_cast<uint32_t>(config.patch_dim()),
                                        static_cast<uint32_t>(config.dim),
                                        DenseGemmMode::kFloat16Vae, DenseGemmBias::kNone, true});
    graph.prepare_shape(slot->sequence, slot->patches);
    ShapeSlot* result = slot.get();
    shapes.push_back(std::move(slot));
    return *result;
  }

  Document make_document(const ShapeSlot& slot) {
    Document doc;
    doc.latent = context.allocate(matrix(config.in_channels, slot.patches));
    doc.patch = context.allocate(matrix(slot.patches, config.in_channels));
    doc.quantized = context.allocate(matrix(slot.patches, config.in_channels));
    doc.tokens = context.allocate(matrix(slot.sequence, config.dim));
    doc.patch_tokens = context.allocate(matrix(slot.patches, config.dim));
    doc.normed = context.allocate(matrix(slot.patches, config.dim));
    doc.projected = context.allocate(matrix(slot.patches, config.patch_dim()));
    const uint64_t pixels = static_cast<uint64_t>(config.out_channels) * slot.time *
                            config.patch_t * slot.height * config.patch * slot.width * config.patch;
    doc.pixels = context.allocate(vector(pixels));
    return doc;
  }

  void ensure_documents(ShapeSlot& slot, uint32_t count) {
    while (slot.documents.size() < count)
      slot.documents.push_back(make_document(slot));
  }

  void record_document(ShapeSlot& slot, Document& doc, TensorBatch& batch) {
    batch.transpose_2d(doc.latent, doc.patch);
    PreparedF16ActivationView post = slot.post_prepared.prepare(batch, doc.patch, slot.patches);
    slot.post_plan.record(batch, post, post_weight, doc.quantized);
    batch.add_bias(doc.quantized, post_bias, doc.quantized);
    PreparedF16ActivationView embed =
        slot.embed_prepared.prepare(batch, doc.quantized, slot.patches);
    slot.embed_plan.record(batch, embed, embed_weight, doc.tokens);
    batch.add_bias(doc.tokens, embed_bias, doc.tokens);
    batch.scatter_rows(register_tokens, slot.register_indices, doc.tokens);
    batch.scatter_rows(zero_token, slot.zero_index, doc.tokens);
    graph.record(batch, doc.tokens, slot.cosine, slot.sine);
    batch.gather_rows(doc.tokens, slot.patch_indices, doc.patch_tokens);
    batch.layer_norm(doc.patch_tokens, norm_weight, norm_bias, doc.normed, config.eps);
    PreparedF16ActivationView projection =
        slot.proj_prepared.prepare(batch, doc.normed, slot.patches);
    slot.proj_plan.record(batch, projection, proj_weight, doc.projected);
    batch.add_bias(doc.projected, proj_bias, doc.projected);
    batch.depth_to_space(doc.projected, doc.pixels, slot.time, slot.height, slot.width,
                         config.out_channels, config.patch_t, config.patch);
  }

  uint64_t shape_bytes() const noexcept {
    uint64_t total = 0;
    for (const std::unique_ptr<ShapeSlot>& slot : shapes) {
      total += tensor_bytes(slot->cosine) + tensor_bytes(slot->sine) +
               tensor_bytes(slot->register_indices) + tensor_bytes(slot->zero_index) +
               tensor_bytes(slot->patch_indices) + slot->post_prepared.reserved_bytes() +
               slot->embed_prepared.reserved_bytes() + slot->proj_prepared.reserved_bytes();
      for (const Document& doc : slot->documents) {
        total += tensor_bytes(doc.latent) + tensor_bytes(doc.patch) + tensor_bytes(doc.quantized) +
                 tensor_bytes(doc.tokens) + tensor_bytes(doc.patch_tokens) +
                 tensor_bytes(doc.normed) + tensor_bytes(doc.projected) + tensor_bytes(doc.pixels);
      }
    }
    return total;
  }
};

VideoVaeDecoder::VideoVaeDecoder() = default;
VideoVaeDecoder::~VideoVaeDecoder() = default;

VideoVaeDecoder::VideoVaeDecoder(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {
}

VideoVaeDecoder::VideoVaeDecoder(VideoVaeDecoder&&) noexcept = default;
VideoVaeDecoder& VideoVaeDecoder::operator=(VideoVaeDecoder&&) noexcept = default;

VideoVaeDecoder VideoVaeDecoder::create(const Device& device, const vae::ViTConfig& config) {
  (void)decoder_operator_count(config);
  return VideoVaeDecoder(std::make_unique<Impl>(device, config));
}

void VideoVaeDecoder::load(const SafeTensors& checkpoint) {
  if (!impl_)
    throw std::logic_error("Vulkan video VAE: empty decoder");
  if (!checkpoint.is_open())
    throw std::invalid_argument("Vulkan video VAE: checkpoint is not open");
  if (is_w4a8_weight(checkpoint, "decoder.transformer_blocks.0.attn.to_qkv.weight")) {
    throw std::runtime_error(
        "Vulkan video VAE: W4A8 checkpoints currently require the CUDA inference backend");
  }
  Impl& d = *impl_;
  d.loaded = false;
  const uint64_t c = static_cast<uint64_t>(d.config.in_channels);
  const uint64_t dim = static_cast<uint64_t>(d.config.dim);
  const uint64_t patch_dim = static_cast<uint64_t>(d.config.patch_dim());
  d.post_weight = d.context.allocate(matrix(c, c), ScalarType::kFloat16);
  d.post_bias = d.context.allocate(vector(c));
  d.embed_weight = d.context.allocate(matrix(dim, c), ScalarType::kFloat16);
  d.embed_bias = d.context.allocate(vector(dim));
  d.register_tokens = d.context.allocate(matrix(d.config.num_register, dim));
  d.zero_token = d.context.allocate(matrix(1, dim));
  d.norm_weight = d.context.allocate(vector(dim));
  d.norm_bias = d.context.allocate(vector(dim));
  d.proj_weight = d.context.allocate(matrix(patch_dim, dim), ScalarType::kFloat16);
  d.proj_bias = d.context.allocate(vector(patch_dim));
  upload_matrix(d.context, d.post_weight,
                load_matrix(checkpoint, "post_quant_conv.weight", c, c, true));
  upload_vector(d.context, d.post_bias, load_vector(checkpoint, "post_quant_conv.bias", c));
  upload_matrix(d.context, d.embed_weight,
                load_matrix(checkpoint, "decoder.x_embedder.weight", dim, c));
  upload_vector(d.context, d.embed_bias, load_vector(checkpoint, "decoder.x_embedder.bias", dim));
  upload_vector(d.context, d.register_tokens,
                load_vector(checkpoint, "decoder.register_tokens", d.config.num_register * dim));
  std::vector<float> zeros(dim, 0.0f);
  upload_vector(d.context, d.zero_token, zeros);
  upload_vector(d.context, d.norm_weight, load_vector(checkpoint, "decoder.norm_out.weight", dim));
  upload_vector(d.context, d.norm_bias, load_vector(checkpoint, "decoder.norm_out.bias", dim));
  upload_matrix(d.context, d.proj_weight,
                load_matrix(checkpoint, "decoder.proj_out.weight", patch_dim, dim));
  upload_vector(d.context, d.proj_bias,
                load_vector(checkpoint, "decoder.proj_out.bias", patch_dim));
  d.graph.load(checkpoint);
  d.common_weight_bytes = tensor_bytes(d.post_weight) + tensor_bytes(d.post_bias) +
                          tensor_bytes(d.embed_weight) + tensor_bytes(d.embed_bias) +
                          tensor_bytes(d.register_tokens) + tensor_bytes(d.zero_token) +
                          tensor_bytes(d.norm_weight) + tensor_bytes(d.norm_bias) +
                          tensor_bytes(d.proj_weight) + tensor_bytes(d.proj_bias);
  d.loaded = true;
}

const vae::ViTConfig& VideoVaeDecoder::config() const {
  if (!impl_)
    throw std::logic_error("Vulkan video VAE: empty decoder");
  return impl_->config;
}

void VideoVaeDecoder::forward_window(const float* latent, int time, int height, int width,
                                     std::vector<float>& output) {
  std::vector<std::vector<float>> batch_output(1);
  const size_t slot = 0;
  forward_windows(latent, 1, time, height, width, batch_output, &slot);
  output = std::move(batch_output.front());
}

void VideoVaeDecoder::forward_windows(const float* latent, int batch, int time, int height,
                                      int width, std::vector<std::vector<float>>& output,
                                      const size_t* slots) {
  if (!impl_ || !impl_->loaded)
    throw std::logic_error("Vulkan video VAE: weights not loaded");
  if (!latent || !slots || batch <= 0 || time <= 0 || height <= 0 || width <= 0)
    throw std::invalid_argument("Vulkan video VAE: invalid window input");
  // Validate the whole call before shape allocation, uploads, or command
  // recording. A bad later slot must leave the decoder unchanged.
  for (int i = 0; i < batch; ++i) {
    if (slots[i] >= output.size())
      throw std::out_of_range("Vulkan video VAE: output slot out of range");
  }
  Impl& d = *impl_;
  Impl::ShapeSlot& shape = d.select_shape(
      static_cast<uint32_t>(time), static_cast<uint32_t>(height), static_cast<uint32_t>(width));
  d.ensure_documents(shape, static_cast<uint32_t>(batch));
  const uint64_t latent_words = static_cast<uint64_t>(d.config.in_channels) * shape.patches;
  const uint64_t pixel_words = static_cast<uint64_t>(d.config.out_channels) * time *
                               d.config.patch_t * height * d.config.patch * width * d.config.patch;
  for (int i = 0; i < batch; ++i) {
    d.context.upload(shape.documents[static_cast<size_t>(i)].latent,
                     latent + static_cast<uint64_t>(i) * latent_words, latent_words);
  }
  const uint32_t documents_per_transaction = kMaxBatchOperators / d.operators_per_document;
  for (uint32_t first = 0; first < static_cast<uint32_t>(batch);
       first += documents_per_transaction) {
    const uint32_t count =
        std::min(documents_per_transaction, static_cast<uint32_t>(batch) - first);
    TensorBatch commands = d.context.begin_batch();
    if (commands.remaining_operator_capacity() < count * d.operators_per_document) {
      throw std::logic_error("Vulkan video VAE: insufficient transaction capacity");
    }
    for (uint32_t i = 0; i < count; ++i)
      d.record_document(shape, shape.documents[first + i], commands);
    commands.submit().wait();
    for (uint32_t i = 0; i < count; ++i) {
      std::vector<float>& destination = output[slots[first + i]];
      destination.resize(static_cast<size_t>(pixel_words));
      d.context.download(shape.documents[first + i].pixels, destination.data(), pixel_words);
    }
  }
}

uint64_t VideoVaeDecoder::persistent_bytes() const noexcept {
  return impl_ ? impl_->common_weight_bytes + impl_->graph.persistent_bytes() : 0;
}

uint64_t VideoVaeDecoder::peak_device_bytes() const noexcept {
  return impl_ ? impl_->common_weight_bytes + impl_->graph.peak_device_bytes() +
                     impl_->shape_bytes() + tensor_bytes(impl_->denorm_input) +
                     tensor_bytes(impl_->denorm_mean) + tensor_bytes(impl_->denorm_std) +
                     tensor_bytes(impl_->denorm_output)
               : 0;
}

uint64_t VideoVaeDecoder::allocator_reserved_bytes() const noexcept {
  return impl_ ? impl_->context.reserved_bytes() : 0;
}

uint64_t VideoVaeDecoder::allocator_used_bytes() const noexcept {
  return impl_ ? impl_->context.pooled_used_bytes() : 0;
}

uint64_t VideoVaeDecoder::descriptor_set_allocations() const noexcept {
  return impl_ ? impl_->context.descriptor_set_allocations() : 0;
}

uint32_t VideoVaeDecoder::cached_shapes() const noexcept {
  return impl_ ? static_cast<uint32_t>(impl_->shapes.size()) : 0;
}

uint32_t VideoVaeDecoder::operators_per_document() const noexcept {
  return impl_ ? impl_->operators_per_document : 0;
}

void VideoVaeDecoder::denormalize_latents(const float* normalized, int channels, uint64_t voxels,
                                          const std::vector<float>& mean,
                                          const std::vector<float>& std_dev,
                                          std::vector<float>& output) {
  if (!impl_ || !normalized || channels != impl_->config.in_channels || voxels == 0 ||
      mean.size() != size_t(channels) || std_dev.size() != size_t(channels)) {
    throw std::invalid_argument("Vulkan video VAE: invalid latent denormalization");
  }
  Impl& d = *impl_;
  if (voxels > d.denorm_capacity) {
    d.denorm_input = d.context.allocate(matrix(channels, voxels));
    d.denorm_output = d.context.allocate(matrix(channels, voxels));
    d.denorm_mean = d.context.allocate(vector(channels));
    d.denorm_std = d.context.allocate(vector(channels));
    d.denorm_capacity = voxels;
  }
  d.context.upload(d.denorm_input, normalized, static_cast<uint64_t>(channels) * voxels);
  d.context.upload(d.denorm_mean, mean.data(), mean.size());
  d.context.upload(d.denorm_std, std_dev.data(), std_dev.size());
  TensorBatch batch = d.context.begin_batch();
  batch.latent_denorm_f32(d.denorm_input, d.denorm_mean, d.denorm_std, d.denorm_output);
  batch.submit().wait();
  output.resize(static_cast<size_t>(channels) * voxels);
  d.context.download(d.denorm_output, output.data(), output.size());
}

vae::DecodedVideo VideoVaeDecoder::decode(const float* normalized_latent, int time, int height,
                                          int width, const std::vector<float>& mean,
                                          const std::vector<float>& std_dev,
                                          const vae::DecodeSchedule& schedule) {
  return vae::decode_video(*this, normalized_latent, time, height, width, mean, std_dev, schedule);
}

} // namespace slopfab::vulkan
