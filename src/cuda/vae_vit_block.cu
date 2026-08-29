#include "vidfab/cuda/vae_vit_block.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstring>
#include <stdexcept>

#include "vidfab/cuda/deterministic_attention.cuh"
#include "vidfab/cuda/deterministic_gemm.cuh"
#include "vidfab/cuda/device.h"
#include "vidfab/cuda/linear.cuh"
#include "vidfab/cuda/nn_kernels.cuh"
#include "vidfab/cuda/vae_kernels.cuh"

namespace vidfab::cuda {
namespace {

using vae::ViTBlockConfig;
using vae::ViTBlockWeightsView;

void validate_config(const ViTBlockConfig& c) {
  if (c.sequence == 0 || c.num_patches > c.sequence || c.dim == 0 ||
      c.heads == 0 || c.head_dim != 64 || c.heads * c.head_dim != c.dim ||
      c.ffn_inner == 0 || c.rope_dim == 0 || c.rope_dim > c.head_dim ||
      (c.rope_dim & 1u) != 0 || !std::isfinite(c.epsilon) || c.epsilon <= 0.0f) {
    throw std::invalid_argument("exact CUDA VAE ViT block: invalid configuration");
  }
}

void validate_weights(const ViTBlockWeightsView& w) {
  if (!w.norm1 || !w.norm2 || !w.scale1 || !w.scale2 || !w.qkv_weight ||
      !w.qkv_bias || !w.out_weight || !w.out_bias || !w.w1_weight ||
      !w.w1_bias || !w.w2_weight || !w.w2_bias) {
    throw std::invalid_argument("exact CUDA VAE ViT block: null weight view");
  }
}

struct BlockWeightsDevice {
  DeviceBuffer<float> norm1, norm2, scale1, scale2, qkv_bias, out_bias,
      w1_bias, w2_bias;
  DeviceBuffer<uint16_t> qkv_weight, out_weight, w1_weight, w2_weight;
  bool loaded = false;

  void load(const ViTBlockWeightsView& w, const ViTBlockConfig& config,
            cudaStream_t stream) {
    loaded = false;
    validate_weights(w);
    const size_t d = config.dim, inner = config.ffn_inner;
    auto require_canonical_half = [](const uint16_t* values, size_t count) {
      for (size_t i = 0; i < count; ++i) {
        if ((values[i] & 0x7c00u) == 0 && (values[i] & 0x03ffu) != 0)
          throw std::invalid_argument(
              "exact CUDA VAE ViT block: fp16 subnormal weight is not canonicalized");
      }
    };
    require_canonical_half(w.qkv_weight, 3 * d * d);
    require_canonical_half(w.out_weight, d * d);
    require_canonical_half(w.w1_weight, 2 * inner * d);
    require_canonical_half(w.w2_weight, d * inner);
    norm1.allocate(d); norm2.allocate(d); scale1.allocate(d); scale2.allocate(d);
    qkv_weight.allocate(3 * d * d); qkv_bias.allocate(3 * d);
    out_weight.allocate(d * d); out_bias.allocate(d);
    w1_weight.allocate(2 * inner * d); w1_bias.allocate(2 * inner);
    w2_weight.allocate(d * inner); w2_bias.allocate(d);
    norm1.copy_from_host(w.norm1, d, stream);
    norm2.copy_from_host(w.norm2, d, stream);
    scale1.copy_from_host(w.scale1, d, stream);
    scale2.copy_from_host(w.scale2, d, stream);
    qkv_weight.copy_from_host(w.qkv_weight, 3 * d * d, stream);
    qkv_bias.copy_from_host(w.qkv_bias, 3 * d, stream);
    out_weight.copy_from_host(w.out_weight, d * d, stream);
    out_bias.copy_from_host(w.out_bias, d, stream);
    w1_weight.copy_from_host(w.w1_weight, 2 * inner * d, stream);
    w1_bias.copy_from_host(w.w1_bias, 2 * inner, stream);
    w2_weight.copy_from_host(w.w2_weight, d * inner, stream);
    w2_bias.copy_from_host(w.w2_bias, d, stream);
    loaded = true;
  }

  uint64_t bytes() const noexcept {
    return norm1.nbytes() + norm2.nbytes() + scale1.nbytes() + scale2.nbytes() +
        qkv_weight.nbytes() + qkv_bias.nbytes() + out_weight.nbytes() +
        out_bias.nbytes() + w1_weight.nbytes() + w1_bias.nbytes() +
        w2_weight.nbytes() + w2_bias.nbytes();
  }
};

struct BlockScratchDevice {
  DeviceBuffer<float> normed, qkv, q, k, v, projected, ffn, activation;
  DeviceBuffer<uint16_t> q_bf16, k_bf16, v_bf16, attention_bf16,
      prepared_q, prepared_k, prepared_v, prepared_dim, prepared_inner;

  explicit BlockScratchDevice(const ViTBlockConfig& c) {
    const size_t rows = c.sequence, d = c.dim, inner = c.ffn_inner;
    normed.allocate(rows * d); qkv.allocate(rows * 3 * d);
    q.allocate(rows * d); k.allocate(rows * d); v.allocate(rows * d);
    q_bf16.allocate(rows * d); k_bf16.allocate(rows * d); v_bf16.allocate(rows * d);
    attention_bf16.allocate(rows * d); prepared_q.allocate(rows * d);
    prepared_k.allocate(rows * d); prepared_v.allocate(rows * d);
    projected.allocate(rows * d); ffn.allocate(rows * 2 * inner);
    activation.allocate(rows * inner); prepared_dim.allocate(rows * d);
    prepared_inner.allocate(rows * inner);
  }

  uint64_t bytes() const noexcept {
    return normed.nbytes() + qkv.nbytes() + q.nbytes() + k.nbytes() + v.nbytes() +
        projected.nbytes() + ffn.nbytes() + activation.nbytes() +
        q_bf16.nbytes() + k_bf16.nbytes() + v_bf16.nbytes() +
        attention_bf16.nbytes() + prepared_q.nbytes() + prepared_k.nbytes() +
        prepared_v.nbytes() + prepared_dim.nbytes() + prepared_inner.nbytes();
  }
};

void exact_gemm(DeviceBuffer<float>& input, DeviceBuffer<uint16_t>& prepared,
                const DeviceBuffer<uint16_t>& weight, DeviceBuffer<float>& output,
                uint32_t rows, uint32_t out_features, uint32_t in_features,
                cudaStream_t stream) {
  launch_narrow_f16(input.get(), prepared.get(),
                    static_cast<size_t>(rows) * in_features, stream);
  launch_deterministic_scalar_gemm_nt(
      reinterpret_cast<const __half*>(prepared.get()),
      reinterpret_cast<const __half*>(weight.get()), nullptr, output.get(), rows,
      out_features, in_features, DenseGemmMode::kFloat16Vae,
      DenseGemmBias::kNone, 0, 0, stream);
}

void run_exact_block(const ViTBlockConfig& c, const BlockWeightsDevice& w,
                     BlockScratchDevice& s, float* tokens, const float* cosine,
                     const float* sine, cudaStream_t stream) {
  if (!w.loaded) throw std::logic_error("exact CUDA VAE ViT block: weights not loaded");
  const uint32_t rows = c.sequence, d = c.dim, inner = c.ffn_inner;
  const size_t count = static_cast<size_t>(rows) * d;
  launch_rmsnorm(tokens, w.norm1.get(), s.normed.get(), rows, d, c.epsilon, stream);
  exact_gemm(s.normed, s.prepared_dim, w.qkv_weight,
             s.qkv, rows, 3 * d, d, stream);
  launch_split_qkv_norm_rope(s.qkv.get(), w.qkv_bias.get(), cosine, sine,
                             s.q.get(), s.k.get(), s.v.get(), rows, c.heads,
                             c.head_dim, c.rope_dim, c.num_patches, c.epsilon, stream);
  launch_heads_to_tokens_bf16(s.q.get(), reinterpret_cast<__nv_bfloat16*>(s.q_bf16.get()),
                              rows, c.heads, c.head_dim, stream);
  launch_heads_to_tokens_bf16(s.k.get(), reinterpret_cast<__nv_bfloat16*>(s.k_bf16.get()),
                              rows, c.heads, c.head_dim, stream);
  launch_heads_to_tokens_bf16(s.v.get(), reinterpret_cast<__nv_bfloat16*>(s.v_bf16.get()),
                              rows, c.heads, c.head_dim, stream);
  launch_prepare_deterministic_attention_inputs(
      stream, reinterpret_cast<const __nv_bfloat16*>(s.q_bf16.get()),
      reinterpret_cast<const __nv_bfloat16*>(s.k_bf16.get()),
      reinterpret_cast<const __nv_bfloat16*>(s.v_bf16.get()),
      reinterpret_cast<__half*>(s.prepared_q.get()),
      reinterpret_cast<__half*>(s.prepared_k.get()),
      reinterpret_cast<__half*>(s.prepared_v.get()), count);
  launch_deterministic_blocked_attention_f16(
      stream, reinterpret_cast<const __half*>(s.prepared_q.get()),
      reinterpret_cast<const __half*>(s.prepared_k.get()),
      reinterpret_cast<const __half*>(s.prepared_v.get()),
      reinterpret_cast<__nv_bfloat16*>(s.attention_bf16.get()), rows, c.heads,
      c.head_dim, 0.125f);
  launch_widen_bf16(reinterpret_cast<const __nv_bfloat16*>(s.attention_bf16.get()),
                    s.normed.get(), count, stream);
  exact_gemm(s.normed, s.prepared_dim, w.out_weight,
             s.projected, rows, d, d, stream);
  launch_layerscale_residual(tokens, s.projected.get(), w.out_bias.get(),
                             w.scale1.get(), rows, d, stream);
  launch_rmsnorm(tokens, w.norm2.get(), s.normed.get(), rows, d, c.epsilon, stream);
  exact_gemm(s.normed, s.prepared_dim, w.w1_weight,
             s.ffn, rows, 2 * inner, d, stream);
  launch_swiglu(s.ffn.get(), w.w1_bias.get(), s.activation.get(), rows, inner, stream);
  exact_gemm(s.activation, s.prepared_inner,
             w.w2_weight, s.projected,
             rows, d, inner, stream);
  launch_layerscale_residual(tokens, s.projected.get(), w.w2_bias.get(),
                             w.scale2.get(), rows, d, stream);
}

class CudaExactViTBlockStage final : public vae::ExactViTBlockStage {
 public:
  explicit CudaExactViTBlockStage(const ViTBlockConfig& config)
      : config_(config), scratch_(config) {
    validate_config(config_);
    const size_t rows = config_.sequence;
    const size_t dim = config_.dim;
    tokens_.allocate(rows * dim);
    cosine_.allocate(rows * config_.rope_dim);
    sine_.allocate(rows * config_.rope_dim);
  }

  DeviceBackend backend() const noexcept override { return DeviceBackend::kCuda; }
  const ViTBlockConfig& config() const noexcept override { return config_; }

  void load(const ViTBlockWeightsView& w) override {
    weights_.load(w, config_, stream_.get());
    stream_.synchronize();
  }

  void forward(const float* tokens, const float* cosine, const float* sine,
               float* output) override {
    if (!weights_.loaded) throw std::logic_error("exact CUDA VAE ViT block: weights not loaded");
    if (!tokens || !cosine || !sine || !output)
      throw std::invalid_argument("exact CUDA VAE ViT block: null activation");
    const uint32_t rows = config_.sequence, d = config_.dim;
    const size_t token_count = static_cast<size_t>(rows) * d;
    tokens_.copy_from_host(tokens, token_count, stream_.get());
    cosine_.copy_from_host(cosine, static_cast<size_t>(rows) * config_.rope_dim,
                           stream_.get());
    sine_.copy_from_host(sine, static_cast<size_t>(rows) * config_.rope_dim,
                         stream_.get());

    run_exact_block(config_, weights_, scratch_, tokens_.get(), cosine_.get(),
                    sine_.get(), stream_.get());
    tokens_.copy_to_host(output, token_count, stream_.get());
    stream_.synchronize();
  }

  uint64_t persistent_bytes() const noexcept override {
    return weights_.bytes();
  }

  uint64_t peak_device_bytes() const noexcept override {
    return persistent_bytes() + scratch_.bytes() + tokens_.nbytes() +
        cosine_.nbytes() + sine_.nbytes();
  }

 private:
  ViTBlockConfig config_;
  Stream stream_;
  BlockWeightsDevice weights_;
  BlockScratchDevice scratch_;
  DeviceBuffer<float> tokens_, cosine_, sine_;
};

}  // namespace

struct ExactViTBlockGraph::Impl {
  ViTBlockConfig config;
  uint32_t layer_count = 0;
  Stream stream;
  std::vector<BlockWeightsDevice> blocks;
  mutable BlockScratchDevice scratch;
  struct HostState {
    DeviceBuffer<float> tokens, cosine, sine;
  };
  std::unique_ptr<HostState> host;

  Impl(const ViTBlockConfig& c, uint32_t layers)
      : config(c), layer_count(layers), blocks(layers), scratch(c) {}
};

ExactViTBlockGraph::ExactViTBlockGraph() = default;
ExactViTBlockGraph::~ExactViTBlockGraph() = default;
ExactViTBlockGraph::ExactViTBlockGraph(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
ExactViTBlockGraph::ExactViTBlockGraph(ExactViTBlockGraph&&) noexcept = default;
ExactViTBlockGraph& ExactViTBlockGraph::operator=(ExactViTBlockGraph&&) noexcept = default;

ExactViTBlockGraph ExactViTBlockGraph::create(const ViTBlockConfig& config,
                                              uint32_t layers) {
  validate_config(config);
  if (layers == 0 || layers > 256)
    throw std::invalid_argument("exact CUDA VAE ViT graph: invalid layer count");
  return ExactViTBlockGraph(std::make_unique<Impl>(config, layers));
}

void ExactViTBlockGraph::load(const SafeTensors& checkpoint) {
  if (!impl_) throw std::logic_error("exact CUDA VAE ViT graph: empty graph");
  for (uint32_t layer = 0; layer < impl_->layer_count; ++layer) {
    vae::ViTBlockWeights weights =
        vae::load_vit_block_weights(checkpoint, layer, impl_->config);
    load_layer(layer, weights.view());
    // The temporary owns pageable vectors used by async copies.
    impl_->stream.synchronize();
  }
}

void ExactViTBlockGraph::load_layer(
    uint32_t layer, const vae::ViTBlockWeightsView& weights) {
  if (!impl_) throw std::logic_error("exact CUDA VAE ViT graph: empty graph");
  if (layer >= impl_->layer_count)
    throw std::out_of_range("exact CUDA VAE ViT graph: layer out of range");
  impl_->blocks[layer].load(weights, impl_->config, impl_->stream.get());
}

void ExactViTBlockGraph::forward_device(float* tokens, const float* cosine,
                                        const float* sine,
                                        cudaStream_t stream) const {
  if (!impl_) throw std::logic_error("exact CUDA VAE ViT graph: empty graph");
  if (!tokens || !cosine || !sine)
    throw std::invalid_argument("exact CUDA VAE ViT graph: null activation");
  for (const BlockWeightsDevice& block : impl_->blocks) {
    if (!block.loaded)
      throw std::logic_error("exact CUDA VAE ViT graph: weights not loaded");
    run_exact_block(impl_->config, block, impl_->scratch, tokens, cosine, sine,
                    stream);
  }
}

void ExactViTBlockGraph::forward_layer_device(
    uint32_t layer, float* tokens, const float* cosine, const float* sine,
    cudaStream_t stream) const {
  if (!impl_) throw std::logic_error("exact CUDA VAE ViT graph: empty graph");
  if (layer >= impl_->layer_count)
    throw std::out_of_range("exact CUDA VAE ViT graph: layer out of range");
  if (!tokens || !cosine || !sine)
    throw std::invalid_argument("exact CUDA VAE ViT graph: null activation");
  const BlockWeightsDevice& block = impl_->blocks[layer];
  if (!block.loaded)
    throw std::logic_error("exact CUDA VAE ViT graph: weights not loaded");
  run_exact_block(impl_->config, block, impl_->scratch, tokens, cosine, sine,
                  stream);
}

void ExactViTBlockGraph::forward(const float* tokens, const float* cosine,
                                 const float* sine, float* output) {
  if (!impl_) throw std::logic_error("exact CUDA VAE ViT graph: empty graph");
  if (!tokens || !cosine || !sine || !output)
    throw std::invalid_argument("exact CUDA VAE ViT graph: null activation");
  if (!impl_->host) {
    impl_->host = std::make_unique<Impl::HostState>();
    impl_->host->tokens.allocate(static_cast<size_t>(impl_->config.sequence) *
                                 impl_->config.dim);
    impl_->host->cosine.allocate(static_cast<size_t>(impl_->config.sequence) *
                                 impl_->config.rope_dim);
    impl_->host->sine.allocate(static_cast<size_t>(impl_->config.sequence) *
                               impl_->config.rope_dim);
  }
  const size_t token_count = static_cast<size_t>(impl_->config.sequence) *
                             impl_->config.dim;
  const size_t rope_count = static_cast<size_t>(impl_->config.sequence) *
                            impl_->config.rope_dim;
  impl_->host->tokens.copy_from_host(tokens, token_count, impl_->stream.get());
  impl_->host->cosine.copy_from_host(cosine, rope_count, impl_->stream.get());
  impl_->host->sine.copy_from_host(sine, rope_count, impl_->stream.get());
  forward_device(impl_->host->tokens.get(), impl_->host->cosine.get(),
                 impl_->host->sine.get(), impl_->stream.get());
  impl_->host->tokens.copy_to_host(output, token_count, impl_->stream.get());
  impl_->stream.synchronize();
}

uint32_t ExactViTBlockGraph::layers() const noexcept {
  return impl_ ? impl_->layer_count : 0;
}

uint64_t ExactViTBlockGraph::persistent_bytes() const noexcept {
  if (!impl_) return 0;
  uint64_t total = 0;
  for (const BlockWeightsDevice& block : impl_->blocks) total += block.bytes();
  return total;
}

uint64_t ExactViTBlockGraph::peak_device_bytes() const noexcept {
  if (!impl_) return 0;
  uint64_t total = persistent_bytes() + impl_->scratch.bytes();
  if (impl_->host) {
    total += impl_->host->tokens.nbytes() + impl_->host->cosine.nbytes() +
             impl_->host->sine.nbytes();
  }
  return total;
}

std::unique_ptr<vae::ExactViTBlockStage> create_exact_vae_vit_block_stage(
    const vae::ViTBlockConfig& config) {
  return std::make_unique<CudaExactViTBlockStage>(config);
}

}  // namespace vidfab::cuda
