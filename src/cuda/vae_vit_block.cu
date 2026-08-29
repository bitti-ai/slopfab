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

class CudaExactViTBlockStage final : public vae::ExactViTBlockStage {
 public:
  explicit CudaExactViTBlockStage(const ViTBlockConfig& config) : config_(config) {
    validate_config(config_);
    const size_t rows = config_.sequence;
    const size_t dim = config_.dim;
    const size_t heads = config_.heads;
    const size_t head_dim = config_.head_dim;
    const size_t inner = config_.ffn_inner;
    tokens_.allocate(rows * dim);
    normed_.allocate(rows * dim);
    qkv_.allocate(rows * 3 * dim);
    q_.allocate(heads * rows * head_dim);
    k_.allocate(heads * rows * head_dim);
    v_.allocate(heads * rows * head_dim);
    q_bf16_.allocate(rows * dim);
    k_bf16_.allocate(rows * dim);
    v_bf16_.allocate(rows * dim);
    attention_bf16_.allocate(rows * dim);
    prepared_q_.allocate(rows * dim);
    prepared_k_.allocate(rows * dim);
    prepared_v_.allocate(rows * dim);
    projected_.allocate(rows * dim);
    ffn_.allocate(rows * 2 * inner);
    activation_.allocate(rows * inner);
    prepared_dim_.allocate(rows * dim);
    prepared_inner_.allocate(rows * inner);
    cosine_.allocate(rows * config_.rope_dim);
    sine_.allocate(rows * config_.rope_dim);
  }

  DeviceBackend backend() const noexcept override { return DeviceBackend::kCuda; }
  const ViTBlockConfig& config() const noexcept override { return config_; }

  void load(const ViTBlockWeightsView& w) override {
    validate_weights(w);
    const size_t d = config_.dim, inner = config_.ffn_inner;
    norm1_.allocate(d); norm2_.allocate(d); scale1_.allocate(d); scale2_.allocate(d);
    qkv_weight_.allocate(3 * d * d); qkv_bias_.allocate(3 * d);
    out_weight_.allocate(d * d); out_bias_.allocate(d);
    w1_weight_.allocate(2 * inner * d); w1_bias_.allocate(2 * inner);
    w2_weight_.allocate(d * inner); w2_bias_.allocate(d);
    norm1_.copy_from_host(w.norm1, d, stream_.get());
    norm2_.copy_from_host(w.norm2, d, stream_.get());
    scale1_.copy_from_host(w.scale1, d, stream_.get());
    scale2_.copy_from_host(w.scale2, d, stream_.get());
    qkv_weight_.copy_from_host(w.qkv_weight, 3 * d * d, stream_.get());
    qkv_bias_.copy_from_host(w.qkv_bias, 3 * d, stream_.get());
    out_weight_.copy_from_host(w.out_weight, d * d, stream_.get());
    out_bias_.copy_from_host(w.out_bias, d, stream_.get());
    w1_weight_.copy_from_host(w.w1_weight, 2 * inner * d, stream_.get());
    w1_bias_.copy_from_host(w.w1_bias, 2 * inner, stream_.get());
    w2_weight_.copy_from_host(w.w2_weight, d * inner, stream_.get());
    w2_bias_.copy_from_host(w.w2_bias, d, stream_.get());
    stream_.synchronize();
    loaded_ = true;
  }

  void forward(const float* tokens, const float* cosine, const float* sine,
               float* output) override {
    if (!loaded_) throw std::logic_error("exact CUDA VAE ViT block: weights not loaded");
    if (!tokens || !cosine || !sine || !output)
      throw std::invalid_argument("exact CUDA VAE ViT block: null activation");
    const uint32_t rows = config_.sequence, d = config_.dim;
    const uint32_t inner = config_.ffn_inner;
    const size_t token_count = static_cast<size_t>(rows) * d;
    tokens_.copy_from_host(tokens, token_count, stream_.get());
    cosine_.copy_from_host(cosine, static_cast<size_t>(rows) * config_.rope_dim,
                           stream_.get());
    sine_.copy_from_host(sine, static_cast<size_t>(rows) * config_.rope_dim,
                         stream_.get());

    launch_rmsnorm(tokens_.get(), norm1_.get(), normed_.get(), rows, d,
                   config_.epsilon, stream_.get());
    exact_gemm(normed_, prepared_dim_, qkv_weight_, qkv_, 3 * d, d);
    launch_split_qkv_norm_rope(qkv_.get(), qkv_bias_.get(), cosine_.get(),
                               sine_.get(), q_.get(), k_.get(), v_.get(), rows,
                               config_.heads, config_.head_dim, config_.rope_dim,
                               config_.num_patches, config_.epsilon, stream_.get());
    launch_heads_to_tokens_bf16(q_.get(),
        reinterpret_cast<__nv_bfloat16*>(q_bf16_.get()), rows, config_.heads,
        config_.head_dim, stream_.get());
    launch_heads_to_tokens_bf16(k_.get(),
        reinterpret_cast<__nv_bfloat16*>(k_bf16_.get()), rows, config_.heads,
        config_.head_dim, stream_.get());
    launch_heads_to_tokens_bf16(v_.get(),
        reinterpret_cast<__nv_bfloat16*>(v_bf16_.get()), rows, config_.heads,
        config_.head_dim, stream_.get());
    launch_prepare_deterministic_attention_inputs(
        stream_.get(), reinterpret_cast<const __nv_bfloat16*>(q_bf16_.get()),
        reinterpret_cast<const __nv_bfloat16*>(k_bf16_.get()),
        reinterpret_cast<const __nv_bfloat16*>(v_bf16_.get()),
        reinterpret_cast<__half*>(prepared_q_.get()),
        reinterpret_cast<__half*>(prepared_k_.get()),
        reinterpret_cast<__half*>(prepared_v_.get()), token_count);
    launch_deterministic_blocked_attention_f16(
        stream_.get(), reinterpret_cast<const __half*>(prepared_q_.get()),
        reinterpret_cast<const __half*>(prepared_k_.get()),
        reinterpret_cast<const __half*>(prepared_v_.get()),
        reinterpret_cast<__nv_bfloat16*>(attention_bf16_.get()), rows,
        config_.heads, config_.head_dim, 0.125f);
    launch_widen_bf16(
        reinterpret_cast<const __nv_bfloat16*>(attention_bf16_.get()),
        normed_.get(), token_count, stream_.get());
    exact_gemm(normed_, prepared_dim_, out_weight_, projected_, d, d);
    launch_layerscale_residual(tokens_.get(), projected_.get(), out_bias_.get(),
                               scale1_.get(), rows, d, stream_.get());
    launch_rmsnorm(tokens_.get(), norm2_.get(), normed_.get(), rows, d,
                   config_.epsilon, stream_.get());
    exact_gemm(normed_, prepared_dim_, w1_weight_, ffn_, 2 * inner, d);
    launch_swiglu(ffn_.get(), w1_bias_.get(), activation_.get(), rows, inner,
                  stream_.get());
    exact_gemm(activation_, prepared_inner_, w2_weight_, projected_, d, inner);
    launch_layerscale_residual(tokens_.get(), projected_.get(), w2_bias_.get(),
                               scale2_.get(), rows, d, stream_.get());
    tokens_.copy_to_host(output, token_count, stream_.get());
    stream_.synchronize();
  }

  uint64_t persistent_bytes() const noexcept override {
    return norm1_.nbytes() + norm2_.nbytes() + scale1_.nbytes() + scale2_.nbytes() +
        qkv_weight_.nbytes() + qkv_bias_.nbytes() + out_weight_.nbytes() +
        out_bias_.nbytes() + w1_weight_.nbytes() + w1_bias_.nbytes() +
        w2_weight_.nbytes() + w2_bias_.nbytes();
  }

  uint64_t peak_device_bytes() const noexcept override {
    return persistent_bytes() + tokens_.nbytes() + normed_.nbytes() + qkv_.nbytes() +
        q_.nbytes() + k_.nbytes() + v_.nbytes() + q_bf16_.nbytes() +
        k_bf16_.nbytes() + v_bf16_.nbytes() + attention_bf16_.nbytes() +
        prepared_q_.nbytes() + prepared_k_.nbytes() + prepared_v_.nbytes() +
        projected_.nbytes() + ffn_.nbytes() + activation_.nbytes() +
        prepared_dim_.nbytes() + prepared_inner_.nbytes() + cosine_.nbytes() +
        sine_.nbytes();
  }

 private:
  template <typename WeightBuffer>
  void exact_gemm(DeviceBuffer<float>& input, DeviceBuffer<uint16_t>& prepared,
                  WeightBuffer& weight, DeviceBuffer<float>& output,
                  uint32_t out_features, uint32_t in_features) {
    const uint32_t rows = config_.sequence;
    launch_narrow_f16(input.get(), prepared.get(),
                      static_cast<size_t>(rows) * in_features, stream_.get());
    const uint32_t tiled = rows / 64u * 64u;
    if (tiled != 0) {
      launch_deterministic_f16_gemm_nt(
          reinterpret_cast<const __half*>(prepared.get()),
          reinterpret_cast<const __half*>(weight.get()), output.get(), tiled,
          out_features, in_features, 0, stream_.get());
    }
    if (tiled != rows) {
      launch_deterministic_scalar_gemm_nt(
          reinterpret_cast<const __half*>(prepared.get()) +
              static_cast<size_t>(tiled) * in_features,
          reinterpret_cast<const __half*>(weight.get()), nullptr, output.get(),
          rows - tiled, out_features, in_features, DenseGemmMode::kFloat16Vae,
          DenseGemmBias::kNone, 0, tiled, stream_.get());
    }
  }

  ViTBlockConfig config_;
  bool loaded_ = false;
  Stream stream_;
  DeviceBuffer<float> norm1_, norm2_, scale1_, scale2_, qkv_bias_, out_bias_,
      w1_bias_, w2_bias_;
  DeviceBuffer<uint16_t> qkv_weight_, out_weight_, w1_weight_, w2_weight_;
  DeviceBuffer<float> tokens_, normed_, qkv_, q_, k_, v_, projected_, ffn_,
      activation_, cosine_, sine_;
  DeviceBuffer<uint16_t> q_bf16_, k_bf16_, v_bf16_, attention_bf16_,
      prepared_q_, prepared_k_, prepared_v_, prepared_dim_, prepared_inner_;
};

}  // namespace

std::unique_ptr<vae::ExactViTBlockStage> create_exact_vae_vit_block_stage(
    const vae::ViTBlockConfig& config) {
  return std::make_unique<CudaExactViTBlockStage>(config);
}

}  // namespace vidfab::cuda
