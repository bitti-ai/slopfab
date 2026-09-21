#include "attention_internal.cuh"

namespace slopfab::cuda {
using namespace attention_detail;

namespace {
size_t checked_product(size_t a, size_t b) {
  if (a != 0 && b > std::numeric_limits<size_t>::max() / a)
    throw std::overflow_error("CUDA attention: workspace shape overflow");
  return a * b;
}
} // namespace

AttentionBackend attention_preferred_backend(const AttentionConfig& cfg) {
  return fused_supported(cfg) ? AttentionBackend::kFused : AttentionBackend::kBlocked;
}

float AttentionConfig::effective_scale() const {
  return scale > 0.0f ? scale : 1.0f / std::sqrt(static_cast<float>(head_dim));
}

size_t attention_workspace_bytes(const AttentionConfig& cfg, AttentionBackend backend) {
  return attention_workspace_bytes(cfg, cfg.num_heads, backend);
}

size_t attention_workspace_bytes(const AttentionConfig& cfg, int num_kv_heads,
                                 AttentionBackend backend) {
  // The fused backend keeps the score tile, the probabilities and the
  // accumulator in shared memory, and reads q, k and v as bf16 without the
  // fp16 conversion copies the blocked path needs. It therefore wants no
  // workspace at all -- not a smaller one.
  if (backend == AttentionBackend::kFused)
    return 0;
  if (backend == AttentionBackend::kSage2)
    return sage2_workspace_bytes(cfg, num_kv_heads);
  if (backend == AttentionBackend::kSol)
    return sol_attention_workspace_bytes(cfg);
  if (cfg.seq_len <= 0 || cfg.num_heads <= 0 || cfg.head_dim <= 0)
    return 0;
  check_config(cfg, num_kv_heads);

  const int bq = effective_query_block(cfg);
  const int bk = choose_key_block(cfg);
  const size_t tile = checked_product(checked_product(cfg.num_heads, bq), bk);
  const size_t stats = checked_product(cfg.num_heads, bq);
  const size_t width = checked_product(cfg.num_heads, cfg.head_dim);

  size_t total = 0;
  auto add = [&](size_t elements, size_t bytes) {
    const size_t size = checked_product(elements, bytes);
    if (size > std::numeric_limits<size_t>::max() - 255)
      throw std::overflow_error("CUDA attention: workspace alignment overflow");
    const size_t aligned = align_up(size);
    if (aligned > std::numeric_limits<size_t>::max() - total)
      throw std::overflow_error("CUDA attention: workspace size overflow");
    total += aligned;
  };
  add(tile, sizeof(__half));                                // scores, then probabilities
  add(checked_product(stats, cfg.head_dim), sizeof(float)); // accumulator
  add(stats, sizeof(float));                                // running max
  add(stats, sizeof(float));                                // running sum
  // K/V keep their compact grouped-query width; Q uses the query-head width.
  const size_t kv_width = checked_product(num_kv_heads, cfg.head_dim);
  add(checked_product(cfg.seq_len, kv_width), sizeof(__half));
  add(checked_product(cfg.seq_len, kv_width), sizeof(__half));
  add(checked_product(bq, width), sizeof(__half));
  return total;
}

AttentionPlan AttentionPlan::compile(const AttentionDescriptor& descriptor,
                                     const AttentionExecutionPolicy& policy) {
  validate_attention_descriptor(descriptor);
  if (descriptor.query_tokens != descriptor.key_value_tokens ||
      descriptor.mask == AttentionMask::kCausal)
    throw std::invalid_argument(
        "CUDA attention: this plan supports full/banded self-attention only");
  if (descriptor.arithmetic == AttentionArithmetic::kExact)
    throw std::invalid_argument(
        "CUDA attention: use the exact operator plan for pinned arithmetic");
  if ((descriptor.mask == AttentionMask::kFrameBanded) != (policy.band_ranges != nullptr))
    throw std::invalid_argument("CUDA attention: mask and band ranges disagree");
  const bool approximate = policy.band_ranges != nullptr ||
                           policy.backend == AttentionBackend::kSage2 ||
                           policy.backend == AttentionBackend::kSol;
  if (approximate && descriptor.arithmetic != AttentionArithmetic::kApproximate)
    throw std::invalid_argument(
        "CUDA attention: selected execution policy requires approximate arithmetic");
  AttentionConfig cfg;
  cfg.seq_len = static_cast<int>(descriptor.query_tokens);
  cfg.num_heads = static_cast<int>(descriptor.query_heads);
  cfg.head_dim = static_cast<int>(descriptor.head_dim);
  cfg.scale = descriptor.scale;
  cfg.query_block = policy.query_block;
  cfg.key_block = policy.key_block;
  cfg.band_ranges = policy.band_ranges;
  cfg.exact_prefix = policy.exact_prefix;
  cfg.sol_beta = policy.sol_beta;
  cfg.sol_error_k = policy.sol_error_k;
  cfg.sol_error_v = policy.sol_error_v;
  cfg.sol_route_counts = policy.sol_route_counts;
  cfg.sol_phase_ms = policy.sol_phase_ms;
  cfg.sol_pipeline = policy.sol_pipeline;
  auto plan = compile(cfg, static_cast<int>(descriptor.key_value_heads), policy.backend);
  plan.descriptor_ = descriptor;
  return plan;
}

AttentionPlan AttentionPlan::compile(const AttentionConfig& cfg, int kv_heads,
                                     AttentionBackend backend) {
  check_config(cfg, kv_heads);
  AttentionPlan plan;
  plan.descriptor_ = {static_cast<uint32_t>(cfg.seq_len),
                      static_cast<uint32_t>(cfg.seq_len),
                      static_cast<uint32_t>(cfg.num_heads),
                      static_cast<uint32_t>(kv_heads),
                      static_cast<uint32_t>(cfg.head_dim),
                      AttentionLayout::kTokensHeadsChannels,
                      cfg.band_ranges ? AttentionMask::kFrameBanded : AttentionMask::kFull,
                      cfg.band_ranges || backend == AttentionBackend::kSage2 ||
                              backend == AttentionBackend::kSol
                          ? AttentionArithmetic::kApproximate
                          : AttentionArithmetic::kEquivalent,
                      cfg.scale};
  validate_attention_descriptor(plan.descriptor_);
  if (cfg.band_ranges && backend != AttentionBackend::kFused)
    throw std::invalid_argument("CUDA attention: band ranges require fused attention");
  switch (backend) {
  case AttentionBackend::kBlocked:
    break;
  case AttentionBackend::kFused:
    if (!fused_supported(cfg))
      throw std::invalid_argument(
          "CUDA attention: fused attention supports head widths 64 and 128");
    break;
  case AttentionBackend::kSage2: {
    SLOPFAB_CUDA_CHECK(cudaGetDevice(&plan.device_));
    const char* reason = nullptr;
    if (!sage2_supported(cfg, plan.device_, &reason))
      throw std::invalid_argument(std::string("CUDA attention: Sage2 ") + reason);
    break;
  }
  case AttentionBackend::kSol:
    if (kv_heads != cfg.num_heads || cfg.head_dim != 128 || cfg.exact_prefix < 0 ||
        cfg.exact_prefix > cfg.seq_len || !std::isfinite(cfg.sol_beta) ||
        !std::isfinite(cfg.sol_error_k) || cfg.sol_error_k < 0.0f ||
        !std::isfinite(cfg.sol_error_v) || cfg.sol_error_v < 0.0f)
      throw std::invalid_argument("CUDA attention: invalid Sol attention contract");
    break;
  default:
    throw std::invalid_argument("CUDA attention: unknown backend");
  }
  plan.config_ = cfg;
  plan.backend_ = backend;
  plan.workspace_bytes_ = attention_workspace_bytes(cfg, kv_heads, backend);
  return plan;
}

void AttentionPlan::forward(cublasHandle_t handle, cudaStream_t stream, const __nv_bfloat16* query,
                            const __nv_bfloat16* key, const __nv_bfloat16* value,
                            __nv_bfloat16* output, Workspace& workspace) const {
  if (!query || !key || !value || !output || output == query || output == key || output == value)
    throw std::invalid_argument(
        "CUDA attention: inputs must exist and output must not alias inputs");
  if (device_ >= 0) {
    int current = -1;
    SLOPFAB_CUDA_CHECK(cudaGetDevice(&current));
    if (current != device_)
      throw std::invalid_argument("CUDA attention: plan belongs to another device");
  }
  if (workspace.used() > workspace.capacity() ||
      workspace_bytes_ > workspace.capacity() - workspace.used())
    throw std::invalid_argument("CUDA attention: insufficient plan workspace");
  attention_forward_gqa(handle, stream, query, key, value, output, config_,
                        static_cast<int>(descriptor_.key_value_heads), backend_, workspace);
}

void attention_forward(cublasHandle_t handle, cudaStream_t stream, const __nv_bfloat16* q,
                       const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* out,
                       const AttentionConfig& cfg, AttentionBackend backend, Workspace& ws) {
  if (backend == AttentionBackend::kFused) {
    run_fused(stream, q, k, v, out, cfg, cfg.num_heads);
    return;
  }
  if (backend == AttentionBackend::kSage2) {
    sage2_attention_forward(stream, q, k, v, out, cfg, cfg.num_heads, ws);
    return;
  }
  if (backend == AttentionBackend::kSol) {
    sol_attention_forward(stream, q, k, v, out, cfg, ws);
    return;
  }
  run_blocked(handle, stream, q, k, v, out, cfg, cfg.num_heads, ws);
}

void attention_forward_query_chunk(cudaStream_t stream, const __nv_bfloat16* q,
                                   const __nv_bfloat16* k, const __nv_bfloat16* v,
                                   __nv_bfloat16* out, const AttentionConfig& cfg, int query_offset,
                                   int query_rows) {
  if (!q || !k || !v || !out || query_offset < 0 || query_rows <= 0 || query_offset > cfg.seq_len ||
      query_rows > cfg.seq_len - query_offset)
    throw std::invalid_argument("attention: invalid compact query range");
  AttentionConfig part = cfg;
  if (cfg.band_ranges != nullptr) {
    if (query_offset % attention_fused_query_tile() != 0)
      throw std::invalid_argument(
          "attention: compact banded query offset must align to query tile");
    part.band_ranges += static_cast<size_t>(query_offset / attention_fused_query_tile()) * 4;
  }
  run_fused(stream, q, k, v, out, part, cfg.num_heads, query_rows);
}

void attention_forward_gqa(cublasHandle_t handle, cudaStream_t stream, const __nv_bfloat16* q,
                           const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* out,
                           const AttentionConfig& cfg, int num_kv_heads, AttentionBackend backend,
                           Workspace& ws) {
  if (backend == AttentionBackend::kFused) {
    run_fused(stream, q, k, v, out, cfg, num_kv_heads);
    return;
  }
  if (backend == AttentionBackend::kSage2) {
    sage2_attention_forward(stream, q, k, v, out, cfg, num_kv_heads, ws);
    return;
  }
  if (backend == AttentionBackend::kSol) {
    if (num_kv_heads != cfg.num_heads)
      throw std::runtime_error("Sol-Attn: grouped-query attention is not supported");
    sol_attention_forward(stream, q, k, v, out, cfg, ws);
    return;
  }
  run_blocked(handle, stream, q, k, v, out, cfg, num_kv_heads, ws);
}

} // namespace slopfab::cuda
