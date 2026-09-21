#include "encoder_internal.cuh"
#include "slopfab/text/backend_capabilities.h"

namespace slopfab::text {
using namespace encoder_detail;

namespace {
// --- weight plumbing --------------------------------------------------------

QuantWeight int8_convrot(const uint8_t* base, const LayerLayout& layout, LayerTensor weight,
                         LayerTensor scale, int out_features, int in_features) {
  QuantWeight w;
  w.format = QuantFormat::kI8;
  w.data = base + layout.offset[static_cast<int>(weight)];
  w.out_features = out_features;
  w.in_features = in_features;
  w.weight_scale = reinterpret_cast<const float*>(base + layout.offset[static_cast<int>(scale)]);
  // Per output channel despite the format tag reading "int8_tensorwise", and
  // there is no input_scale anywhere in this checkpoint: the quantiser is
  // symmetric per row at /127, so `int8 * weight_scale` is exact
  // (spec section 5.1).
  w.per_channel_scale = true;
  w.input_scale = 0.0f;
  w.convrot = true;
  w.convrot_group = kConvRotGroup;
  return w;
}

// `pre_quant` is kCount where the layer has none — which is a positive
// statement that the quantiser folded it into the preceding norm, not that it
// is unknown. Five of the seven are like that; checked per tensor, never
// inferred from the name.
QuantWeight nvfp4_awq(const uint8_t* base, const LayerLayout& layout, LayerTensor weight,
                      LayerTensor scale, LayerTensor pre_quant, float global_scale,
                      int out_features, int in_features) {
  QuantWeight w;
  w.format = QuantFormat::kNVFP4;
  w.data = base + layout.offset[static_cast<int>(weight)];
  w.out_features = out_features;
  w.in_features = in_features;
  w.block_scale = base + layout.offset[static_cast<int>(scale)];
  w.global_scale = global_scale;
  // Every quantised linear of this build declares full_precision_matrix_mult,
  // so none may ever reach a native fp4 GEMM. The flag comes from the file and
  // validation has already insisted all 350 carry it.
  w.full_precision = true;
  // Not rotated, unlike the int8 build. Leaving this true would apply a
  // Hadamard nothing had undone.
  w.convrot = false;
  w.per_channel_scale = false;
  w.weight_scale = nullptr;
  if (pre_quant != LayerTensor::kCount && layout.bytes[static_cast<int>(pre_quant)] != 0) {
    w.pre_quant_scale =
        reinterpret_cast<const __nv_bfloat16*>(base + layout.offset[static_cast<int>(pre_quant)]);
  }
  return w;
}

const __nv_bfloat16* norm_ptr(const uint8_t* base, const LayerLayout& layout, LayerTensor which) {
  return reinterpret_cast<const __nv_bfloat16*>(base + layout.offset[static_cast<int>(which)]);
}

CausalAttentionConfig attention_config(const LayerDims& dims) {
  CausalAttentionConfig cfg;
  cfg.seq_len = dims.num_tokens;
  cfg.num_heads = dims.num_heads;
  cfg.num_kv_heads = dims.num_kv_heads;
  cfg.head_dim = dims.head_dim;
  cfg.query_block = dims.attn_query_block;
  return cfg;
}

// The dequantisation scratch a projection of this shape will want. Shapes only:
// `linear_workspace_bytes` never dereferences the pointers.
//
// Both formats come to the same total — one dense bf16 copy of the weight plus
// one transformed copy of the activation — but for different reasons: the int8
// path rotates the activation, the nvfp4 path scales it. Sizing them separately
// keeps that a coincidence rather than an assumption.
size_t projection_workspace(WeightFormat format, int out_features, int in_features, int rows) {
  const size_t weight =
      align_up(static_cast<size_t>(out_features) * in_features * sizeof(__nv_bfloat16));
  const size_t activation =
      align_up(static_cast<size_t>(rows) * in_features * sizeof(__nv_bfloat16));
  if (format == WeightFormat::kNVFP4Awq)
    return weight + activation;

  QuantWeight w;
  w.format = QuantFormat::kI8;
  w.out_features = out_features;
  w.in_features = in_features;
  w.per_channel_scale = true;
  w.convrot = true;
  w.convrot_group = kConvRotGroup;
  return slopfab::cuda::linear_workspace_bytes(w, rows, ComputeType::kBF16);
}

} // namespace

// --- decoder layer -----------------------------------------------------------

LayerWeights layer_weights_from_blob(const uint8_t* base, const LayerLayout& layout,
                                     const EncoderConfig& config,
                                     const LayerGlobalScales& globals) {
  const int hidden = config.hidden_size;
  const int q_width = config.num_attention_heads * config.head_dim;
  const int kv_width = config.num_key_value_heads * config.head_dim;
  const int inner = config.intermediate_size;

  if (config.format == WeightFormat::kNVFP4Awq) {
    // The seven globals are in the order q, k, v, o, gate, up, down. Only
    // o_proj and down_proj carry a pre_quant_scale; the other five had it
    // folded into input_layernorm and post_attention_layernorm respectively,
    // which is why those two norms differ from the int8 build's while q_norm
    // and k_norm are bitwise identical to it.
    LayerWeights w;
    w.q_proj = nvfp4_awq(base, layout, LayerTensor::kQWeight, LayerTensor::kQScale,
                         LayerTensor::kCount, globals.value[0], q_width, hidden);
    w.k_proj = nvfp4_awq(base, layout, LayerTensor::kKWeight, LayerTensor::kKScale,
                         LayerTensor::kCount, globals.value[1], kv_width, hidden);
    w.v_proj = nvfp4_awq(base, layout, LayerTensor::kVWeight, LayerTensor::kVScale,
                         LayerTensor::kCount, globals.value[2], kv_width, hidden);
    w.o_proj = nvfp4_awq(base, layout, LayerTensor::kOWeight, LayerTensor::kOScale,
                         LayerTensor::kOPreQuantScale, globals.value[3], hidden, q_width);
    w.gate_proj = nvfp4_awq(base, layout, LayerTensor::kGateWeight, LayerTensor::kGateScale,
                            LayerTensor::kCount, globals.value[4], inner, hidden);
    w.up_proj = nvfp4_awq(base, layout, LayerTensor::kUpWeight, LayerTensor::kUpScale,
                          LayerTensor::kCount, globals.value[5], inner, hidden);
    w.down_proj = nvfp4_awq(base, layout, LayerTensor::kDownWeight, LayerTensor::kDownScale,
                            LayerTensor::kDownPreQuantScale, globals.value[6], hidden, inner);
    w.input_layernorm = norm_ptr(base, layout, LayerTensor::kInputLayerNorm);
    w.post_attention_layernorm = norm_ptr(base, layout, LayerTensor::kPostAttentionLayerNorm);
    w.q_norm = norm_ptr(base, layout, LayerTensor::kQNorm);
    w.k_norm = norm_ptr(base, layout, LayerTensor::kKNorm);
    return w;
  }

  LayerWeights w;
  w.q_proj =
      int8_convrot(base, layout, LayerTensor::kQWeight, LayerTensor::kQScale, q_width, hidden);
  w.k_proj =
      int8_convrot(base, layout, LayerTensor::kKWeight, LayerTensor::kKScale, kv_width, hidden);
  w.v_proj =
      int8_convrot(base, layout, LayerTensor::kVWeight, LayerTensor::kVScale, kv_width, hidden);
  w.o_proj =
      int8_convrot(base, layout, LayerTensor::kOWeight, LayerTensor::kOScale, hidden, q_width);
  w.gate_proj =
      int8_convrot(base, layout, LayerTensor::kGateWeight, LayerTensor::kGateScale, inner, hidden);
  w.up_proj =
      int8_convrot(base, layout, LayerTensor::kUpWeight, LayerTensor::kUpScale, inner, hidden);
  w.down_proj =
      int8_convrot(base, layout, LayerTensor::kDownWeight, LayerTensor::kDownScale, hidden, inner);
  w.input_layernorm = norm_ptr(base, layout, LayerTensor::kInputLayerNorm);
  w.post_attention_layernorm = norm_ptr(base, layout, LayerTensor::kPostAttentionLayerNorm);
  w.q_norm = norm_ptr(base, layout, LayerTensor::kQNorm);
  w.k_norm = norm_ptr(base, layout, LayerTensor::kKNorm);
  return w;
}

size_t layer_workspace_bytes(const LayerDims& d) {
  if (d.num_tokens <= 0)
    return 0;
  const size_t L = static_cast<size_t>(d.num_tokens);
  const size_t q_width = static_cast<size_t>(d.num_heads) * d.head_dim;
  const size_t kv_width = static_cast<size_t>(d.num_kv_heads) * d.head_dim;
  const size_t bf = sizeof(__nv_bfloat16);

  // Live across the whole layer, in the order `encoder_layer_forward` carves
  // them. Two [L, 25600] intermediates dominate: 419 MB each at L = 4096.
  size_t activations = 0;
  activations += align_up(L * d.hidden * bf);       // n
  activations += align_up(L * q_width * bf);        // q
  activations += align_up(L * kv_width * bf);       // k
  activations += align_up(L * kv_width * bf);       // v
  activations += align_up(L * q_width * bf);        // attention output
  activations += align_up(L * d.hidden * bf);       // projection output
  activations += align_up(L * d.intermediate * bf); // gate
  activations += align_up(L * d.intermediate * bf); // up

  // Transient, carved on top: one GEMM's dequantisation scratch, or the
  // attention tiles, whichever is larger. The seven GEMMs are strictly
  // sequential, so one weight scratch buffer suffices (spec section 7.1).
  size_t transient = causal_attention_workspace_bytes(attention_config(d));
  const int rows = d.num_tokens;
  const WeightFormat f = d.format;
  transient =
      std::max(transient, projection_workspace(f, static_cast<int>(q_width), d.hidden, rows));
  transient =
      std::max(transient, projection_workspace(f, static_cast<int>(kv_width), d.hidden, rows));
  transient =
      std::max(transient, projection_workspace(f, d.hidden, static_cast<int>(q_width), rows));
  transient = std::max(transient, projection_workspace(f, d.intermediate, d.hidden, rows));
  transient = std::max(transient, projection_workspace(f, d.hidden, d.intermediate, rows));

  return activations + transient;
}

size_t encoder_detail::resident_request_bytes(const EncoderConfig& cfg, size_t weight_bytes,
                                              size_t total_device_bytes) {
  LayerDims dims;
  dims.format = cfg.format;
  dims.num_tokens = cfg.max_prompt_tokens;
  dims.hidden = cfg.hidden_size;
  dims.num_heads = cfg.num_attention_heads;
  dims.num_kv_heads = cfg.num_key_value_heads;
  dims.head_dim = cfg.head_dim;
  dims.intermediate = cfg.intermediate_size;
  dims.rms_norm_eps = cfg.rms_norm_eps;

  const size_t rows = static_cast<size_t>(cfg.max_prompt_tokens);
  const size_t stream = rows * static_cast<size_t>(cfg.hidden_size);
  const size_t rope = rows * static_cast<size_t>(cfg.head_dim);
  const size_t persistent =
      stream * (sizeof(__nv_bfloat16) + sizeof(float)) + 2 * rope * sizeof(float);
  // WDDM can accept large cudaMalloc reservations and fail later when the
  // first DMA commits their pages. Preserve a device/driver budget in addition
  // to the exact max-request graph footprint so load fails synchronously.
  const size_t driver_reserve = std::max<size_t>(2ull << 30, total_device_bytes / 5u);
  const size_t workspace = layer_workspace_bytes(dims);
  if (weight_bytes > std::numeric_limits<size_t>::max() - workspace ||
      weight_bytes + workspace > std::numeric_limits<size_t>::max() - persistent ||
      weight_bytes + workspace + persistent > std::numeric_limits<size_t>::max() - driver_reserve)
    return std::numeric_limits<size_t>::max();
  return weight_bytes + workspace + persistent + driver_reserve;
}

void encoder_layer_forward(cublasHandle_t handle, cudaStream_t stream,
                           slopfab::cuda::LinearRunner& linear, const LayerWeights& w,
                           const LayerDims& d, const float* cos, const float* sin, __nv_bfloat16* x,
                           Workspace& ws) {
  require(d.num_tokens > 0, "encoder_layer_forward: num_tokens must be positive");
  const size_t L = static_cast<size_t>(d.num_tokens);
  const int rows = d.num_tokens;
  const size_t q_width = static_cast<size_t>(d.num_heads) * d.head_dim;
  const size_t kv_width = static_cast<size_t>(d.num_kv_heads) * d.head_dim;

  Workspace::Scope scope(ws);
  __nv_bfloat16* n = ws.alloc_n<__nv_bfloat16>(L * d.hidden);
  __nv_bfloat16* q = ws.alloc_n<__nv_bfloat16>(L * q_width);
  __nv_bfloat16* k = ws.alloc_n<__nv_bfloat16>(L * kv_width);
  __nv_bfloat16* v = ws.alloc_n<__nv_bfloat16>(L * kv_width);
  __nv_bfloat16* attn = ws.alloc_n<__nv_bfloat16>(L * q_width);
  __nv_bfloat16* proj = ws.alloc_n<__nv_bfloat16>(L * d.hidden);
  __nv_bfloat16* gate = ws.alloc_n<__nv_bfloat16>(L * d.intermediate);
  __nv_bfloat16* up = ws.alloc_n<__nv_bfloat16>(L * d.intermediate);

  // --- attention half. Pre-norm: the residual carries the *unnormalised*
  // stream and is never gated or scaled (spec section 4.4).
  slopfab::cuda::launch_rmsnorm(x, w.input_layernorm, n, rows, d.hidden, d.rms_norm_eps, stream);

  // ConvRot rotates the contraction axis, so it belongs to the activation, not
  // to the GEMM: q/k/v share one rotation of `n`, and LinearRunner applies it
  // per call. The rotation must see the *complete* RMSNorm output — H does not
  // commute with diag(w) (spec section 5.3).
  linear.forward(w.q_proj, n, rows, q, ws);
  linear.forward(w.k_proj, n, rows, k, ws);
  linear.forward(w.v_proj, n, rows, v, ws);

  // QK-norm BEFORE RoPE. Reversing the two is a silent quality bug: RMSNorm
  // scales channel j by w[j], RoPE mixes j with j+64, and those two weights
  // differ by up to 440x on k_norm (spec section 4.2).
  slopfab::cuda::launch_head_rmsnorm(q, w.q_norm, rows, d.num_heads, d.head_dim, d.rms_norm_eps,
                                     stream);
  slopfab::cuda::launch_head_rmsnorm(k, w.k_norm, rows, d.num_kv_heads, d.head_dim, d.rms_norm_eps,
                                     stream);
  // v is not normalised. Only q and k.

  // All 128 head dims rotate, pairing j with j + 64 — unlike the H3 DiT, which
  // rotates 96 of 128 and pairs j with j + 48 (spec section 2.4).
  slopfab::cuda::launch_rope_neox(q, cos, sin, rows, d.num_heads, d.head_dim, stream);
  slopfab::cuda::launch_rope_neox(k, cos, sin, rows, d.num_kv_heads, d.head_dim, stream);

  causal_attention_forward(handle, stream, q, k, v, attn, attention_config(d), ws);
  linear.forward(w.o_proj, attn, rows, proj, ws);
  launch_residual_add(x, proj, L * d.hidden, stream);

  // --- MLP half.
  slopfab::cuda::launch_rmsnorm(x, w.post_attention_layernorm, n, rows, d.hidden, d.rms_norm_eps,
                                stream);
  linear.forward(w.gate_proj, n, rows, gate, ws);
  linear.forward(w.up_proj, n, rows, up, ws);
  // gate_proj goes through SiLU; up_proj does not.
  launch_swiglu_split(gate, up, gate, L * d.intermediate, stream);
  linear.forward(w.down_proj, gate, rows, proj, ws);
  launch_residual_add(x, proj, L * d.hidden, stream);
}

size_t exact_layer_workspace_bytes(const LayerWeights& w, const LayerDims& d) {
  require(d.num_tokens > 0, "exact layer workspace: num_tokens must be positive");
  const size_t rows = static_cast<size_t>(d.num_tokens);
  const size_t q_width = static_cast<size_t>(d.num_heads) * d.head_dim;
  const size_t kv_width = static_cast<size_t>(d.num_kv_heads) * d.head_dim;
  const size_t bf = sizeof(__nv_bfloat16);
  size_t activations = 0;
  for (size_t elements :
       {rows * d.hidden, rows * q_width, rows * kv_width, rows * kv_width, rows * q_width,
        rows * d.hidden, rows * d.intermediate, rows * d.intermediate, rows * d.intermediate}) {
    activations += align_up(elements * bf);
  }
  size_t transient = 0;
  for (const slopfab::cuda::QuantWeight* weight :
       {&w.q_proj, &w.k_proj, &w.v_proj, &w.o_proj, &w.gate_proj, &w.up_proj, &w.down_proj}) {
    const size_t dense =
        align_up(static_cast<size_t>(weight->out_features) * weight->in_features * bf);
    const bool transform = weight->pre_quant_scale != nullptr ||
                           (weight->convrot && weight->in_features % weight->convrot_group == 0);
    const size_t transformed = transform ? align_up(rows * weight->in_features * bf) : 0;
    transient = std::max(transient, dense + transformed);
  }
  return activations + transient;
}

void encoder_layer_forward_exact(cudaStream_t stream, const LayerWeights& w, const LayerDims& d,
                                 const float* cos, const float* sin, __nv_bfloat16* x,
                                 Workspace& ws, const ExactLayerTaps* taps) {
  require(supports_exact_text_layer({d.num_tokens, d.hidden, d.num_heads, d.num_kv_heads,
                                     d.head_dim, d.intermediate, d.rms_norm_eps}),
          "encoder_layer_forward_exact: invalid Qwen production dimensions");
  const size_t L = static_cast<size_t>(d.num_tokens);
  const uint32_t rows = static_cast<uint32_t>(d.num_tokens);
  const uint32_t q_width = static_cast<uint32_t>(d.num_heads * d.head_dim);
  const uint32_t kv_width = static_cast<uint32_t>(d.num_kv_heads * d.head_dim);
  Workspace::Scope scope(ws);
  __nv_bfloat16* n = ws.alloc_n<__nv_bfloat16>(L * d.hidden);
  __nv_bfloat16* q = ws.alloc_n<__nv_bfloat16>(L * q_width);
  __nv_bfloat16* k = ws.alloc_n<__nv_bfloat16>(L * kv_width);
  __nv_bfloat16* v = ws.alloc_n<__nv_bfloat16>(L * kv_width);
  __nv_bfloat16* attention = ws.alloc_n<__nv_bfloat16>(L * q_width);
  __nv_bfloat16* branch = ws.alloc_n<__nv_bfloat16>(L * d.hidden);
  __nv_bfloat16* gate = ws.alloc_n<__nv_bfloat16>(L * d.intermediate);
  __nv_bfloat16* up = ws.alloc_n<__nv_bfloat16>(L * d.intermediate);
  __nv_bfloat16* activation = ws.alloc_n<__nv_bfloat16>(L * d.intermediate);
  auto copy_tap = [&](const __nv_bfloat16* source, __nv_bfloat16* destination, size_t count) {
    if (destination != nullptr) {
      SLOPFAB_CUDA_CHECK(cudaMemcpyAsync(destination, source, count * sizeof(__nv_bfloat16),
                                         cudaMemcpyDeviceToDevice, stream));
    }
  };
  auto projection = [&](const slopfab::cuda::QuantWeight& weight, const __nv_bfloat16* input,
                        __nv_bfloat16* output) {
    Workspace::Scope projection_scope(ws);
    const __nv_bfloat16* source = input;
    if (weight.pre_quant_scale != nullptr) {
      __nv_bfloat16* transformed = ws.alloc_n<__nv_bfloat16>(L * weight.in_features);
      slopfab::cuda::launch_pre_quant_scale(input, weight.pre_quant_scale, transformed,
                                            d.num_tokens, weight.in_features, stream);
      source = transformed;
    } else if (weight.convrot && weight.in_features % weight.convrot_group == 0) {
      __nv_bfloat16* transformed = ws.alloc_n<__nv_bfloat16>(L * weight.in_features);
      slopfab::cuda::launch_convrot(input, transformed, d.num_tokens, weight.in_features,
                                    weight.convrot_group, stream);
      source = transformed;
    }
    const __nv_bfloat16* dense = slopfab::cuda::materialize_bf16_exact(weight, ws, stream);
    const uint32_t tiled = rows / 64u * 64u;
    if (tiled != 0) {
      slopfab::cuda::launch_deterministic_bf16_gemm_nt(source, dense, nullptr, output, tiled,
                                                       weight.out_features, weight.in_features,
                                                       DenseGemmBias::kNone, 0, 0, stream);
    }
    if (tiled != rows) {
      slopfab::cuda::launch_deterministic_scalar_gemm_nt(
          source, dense, nullptr, output, rows - tiled, weight.out_features, weight.in_features,
          DenseGemmMode::kBFloat16, DenseGemmBias::kNone, tiled, tiled, stream);
    }
  };

  slopfab::cuda::launch_rmsnorm(x, w.input_layernorm, n, d.num_tokens, d.hidden, d.rms_norm_eps,
                                stream);
  if (taps)
    copy_tap(n, taps->input_norm, L * d.hidden);
  projection(w.q_proj, n, q);
  projection(w.k_proj, n, k);
  projection(w.v_proj, n, v);
  slopfab::cuda::launch_head_rmsnorm(q, w.q_norm, d.num_tokens, d.num_heads, d.head_dim,
                                     d.rms_norm_eps, stream);
  slopfab::cuda::launch_head_rmsnorm(k, w.k_norm, d.num_tokens, d.num_kv_heads, d.head_dim,
                                     d.rms_norm_eps, stream);
  slopfab::cuda::launch_rope_neox(q, cos, sin, d.num_tokens, d.num_heads, d.head_dim, stream);
  slopfab::cuda::launch_rope_neox(k, cos, sin, d.num_tokens, d.num_kv_heads, d.head_dim, stream);
  if (taps) {
    copy_tap(q, taps->query, L * q_width);
    copy_tap(k, taps->key, L * kv_width);
    copy_tap(v, taps->value, L * kv_width);
  }
  slopfab::cuda::launch_deterministic_causal_gqa_attention(stream, q, k, v, attention, rows,
                                                           d.num_heads, d.num_kv_heads, d.head_dim,
                                                           exact_attention_scale(d.head_dim));
  if (taps)
    copy_tap(attention, taps->attention, L * q_width);
  projection(w.o_proj, attention, branch);
  launch_residual_add_exact(x, branch, L * d.hidden, stream);
  if (taps)
    copy_tap(x, taps->attention_residual, L * d.hidden);
  slopfab::cuda::launch_rmsnorm(x, w.post_attention_layernorm, n, d.num_tokens, d.hidden,
                                d.rms_norm_eps, stream);
  if (taps)
    copy_tap(n, taps->post_attention_norm, L * d.hidden);
  projection(w.gate_proj, n, gate);
  projection(w.up_proj, n, up);
  if (taps) {
    copy_tap(gate, taps->gate, L * d.intermediate);
    copy_tap(up, taps->up, L * d.intermediate);
  }
  launch_swiglu_split_exact(gate, up, activation, L * d.intermediate, stream);
  if (taps)
    copy_tap(activation, taps->activation, L * d.intermediate);
  projection(w.down_proj, activation, branch);
  launch_residual_add_exact(x, branch, L * d.hidden, stream);
  if (taps)
    copy_tap(x, taps->final_residual, L * d.hidden);
}

} // namespace slopfab::text
