#include "linear_internal.cuh"

namespace slopfab::cuda {
using namespace linear_detail;

// --- QuantWeight ------------------------------------------------------------

size_t QuantWeight::stored_bytes() const {
  const size_t n = static_cast<size_t>(out_features) * in_features;
  // Two nibbles per byte. Both shipped 4-bit formats place the even element in
  // the high nibble; their dequantisers pin that convention independently.
  if (format == QuantFormat::kNVFP4 || format == QuantFormat::kNF4)
    return (n + 1) / 2;
  return element_bytes(format) * n;
}

size_t linear_workspace_bytes(const QuantWeight& w, int rows, ComputeType compute) {
  const size_t weights = static_cast<size_t>(w.out_features) * w.in_features;
  const size_t act = static_cast<size_t>(rows) * w.in_features;
  size_t total = 0;

  if (compute == ComputeType::kF32) {
    // fp32 compute needs an fp32 copy, and anything not already fp32 reaches it
    // through the bf16 dequantiser, so both buffers can be live at once.
    if (w.format != QuantFormat::kF32) {
      total += align_up(weights * sizeof(__nv_bfloat16));
      total += align_up(weights * sizeof(float));
    }
    // The AWQ scale writes a scaled copy of the activation, which ConvRot then
    // reads and rotates into a second copy, so on a layer with both the two
    // buffers are live at once and add rather than overlap.
    if (w.pre_quant_scale != nullptr)
      total += align_up(act * sizeof(float));
    if (convrot_applies(w))
      total += align_up(act * sizeof(float));
  } else {
    if (w.format != QuantFormat::kBF16)
      total += align_up(weights * sizeof(__nv_bfloat16));
    if (w.pre_quant_scale != nullptr)
      total += align_up(act * sizeof(__nv_bfloat16));
    if (convrot_applies(w))
      total += align_up(act * sizeof(__nv_bfloat16));
  }

  // The native nvfp4 GEMM carves a quantised copy of the activation instead of
  // a dequantised copy of the weight, so the two are alternatives and this is a
  // max rather than a sum. Sized unconditionally because this function is not
  // told whether `set_native` is on, and because the difference only ever
  // matters for a weight the native path could take at all.
  //
  // At every production shape the dequantised weight is the larger of the two,
  // so this changes nothing today. It is here because "the other buffer happens
  // to be bigger" is a coincidence, not an invariant, and a caller with many
  // rows and few output features would otherwise throw from Workspace::alloc.
  if (w.format == QuantFormat::kNVFP4 && compute == ComputeType::kBF16 &&
      nvfp4_gemm_shape_supported(w.out_features, w.in_features)) {
    total = std::max(total, nvfp4_gemm_workspace_bytes(rows, w.in_features));
  }
  return total;
}

size_t linear_dense_weight_bytes(const QuantWeight& w) {
  if (w.format == QuantFormat::kBF16)
    return 0; // already dense; `prepare` carves nothing
  return align_up(static_cast<size_t>(w.out_features) * w.in_features * sizeof(__nv_bfloat16));
}

size_t linear_activation_workspace_bytes(const QuantWeight& w, int rows, ComputeType compute) {
  const size_t weights = static_cast<size_t>(w.out_features) * w.in_features;
  const size_t act = static_cast<size_t>(rows) * w.in_features;
  size_t total = 0;

  // Everything `linear_workspace_bytes` counts except the dense bf16 copy: the
  // fp32 widening of it, and the activation copies. The two functions add up to
  // that one for every format the block stacks use.
  if (compute == ComputeType::kF32) {
    if (w.format != QuantFormat::kF32)
      total += align_up(weights * sizeof(float));
    if (w.pre_quant_scale != nullptr)
      total += align_up(act * sizeof(float));
    if (convrot_applies(w))
      total += align_up(act * sizeof(float));
  } else {
    if (w.pre_quant_scale != nullptr)
      total += align_up(act * sizeof(__nv_bfloat16));
    if (convrot_applies(w))
      total += align_up(act * sizeof(__nv_bfloat16));
  }

  // Same alternative as in `linear_workspace_bytes`: with the native path on,
  // `prepare` carves nothing and the quantised activation copy is the whole
  // requirement, so it belongs on this side of the split.
  if (w.format == QuantFormat::kNVFP4 && compute == ComputeType::kBF16 &&
      nvfp4_gemm_shape_supported(w.out_features, w.in_features)) {
    total = std::max(total, nvfp4_gemm_workspace_bytes(rows, w.in_features));
  }
  return total;
}

// --- LinearRunner -----------------------------------------------------------

void LinearRunner::init(cublasHandle_t handle, cudaStream_t stream) {
  handle_ = handle;
  stream_ = stream;
  native_nvfp4_device_ = current_device_compute_capability() == 120;
  SLOPFAB_CUBLAS_CHECK(cublas_set_stream(handle_, stream_));
}

bool LinearRunner::takes_native_nvfp4(const QuantWeight& w) const {
  // Native low-precision GEMM is not wired up yet for fp8 — both settings
  // dequantise there. The guard is written now so the invariant survives that
  // change: a weight the checkpoint flagged full_precision_matrix_mult, or an
  // fp8 weight that ships no input_scale (the same statement in the older
  // files' vocabulary), must never reach a low-precision GEMM (spec 8.2).
  const bool native_path = native_ && !w.full_precision &&
                           ((w.format == QuantFormat::kF8E4M3 && w.input_scale != 0.0f) ||
                            w.format == QuantFormat::kNVFP4);

  // The extra conditions are not a weakening of the guard above: ConvRot never
  // coincides with an nvfp4 transformer weight and the rotation belongs to the
  // activation, `pre_quant_scale` is an AWQ text-encoder field that
  // `full_precision` already excludes, and `nvfp4_gemm_supported` is the
  // block-scale swizzle's no-padding precondition. Each falls through to the
  // reference path, which is always correct, never to something approximate.
  return native_path && native_nvfp4_device_ && w.format == QuantFormat::kNVFP4 &&
         w.block_scale != nullptr && !convrot_applies(w) && w.pre_quant_scale == nullptr &&
         nvfp4_gemm_shape_supported(w.out_features, w.in_features);
}

const __nv_bfloat16* LinearRunner::prepare(const QuantWeight& w, Workspace& ws) {
  if (handle_ == nullptr)
    throw std::runtime_error("LinearRunner::prepare: init() not called");
  if (takes_native_nvfp4(w))
    return nullptr;
  return materialise_bf16(w, ws, stream_);
}

void LinearRunner::forward(const QuantWeight& w, const __nv_bfloat16* x, int rows, __nv_bfloat16* y,
                           Workspace& ws) {
  if (handle_ == nullptr)
    throw std::runtime_error("LinearRunner::forward: init() not called");
  if (rows <= 0)
    return;

  // The arena is only rewound, never freed, and everything below is issued on
  // one stream, so releasing the cursor at return cannot race the GEMM.
  Workspace::Scope scope(ws);
  forward_prepared(w, prepare(w, ws), x, rows, y, ws);
}

void LinearRunner::forward_prepared(const QuantWeight& w, const __nv_bfloat16* weight,
                                    const __nv_bfloat16* x, int rows, __nv_bfloat16* y,
                                    Workspace& ws) {
  if (handle_ == nullptr) {
    throw std::runtime_error("LinearRunner::forward_prepared: init() not called");
  }
  if (rows <= 0)
    return;

  Workspace::Scope scope(ws);

  // Null is what `prepare` returns for a weight with no dense form, so this is
  // the native nvfp4 GEMM rather than an error — but only for a weight that
  // would actually take it. Without the check, a caller that forgot to call
  // `prepare` at all would hand a bf16 weight to the fp4 GEMM and get plausible
  // wrong numbers, which is the failure this whole format invites.
  if (weight == nullptr && !takes_native_nvfp4(w)) {
    throw std::runtime_error(
        "LinearRunner::forward_prepared: null dense weight for a layer that does not take the "
        "native nvfp4 GEMM — call prepare() and pass what it returns");
  }
  if (weight == nullptr) {
    nvfp4_gemm_forward_prevalidated(x, static_cast<const uint8_t*>(w.data), w.block_scale,
                                    w.global_scale, y, rows, w.out_features, w.in_features, ws,
                                    stream_);
    add_bias(y, /*y_is_f32=*/false, w, rows, stream_);
    return;
  }

  // AWQ scales the activation per input channel ahead of everything else. A
  // null pointer means the quantiser folded the scale into the preceding norm's
  // weight, so it is "already accounted for" rather than "unknown" — check the
  // tensor, never the layer's name. nvfp4 weights are never ConvRot, so the two
  // never actually compose; the order below is the one that would be right if
  // they ever did.
  const __nv_bfloat16* xin = x;
  if (w.pre_quant_scale != nullptr) {
    __nv_bfloat16* xs = ws.alloc_n<__nv_bfloat16>(static_cast<size_t>(rows) * w.in_features);
    launch_pre_quant_scale(xin, w.pre_quant_scale, xs, rows, w.in_features, stream_);
    xin = xs;
  }
  if (convrot_applies(w)) {
    __nv_bfloat16* xr = ws.alloc_n<__nv_bfloat16>(static_cast<size_t>(rows) * w.in_features);
    launch_convrot(xin, xr, rows, w.in_features, w.convrot_group, stream_);
    xin = xr;
  }

  // Row-major y[rows,out] = x[rows,in] @ W[out,in]^T. Column-major sees W as
  // [in,out] and x as [in,rows]; op_T on W then gives [out,in] * [in,rows].
  const float alpha = 1.0f;
  const float beta = 0.0f;
  SLOPFAB_CUBLAS_CHECK(cublas_gemm_ex(handle_, CUBLAS_OP_T, CUBLAS_OP_N, w.out_features, rows,
                                      w.in_features, &alpha, weight, CUDA_R_16BF, w.in_features,
                                      xin, CUDA_R_16BF, w.in_features, &beta, y, CUDA_R_16BF,
                                      w.out_features, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

  add_bias(y, /*y_is_f32=*/false, w, rows, stream_);
}

void LinearRunner::forward_f32(const QuantWeight& w, const float* x, int rows, float* y,
                               Workspace& ws) {
  if (handle_ == nullptr)
    throw std::runtime_error("LinearRunner::forward_f32: init() not called");
  if (rows <= 0)
    return;

  Workspace::Scope scope(ws);
  const size_t n = static_cast<size_t>(w.out_features) * w.in_features;

  const float* weight = nullptr;
  if (w.format == QuantFormat::kF32) {
    weight = static_cast<const float*>(w.data);
  } else {
    const __nv_bfloat16* narrow = materialise_bf16(w, ws, stream_);
    float* wide = ws.alloc_n<float>(n);
    launch_widen_bf16(narrow, wide, n, stream_);
    weight = wide;
  }

  const float* xin = x;
  if (w.pre_quant_scale != nullptr) {
    float* xs = ws.alloc_n<float>(static_cast<size_t>(rows) * w.in_features);
    pre_quant_scale_f32(xin, w.pre_quant_scale, xs, rows, w.in_features, stream_);
    xin = xs;
  }
  if (convrot_applies(w)) {
    float* xr = ws.alloc_n<float>(static_cast<size_t>(rows) * w.in_features);
    launch_convrot_f32(xin, xr, rows, w.in_features, w.convrot_group, stream_);
    xin = xr;
  }

  gemm_nt(handle_, xin, weight, y, rows, w.out_features, w.in_features);
  add_bias(y, /*y_is_f32=*/true, w, rows, stream_);
}

} // namespace slopfab::cuda
