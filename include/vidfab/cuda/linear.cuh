// Quantised linear layers.
//
// Three checkpoints, three storage formats, one operation. This header is the
// single place that knows how a stored weight becomes `y = x W^T + b`:
//
//   BF16/F16/F32   dense, no scales                    (norms, refiner, VAE)
//   F8_E4M3        per-tensor weight_scale,            (H3 transformer blocks)
//                  optional per-tensor input_scale
//   I8 + ConvRot   per-output-channel weight_scale,    (Qwen3-VL text encoder)
//                  Hadamard rotation on the contraction axis
//   NVFP4          E2M1 nibbles, an e4m3 scale per 16  (both nvfp4 checkpoints)
//                  contracted elements, one fp32 global scale on top, and on
//                  the AWQ text encoder an optional per-input-channel
//                  activation scale
//   NF4            bitsandbytes nibbles, an absmax per 64 elements, with those
//                  absmax values themselves quantised in blocks of 256
//
// Every weight is stored PyTorch-style `[out_features, in_features]` row-major
// and there are no transposes anywhere in either checkpoint, so the contraction
// is always over the second axis.
//
// **Correctness before speed.** The first implementation dequantises the weight
// into workspace and runs an ordinary cuBLAS GEMM. That is numerically *better*
// than a native low-precision GEMM, not worse, and it is what the transformer
// spec recommends for a port that does not want fp8 tensor cores (§8.2). It
// costs one extra read+write of the weight per GEMM. Native fp8/int8 GEMM goes
// behind the same API later, selected by `LinearRunner::set_native`.
//
// Two traps, both of which produce plausible output rather than a crash:
//
//   - `mlp.fc2` in the H3 transformer ships **no input_scale** and is tagged
//     `"full_precision_matrix_mult": true`. It must never take an fp8 GEMM
//     path. `QuantWeight::input_scale == 0` marks this and is load-bearing.
//   - ConvRot is a rotation applied offline to the weight. Skipping the
//     matching online rotation of the activation computes `x H W^T`, which is
//     well-scaled noise. See docs/convrot_notes.md.
#pragma once

#include <cublas_v2.h>
#include <cuda_bf16.h>

#include <cstddef>
#include <cstdint>

#include "vidfab/cuda/workspace.cuh"

namespace vidfab::cuda {

enum class QuantFormat {
  kF32,
  kF16,
  kBF16,
  kF8E4M3,   // per-tensor scales
  kI8,       // per-output-channel weight_scale
  kNVFP4,    // 4-bit, block-scaled; `data` is half as many bytes as elements
  kNF4,      // bitsandbytes NF4, double-quantised 64-element block scales
};

// Elements of the contraction axis sharing one e4m3 block scale. Fixed, not a
// parameter: 16 is what `scale_vec::4X` on m16n8k64 implements and what both
// shipped checkpoints are packed for, and a checkpoint using anything else
// would need a different instruction, not a different constant.
constexpr int kNVFP4BlockSize = 16;

// Compute precision for the GEMM itself. Accumulation is fp32 in every case;
// this selects the operand precision.
enum class ComputeType {
  kBF16,  // default for both block stacks; matches the reference's dtype
  kF32,   // patch projections, output heads, AdaLN — see spec section 9.1
};

// A weight as it sits on the device, plus everything needed to interpret it.
// Owns nothing: the pointers belong to whoever loaded the checkpoint.
struct QuantWeight {
  QuantFormat format = QuantFormat::kBF16;
  const void* data = nullptr;  // device, [out_features, in_features] row-major
  int out_features = 0;
  int in_features = 0;

  // Device pointer. One element when `per_channel_scale` is false, otherwise
  // `out_features` elements. Null for unquantised formats.
  const float* weight_scale = nullptr;
  bool per_channel_scale = false;

  // Host scalar. Zero means "absent", which for an fp8 weight is a positive
  // statement that the layer must run at full precision — see the header
  // comment. Never synthesise a value for it.
  float input_scale = 0.0f;

  // The checkpoint's own `full_precision_matrix_mult` flag, read out of the
  // `comfy_quant` blob. Every quantised layer of the nvfp4 text encoder sets
  // it and no layer of the nvfp4 transformer does, so it is the file — not a
  // heuristic on which scales are present — that decides whether a native
  // low-precision GEMM is allowed. Nothing may clear it.
  bool full_precision = false;

  // --- nvfp4 ----------------------------------------------------------------

  // Device, `out_features * in_features / kNVFP4BlockSize` raw e4m3 bytes.
  // Block j of row o scales stored elements [j*16, j*16+16).
  const uint8_t* block_scale = nullptr;

  // Second-level scale, host side, multiplying the whole tensor. Stored as
  // `.weight_scale_2`; one per tensor, so it costs nothing to fold in.
  float global_scale = 1.0f;

  // AWQ, device, `in_features` elements. The activation is scaled by this per
  // input channel before the GEMM. Present on exactly the text-encoder layers
  // whose input does not come straight from a norm — absent elsewhere because
  // the quantiser folded it into the preceding norm's weight, so a null
  // pointer here means "already accounted for", not "unknown". Check per
  // tensor; do not infer it from the layer's name.
  const __nv_bfloat16* pre_quant_scale = nullptr;

  // --- bitsandbytes NF4 ----------------------------------------------------
  // Codes are packed high nibble first. Each 64 weights share an absmax; the
  // absmax bytes are themselves quantised in groups of 256.
  const uint8_t* nf4_absmax = nullptr;
  const float* nf4_quant_map = nullptr;         // 16 entries
  const float* nf4_nested_quant_map = nullptr;  // 256 entries
  const float* nf4_nested_absmax = nullptr;
  int nf4_block_size = 64;
  int nf4_nested_block_size = 256;
  float nf4_nested_offset = 0.0f;

  // ConvRot: the contraction axis was rotated offline in groups of
  // `convrot_group`, so the activation must be rotated the same way online.
  // Only meaningful when `in_features % convrot_group == 0`; the quantiser
  // skips rotation otherwise, so check per tensor rather than assuming.
  bool convrot = false;
  int convrot_group = 256;

  // Optional, device, `out_features` elements. Biases are never quantised and
  // never rotated.
  const void* bias = nullptr;
  QuantFormat bias_format = QuantFormat::kF32;

  bool has_bias() const { return bias != nullptr; }
  size_t stored_bytes() const;
};

// Bytes of workspace `forward` needs for a given weight and batch size. Call
// this over every layer at load time and reserve the maximum once.
size_t linear_workspace_bytes(const QuantWeight& w, int rows, ComputeType compute);

// `linear_workspace_bytes` split at the boundary a hoisting caller needs: the
// dense bf16 copy `prepare` carves and holds for a whole loop, and the per-call
// buffers `forward_prepared` carves and rewinds inside it.
//
// The split is not an exact partition of `linear_workspace_bytes`, because that
// function over-reserves in two cases this one does not:
//   - bf16 weight at fp32 compute: it counts a bf16 copy that `materialise_bf16`
//     never allocates, since a bf16 weight is already dense. The split is
//     smaller here, and the split is the correct number.
//   - fp32 weight at fp32 compute: the split counts a bf16 copy that nothing
//     carves. Harmless over-reservation, kept so the dense side never
//     under-states what a non-bf16 format needs.
// Everywhere else the two sides sum to it. Size an arena from the sum, never
// from one side alone.
size_t linear_dense_weight_bytes(const QuantWeight& w);
size_t linear_activation_workspace_bytes(const QuantWeight& w, int rows, ComputeType compute);

class LinearRunner {
 public:
  LinearRunner() = default;

  void init(cublasHandle_t handle, cudaStream_t stream);

  // Selects native low-precision GEMM where the format supports it. Off by
  // default: the dequantise-then-GEMM path is the reference behaviour and the
  // one the unit tests pin. Turning this on must not change results by more
  // than the per-tensor tolerance (1e-3 abs / 1e-2 rel).
  void set_native(bool enable) { native_ = enable; }
  bool native() const { return native_; }

  // y[rows, out_features] = x[rows, in_features] @ W^T + bias
  //
  // `x` and `y` are row-major and must not alias. Applies ConvRot to a copy of
  // `x` in workspace when the weight requires it; `x` itself is not modified.
  void forward(const QuantWeight& w, const __nv_bfloat16* x, int rows, __nv_bfloat16* y,
               Workspace& ws);

  // Dequantises `w` into `ws` once, for a caller that is about to run several
  // `forward`s against it — the row-chunk loops in the transformer block, where
  // the dense copy is the same every chunk and re-deriving it per chunk is the
  // single largest piece of redundant memory traffic in a denoise step.
  //
  // The returned pointer stays valid until the caller's `Workspace::Scope`
  // rewinds, and must be passed to `forward_prepared` alongside the same `w`.
  // Returns null when `w` takes the native low-precision GEMM instead, which
  // never materialises a dense copy; `forward_prepared` handles that pointer
  // and is the reason this returns rather than throws.
  const __nv_bfloat16* prepare(const QuantWeight& w, Workspace& ws);

  // `forward` with the dequantisation already done. `dense_w` must be what
  // `prepare(w, ...)` returned for this same weight. Null is not "unknown, work
  // it out": it is the positive statement that `w` takes the native nvfp4 GEMM,
  // which reads the stored nibbles and has no dense form, and this call goes
  // straight there. Passing null for a weight that has a dense form throws
  // rather than silently running the wrong GEMM.
  void forward_prepared(const QuantWeight& w, const __nv_bfloat16* dense_w, const __nv_bfloat16* x,
                        int rows, __nv_bfloat16* y, Workspace& ws);

  // fp32 in, fp32 out. For the four fp32 tensors in the transformer — the two
  // patch projections and the two output heads — where the reference aligns
  // the activation with the parameter dtype.
  void forward_f32(const QuantWeight& w, const float* x, int rows, float* y, Workspace& ws);

 private:
  // Whether `w` goes to the native nvfp4 GEMM, which consumes the stored
  // nibbles directly and so has no dense bf16 copy to prepare.
  bool takes_native_nvfp4(const QuantWeight& w) const;

  cublasHandle_t handle_ = nullptr;
  cudaStream_t stream_ = nullptr;
  bool native_ = false;
};

// --- ConvRot ----------------------------------------------------------------
//
// H = kron(h4, h4, h4, h4) / 16 over a 256-wide group, where h4 is the regular
// (not Sylvester) Hadamard matrix. H is symmetric, orthogonal and involutory,
// so the same transform serves both directions. Implemented as a four-stage
// radix-4 butterfly with a single 1/16 at the end. See docs/convrot_notes.md
// — the wrong Hadamard gives relative error 1.4, not a crash.
void launch_convrot(const __nv_bfloat16* in, __nv_bfloat16* out, int rows, int dim, int group,
                    cudaStream_t stream);
void launch_convrot_f32(const float* in, float* out, int rows, int dim, int group,
                        cudaStream_t stream);

// --- dequantisation ---------------------------------------------------------

// dst[i] = f8_e4m3(src[i]) * (*scale). `scale` is a device scalar.
void launch_dequant_f8e4m3(const uint8_t* src, const float* scale, __nv_bfloat16* dst, size_t n,
                           cudaStream_t stream);

// dst[o, i] = src[o, i] * scale[o]. Per-output-channel, as ConvRot int8 uses
// despite its format tag reading "int8_tensorwise".
void launch_dequant_i8_per_channel(const int8_t* src, const float* scale, __nv_bfloat16* dst,
                                   int out_features, int in_features, cudaStream_t stream);

// dst[o, i] = e2m1(nibble i of row o) * e4m3(block scale of (o, i/16)) * global.
// `src` holds `out_features * in_features / 2` bytes; `block_scale` holds
// `out_features * in_features / 16` raw e4m3 bytes.
//
// Two things about the stored bytes that the shapes do not tell you, both
// measured rather than assumed, and the earlier version of this comment had the
// first of them backwards:
//
//   - the **high** nibble of a byte is the even-indexed element (dtype.h
//     records how that was established, and why no summary statistic caught it);
//   - `block_scale` is **not** plain `[out, in/16]` row-major. It is swizzled
//     into 512-byte tiles, so the implementation unswizzles on the way in.
//
// Both hold on both checkpoints despite their different quantiser provenance,
// so they are constants and this signature takes no layout argument. Shipping a
// parameter with one valid value would only be an invitation to pass the wrong
// one; the wrong forms stay computable in the CPU-reference test instead, so a
// toolkit that ever flips the convention fails by name rather than silently.
//
// `global_scale` multiplies: `w = e2m1 * e4m3_block * weight_scale_2`. Checked,
// not inferred from the vendor's convention — `6 * 448 * weight_scale_2`
// reproduces the fp8 checkpoint's amax for the same tensor to four decimals.
void launch_dequant_nvfp4(const uint8_t* src, const uint8_t* block_scale, float global_scale,
                          __nv_bfloat16* dst, int out_features, int in_features,
                          cudaStream_t stream);

void launch_dequant_nf4(const uint8_t* src, const uint8_t* absmax, const float* quant_map,
                        const float* nested_quant_map, const float* nested_absmax,
                        int block_size, int nested_block_size, float nested_offset,
                        __nv_bfloat16* dst, int out_features, int in_features,
                        cudaStream_t stream);

void launch_dequant_nf4_f16(const uint8_t* src, const uint8_t* absmax, const float* quant_map,
                            const float* nested_quant_map, const float* nested_absmax,
                            int block_size, int nested_block_size, float nested_offset,
                            __half* dst, size_t n, cudaStream_t stream);

// dst[r, i] = src[r, i] * scale[i]. The AWQ activation scaling; separate from
// the GEMM because it also has to happen ahead of a native fp4 path.
void launch_pre_quant_scale(const __nv_bfloat16* src, const __nv_bfloat16* scale,
                            __nv_bfloat16* dst, int rows, int dim, cudaStream_t stream);

// dst[i] = clamp(src[i] / input_scale, -448, 448) rounded to e4m3.
void launch_quantize_f8e4m3(const __nv_bfloat16* src, float input_scale, uint8_t* dst, size_t n,
                            cudaStream_t stream);

void launch_widen_bf16(const __nv_bfloat16* src, float* dst, size_t n, cudaStream_t stream);
void launch_narrow_to_bf16(const float* src, __nv_bfloat16* dst, size_t n, cudaStream_t stream);

}  // namespace vidfab::cuda
