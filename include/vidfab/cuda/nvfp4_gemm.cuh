// Native block-scaled nvfp4 GEMM.
//
// `y[rows, out] = x[rows, in] @ W^T`, with W as the checkpoint stores it —
// E2M1 nibbles, an E4M3 scale per 16 contracted elements, one fp32 scale on
// the whole tensor — and nothing dequantised on the way. The reference path in
// linear.cu expands W to bf16 in workspace and calls cuBLAS; this exists to
// delete that expansion, which for `mlp.fc1` writes and re-reads 308 MB per
// call to save reading 77 MB.
//
// The activation has to be quantised to match. There is no `input_scale` in
// either nvfp4 checkpoint — it is not that the loader has not found it, it is
// not in the files — so the quantisation is dynamic, per 16-element block
// along the contraction axis, and `launch_quantize_nvfp4_activations` is where
// that rule lives.
//
// **Shape requirement.** `in_features % 64 == 0`. Not a tensor-core
// constraint: it is what makes the packed row (`in/2` bytes) 32-byte aligned
// and the scale row (`in/16` bytes) 4-byte aligned, so both stage into shared
// memory with vector loads and no partial-word tail. Every H3 transformer
// tensor satisfies it (5376, 7168, 14336, 28672). `nvfp4_gemm_supported`
// reports it and the caller must fall back to dequantise-then-cuBLAS when it
// is false — a shape this cannot take is not an error, it is a shape for the
// reference path.
#pragma once

#include <cuda_bf16.h>

#include <cstddef>
#include <cstdint>

#include "vidfab/cuda/workspace.cuh"

namespace vidfab::cuda {

// Rows quantised per internal pass. The activation buffer is what makes the
// native path's workspace grow with the batch, and bounding it here keeps that
// demand under what `linear_workspace_bytes` already reserves for the
// dequantised weight at every production shape. The transformer never asks for
// more than this in one call anyway (transformer.cpp::kRowChunk).
constexpr int kNVFP4RowChunk = 8192;

// --- how the weight sits on disk ---------------------------------------------
//
// How the quantiser wrote the checkpoint's bytes is a different question from
// how the tensor core reads its registers. The register layout is pinned by
// `nvfp4_mma_operand_layout`; this is the other end, and the load path between
// the two is not the identity. Two things about it, both established from data
// rather than from the shape metadata:
//
//   * **The high nibble is the even-indexed element**, not the low one.
//   * **`.weight_scale` is swizzled**, not `[out, in/16]` row-major. The scale
//     for output row `m`, block `j` of `Kb = in/16` sits at byte
//
//       ((m/128) * (Kb/4) + j/4) * 512 + (m%32) * 16 + ((m%128)/32) * 4 + j%4
//
// Neither is a per-file property: the two independently quantised checkpoints
// agree, so both are constants here rather than arguments. The *wrong* form of
// each is still computed in `nvfp4_gemm_disk_layout`, which asserts the kernel
// matches one and not the other — a convention that flips under a future
// toolkit should fail with a named test, not decay into plausible output.
//
// The swizzle costs this kernel an address computation and nothing else: a
// `scale_vec::4X` operand consumes four consecutive blocks starting at a
// multiple of four, and the swizzle keeps exactly those four bytes contiguous
// and 4-byte aligned. Only the weight is affected — the activation is
// quantised by this file, so it is written the way the instruction reads it.
//
// The swizzle's no-padding precondition is `out % 128 == 0` and `Kb % 4 == 0`.
// Every quantised tensor in both checkpoints satisfies both, so the padded case
// has never been observed and its convention is not guessed at: it throws.
bool nvfp4_gemm_supported(int out_features, int in_features);

// Workspace the GEMM carves for the quantised activation and its scales.
size_t nvfp4_gemm_workspace_bytes(int rows, int in_features);

// x[rows, dim] bf16 -> `packed` (dim/2 bytes per row, low nibble = even index)
// and `scales` (dim/16 raw e4m3 bytes per row). `dim % 16 == 0` is required;
// blocks never straddle a row, so the pair is addressed as one flat run of
// `rows * dim / 16` blocks.
void launch_quantize_nvfp4_activations(const __nv_bfloat16* x, uint8_t* packed, uint8_t* scales,
                                       int rows, int dim, cudaStream_t stream);

// y[rows, out_features] = x[rows, in_features] @ W^T * global_scale.
//
// No bias — the caller adds it, as it does on the reference path. `w_packed`
// is `[out_features, in_features/2]` bytes and `w_scale` is
// `[out_features, in_features/16]` raw e4m3 bytes, both exactly as stored.
void nvfp4_gemm_forward(const __nv_bfloat16* x, const uint8_t* w_packed, const uint8_t* w_scale,
                        float global_scale, __nv_bfloat16* y, int rows, int out_features,
                        int in_features, Workspace& ws, cudaStream_t stream);

}  // namespace vidfab::cuda
