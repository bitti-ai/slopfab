// Kernel launchers for the VAE decoder. All buffers are device fp32.
#pragma once

#include <cuda_runtime.h>

#include "vidfab/cuda/device.h"

namespace vidfab::cuda {

// y = x / sqrt(mean(x^2) + eps) * weight, per row of `dim` elements.
void launch_rmsnorm(const float* x, const float* weight, float* out, int rows, int dim, float eps,
                    cudaStream_t stream);

// y = (x - mean)/sqrt(var + eps) * weight + bias, biased variance.
void launch_layernorm(const float* x, const float* weight, const float* bias, float* out, int rows,
                      int dim, float eps, cudaStream_t stream);

void launch_add_bias(float* y, const float* bias, int rows, int cols, cudaStream_t stream);

// Splits interleaved fused QKV, applies non-affine RMSNorm over head_dim, then
// RoPE over the first `rope_dim` dims. Outputs are head-major [H][S][D].
// Tokens at index >= num_patches are suffix tokens and skip rotation.
void launch_split_qkv_norm_rope(const float* qkv, const float* cos_tab, const float* sin_tab,
                                float* q, float* k, float* v, int seq, int heads, int head_dim,
                                int rope_dim, int num_patches, float eps, cudaStream_t stream);

void launch_softmax_rows(float* scores, int rows, int cols, float scale, cudaStream_t stream);

void launch_merge_heads(const float* in, float* out, int seq, int heads, int head_dim,
                        cudaStream_t stream);

// x += (y + bias) * scale, bias and scale broadcast over columns (LayerScale).
// `bias` may be null when it has already been applied.
void launch_layerscale_residual(float* x, const float* y, const float* bias, const float* scale,
                                int rows, int cols, cudaStream_t stream);

// out = silu(in[:, :inner] + bias[:inner]) * (in[:, inner:] + bias[inner:])
// `bias` may be null.
void launch_swiglu(const float* in, const float* bias, float* out, int rows, int inner,
                   cudaStream_t stream);

// Widens `count` fp16 values to fp32 on the device.
void launch_widen_f16(const void* src, float* dst, size_t count, cudaStream_t stream);

// [channels, voxels] -> [voxels, channels]
void launch_transpose_cn_to_nc(const float* src, float* dst, int channels, int voxels,
                               cudaStream_t stream);

// tokens [T*H*W, channels*patch_t*patch*patch] -> out [channels, T*patch_t, H*patch, W*patch]
void launch_depth_to_space(const float* tokens, float* out, int T, int H, int W, int channels,
                           int patch_t, int patch, cudaStream_t stream);

// out = z_norm * std + mean, per channel over [channels, voxels].
void launch_latent_denorm(const float* z_norm, const float* mean, const float* std_dev, float* out,
                          int channels, int voxels, cudaStream_t stream);

}  // namespace vidfab::cuda
