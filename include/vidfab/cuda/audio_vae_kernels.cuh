// Kernel launchers for the MiniMax H3 audio VAE decoder (DAC front end plus a
// BigVGAN vocoder). All buffers are device fp32 in [B, C, T] layout: channel
// major with contiguous time, element (b, c, t) at ((b*C) + c)*T + t.
//
// The full derivation of every formula here — with citations into
// ref/FL2VA/audio_vae/*.py — is docs/audio_vae_spec.md. The short version of
// why these are direct kernels rather than cuBLAS calls: channel counts run
// from 2048 down to 1 while lengths run to 324,000, so the natural GEMM shapes
// are degenerate at both ends of the stack.
//
// Three of these are the ones that go wrong quietly:
//
//   * snake_beta takes alpha and beta in LOG space, and divides by beta, not
//     by alpha (docs/audio_vae_spec.md §6). Feeding it linear-scale parameters
//     clips 90% of the output; dividing by alpha sounds fine and is wrong.
//   * aa_upsample_snake and aa_downsample use REPLICATE padding with
//     asymmetric crops (15/15 after a 2x expansion, 5/6 before a 2x decimation)
//     that put the resampled signal on the correct half-sample phase. A
//     symmetric pad runs, returns the right length, and aliases.
//   * conv1d is a cross-correlation, matching torch. The kernel is not
//     reversed.
#pragma once

#include <cuda_runtime.h>

#include "vidfab/cuda/device.h"

namespace vidfab::cuda {

// Anti-alias resamplers are hard-wired to BigVGAN's Activation1d defaults:
// ratio 2 up and down, 12-tap Kaiser-windowed sinc shipped as a weight.
constexpr int kAudioAAKernel = 2 * 6;
constexpr int kAudioAARatio = 2;

// y[b][co][n] = bias[co] + sum_ci sum_k x[b][ci][n + k*dilation - pad] * w[co][ci][k]
//
// `w` is [Cout, Cin, K] and `bias` may be null (conv_post ships without one).
// Reads outside [0, len_in) contribute zero, i.e. torch's default zero padding.
// STRIDE IS ALWAYS 1 — no convolution in the decode path strides, and the
// launcher rejects any other request rather than silently ignoring it.
// `len_out` must equal len_in + 2*pad - dilation*(K-1); the caller passes it so
// the arithmetic is checked in one place.
void launch_conv1d(const float* x, const float* w, const float* bias, float* y, int batch,
                   int in_channels, int out_channels, int len_in, int len_out, int kernel,
                   int pad, int dilation, cudaStream_t stream);

// Transposed convolution, the BigVGAN upsamplers.
//
//   y[b][co][i*stride + k - pad] += x[b][ci][i] * w[ci][co][k]
//
// NOTE the weight layout: [Cin, Cout, K], input channels first, the opposite of
// launch_conv1d. That is torch's ConvTranspose1d convention and the checkpoint
// follows it (decoder.ups.0.0.weight is [1024, 512, 9] for a 1024 -> 512 stage).
//
// output_padding and dilation are fixed at 0 and 1, so
// len_out = (len_in - 1)*stride - 2*pad + kernel. With pad = (kernel-stride)/2
// as the reference builds them, that is exactly len_in*stride.
void launch_conv_transpose1d(const float* x, const float* w, const float* bias, float* y, int batch,
                             int in_channels, int out_channels, int len_in, int len_out,
                             int kernel, int stride, int pad, cudaStream_t stream);

// y = x + sin(exp(log_alpha[c]) * x)^2 / (exp(log_beta[c]) + 1e-9), in place.
//
// log_alpha and log_beta are [channels], exactly as stored in the checkpoint —
// the exp is done here so no caller can forget it. The 1e-9 guard goes on the
// exponentiated beta, not on the log.
void launch_snake_beta(float* x, const float* log_alpha, const float* log_beta, int batch,
                       int channels, int len, cudaStream_t stream);

// The first two thirds of Activation1d: 2x anti-aliased upsample followed by
// SnakeBeta, fused because the intermediate is 2x the largest tensor in the
// pipeline and is read exactly once.
//
//   y[b][c][n] = snake( 2 * sum_{i<6} x[b][c][clamp(((n+15)>>1) - i - 5)] * f[((n+15)&1) + 2i] )
//
// `filter` is the shipped 12-tap buffer. `y` holds 2*len_in samples per channel.
void launch_aa_upsample_snake(const float* x, const float* filter, const float* log_alpha,
                              const float* log_beta, float* y, int batch, int channels, int len_in,
                              cudaStream_t stream);

// The last third of Activation1d: 12-tap lowpass with replicate padding (5 left,
// 6 right) decimated by 2.
//
//   y[b][c][n] = sum_{k<12} x[b][c][clamp(2n + k - 5, 0, len_in-1)] * filter[k]
//
// len_out is (len_in - 1)/2 + 1, which for the even len_in this always sees is
// len_in/2 — the wrapper as a whole preserves length.
void launch_aa_downsample(const float* x, const float* filter, float* y, int batch, int channels,
                          int len_in, int len_out, cudaStream_t stream);

// x += y, elementwise. The AMPBlock1 residual.
void launch_add_inplace(float* x, const float* y, size_t count, cudaStream_t stream);

// x *= scale. BigVGAN averages its three resblocks rather than summing them.
void launch_scale_inplace(float* x, float scale, size_t count, cudaStream_t stream);

// x = min(max(x, lo), hi). use_tanh_at_final is false for this checkpoint, so
// the output bound is a clamp, not a tanh.
void launch_clamp_inplace(float* x, float lo, float hi, size_t count, cudaStream_t stream);

// [batch, 1, frames] planar -> [frames, batch] interleaved, which is what the
// WAV writer and the muxer both want.
void launch_interleave(const float* planar, float* interleaved, int batch, int frames,
                       cudaStream_t stream);

}  // namespace vidfab::cuda
