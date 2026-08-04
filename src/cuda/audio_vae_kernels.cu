// Kernels for the MiniMax H3 audio VAE decoder.
//
// Everything computes in fp32; the checkpoint is fp32 throughout, so this is
// the arithmetic being matched rather than a conservative choice.
//
// The shapes here are the opposite of the video VAE's: few channels, very long
// sequences. The last upsample stage is 8 channels by 324,000 samples for a
// ten-second clip. That rules out the im2col + GEMM route (the contraction
// dimension would be 8) and rules in direct convolution with a shared-memory
// tile over time. See docs/audio_vae_spec.md §13.

#include "vidfab/cuda/audio_vae_kernels.cuh"

#include <cmath>
#include <stdexcept>
#include <string>

namespace vidfab::cuda {
namespace {

// --- conv1d tiling ----------------------------------------------------------
//
// One block owns kConvTime consecutive output positions and kConvOut output
// channels; one thread owns kConvUnroll of those positions (strided by the
// block width so shared loads stay coalesced) across all kConvOut channels.
// Holding the whole output-channel group in registers is what buys the reuse:
// each shared-memory weight read feeds kConvUnroll fused multiply-adds and each
// shared-memory activation read feeds kConvOut of them.
constexpr int kConvThreads = 128;
constexpr int kConvUnroll = 4;
constexpr int kConvTime = kConvThreads * kConvUnroll;  // 512
constexpr int kConvOut = 8;
constexpr int kConvIn = 4;  // input channels staged per pass

// Widest activation window a single block needs: kConvTime taps plus the
// dilated kernel's reach. The decoder's worst case is kernel 11 at dilation 5.
__host__ __device__ inline int conv_span(int kernel, int dilation) {
  return kConvTime + (kernel - 1) * dilation;
}

__global__ void conv1d_kernel(const float* __restrict__ x, const float* __restrict__ w,
                              const float* __restrict__ bias, float* __restrict__ y, int in_ch,
                              int out_ch, int len_in, int len_out, int kernel, int pad,
                              int dilation) {
  extern __shared__ float smem[];
  const int span = conv_span(kernel, dilation);
  float* sx = smem;                   // [kConvIn][span]
  float* sw = smem + kConvIn * span;  // [kConvIn][kConvOut][kernel]

  const int b = static_cast<int>(blockIdx.z);
  const int co0 = static_cast<int>(blockIdx.y) * kConvOut;
  const int t0 = static_cast<int>(blockIdx.x) * kConvTime;
  const int tid = static_cast<int>(threadIdx.x);

  float acc[kConvUnroll][kConvOut];
#pragma unroll
  for (int u = 0; u < kConvUnroll; ++u) {
#pragma unroll
    for (int o = 0; o < kConvOut; ++o) {
      const int co = co0 + o;
      acc[u][o] = (bias != nullptr && co < out_ch) ? bias[co] : 0.0f;
    }
  }

  const size_t x_batch = static_cast<size_t>(b) * in_ch * len_in;
  const int weights_per_pass = kConvIn * kConvOut * kernel;

  for (int ci0 = 0; ci0 < in_ch; ci0 += kConvIn) {
    __syncthreads();
    for (int idx = tid; idx < kConvIn * span; idx += kConvThreads) {
      const int c = idx / span;
      const int s = idx - c * span;
      const int ci = ci0 + c;
      const int t = t0 + s - pad;
      float v = 0.0f;
      if (ci < in_ch && t >= 0 && t < len_in) v = x[x_batch + static_cast<size_t>(ci) * len_in + t];
      sx[idx] = v;
    }
    for (int idx = tid; idx < weights_per_pass; idx += kConvThreads) {
      const int c = idx / (kConvOut * kernel);
      const int rem = idx - c * kConvOut * kernel;
      const int o = rem / kernel;
      const int k = rem - o * kernel;
      const int ci = ci0 + c;
      const int co = co0 + o;
      sw[idx] =
          (ci < in_ch && co < out_ch) ? w[(static_cast<size_t>(co) * in_ch + ci) * kernel + k] : 0.0f;
    }
    __syncthreads();

    for (int c = 0; c < kConvIn; ++c) {
      const float* sxc = sx + c * span;
      const float* swc = sw + c * kConvOut * kernel;
      for (int k = 0; k < kernel; ++k) {
        float wv[kConvOut];
#pragma unroll
        for (int o = 0; o < kConvOut; ++o) wv[o] = swc[o * kernel + k];
        const int base = tid + k * dilation;
#pragma unroll
        for (int u = 0; u < kConvUnroll; ++u) {
          const float xv = sxc[base + u * kConvThreads];
#pragma unroll
          for (int o = 0; o < kConvOut; ++o) acc[u][o] = fmaf(xv, wv[o], acc[u][o]);
        }
      }
    }
  }

  const size_t y_batch = static_cast<size_t>(b) * out_ch * len_out;
#pragma unroll
  for (int u = 0; u < kConvUnroll; ++u) {
    const int n = t0 + tid + u * kConvThreads;
    if (n >= len_out) continue;
#pragma unroll
    for (int o = 0; o < kConvOut; ++o) {
      const int co = co0 + o;
      if (co < out_ch) y[y_batch + static_cast<size_t>(co) * len_out + n] = acc[u][o];
    }
  }
}

// --- transposed conv --------------------------------------------------------
//
// Gather form (docs/audio_vae_spec.md §5): output n draws on input j0 - t at
// tap phase + t*stride, where phase = (n + pad) % stride and j0 = (n + pad) /
// stride. With the shipped (kernel, stride) pairs — (9, 5) and (4, 2) — only two
// taps survive per output, so this is ~3% of the decoder's arithmetic and is
// kept simple: weights through shared memory, activations straight from L2.
constexpr int kUpThreads = 128;
constexpr int kUpOut = 8;
constexpr int kUpIn = 8;

__global__ void conv_transpose1d_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                        const float* __restrict__ bias, float* __restrict__ y,
                                        int in_ch, int out_ch, int len_in, int len_out, int kernel,
                                        int stride, int pad) {
  extern __shared__ float sw[];  // [kUpIn][kUpOut][kernel], one pass at a time

  const int b = static_cast<int>(blockIdx.z);
  const int co0 = static_cast<int>(blockIdx.y) * kUpOut;
  const int tid = static_cast<int>(threadIdx.x);
  const int n = static_cast<int>(blockIdx.x) * kUpThreads + tid;

  float acc[kUpOut];
#pragma unroll
  for (int o = 0; o < kUpOut; ++o) {
    const int co = co0 + o;
    acc[o] = (bias != nullptr && co < out_ch) ? bias[co] : 0.0f;
  }

  // The phase is per thread, not per block: kUpThreads is not a multiple of a
  // stride of 5, so neighbouring outputs draw on different taps.
  const int m = n + pad;
  const int phase = m % stride;
  const int j0 = m / stride;

  const int weights_per_pass = kUpIn * kUpOut * kernel;
  const size_t x_batch = static_cast<size_t>(b) * in_ch * len_in;

  for (int ci0 = 0; ci0 < in_ch; ci0 += kUpIn) {
    __syncthreads();
    for (int idx = tid; idx < weights_per_pass; idx += kUpThreads) {
      const int c = idx / (kUpOut * kernel);
      const int rem = idx - c * kUpOut * kernel;
      const int o = rem / kernel;
      const int k = rem - o * kernel;
      const int ci = ci0 + c;
      const int co = co0 + o;
      sw[idx] =
          (ci < in_ch && co < out_ch) ? w[(static_cast<size_t>(ci) * out_ch + co) * kernel + k] : 0.0f;
    }
    __syncthreads();
    // Every thread keeps iterating the (uniform) ci0 loop so the barriers above
    // stay collective; only the arithmetic is skipped.
    if (n >= len_out) continue;

    for (int c = 0; c < kUpIn; ++c) {
      if (ci0 + c >= in_ch) break;
      const float* swc = sw + c * kUpOut * kernel;
      const size_t x_row = x_batch + static_cast<size_t>(ci0 + c) * len_in;
      for (int k = phase, t = 0; k < kernel; k += stride, ++t) {
        const int j = j0 - t;
        if (j < 0) break;
        if (j >= len_in) continue;
        const float xv = x[x_row + j];
#pragma unroll
        for (int o = 0; o < kUpOut; ++o) acc[o] = fmaf(xv, swc[o * kernel + k], acc[o]);
      }
    }
  }

  if (n >= len_out) return;
  const size_t y_batch = static_cast<size_t>(b) * out_ch * len_out;
#pragma unroll
  for (int o = 0; o < kUpOut; ++o) {
    const int co = co0 + o;
    if (co < out_ch) y[y_batch + static_cast<size_t>(co) * len_out + n] = acc[o];
  }
}

// --- SnakeBeta and the anti-alias resamplers --------------------------------

// sinf, not __sinf: the argument is alpha*x with alpha up to 4.4 and x up to
// ~3.5, and the fast intrinsic's relative error would eat most of the 1e-3
// absolute budget after 127 activations.
__device__ __forceinline__ float snake(float v, float alpha, float beta_recip) {
  const float s = sinf(alpha * v);
  return fmaf(s * s, beta_recip, v);
}

__global__ void snake_beta_kernel(float* __restrict__ x, const float* __restrict__ log_alpha,
                                  const float* __restrict__ log_beta, int channels, int len) {
  const int c = static_cast<int>(blockIdx.y);
  const int b = static_cast<int>(blockIdx.z);
  const float alpha = expf(log_alpha[c]);
  const float beta_recip = 1.0f / (expf(log_beta[c]) + 1e-9f);
  float* row = x + (static_cast<size_t>(b) * channels + c) * len;
  const int stride = static_cast<int>(gridDim.x) * static_cast<int>(blockDim.x);
  for (int n = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
               static_cast<int>(threadIdx.x);
       n < len; n += stride) {
    row[n] = snake(row[n], alpha, beta_recip);
  }
}

// docs/audio_vae_spec.md §7.2. The 15-sample left crop and the replicate
// padding collapse into a clamped index; the six taps that survive are the ones
// whose parity matches n + 15.
__global__ void aa_upsample_snake_kernel(const float* __restrict__ x,
                                         const float* __restrict__ filter,
                                         const float* __restrict__ log_alpha,
                                         const float* __restrict__ log_beta, float* __restrict__ y,
                                         int channels, int len_in) {
  __shared__ float f[kAudioAAKernel];
  if (static_cast<int>(threadIdx.x) < kAudioAAKernel) f[threadIdx.x] = filter[threadIdx.x];
  __syncthreads();

  const int c = static_cast<int>(blockIdx.y);
  const int b = static_cast<int>(blockIdx.z);
  const float alpha = expf(log_alpha[c]);
  const float beta_recip = 1.0f / (expf(log_beta[c]) + 1e-9f);

  const int len_out = kAudioAARatio * len_in;
  const float* row = x + (static_cast<size_t>(b) * channels + c) * len_in;
  float* out = y + (static_cast<size_t>(b) * channels + c) * len_out;

  constexpr int kHalfTaps = kAudioAAKernel / 2;  // 6 of the 12 taps, by parity
  constexpr int kCrop = 15;  // pad*stride + (kernel - stride)/2, the left crop
  constexpr int kLag = 5;    // the replicate pad, kernel/ratio - 1

  const int stride = static_cast<int>(gridDim.x) * static_cast<int>(blockDim.x);
  for (int n = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
               static_cast<int>(threadIdx.x);
       n < len_out; n += stride) {
    const int m = n + kCrop;
    const int parity = m & 1;
    const int j = (m >> 1) - kLag;
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < kHalfTaps; ++i) {
      int s = j - i;
      s = s < 0 ? 0 : (s >= len_in ? len_in - 1 : s);
      sum = fmaf(row[s], f[parity + 2 * i], sum);
    }
    out[n] = snake(sum * static_cast<float>(kAudioAARatio), alpha, beta_recip);
  }
}

// docs/audio_vae_spec.md §7.3. Replicate pad 5 left / 6 right, then a stride-2
// correlation. The asymmetry is the phase and must not be tidied up.
__global__ void aa_downsample_kernel(const float* __restrict__ x, const float* __restrict__ filter,
                                     float* __restrict__ y, int channels, int len_in, int len_out) {
  __shared__ float f[kAudioAAKernel];
  if (static_cast<int>(threadIdx.x) < kAudioAAKernel) f[threadIdx.x] = filter[threadIdx.x];
  __syncthreads();

  const int c = static_cast<int>(blockIdx.y);
  const int b = static_cast<int>(blockIdx.z);
  const float* row = x + (static_cast<size_t>(b) * channels + c) * len_in;
  float* out = y + (static_cast<size_t>(b) * channels + c) * len_out;

  const int stride = static_cast<int>(gridDim.x) * static_cast<int>(blockDim.x);
  for (int n = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
               static_cast<int>(threadIdx.x);
       n < len_out; n += stride) {
    float sum = 0.0f;
#pragma unroll
    for (int k = 0; k < kAudioAAKernel; ++k) {
      int s = kAudioAARatio * n + k - 5;
      s = s < 0 ? 0 : (s >= len_in ? len_in - 1 : s);
      sum = fmaf(row[s], f[k], sum);
    }
    out[n] = sum;
  }
}

// --- elementwise ------------------------------------------------------------

__global__ void add_inplace_kernel(float* __restrict__ x, const float* __restrict__ y,
                                   size_t count) {
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < count;
       i += stride) {
    x[i] += y[i];
  }
}

__global__ void scale_inplace_kernel(float* __restrict__ x, float scale, size_t count) {
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < count;
       i += stride) {
    x[i] *= scale;
  }
}

__global__ void clamp_inplace_kernel(float* __restrict__ x, float lo, float hi, size_t count) {
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < count;
       i += stride) {
    x[i] = fminf(fmaxf(x[i], lo), hi);
  }
}

__global__ void interleave_kernel(const float* __restrict__ planar, float* __restrict__ out,
                                  int batch, int frames) {
  const size_t count = static_cast<size_t>(batch) * frames;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < count;
       i += stride) {
    const int b = static_cast<int>(i / static_cast<size_t>(frames));
    const int t = static_cast<int>(i - static_cast<size_t>(b) * frames);
    out[static_cast<size_t>(t) * batch + b] = planar[i];
  }
}

int elementwise_blocks(size_t count) {
  const size_t blocks = (count + 255) / 256;
  if (blocks == 0) return 1;
  return static_cast<int>(blocks > 4096 ? 4096 : blocks);
}

int ceil_div(int a, int b) { return (a + b - 1) / b; }

// Anti-alias and Snake kernels are memory bound; capping the time axis keeps
// the grid-stride loop from degenerating into one element per block.
dim3 elementwise_grid(int len, int channels, int batch) {
  const int tiles = ceil_div(len, 256);
  return dim3(static_cast<unsigned>(tiles > 256 ? 256 : tiles), static_cast<unsigned>(channels),
              static_cast<unsigned>(batch));
}

}  // namespace

void launch_conv1d(const float* x, const float* w, const float* bias, float* y, int batch,
                   int in_channels, int out_channels, int len_in, int len_out, int kernel, int pad,
                   int dilation, cudaStream_t stream) {
  if (kernel <= 0 || dilation <= 0 || len_in <= 0 || len_out <= 0 || in_channels <= 0 ||
      out_channels <= 0 || batch <= 0) {
    throw std::runtime_error("launch_conv1d: non-positive extent");
  }
  const int expected = len_in + 2 * pad - dilation * (kernel - 1);
  if (expected != len_out) {
    throw std::runtime_error("launch_conv1d: len_out " + std::to_string(len_out) +
                             " disagrees with the padding arithmetic (" + std::to_string(expected) +
                             ")");
  }
  const dim3 grid(static_cast<unsigned>(ceil_div(len_out, kConvTime)),
                  static_cast<unsigned>(ceil_div(out_channels, kConvOut)),
                  static_cast<unsigned>(batch));
  const size_t shared = sizeof(float) * (static_cast<size_t>(kConvIn) * conv_span(kernel, dilation) +
                                         static_cast<size_t>(kConvIn) * kConvOut * kernel);
  conv1d_kernel<<<grid, kConvThreads, shared, stream>>>(x, w, bias, y, in_channels, out_channels,
                                                        len_in, len_out, kernel, pad, dilation);
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

void launch_conv_transpose1d(const float* x, const float* w, const float* bias, float* y, int batch,
                             int in_channels, int out_channels, int len_in, int len_out, int kernel,
                             int stride, int pad, cudaStream_t stream) {
  if (kernel <= 0 || stride <= 0 || len_in <= 0 || len_out <= 0 || in_channels <= 0 ||
      out_channels <= 0 || batch <= 0) {
    throw std::runtime_error("launch_conv_transpose1d: non-positive extent");
  }
  const int expected = (len_in - 1) * stride - 2 * pad + kernel;
  if (expected != len_out) {
    throw std::runtime_error("launch_conv_transpose1d: len_out " + std::to_string(len_out) +
                             " disagrees with the stride arithmetic (" + std::to_string(expected) +
                             ")");
  }
  const dim3 grid(static_cast<unsigned>(ceil_div(len_out, kUpThreads)),
                  static_cast<unsigned>(ceil_div(out_channels, kUpOut)),
                  static_cast<unsigned>(batch));
  const size_t shared = sizeof(float) * static_cast<size_t>(kUpIn) * kUpOut * kernel;
  conv_transpose1d_kernel<<<grid, kUpThreads, shared, stream>>>(
      x, w, bias, y, in_channels, out_channels, len_in, len_out, kernel, stride, pad);
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

void launch_snake_beta(float* x, const float* log_alpha, const float* log_beta, int batch,
                       int channels, int len, cudaStream_t stream) {
  snake_beta_kernel<<<elementwise_grid(len, channels, batch), 256, 0, stream>>>(x, log_alpha,
                                                                               log_beta, channels,
                                                                               len);
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

void launch_aa_upsample_snake(const float* x, const float* filter, const float* log_alpha,
                              const float* log_beta, float* y, int batch, int channels, int len_in,
                              cudaStream_t stream) {
  const dim3 grid = elementwise_grid(kAudioAARatio * len_in, channels, batch);
  aa_upsample_snake_kernel<<<grid, 256, 0, stream>>>(x, filter, log_alpha, log_beta, y, channels,
                                                     len_in);
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

void launch_aa_downsample(const float* x, const float* filter, float* y, int batch, int channels,
                          int len_in, int len_out, cudaStream_t stream) {
  const int expected = (len_in - 1) / kAudioAARatio + 1;
  if (expected != len_out) {
    throw std::runtime_error("launch_aa_downsample: len_out " + std::to_string(len_out) +
                             " disagrees with the decimation arithmetic (" +
                             std::to_string(expected) + ")");
  }
  aa_downsample_kernel<<<elementwise_grid(len_out, channels, batch), 256, 0, stream>>>(
      x, filter, y, channels, len_in, len_out);
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

void launch_add_inplace(float* x, const float* y, size_t count, cudaStream_t stream) {
  add_inplace_kernel<<<elementwise_blocks(count), 256, 0, stream>>>(x, y, count);
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

void launch_scale_inplace(float* x, float scale, size_t count, cudaStream_t stream) {
  scale_inplace_kernel<<<elementwise_blocks(count), 256, 0, stream>>>(x, scale, count);
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

void launch_clamp_inplace(float* x, float lo, float hi, size_t count, cudaStream_t stream) {
  clamp_inplace_kernel<<<elementwise_blocks(count), 256, 0, stream>>>(x, lo, hi, count);
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

void launch_interleave(const float* planar, float* interleaved, int batch, int frames,
                       cudaStream_t stream) {
  const size_t count = static_cast<size_t>(batch) * frames;
  interleave_kernel<<<elementwise_blocks(count), 256, 0, stream>>>(planar, interleaved, batch,
                                                                   frames);
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

}  // namespace vidfab::cuda
