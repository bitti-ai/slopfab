// Kernels that exist only for the H3 DiT.
//
// Everything genuinely shared with the text encoder or the VAE lives in
// nn_kernels.cu; what is left here is the rank-8 AdaLN expansion, which has no
// analogue anywhere else, and one ungated residual add for the token refiner.
//
// The launchers are declared where they are used (src/dit/transformer.cpp)
// rather than in a header: they take plain host types, they have exactly one
// caller, and adding a public CUDA header for two functions would pull
// cuda_bf16.h into every translation unit that includes the transformer.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

#include "vidfab/cuda/device.h"

namespace vidfab::cuda {
namespace {

constexpr int kThreads = 256;

inline int grid_1d(size_t n, int block) { return static_cast<int>((n + block - 1) / block); }

// --- rank-8 AdaLN expansion -------------------------------------------------
//
// m = W_8 @ c(t) + b, evaluated in fp32 (spec 9.1). W_8 is `[out_features,
// rank]` row-major and `out_features` decomposes as
//
//     p = modality * (num_param * channels) + param * channels + channel
//
// (spec 3.2), which is modality-outer and parameter-inner. The consumers —
// `launch_rmsnorm_modulate` and `launch_add_gated` — index a *single*
// parameter's table by the per-row modulation index `a[row] = t*3 + tag`, with
// a row stride of `channels`. So the write here is a transpose: the parameter
// axis moves outermost and the modality axis moves inside the timestep axis,
// giving `[num_param][num_t * num_modality][channels]`. Six pointers into that
// buffer, spaced `num_t * num_modality * channels` apart, are the six
// modulation tables of one block.
//
// The final layer goes through the same kernel with `num_modality = 1` and
// `num_param = 2`, because it selects on `timestep_indices` alone and has no
// modality dependence (spec 3.2).
//
// The write is fully coalesced: consecutive threads own consecutive `channel`
// values, and `channels = 5376 = 21 * blockDim.x`, so no warp ever straddles a
// (modality, param) boundary and every store is one aligned 128-byte
// transaction. That exactness is a property of 5376, not a requirement — a
// hidden size that is not a multiple of the block size costs one split
// transaction per group and nothing else.
//
// Note the bias starts the fma chain rather than being added at the end. That
// is one rounding *fewer* than a `linear`-then-add-bias reference, so this is
// not bit-identical to a torch dump of the same operation; it is closer to the
// exact value, which is the point of spec 9.1.
__global__ void adaln_expand_kernel(const float* __restrict__ w, const float* __restrict__ bias,
                                    const float* __restrict__ code, float* __restrict__ out,
                                    int num_t, int num_modality, int num_param, int channels,
                                    int rank, int out_features) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= out_features) return;
  const int ti = blockIdx.y;

  const float* c = code + static_cast<size_t>(ti) * rank;
  const float* row = w + static_cast<size_t>(p) * rank;

  // Accumulate in fp32 starting from the bias. The same c(t) feeds all 51
  // consumers at every sampling step, so a rounding here biases every block's
  // modulation identically and accumulates coherently over the trajectory.
  float acc = bias[p];
  for (int k = 0; k < rank; ++k) acc = fmaf(row[k], c[k], acc);

  const int channel = p % channels;
  const int rest = p / channels;
  const int param = rest % num_param;
  const int modality = rest / num_param;

  const size_t dst = (static_cast<size_t>(param) * num_t * num_modality +
                      static_cast<size_t>(ti) * num_modality + modality) *
                         channels +
                     channel;
  out[dst] = acc;
}

// --- ungated residual -------------------------------------------------------
//
// The token refiner has no AdaLN, so its residual adds carry no gate (spec 6).
// Using `launch_add_gated` with a synthetic all-ones gate would work but would
// need a `[1, hidden]` fp32 buffer and a zero index array per call, which is
// more machinery than the add itself.
__global__ void add_rows_bf16_kernel(__nv_bfloat16* __restrict__ x,
                                     const __nv_bfloat16* __restrict__ branch, size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  x[i] = __float2bfloat16(__bfloat162float(x[i]) + __bfloat162float(branch[i]));
}

}  // namespace

void launch_adaln_expand(const float* w, const float* bias, const float* code, float* out,
                         int num_t, int num_modality, int num_param, int channels, int rank,
                         cudaStream_t stream) {
  if (num_t <= 0 || num_modality <= 0 || num_param <= 0 || channels <= 0 || rank <= 0) {
    throw std::runtime_error("launch_adaln_expand: every extent must be positive");
  }
  // The product, not just the factors. A signed overflow here would make the
  // `p >= out_features` guard compare against a negative bound, so every thread
  // would pass it and write outside the allocation — silent corruption rather
  // than a fault.
  const size_t features = static_cast<size_t>(num_modality) * num_param * channels;
  if (features > 0x7FFFFFFFull) {
    throw std::runtime_error("launch_adaln_expand: " + std::to_string(features) +
                             " output features overflows the kernel's index type");
  }
  if (num_t > 65535) {
    throw std::runtime_error("launch_adaln_expand: " + std::to_string(num_t) +
                             " timesteps exceeds the gridDim.y limit");
  }
  const int out_features = static_cast<int>(features);
  const dim3 grid(grid_1d(features, kThreads), num_t);
  adaln_expand_kernel<<<grid, kThreads, 0, stream>>>(w, bias, code, out, num_t, num_modality,
                                                     num_param, channels, rank, out_features);
  // A launch-configuration check, not an execution check: an asynchronous fault
  // from any earlier kernel on this stream also latches here and will be
  // reported with this file's line number.
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

void launch_add_rows_bf16(__nv_bfloat16* x, const __nv_bfloat16* branch, size_t n,
                          cudaStream_t stream) {
  if (n == 0) return;
  add_rows_bf16_kernel<<<grid_1d(n, kThreads), kThreads, 0, stream>>>(x, branch, n);
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

}  // namespace vidfab::cuda
