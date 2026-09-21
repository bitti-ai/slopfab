#include "nn_internal.cuh"

namespace slopfab::cuda {
using namespace nn_detail;

namespace {
__global__ void scatter_add_rows_kernel(const __nv_bfloat16* src, const int32_t* index,
                                        __nv_bfloat16* dst, int dim) {
  const int r = blockIdx.x;
  const int d = blockIdx.y * blockDim.x + threadIdx.x;
  if (d < dim) {
    const size_t to = static_cast<size_t>(index[r]) * dim + d;
    dst[to] = __float2bfloat16(__bfloat162float(dst[to]) +
                               __bfloat162float(src[static_cast<size_t>(r) * dim + d]));
  }
}

__global__ void merge_four_rows_kernel(const __nv_bfloat16* src, __nv_bfloat16* dst, int dim) {
  const int g = blockIdx.x;
  const int d = blockIdx.y * blockDim.x + threadIdx.x;
  if (d < 4 * dim)
    dst[static_cast<size_t>(g) * 4 * dim + d] = src[static_cast<size_t>(g) * 4 * dim + d];
}

// --- row permutation --------------------------------------------------------
//
// A permutation is a pure move, so the packed path copies raw uint4 rather than
// converting anything: 8 bf16 or 4 fp32 per thread instead of one element.
// Measured on the bf16 gather at n*dim = 270M: 0.618 ms / 1312 GB/s scalar
// against 0.543 / 1494 packed.

// Elements of T in one 16-byte transaction.
template <typename T> struct PackWidth {
  static constexpr int value = static_cast<int>(16 / sizeof(T));
};

template <typename T>
__global__ void gather_rows_kernel(const T* __restrict__ src, const int32_t* __restrict__ index,
                                   T* __restrict__ dst, int n, int dim) {
  const int row = blockIdx.x;
  const int col = blockIdx.y * blockDim.x + threadIdx.x;
  if (row >= n || col >= dim)
    return;
  dst[static_cast<size_t>(row) * dim + col] = src[static_cast<size_t>(index[row]) * dim + col];
}

template <typename T>
__global__ void gather_rows_packed_kernel(const T* __restrict__ src,
                                          const int32_t* __restrict__ index, T* __restrict__ dst,
                                          int n, int packs) {
  const int row = blockIdx.x;
  const int p = blockIdx.y * blockDim.x + threadIdx.x;
  if (row >= n || p >= packs)
    return;
  const size_t dim = static_cast<size_t>(packs) * PackWidth<T>::value;
  reinterpret_cast<uint4*>(dst + static_cast<size_t>(row) * dim)[p] =
      reinterpret_cast<const uint4*>(src + static_cast<size_t>(index[row]) * dim)[p];
}

template <typename T>
__global__ void scatter_rows_kernel(const T* __restrict__ src, const int32_t* __restrict__ index,
                                    T* __restrict__ dst, int n, int dim) {
  const int row = blockIdx.x;
  const int col = blockIdx.y * blockDim.x + threadIdx.x;
  if (row >= n || col >= dim)
    return;
  dst[static_cast<size_t>(index[row]) * dim + col] = src[static_cast<size_t>(row) * dim + col];
}

template <typename T>
__global__ void scatter_rows_packed_kernel(const T* __restrict__ src,
                                           const int32_t* __restrict__ index, T* __restrict__ dst,
                                           int n, int packs) {
  const int row = blockIdx.x;
  const int p = blockIdx.y * blockDim.x + threadIdx.x;
  if (row >= n || p >= packs)
    return;
  const size_t dim = static_cast<size_t>(packs) * PackWidth<T>::value;
  reinterpret_cast<uint4*>(dst + static_cast<size_t>(index[row]) * dim)[p] =
      reinterpret_cast<const uint4*>(src + static_cast<size_t>(row) * dim)[p];
}

// Row bases are `row * dim`, so a `dim` divisible by the pack width settles
// alignment given a 16-byte-aligned buffer — cudaMalloc gives 256 and the
// workspace gives 256.
template <typename T> bool row_packable(const T* a, const T* b, int dim) {
  return dim % PackWidth<T>::value == 0 && reinterpret_cast<uintptr_t>(a) % 16 == 0 &&
         reinterpret_cast<uintptr_t>(b) % 16 == 0;
}

} // namespace

namespace {

template <typename T>
void launch_gather_impl(const T* src, const int32_t* index, T* dst, int n, int dim,
                        cudaStream_t stream) {
  if (row_packable(src, dst, dim)) {
    const int packs = dim / PackWidth<T>::value;
    const dim3 grid(n, grid_1d(static_cast<size_t>(packs), kRowThreads));
    gather_rows_packed_kernel<<<grid, kRowThreads, 0, stream>>>(src, index, dst, n, packs);
  } else {
    const dim3 grid(n, grid_1d(static_cast<size_t>(dim), kRowThreads));
    gather_rows_kernel<<<grid, kRowThreads, 0, stream>>>(src, index, dst, n, dim);
  }
  SLOPFAB_CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void launch_scatter_impl(const T* src, const int32_t* index, T* dst, int n, int dim,
                         cudaStream_t stream) {
  if (row_packable(src, dst, dim)) {
    const int packs = dim / PackWidth<T>::value;
    const dim3 grid(n, grid_1d(static_cast<size_t>(packs), kRowThreads));
    scatter_rows_packed_kernel<<<grid, kRowThreads, 0, stream>>>(src, index, dst, n, packs);
  } else {
    const dim3 grid(n, grid_1d(static_cast<size_t>(dim), kRowThreads));
    scatter_rows_kernel<<<grid, kRowThreads, 0, stream>>>(src, index, dst, n, dim);
  }
  SLOPFAB_CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch_gather_rows(const __nv_bfloat16* src, const int32_t* index, __nv_bfloat16* dst, int n,
                        int dim, cudaStream_t stream) {
  require_positive(n, dim, "launch_gather_rows");
  launch_gather_impl(src, index, dst, n, dim, stream);
}

void launch_gather_rows_f32(const float* src, const int32_t* index, float* dst, int n, int dim,
                            cudaStream_t stream) {
  require_positive(n, dim, "launch_gather_rows_f32");
  launch_gather_impl(src, index, dst, n, dim, stream);
}

void launch_scatter_rows(const __nv_bfloat16* src, const int32_t* index, __nv_bfloat16* dst, int n,
                         int dim, cudaStream_t stream) {
  require_positive(n, dim, "launch_scatter_rows");
  launch_scatter_impl(src, index, dst, n, dim, stream);
}

void launch_scatter_rows_f32(const float* src, const int32_t* index, float* dst, int n, int dim,
                             cudaStream_t stream) {
  require_positive(n, dim, "launch_scatter_rows_f32");
  launch_scatter_impl(src, index, dst, n, dim, stream);
}

void launch_scatter_add_rows(const __nv_bfloat16* src, const int32_t* index, __nv_bfloat16* dst,
                             int n, int dim, cudaStream_t stream) {
  require_positive(n, dim, "launch_scatter_add_rows");
  scatter_add_rows_kernel<<<dim3(n, grid_1d(dim, kRowThreads)), kRowThreads, 0, stream>>>(
      src, index, dst, dim);
  SLOPFAB_CUDA_CHECK(cudaGetLastError());
}

void launch_merge_four_rows(const __nv_bfloat16* src, __nv_bfloat16* dst, int groups, int dim,
                            cudaStream_t stream) {
  require_positive(groups, dim, "launch_merge_four_rows");
  merge_four_rows_kernel<<<dim3(groups, grid_1d(4 * dim, kRowThreads)), kRowThreads, 0, stream>>>(
      src, dst, dim);
  SLOPFAB_CUDA_CHECK(cudaGetLastError());
}

} // namespace slopfab::cuda
