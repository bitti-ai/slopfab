// Row-major GEMM wrappers over cuBLAS.
//
// cuBLAS is column-major. A row-major [M,K] buffer is bit-identical to a
// column-major [K,M] buffer, so every call below swaps the operands and
// dimensions rather than transposing any data. These wrappers are unit-tested
// against a naive CPU matmul because a layout mistake here produces numbers
// that are wrong but entirely plausible.
#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "slopfab/cuda/cublas_dispatch.h"

namespace slopfab::cuda {

inline void cublas_check(cublasStatus_t status, const char* expr, const char* file, int line) {
  if (status == CUBLAS_STATUS_SUCCESS)
    return;
  throw std::runtime_error("cublas: status " + std::to_string(static_cast<int>(status)) + " in " +
                           expr + " at " + file + ":" + std::to_string(line));
}

#define SLOPFAB_CUBLAS_CHECK(expr) ::slopfab::cuda::cublas_check((expr), #expr, __FILE__, __LINE__)

// C[M,N] = A[M,K] * B[N,K]^T   (B stored row-major as [N,K])
inline void gemm_nt(cublasHandle_t h, const float* A, const float* B, float* C, int M, int N,
                    int K) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  SLOPFAB_CUBLAS_CHECK(
      cublas_sgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, B, K, A, K, &beta, C, N));
}

// C[M,N] = A[M,K] * B[K,N]
inline void gemm_nn(cublasHandle_t h, const float* A, const float* B, float* C, int M, int N,
                    int K) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  SLOPFAB_CUBLAS_CHECK(
      cublas_sgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N));
}

// Batched C[b][M,N] = A[b][M,K] * B[b][N,K]^T
inline void gemm_nt_batched(cublasHandle_t h, const float* A, const float* B, float* C, int M,
                            int N, int K, int batch, long long strideA, long long strideB,
                            long long strideC) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  SLOPFAB_CUBLAS_CHECK(cublas_sgemm_strided_batched(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, B,
                                                    K, strideB, A, K, strideA, &beta, C, N, strideC,
                                                    batch));
}

// Batched C[b][M,N] = A[b][M,K] * B[b][K,N]
inline void gemm_nn_batched(cublasHandle_t h, const float* A, const float* B, float* C, int M,
                            int N, int K, int batch, long long strideA, long long strideB,
                            long long strideC) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  SLOPFAB_CUBLAS_CHECK(cublas_sgemm_strided_batched(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B,
                                                    N, strideB, A, K, strideA, &beta, C, N, strideC,
                                                    batch));
}

// As gemm_nn_batched, but with an explicit row stride for C. Setting ldc wider
// than N lets a batched GEMM scatter its per-batch results into interleaved
// columns of one matrix — used to write attention output directly in
// token-major layout instead of transposing afterwards. `ldc` is the row
// length of the destination; `strideC` is the per-batch column offset.
inline void gemm_nn_batched_ld(cublasHandle_t h, const float* A, const float* B, float* C, int M,
                               int N, int K, int batch, long long strideA, long long strideB,
                               long long strideC, int ldc) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  SLOPFAB_CUBLAS_CHECK(cublas_sgemm_strided_batched(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B,
                                                    N, strideB, A, K, strideA, &beta, C, ldc,
                                                    strideC, batch));
}

} // namespace slopfab::cuda
