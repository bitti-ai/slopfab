#pragma once

#include <cublas_v2.h>

#include <string>

namespace slopfab::cuda {

// Must be called before the first cuBLAS operation. The CLI uses this for
// --cuda-version; the C ABI exposes it as slopfab_cuda_set_version.
void set_cublas_version_request(const std::string& requested);

// Loads cuBLAS if necessary and reports the selected toolkit major/path.
int cublas_loaded_major();
std::wstring cublas_loaded_path();

cublasStatus_t cublas_create(cublasHandle_t* handle);
cublasStatus_t cublas_destroy(cublasHandle_t handle);
cublasStatus_t cublas_set_stream(cublasHandle_t handle, cudaStream_t stream);
cublasStatus_t cublas_set_math_mode(cublasHandle_t handle, cublasMath_t mode);
cublasStatus_t cublas_sgemm(cublasHandle_t handle, cublasOperation_t transa,
                            cublasOperation_t transb, int m, int n, int k, const float* alpha,
                            const float* a, int lda, const float* b, int ldb, const float* beta,
                            float* c, int ldc);
cublasStatus_t cublas_sgemm_strided_batched(cublasHandle_t handle, cublasOperation_t transa,
                                            cublasOperation_t transb, int m, int n, int k,
                                            const float* alpha, const float* a, int lda,
                                            long long stride_a, const float* b, int ldb,
                                            long long stride_b, const float* beta, float* c,
                                            int ldc, long long stride_c, int batch_count);
cublasStatus_t cublas_gemm_ex(cublasHandle_t handle, cublasOperation_t transa,
                              cublasOperation_t transb, int m, int n, int k, const void* alpha,
                              const void* a, cudaDataType a_type, int lda, const void* b,
                              cudaDataType b_type, int ldb, const void* beta, void* c,
                              cudaDataType c_type, int ldc, cublasComputeType_t compute_type,
                              cublasGemmAlgo_t algorithm);
cublasStatus_t cublas_gemm_strided_batched_ex(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
    const void* alpha, const void* a, cudaDataType a_type, int lda, long long stride_a,
    const void* b, cudaDataType b_type, int ldb, long long stride_b, const void* beta, void* c,
    cudaDataType c_type, int ldc, long long stride_c, int batch_count,
    cublasComputeType_t compute_type, cublasGemmAlgo_t algorithm);

} // namespace slopfab::cuda
