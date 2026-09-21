#include <cublas_v2.h>

#include <type_traits>

using Create = cublasStatus_t(CUBLASWINAPI*)(cublasHandle_t*);
using Destroy = cublasStatus_t(CUBLASWINAPI*)(cublasHandle_t);
using SetStream = cublasStatus_t(CUBLASWINAPI*)(cublasHandle_t, cudaStream_t);
using SetMathMode = cublasStatus_t(CUBLASWINAPI*)(cublasHandle_t, cublasMath_t);
using Sgemm = cublasStatus_t(CUBLASWINAPI*)(cublasHandle_t, cublasOperation_t, cublasOperation_t,
                                            int, int, int, const float*, const float*, int,
                                            const float*, int, const float*, float*, int);
using SgemmStridedBatched = cublasStatus_t(CUBLASWINAPI*)(
    cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const float*, const float*,
    int, long long, const float*, int, long long, const float*, float*, int, long long, int);
using GemmEx = cublasStatus_t(CUBLASWINAPI*)(cublasHandle_t, cublasOperation_t, cublasOperation_t,
                                             int, int, int, const void*, const void*, cudaDataType,
                                             int, const void*, cudaDataType, int, const void*,
                                             void*, cudaDataType, int, cublasComputeType_t,
                                             cublasGemmAlgo_t);
using GemmStridedBatchedEx = cublasStatus_t(CUBLASWINAPI*)(
    cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const void*, const void*,
    cudaDataType, int, long long, const void*, cudaDataType, int, long long, const void*, void*,
    cudaDataType, int, long long, int, cublasComputeType_t, cublasGemmAlgo_t);

static_assert(std::is_same_v<decltype(&::cublasCreate_v2), Create>);
static_assert(std::is_same_v<decltype(&::cublasDestroy_v2), Destroy>);
static_assert(std::is_same_v<decltype(&::cublasSetStream_v2), SetStream>);
static_assert(std::is_same_v<decltype(&::cublasSetMathMode), SetMathMode>);
static_assert(std::is_same_v<decltype(&::cublasSgemm_v2), Sgemm>);
static_assert(std::is_same_v<decltype(&::cublasSgemmStridedBatched), SgemmStridedBatched>);
constexpr GemmEx kGemmEx = &::cublasGemmEx;
constexpr GemmStridedBatchedEx kGemmStridedBatchedEx = &::cublasGemmStridedBatchedEx;
static_assert(kGemmEx != nullptr && kGemmStridedBatchedEx != nullptr);

static_assert(CUBLAS_STATUS_SUCCESS == 0);
static_assert(CUBLAS_OP_N == 0 && CUBLAS_OP_T == 1);
static_assert(CUBLAS_GEMM_DEFAULT == -1 && CUBLAS_GEMM_DEFAULT_TENSOR_OP == 99);
static_assert(CUBLAS_DEFAULT_MATH == 0 && CUBLAS_PEDANTIC_MATH == 2);
static_assert(CUBLAS_COMPUTE_32F == 68);
static_assert(CUDA_R_32F == 0 && CUDA_R_16F == 2 && CUDA_R_16BF == 14);

int main() {
  return 0;
}
