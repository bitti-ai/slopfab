#include "lt_linear.cuh"
#include "slopfab/cuda/device.h"
#include "slopfab/cuda/gemm.cuh"
#include <cublasLt.h>
#include <map>
#include <tuple>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace slopfab::seedvr2 {
namespace {
#define LT_FUNCTIONS(X)                                                                            \
  X(Create)                                                                                        \
  X(Destroy) X(MatmulDescCreate) X(MatmulDescDestroy) X(MatmulDescSetAttribute)                    \
      X(MatrixLayoutCreate) X(MatrixLayoutDestroy) X(MatmulPreferenceCreate)                       \
          X(MatmulPreferenceDestroy) X(MatmulPreferenceSetAttribute) X(MatmulAlgoGetHeuristic)     \
              X(Matmul)

struct Api {
#define DECLARE(name) decltype(&::cublasLt##name) name = nullptr;
  LT_FUNCTIONS(DECLARE)
#undef DECLARE

  bool load() {
#ifdef _WIN32
    // The existing dispatch layer has loaded this exact toolkit's Lt DLL.
    // Never let the executable import a different toolkit's library by name.
    auto path = cuda::cublas_loaded_path();
    const auto start = path.find_last_of(L"/\\");
    path.erase(start == std::wstring::npos ? 0 : start + 1);
    path += L"cublasLt64_" + std::to_wstring(cuda::cublas_loaded_major()) + L".dll";
    HMODULE module = GetModuleHandleW(path.c_str());
    if (!module)
      return false;
#define LOAD(name)                                                                                 \
  name = reinterpret_cast<decltype(name)>(GetProcAddress(module, "cublasLt" #name));
#else
#define LOAD(name) name = &::cublasLt##name;
#endif
    LT_FUNCTIONS(LOAD)
#undef LOAD
#define REQUIRE(name)                                                                              \
  if (!name)                                                                                       \
    return false;
    LT_FUNCTIONS(REQUIRE)
#undef REQUIRE
    return true;
  }
};

#undef LT_FUNCTIONS

constexpr size_t workspace_bytes = size_t(8) << 20;
}

struct LtLinear::Impl {
  Api api;
  bool attempted = false;
  int device = -1;
  cublasLtHandle_t handle = nullptr;
  void* workspace = nullptr;

  struct Plan {
    Api& api;
    cublasLtMatmulDesc_t operation = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, output = nullptr;
    cublasLtMatmulAlgo_t algorithm{};
    size_t workspace_size = 0;
    bool valid = false;

    explicit Plan(Api& api_) : api(api_) {
    }

    ~Plan() {
      if (operation)
        api.MatmulDescDestroy(operation);
      if (a)
        api.MatrixLayoutDestroy(a);
      if (b)
        api.MatrixLayoutDestroy(b);
      if (output)
        api.MatrixLayoutDestroy(output);
    }
  };

  std::map<std::tuple<int, int, int>, std::unique_ptr<Plan>> plans;

  ~Impl() {
    int previous = -1;
    if (device >= 0 && cudaGetDevice(&previous) == cudaSuccess && previous != device)
      cudaSetDevice(device);
    plans.clear();
    if (handle)
      api.Destroy(handle);
    if (workspace)
      cudaFree(workspace);
    if (previous >= 0 && previous != device)
      cudaSetDevice(previous);
  }

  bool initialize() {
    if (attempted)
      return handle != nullptr;
    attempted = true;
    SLOPFAB_CUDA_CHECK(cudaGetDevice(&device));
    if (!api.load() || api.Create(&handle) != CUBLAS_STATUS_SUCCESS) {
      handle = nullptr;
      return false;
    }
    auto status = cudaMalloc(&workspace, workspace_bytes);
    if (status == cudaErrorMemoryAllocation) {
      cudaGetLastError();
      api.Destroy(handle);
      handle = nullptr;
      return false;
    }
    SLOPFAB_CUDA_CHECK(status);
    return true;
  }

  Plan& plan(int rows, int ci, int co, const __nv_bfloat16* bias) {
    const auto key = std::make_tuple(rows, ci, co);
    auto found = plans.find(key);
    if (found != plans.end())
      return *found->second;
    auto value = std::make_unique<Plan>(api);
    auto& p = *plans.emplace(key, std::move(value)).first->second;
    if (api.MatmulDescCreate(&p.operation, CUBLAS_COMPUTE_32F, CUDA_R_32F) !=
            CUBLAS_STATUS_SUCCESS ||
        api.MatrixLayoutCreate(&p.a, CUDA_R_16BF, ci, co, ci) != CUBLAS_STATUS_SUCCESS ||
        api.MatrixLayoutCreate(&p.b, CUDA_R_16BF, ci, rows, ci) != CUBLAS_STATUS_SUCCESS ||
        api.MatrixLayoutCreate(&p.output, CUDA_R_16BF, co, rows, co) != CUBLAS_STATUS_SUCCESS)
      return p;
    const cublasOperation_t transpose = CUBLAS_OP_T;
    const cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
    const cudaDataType_t bias_type = CUDA_R_16BF;
    if (api.MatmulDescSetAttribute(p.operation, CUBLASLT_MATMUL_DESC_TRANSA, &transpose,
                                   sizeof(transpose)) != CUBLAS_STATUS_SUCCESS ||
        api.MatmulDescSetAttribute(p.operation, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue,
                                   sizeof(epilogue)) != CUBLAS_STATUS_SUCCESS ||
        api.MatmulDescSetAttribute(p.operation, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type,
                                   sizeof(bias_type)) != CUBLAS_STATUS_SUCCESS ||
        api.MatmulDescSetAttribute(p.operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias,
                                   sizeof(bias)) != CUBLAS_STATUS_SUCCESS)
      return p;
    cublasLtMatmulPreference_t preference = nullptr;
    if (api.MatmulPreferenceCreate(&preference) != CUBLAS_STATUS_SUCCESS)
      return p;
    cublasLtMatmulHeuristicResult_t result{};
    int count = 0;
    auto status =
        api.MatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                         &workspace_bytes, sizeof(workspace_bytes));
    if (status == CUBLAS_STATUS_SUCCESS)
      status = api.MatmulAlgoGetHeuristic(handle, p.operation, p.a, p.b, p.output, p.output,
                                          preference, 1, &result, &count);
    api.MatmulPreferenceDestroy(preference);
    if (status == CUBLAS_STATUS_SUCCESS && count && result.state == CUBLAS_STATUS_SUCCESS &&
        result.workspaceSize <= workspace_bytes) {
      p.algorithm = result.algo;
      p.workspace_size = result.workspaceSize;
      p.valid = true;
    }
    return p;
  }
};

LtLinear::LtLinear() : impl_(std::make_unique<Impl>()) {
}

LtLinear::~LtLinear() = default;

bool LtLinear::forward(const __nv_bfloat16* x, const __nv_bfloat16* weights,
                       const __nv_bfloat16* bias, __nv_bfloat16* output, int rows, int ci, int co,
                       cudaStream_t stream) {
  if (rows < 1 || ci < 1 || co < 1 || !impl_->initialize())
    return false;
  auto& p = impl_->plan(rows, ci, co, bias);
  if (!p.valid)
    return false;
  if (impl_->api.MatmulDescSetAttribute(p.operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias,
                                        sizeof(bias)) != CUBLAS_STATUS_SUCCESS)
    return false;
  const float alpha = 1, beta = 0;
  const auto status = impl_->api.Matmul(impl_->handle, p.operation, &alpha, weights, p.a, x, p.b,
                                        &beta, output, p.output, output, p.output, &p.algorithm,
                                        impl_->workspace, p.workspace_size, stream);
  if (status == CUBLAS_STATUS_NOT_SUPPORTED || status == CUBLAS_STATUS_INVALID_VALUE) {
    p.valid = false;
    return false;
  }
  SLOPFAB_CUBLAS_CHECK(status);
  return true;
}
}
