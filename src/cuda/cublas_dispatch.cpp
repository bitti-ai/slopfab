#include "vidfab/cuda/cublas_dispatch.h"

#include <algorithm>
#include <cstdlib>
#include <cwchar>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "vidfab/cuda/cuda_toolkit.h"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace vidfab::cuda {
namespace {

using GemmEx = cublasStatus_t(CUBLASWINAPI*)(
    cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
    const void*, const void*, cudaDataType, int, const void*, cudaDataType, int,
    const void*, void*, cudaDataType, int, cublasComputeType_t, cublasGemmAlgo_t);
using GemmStridedBatchedEx = cublasStatus_t(CUBLASWINAPI*)(
    cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
    const void*, const void*, cudaDataType, int, long long, const void*,
    cudaDataType, int, long long, const void*, void*, cudaDataType, int,
    long long, int, cublasComputeType_t, cublasGemmAlgo_t);

struct CublasApi {
  decltype(&::cublasCreate_v2) create = nullptr;
  decltype(&::cublasDestroy_v2) destroy = nullptr;
  decltype(&::cublasSetStream_v2) set_stream = nullptr;
  decltype(&::cublasSetMathMode) set_math_mode = nullptr;
  decltype(&::cublasSgemm_v2) sgemm = nullptr;
  decltype(&::cublasSgemmStridedBatched) sgemm_strided_batched = nullptr;
  GemmEx gemm_ex = nullptr;
  GemmStridedBatchedEx gemm_strided_batched_ex = nullptr;
  int major = 0;
  std::wstring path;
};

struct CublasState {
  std::once_flag once;
  std::mutex request_mutex;
  std::wstring request;
  bool request_frozen = false;
  CublasApi api;
};

CublasState& state() {
  static CublasState value;
  return value;
}

#if defined(_WIN32)

std::wstring environment(const wchar_t* name) {
  const DWORD length = GetEnvironmentVariableW(name, nullptr, 0);
  if (length == 0) return {};
  std::wstring value(length, L'\0');
  GetEnvironmentVariableW(name, value.data(), length);
  value.resize(std::wcslen(value.c_str()));
  return value;
}

bool absolute_path(const std::wstring& path) {
  return (path.size() >= 3 && path[1] == L':' &&
          (path[2] == L'\\' || path[2] == L'/')) ||
         (path.size() >= 2 && path[0] == L'\\' && path[1] == L'\\');
}

void add_root(std::vector<std::wstring>& roots, std::wstring root) {
  while (!root.empty() && (root.back() == L'\\' || root.back() == L'/'))
    root.pop_back();
  if (root.size() >= 8 &&
      _wcsicmp(root.c_str() + root.size() - 8, L"\\bin\\x64") == 0)
    root.resize(root.size() - 8);
  else if (root.size() >= 4 &&
           _wcsicmp(root.c_str() + root.size() - 4, L"\\bin") == 0)
    root.resize(root.size() - 4);
  if (!absolute_path(root)) return;
  if (std::find_if(roots.begin(), roots.end(), [&](const std::wstring& old) {
        return _wcsicmp(old.c_str(), root.c_str()) == 0;
      }) == roots.end())
    roots.push_back(std::move(root));
}

std::vector<std::wstring> toolkit_roots(int major) {
  std::vector<std::wstring> roots;
  const std::wstring prefix = major == 13 ? L"CUDA_PATH_V13_" : L"CUDA_PATH_V12_";
  // Prefer the highest installed point release within the requested ABI
  // major. Environment-block enumeration order is unspecified and otherwise
  // made a machine with both 12.4 and 12.8 choose whichever was listed first.
  for (int minor = 9; minor >= 0; --minor) {
    const std::wstring name = prefix + std::to_wstring(minor);
    add_root(roots, environment(name.c_str()));
  }
  const wchar_t* block = GetEnvironmentStringsW();
  if (block != nullptr) {
    for (const wchar_t* entry = block; *entry; entry += std::wcslen(entry) + 1) {
      const wchar_t* equals = std::wcschr(entry, L'=');
      if (equals == nullptr) continue;
      const std::wstring name(entry, equals);
      if (name.size() >= prefix.size() &&
          _wcsnicmp(name.c_str(), prefix.c_str(), prefix.size()) == 0)
        add_root(roots, equals + 1);
    }
    FreeEnvironmentStringsW(const_cast<wchar_t*>(block));
  }
  add_root(roots, environment(L"CUDA_PATH"));
  const std::wstring program_files = environment(L"ProgramFiles");
  if (!program_files.empty()) {
    for (int minor = 9; minor >= 0; --minor)
      add_root(roots, program_files + L"\\NVIDIA GPU Computing Toolkit\\CUDA\\v" +
                          std::to_wstring(major) + L"." + std::to_wstring(minor));
  }
  return roots;
}

bool file_exists(const std::wstring& path) {
  const DWORD attributes = GetFileAttributesW(path.c_str());
  return attributes != INVALID_FILE_ATTRIBUTES &&
         (attributes & FILE_ATTRIBUTE_DIRECTORY) == 0;
}

class ScopedLoaderErrorMode {
 public:
  ScopedLoaderErrorMode() {
    changed_ = SetThreadErrorMode(SEM_FAILCRITICALERRORS | SEM_NOOPENFILEERRORBOX,
                                  &previous_) != 0;
  }
  ~ScopedLoaderErrorMode() {
    if (changed_) {
      DWORD ignored = 0;
      SetThreadErrorMode(previous_, &ignored);
    }
  }

  ScopedLoaderErrorMode(const ScopedLoaderErrorMode&) = delete;
  ScopedLoaderErrorMode& operator=(const ScopedLoaderErrorMode&) = delete;

 private:
  DWORD previous_ = 0;
  bool changed_ = false;
};

std::vector<CudaToolkitCandidate> candidates() {
  std::vector<CudaToolkitCandidate> out;
  for (int major : {13, 12}) {
    std::wstring found;
    const std::wstring suffix = std::to_wstring(major) + L".dll";
    for (const std::wstring& root : toolkit_roots(major)) {
      for (const std::wstring& bin : {root + L"\\bin\\x64", root + L"\\bin"}) {
        if (file_exists(bin + L"\\cublas64_" + suffix) &&
            file_exists(bin + L"\\cublasLt64_" + suffix)) {
          found = bin;
          break;
        }
      }
      if (!found.empty()) break;
    }
    out.push_back({major, std::move(found)});
  }
  return out;
}

std::string narrow(const std::wstring& text) {
  if (text.empty()) return {};
  const int size = WideCharToMultiByte(CP_UTF8, 0, text.c_str(), -1, nullptr, 0,
                                       nullptr, nullptr);
  std::string out(static_cast<size_t>(size > 0 ? size : 0), '\0');
  if (size > 1) {
    WideCharToMultiByte(CP_UTF8, 0, text.c_str(), -1, out.data(), size, nullptr, nullptr);
    out.pop_back();
  }
  return out;
}

template <typename Function>
Function symbol(HMODULE module, const char* name, const std::wstring& dll) {
  FARPROC address = GetProcAddress(module, name);
  if (address == nullptr)
    throw std::runtime_error("cuBLAS " + narrow(dll) + " is missing export " + name);
  return reinterpret_cast<Function>(address);
}

void load_candidate(CublasApi& api, const CudaToolkitCandidate& selected) {
  const std::wstring suffix = std::to_wstring(selected.major) + L".dll";
  const std::wstring lt_path = selected.bin + L"\\cublasLt64_" + suffix;
  const std::wstring blas_path = selected.bin + L"\\cublas64_" + suffix;
  constexpr DWORD flags = LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR |
                          LOAD_LIBRARY_SEARCH_DEFAULT_DIRS;
  // A corrupt toolkit DLL must become an exception/fallback, never a modal
  // "Bad Image" dialog that deadlocks a service or an unattended DLL host.
  // Thread-local mode avoids changing the embedding application's policy.
  ScopedLoaderErrorMode error_mode;
  HMODULE lt = LoadLibraryExW(lt_path.c_str(), nullptr, flags);
  if (lt == nullptr)
    throw std::runtime_error("cuBLAS: cannot load " + narrow(lt_path) +
                             " (Windows error " + std::to_string(GetLastError()) + ")");
  HMODULE blas = LoadLibraryExW(blas_path.c_str(), nullptr, flags);
  if (blas == nullptr) {
    const DWORD error = GetLastError();
    FreeLibrary(lt);
    throw std::runtime_error("cuBLAS: cannot load " + narrow(blas_path) +
                             " (Windows error " + std::to_string(error) + ")");
  }
  try {
    api.create = symbol<decltype(api.create)>(blas, "cublasCreate_v2", blas_path);
    api.destroy = symbol<decltype(api.destroy)>(blas, "cublasDestroy_v2", blas_path);
    api.set_stream = symbol<decltype(api.set_stream)>(blas, "cublasSetStream_v2", blas_path);
    api.set_math_mode = symbol<decltype(api.set_math_mode)>(blas, "cublasSetMathMode", blas_path);
    api.sgemm = symbol<decltype(api.sgemm)>(blas, "cublasSgemm_v2", blas_path);
    api.sgemm_strided_batched = symbol<decltype(api.sgemm_strided_batched)>(
        blas, "cublasSgemmStridedBatched", blas_path);
    api.gemm_ex = symbol<decltype(api.gemm_ex)>(blas, "cublasGemmEx", blas_path);
    api.gemm_strided_batched_ex = symbol<decltype(api.gemm_strided_batched_ex)>(
        blas, "cublasGemmStridedBatchedEx", blas_path);
  } catch (...) {
    FreeLibrary(blas);
    FreeLibrary(lt);
    throw;
  }
  // Keep both modules resident for process lifetime. Handles and CUDA work can
  // outlive any individual model object, including during static destruction.
  api.major = selected.major;
  api.path = blas_path;
}

void load_windows(CublasApi& api, const std::wstring& requested) {
  const std::vector<CudaToolkitCandidate> found = candidates();
  int driver_version = 0;
  const cudaError_t driver_status = cudaDriverGetVersion(&driver_version);
  if (driver_status != cudaSuccess) {
    throw std::runtime_error(
        "cuBLAS: cannot query NVIDIA driver capability without a context: " +
        std::string(cudaGetErrorString(driver_status)));
  }
  std::string failures;
  bool matched = false;
  for (const CudaToolkitCandidate& candidate : found) {
    if (candidate.bin.empty()) continue;
    if (requested != L"auto" && requested != std::to_wstring(candidate.major))
      continue;
    matched = true;
    if (!cuda_driver_supports_toolkit(driver_version, candidate.major)) {
      const std::string reason =
          "CUDA " + std::to_string(candidate.major) +
          " cuBLAS requires driver API >= " +
          std::to_string(candidate.major * 1000) +
          "; installed driver reports " + std::to_string(driver_version);
      if (!failures.empty()) failures += "; ";
      failures += reason;
      if (requested != L"auto")
        throw std::runtime_error("cuBLAS: " + reason);
      continue;
    }
    try {
      CublasApi loaded;
      load_candidate(loaded, candidate);
      api = std::move(loaded);
      return;
    } catch (const std::exception& error) {
      if (!failures.empty()) failures += "; ";
      failures += error.what();
      // Auto is intentionally allowed to recover from an incomplete/broken
      // CUDA 13 installation by trying CUDA 12. An explicit request is not.
      if (requested != L"auto") throw;
    }
  }
  if (!matched) {
    failures = "no matching toolkit with both cuBLAS DLLs was discovered";
  }
  throw std::runtime_error(
      "cuBLAS: cannot initialize requested CUDA " + narrow(requested) + ": " +
      failures +
      " (searched absolute CUDA_PATH_V13_*/CUDA_PATH_V12_* and Program Files paths)");
}

#endif

const CublasApi& api() {
  CublasState& current = state();
  std::call_once(current.once, [&] {
    std::wstring request;
    {
      std::lock_guard<std::mutex> lock(current.request_mutex);
      current.request_frozen = true;
      request = current.request;
    }
#if defined(_WIN32)
    request = cuda_version_request(request, environment(L"VIDFAB_CUDA_VERSION"));
#else
    std::wstring environment_request;
    if (const char* value = std::getenv("VIDFAB_CUDA_VERSION"); value != nullptr)
      environment_request.assign(value, value + std::char_traits<char>::length(value));
    request = cuda_version_request(request, environment_request);
#endif
    if (request != L"auto" && request != L"13" && request != L"12")
      throw std::runtime_error("cuBLAS: VIDFAB_CUDA_VERSION must be auto, 13, or 12");
#if defined(_WIN32)
    load_windows(current.api, request);
#else
    constexpr int linked_major = CUDART_VERSION / 1000;
    if (!cuda_version_matches_linked_toolkit(request, linked_major)) {
      throw std::runtime_error(
          "cuBLAS: requested CUDA " + std::string(request.begin(), request.end()) +
          " but this non-Windows build is linked to CUDA " +
          std::to_string(linked_major));
    }
    current.api.create = &::cublasCreate_v2;
    current.api.destroy = &::cublasDestroy_v2;
    current.api.set_stream = &::cublasSetStream_v2;
    current.api.set_math_mode = &::cublasSetMathMode;
    current.api.sgemm = &::cublasSgemm_v2;
    current.api.sgemm_strided_batched = &::cublasSgemmStridedBatched;
    current.api.gemm_ex = static_cast<GemmEx>(&::cublasGemmEx);
    current.api.gemm_strided_batched_ex =
        static_cast<GemmStridedBatchedEx>(&::cublasGemmStridedBatchedEx);
    current.api.major = linked_major;
#endif
  });
  return current.api;
}

}  // namespace

void set_cublas_version_request(const std::string& requested) {
  if (requested != "auto" && requested != "13" && requested != "12")
    throw std::invalid_argument("--cuda-version requires auto, 13, or 12");
  CublasState& current = state();
  std::lock_guard<std::mutex> lock(current.request_mutex);
  if (current.request_frozen)
    throw std::logic_error("cuBLAS version cannot change after initialization");
  current.request.assign(requested.begin(), requested.end());
}

int cublas_loaded_major() { return api().major; }
std::wstring cublas_loaded_path() { return api().path; }

cublasStatus_t cublas_create(cublasHandle_t* handle) { return api().create(handle); }
cublasStatus_t cublas_destroy(cublasHandle_t handle) { return api().destroy(handle); }
cublasStatus_t cublas_set_stream(cublasHandle_t handle, cudaStream_t stream) {
  return api().set_stream(handle, stream);
}
cublasStatus_t cublas_set_math_mode(cublasHandle_t handle, cublasMath_t mode) {
  return api().set_math_mode(handle, mode);
}
cublasStatus_t cublas_sgemm(cublasHandle_t handle, cublasOperation_t transa,
                            cublasOperation_t transb, int m, int n, int k,
                            const float* alpha, const float* a, int lda,
                            const float* b, int ldb, const float* beta, float* c,
                            int ldc) {
  return api().sgemm(handle, transa, transb, m, n, k, alpha, a, lda, b, ldb,
                     beta, c, ldc);
}
cublasStatus_t cublas_sgemm_strided_batched(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const float* alpha, const float* a, int lda,
    long long stride_a, const float* b, int ldb, long long stride_b,
    const float* beta, float* c, int ldc, long long stride_c, int batch_count) {
  return api().sgemm_strided_batched(handle, transa, transb, m, n, k, alpha, a,
                                     lda, stride_a, b, ldb, stride_b, beta, c,
                                     ldc, stride_c, batch_count);
}
cublasStatus_t cublas_gemm_ex(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const void* alpha, const void* a, cudaDataType a_type,
    int lda, const void* b, cudaDataType b_type, int ldb, const void* beta,
    void* c, cudaDataType c_type, int ldc, cublasComputeType_t compute_type,
    cublasGemmAlgo_t algorithm) {
  return api().gemm_ex(handle, transa, transb, m, n, k, alpha, a, a_type, lda,
                       b, b_type, ldb, beta, c, c_type, ldc, compute_type,
                       algorithm);
}
cublasStatus_t cublas_gemm_strided_batched_ex(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const void* alpha, const void* a, cudaDataType a_type,
    int lda, long long stride_a, const void* b, cudaDataType b_type, int ldb,
    long long stride_b, const void* beta, void* c, cudaDataType c_type, int ldc,
    long long stride_c, int batch_count, cublasComputeType_t compute_type,
    cublasGemmAlgo_t algorithm) {
  return api().gemm_strided_batched_ex(
      handle, transa, transb, m, n, k, alpha, a, a_type, lda, stride_a, b,
      b_type, ldb, stride_b, beta, c, c_type, ldc, stride_c, batch_count,
      compute_type, algorithm);
}

}  // namespace vidfab::cuda
