#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <string>

#include "slopfab/cuda/cublas_dispatch.h"
#include "slopfab/cuda/device.h"

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  const std::string mode = argv[1];
  const bool auto_preference = mode == "auto";
  const bool fallback = mode == "auto-fallback";
  const bool broken_explicit = mode == "broken-13";
  const int expected = auto_preference ? 13 : (fallback ? 12 : std::atoi(argv[1]));
  std::filesystem::path fake_root;
#if defined(_WIN32)
  if (fallback || broken_explicit) {
    fake_root = std::filesystem::temp_directory_path() /
                (fallback ? "slopfab-broken-cuda13-auto"
                          : "slopfab-broken-cuda13-explicit");
    const std::filesystem::path bin = fake_root / "bin" / "x64";
    std::filesystem::create_directories(bin);
    if (fallback) {
      // Exercise the production corrupt-image path. The loader suppresses
      // Windows error UI locally, then auto mode must continue to CUDA 12.
      std::ofstream(bin / "cublas64_13.dll", std::ios::binary).put('\0');
      std::ofstream(bin / "cublasLt64_13.dll", std::ios::binary).put('\0');
    } else {
      // A valid PE with the wrong exports gives explicit mode a second class
      // of incomplete-installation diagnostic to exercise.
      const wchar_t* system_root = _wgetenv(L"SystemRoot");
      if (system_root == nullptr) return 10;
      const std::filesystem::path fixture =
          std::filesystem::path(system_root) / "System32" / "version.dll";
      std::filesystem::copy_file(fixture, bin / "cublas64_13.dll",
                                 std::filesystem::copy_options::overwrite_existing);
      std::filesystem::copy_file(fixture, bin / "cublasLt64_13.dll",
                                 std::filesystem::copy_options::overwrite_existing);
    }
    _wputenv_s(L"CUDA_PATH_V13_9", fake_root.c_str());
    _putenv_s("SLOPFAB_CUDA_VERSION", fallback ? "auto" : "13");
  }
#endif
  try {
    cublasHandle_t handle = nullptr;
    const cublasStatus_t created = slopfab::cuda::cublas_create(&handle);
    if (broken_explicit) {
      if (created == CUBLAS_STATUS_SUCCESS && handle != nullptr)
        slopfab::cuda::cublas_destroy(handle);
      if (!fake_root.empty()) std::filesystem::remove_all(fake_root);
      return 7;
    }
    if (created != CUBLAS_STATUS_SUCCESS) return 3;
    if (slopfab::cuda::cublas_loaded_major() != expected) return 4;
    const std::wstring path = slopfab::cuda::cublas_loaded_path();
    if (path.find(L"cublas64_" + std::to_wstring(expected) + L".dll") ==
        std::wstring::npos)
      return 5;

    float* a = nullptr;
    float* b = nullptr;
    float* c = nullptr;
    SLOPFAB_CUDA_CHECK(cudaMalloc(&a, sizeof(float)));
    SLOPFAB_CUDA_CHECK(cudaMalloc(&b, sizeof(float)));
    SLOPFAB_CUDA_CHECK(cudaMalloc(&c, sizeof(float)));
    const float ha = 2.0f, hb = 3.0f;
    SLOPFAB_CUDA_CHECK(cudaMemcpy(a, &ha, sizeof(float), cudaMemcpyHostToDevice));
    SLOPFAB_CUDA_CHECK(cudaMemcpy(b, &hb, sizeof(float), cudaMemcpyHostToDevice));
    const float alpha = 1.0f, beta = 0.0f;
    const cublasStatus_t gemm = slopfab::cuda::cublas_sgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, 1, &alpha, a, 1, b, 1,
        &beta, c, 1);
    float hc = 0.0f;
    SLOPFAB_CUDA_CHECK(cudaMemcpy(&hc, c, sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(c);
    cudaFree(b);
    cudaFree(a);
    slopfab::cuda::cublas_destroy(handle);
    if (gemm != CUBLAS_STATUS_SUCCESS || std::fabs(hc - 6.0f) > 1.0e-6f) return 6;
    std::printf("CUDA %d cuBLAS loaded from %ls\n", expected, path.c_str());
    if (!fake_root.empty()) std::filesystem::remove_all(fake_root);
    return 0;
  } catch (const std::exception& error) {
    if (!fake_root.empty()) std::filesystem::remove_all(fake_root);
    if (broken_explicit &&
        std::string(error.what()).find("slopfab-broken-cuda13-explicit") !=
            std::string::npos)
      return 0;
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
