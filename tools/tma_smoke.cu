// Standalone SM90+ TMA smoke test. Build with:
//   nvcc -std=c++17 -arch=sm_120 -O2 tools/tma_smoke.cu -lcuda -o tma_smoke.exe
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {
constexpr int kTileRows = 64;
constexpr int kTileCols = 128;
constexpr int kSourceRows = 80;
constexpr int kSourcePitch = 160; // Deliberately strided (320 bytes per row).
constexpr int kStartRow = 7;
constexpr int kTileBytes = kTileRows * kTileCols * sizeof(uint16_t);

#define CUDA_CHECK(call)                                                                           \
  do {                                                                                             \
    cudaError_t status_ = (call);                                                                  \
    if (status_ != cudaSuccess) {                                                                  \
      std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(status_));        \
      return 1;                                                                                    \
    }                                                                                              \
  } while (0)

#define CU_CHECK(call)                                                                             \
  do {                                                                                             \
    CUresult status_ = (call);                                                                     \
    if (status_ != CUDA_SUCCESS) {                                                                 \
      const char* message_ = nullptr;                                                              \
      cuGetErrorString(status_, &message_);                                                        \
      std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,                                      \
                   message_ ? message_ : "CUDA driver error");                                     \
      return 1;                                                                                    \
    }                                                                                              \
  } while (0)

__global__ void tma_load(const __grid_constant__ CUtensorMap map, uint16_t* output) {
  __shared__ alignas(128) uint16_t tile[kTileRows * kTileCols];
  __shared__ alignas(8) uint64_t barrier;

  if (threadIdx.x == 0) {
    uint32_t tile_addr = static_cast<uint32_t>(__cvta_generic_to_shared(tile));
    uint32_t barrier_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&barrier));
    uint64_t state;
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n"
                 "fence.proxy.async.shared::cta;\n"
                 "mbarrier.arrive.expect_tx.shared::cta.b64 %1, [%0], %2;\n"
                 "cp.async.bulk.tensor.2d.shared::cta.global.tile."
                 "mbarrier::complete_tx::bytes [%3], [%4, {%5, %6}], [%0];"
                 : "+r"(barrier_addr), "=l"(state)
                 : "r"(kTileBytes), "r"(tile_addr), "l"(&map), "r"(0), "r"(kStartRow)
                 : "memory");
    uint32_t ready = 0;
    do {
      asm volatile("{ .reg .pred p; "
                   "mbarrier.test_wait.shared::cta.b64 p, [%1], %2; "
                   "selp.b32 %0, 1, 0, p; }"
                   : "=r"(ready)
                   : "r"(barrier_addr), "l"(state)
                   : "memory");
    } while (!ready);
  }
  __syncthreads();

  for (int i = threadIdx.x; i < kTileRows * kTileCols; i += blockDim.x) {
    output[i] = tile[i];
  }
}
} // namespace

int main() {
  CUDA_CHECK(cudaFree(nullptr)); // Establish the primary driver context.

  std::vector<uint16_t> source(kSourceRows * kSourcePitch, 0xdead);
  for (int row = 0; row < kSourceRows; ++row) {
    for (int col = 0; col < kTileCols; ++col) {
      source[row * kSourcePitch + col] =
          static_cast<uint16_t>(0x3f00u + ((row * 131 + col) & 0xffu));
    }
  }

  uint16_t *device_source = nullptr, *device_output = nullptr;
  CUDA_CHECK(cudaMalloc(&device_source, source.size() * sizeof(uint16_t)));
  CUDA_CHECK(cudaMalloc(&device_output, kTileBytes));
  CUDA_CHECK(cudaMemcpy(device_source, source.data(), source.size() * sizeof(uint16_t),
                        cudaMemcpyHostToDevice));

  CUtensorMap map{};
  const cuuint64_t global_dims[2] = {kTileCols, kSourceRows};
  const cuuint64_t global_strides[1] = {kSourcePitch * sizeof(uint16_t)};
  const cuuint32_t box_dims[2] = {kTileCols, kTileRows};
  const cuuint32_t element_strides[2] = {1, 1};
  CU_CHECK(cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, device_source, global_dims, global_strides,
      box_dims, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

  tma_load<<<1, 256>>>(map, device_output);
  CUDA_CHECK(cudaGetLastError());
  std::vector<uint16_t> output(kTileRows * kTileCols);
  CUDA_CHECK(cudaMemcpy(output.data(), device_output, kTileBytes, cudaMemcpyDeviceToHost));

  int mismatches = 0;
  for (int row = 0; row < kTileRows; ++row) {
    for (int col = 0; col < kTileCols; ++col) {
      const uint16_t expected = source[(row + kStartRow) * kSourcePitch + col];
      const uint16_t actual = output[row * kTileCols + col];
      if (actual != expected && mismatches++ < 8) {
        std::fprintf(stderr, "mismatch [%d,%d]: got 0x%04x, expected 0x%04x\n", row, col, actual,
                     expected);
      }
    }
  }
  cudaFree(device_output);
  cudaFree(device_source);
  if (mismatches) {
    std::fprintf(stderr, "FAIL: %d mismatches\n", mismatches);
    return 1;
  }
  std::printf("PASS: TMA loaded strided BF16 tile 64x128 (%d bytes)\n", kTileBytes);
  return 0;
}
