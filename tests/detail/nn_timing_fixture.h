#pragma once
#include "nn_kernels_fixture.h"

namespace {
// --- timings ----------------------------------------------------------------
//
// Not an assertion, a measurement. The numbers go to the CUDA critic alongside
// the kernels; the only thing checked here is that the production shapes run at
// all on this card.

struct Timer {
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  Timer() {
    SLOPFAB_CUDA_CHECK(cudaEventCreate(&start));
    SLOPFAB_CUDA_CHECK(cudaEventCreate(&stop));
  }
  ~Timer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }
  template <class F>
  float measure(F&& f, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) f();
    SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
    SLOPFAB_CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) f();
    SLOPFAB_CUDA_CHECK(cudaEventRecord(stop));
    SLOPFAB_CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    SLOPFAB_CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms / float(iters);
  }
};

// --- operands for a timing kernel -------------------------------------------
//
// **Constant operands are not a neutral choice, and this benchmark used to make
// it.** A `cudaMemset` fills a buffer with one repeated byte, so a GEMM over two
// such buffers multiplies the same pair of 16-bit values in every lane on every
// cycle and the tensor-core multiplier array barely toggles. Dynamic power *is*
// switching activity, so the card stops drawing its 575 W limit and boosts to
// the top of its V/F curve — measured here at 2865 MHz against the ~2.5 GHz it
// holds on real data, which is worth about 14% of throughput.
//
// That is enough to have put the reported figures **above this card's own
// sustained bf16 ceiling** (216-222 TFLOP/s): the four linears read 234-237
// TFLOP/s in the table this file feeds, a physical impossibility that sat in
// README.md unnoticed while the same table declared the ceiling two lines above.
//
// So the operands are filled with pseudo-random bits instead. The fill runs on
// the device because the largest of these buffers is 1.08 GB and staging it
// through the host would cost more than the measurement.
__global__ void fill_bf16_kernel(uint16_t* __restrict__ dst, size_t n, uint32_t seed) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint32_t s = static_cast<uint32_t>(i) * 2654435761u + seed;
  s ^= s << 13;
  s ^= s >> 17;
  s ^= s << 5;
  // Sign and all seven mantissa bits random; exponent in [126, 128], i.e. |x| in
  // [0.5, 4). Broad enough to toggle the datapath, tight enough that neither a
  // K = 14336 dot product nor a 37710-key softmax can leave range.
  const uint32_t exp = 126u + (s >> 28) % 3u;
  dst[i] = static_cast<uint16_t>((s & 0x8000u) | (exp << 7) | ((s >> 8) & 0x7Fu));
}

// e4m3 codes. The exponent is kept near the format's bias of 7 and the two NaN
// patterns (|code| == 0x7F) are unreachable by construction rather than masked
// out afterwards, so no fill can turn a timing run into a NaN propagation study.
__global__ void fill_f8_kernel(uint8_t* __restrict__ dst, size_t n, uint32_t seed) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint32_t s = static_cast<uint32_t>(i) * 2246822519u + seed;
  s ^= s << 13;
  s ^= s >> 17;
  s ^= s << 5;
  const uint32_t e = 5u + (s >> 27) % 5u;  // 2^-2 .. 2^2 before the tensor scale
  dst[i] = static_cast<uint8_t>((s & 0x80u) | (e << 3) | ((s >> 4) & 0x07u));
}

void fill_random(BfBuf& b, uint32_t seed) {
  const size_t n = b.raw.size();
  fill_bf16_kernel<<<static_cast<int>((n + 255) / 256), 256>>>(b.raw.get(), n, seed);
  SLOPFAB_CUDA_CHECK(cudaGetLastError());
}

void fill_random_f8(DeviceBuffer<uint8_t>& b, uint32_t seed) {
  const size_t n = b.size();
  fill_f8_kernel<<<static_cast<int>((n + 255) / 256), 256>>>(b.get(), n, seed);
  SLOPFAB_CUDA_CHECK(cudaGetLastError());
}

// The dequantiser runs once per GEMM on every one of the 200 quantised linears,
// fifty times a step, so its cost is not incidental. It is pure streaming and
// should sit near the card's bandwidth: 2 bytes written and 9/16 read per
// element, of which the store is the overwhelming majority.
//
// This one keeps its `cudaMemset` fills deliberately: it is bound by how many
// bytes cross the memory system, not by multiplier switching, and it is the
// control that says so.


}  // namespace
