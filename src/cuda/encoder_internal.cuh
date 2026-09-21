#pragma once
#include "slopfab/text/encoder.h"
#include "slopfab/attention.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "slopfab/cuda/device.h"
#include "slopfab/cuda/deterministic_math.cuh"
#include "slopfab/cuda/deterministic_gemm.cuh"
#include "slopfab/cuda/deterministic_attention.cuh"
#include "slopfab/cuda/gemm.cuh"
#include "slopfab/cuda/linear.cuh"
#include "slopfab/cuda/nn_kernels.cuh"
#include "slopfab/cuda/workspace.cuh"

namespace slopfab::text::encoder_detail {
using slopfab::cuda::ComputeType;
using slopfab::cuda::DeviceBuffer;
using slopfab::cuda::PinnedBuffer;
using slopfab::cuda::QuantFormat;
using slopfab::cuda::QuantWeight;
using slopfab::cuda::Workspace;

// src/text/encoder.cpp works out the on-disk shapes from its own copy of this,
// because it is host-compiled and linear.cuh drags in cuBLAS. If the two ever
// disagree the scale tensor's declared width is wrong and nothing else notices.
static_assert(slopfab::cuda::kNVFP4BlockSize == 16,
              "the encoder's nvfp4 shapes and scale swizzle assume 16 elements per block");

constexpr int kWarp = 32;
constexpr int kThreads = 256;
constexpr int kSoftmaxThreads = 256;
constexpr int kConvRotGroup = 256;
constexpr int kDefaultQueryBlock = 256;

// Score tiles dominate the attention footprint: `heads * bq * bk` elements at
// fp32 plus bf16. A quarter of what attention.cu allows itself, because this
// module may be sharing the card with 24.4 GB of resident weights and its
// sequences are two orders of magnitude shorter.
constexpr size_t kScoreTileBudget = 256ull << 20;

// MSVC's INFINITY macro is a double expression, which nvcc warns about on every
// use in float context. Build the bit pattern instead.
__device__ inline float neg_inf() {
  return __int_as_float(0xFF800000);
}

constexpr float kHostNegInf = -std::numeric_limits<float>::infinity();

inline size_t align_up(size_t n) {
  return (n + 255) / 256 * 256;
}

inline int grid_1d(size_t n, int block) {
  return static_cast<int>((n + block - 1) / block);
}

inline void require(bool ok, const std::string& message) {
  if (!ok)
    throw std::runtime_error("text encoder: " + message);
}

size_t resident_request_bytes(const EncoderConfig&, size_t weight_bytes, size_t total_device_bytes);
} // namespace slopfab::text::encoder_detail
