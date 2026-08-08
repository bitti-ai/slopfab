#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstddef>
namespace vidfab::cuda {
struct TensorScan { unsigned long long nonfinite; unsigned max_bits; unsigned pad; };
void launch_tensor_scan(const __nv_bfloat16* x,size_t n,TensorScan* out,cudaStream_t stream);
}
