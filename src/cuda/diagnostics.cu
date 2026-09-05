#include "slopfab/cuda/diagnostics.cuh"
#include "slopfab/cuda/device.h"
namespace slopfab::cuda {
__global__ void scan(const __nv_bfloat16* x,size_t n,TensorScan* out){
  for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x){
    const float v=__bfloat162float(x[i]);
    if(!isfinite(v)) atomicAdd(&out->nonfinite,1ull);
    else atomicMax(&out->max_bits,__float_as_uint(fabsf(v)));
  }
}
void launch_tensor_scan(const __nv_bfloat16* x,size_t n,TensorScan* out,cudaStream_t stream){
  scan<<<256,256,0,stream>>>(x,n,out);SLOPFAB_CUDA_CHECK(cudaGetLastError());
}
}
