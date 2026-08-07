#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace vidfab::cuda {

// N=1, T=1 specialization of torch Conv3d. Activations are channel-major
// [C,H,W], checkpoint weights are FP16 [Cout,Cin,Kt,Kh,Kw]. For the causal
// encoder only temporal slice Kt-1 overlaps the single input frame.
void launch_keyframe_conv3d(const float* x, const __half* weight, const __half* bias, float* y,
                            int cin, int cout, int in_h, int in_w, int kernel,
                            int stride, bool reflect, bool asymmetric_downsample,
                            cudaStream_t stream);

// Time-isolated GroupNorm for one frame, followed by SiLU. Layout [C,H,W].
void launch_keyframe_groupnorm_silu(const float* x, const __half* weight, const __half* bias,
                                    float* y, int channels, int height, int width,
                                    int groups, float eps, cudaStream_t stream);

void launch_keyframe_add(const float* a, const float* b, float* y, size_t count,
                         cudaStream_t stream);

}  // namespace vidfab::cuda
