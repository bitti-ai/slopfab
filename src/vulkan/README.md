# Embedded output shader

`rgb_to_yuv.comp` converts planar fp32 RGB into packed BT.709 limited-range
YUV420 values. It is used only for output colour conversion; model inference
remains CUDA.

The checked-in SPIR-V was produced with Khronos glslang 16.5.0:

```text
glslang -V --target-env vulkan1.2 -S comp src/vulkan/rgb_to_yuv.comp -o src/vulkan/rgb_to_yuv.comp.spv
```

Expected SHA-256 digests (also checked by CMake before embedding):

```text
rgb_to_yuv.comp      77FA686E8C2CCCA0B62551211DEFB4D47DD941DD617627EFCE56DF35637D2C37
rgb_to_yuv.comp.spv  C4CEE251C15F54371387D2DBACB4637C76C0DBC5DC6C0663C1E1269C703E71E6
```

The runtime and build do not require a shader compiler or Vulkan SDK.

## Tensor primitive shader

`tensor_ops.comp` is the bounded neural-primitive module. One cached pipeline
provides fp32 add/add-bias, exact fp32-to/from-fp16 and bf16 conversion, fp32
transpose, bounds-safe trusted-index gather/scatter, head-major fp32 to
token-major bf16, and fp32 depth-to-space. Calls use the reusable batched tensor
path; exact copies use transfer commands and need no shader. The checked-in
modules were produced with the same Khronos glslang 16.5.0:

```text
glslang -V --target-env vulkan1.2 -S comp src/vulkan/tensor_ops.comp -o tensor_ops.raw.spv
python tools/add_spirv_float_controls.py tensor_ops.raw.spv src/vulkan/tensor_ops.comp.spv src/vulkan/tensor_ops_denorm.comp.spv
```

Expected SHA-256 digests (also pinned by CMake):

```text
tensor_ops.comp              A5E10D2FF909DC131E0E38E4BF5927DF1BC3A9FBB02F283BF5118942DFCAA806
tensor_ops.comp.spv          A11B87B00658E97B8A3B53A24618045EADBBFA929A44AD9BC08997F958FA2106
tensor_ops_denorm.comp.spv   E35519ECE4AF1021BF5056B100FAE32D8C3A8757FC0D439C9EAEDA9613F92AA7
```

The deterministic postprocessor adds explicit fp32 signed-zero/Inf/NaN and
round-to-nearest-even execution modes. The denorm variant additionally adds
`DenormPreserve`; it is selected only when the queried Vulkan 1.2 float-control
properties permit that mode. fp16 and bf16 narrowing use CUDA's canonical NaN
value (`0x7fff`) and round-to-nearest-even. Thus CMake hashes exactly the
modules executed by the driver, not an untracked runtime transformation.

The portable CUDA-exact domain for fp32 add and add-bias requires both operands
and the correctly rounded result to be zero, normal, or infinity; NaN payload
arithmetic is excluded. Subnormal operands/results are exact only when
`full_fp32_arithmetic_exactness()` is true. Callers that require that wider
domain must use `require_full_fp32_arithmetic_exactness()` and fail closed.

## Video-VAE normalization shaders

`tensor_norm.comp` supplies the fp32 RMSNorm and affine LayerNorm reduction
trees used by the shipped video VAE. They are compiled as two static modules:
each workgroup has 256 lanes and reproduces the CUDA warp-shuffle tree with
shared memory and uniform barriers. Keeping the RMS and Layer paths separate
also makes every module's barrier count statically uniform.

```text
glslang -V --target-env vulkan1.2 -S comp -DRMS_NORM=1 src/vulkan/tensor_norm.comp -o tensor_rms_norm.raw.spv
python tools/add_spirv_float_controls.py --preserve-only tensor_rms_norm.raw.spv src/vulkan/tensor_rms_norm.comp.spv
glslang -V --target-env vulkan1.2 -S comp src/vulkan/tensor_norm.comp -o tensor_layer_norm.raw.spv
python tools/add_spirv_float_controls.py --preserve-only tensor_layer_norm.raw.spv src/vulkan/tensor_layer_norm.comp.spv
```

The compiler is Khronos glslang 16.5.0. CMake pins these SHA-256 digests:

```text
tensor_norm.comp                    FD333EEE5C6E321793A107FF844C59E62FD53CF8605E8A200BDE785422DBF7D2
tensor_rms_norm.comp.spv            1B76BCD637A4C35CDC6F5D40E4A5D7855D8F74BC2BFA7E581E4D74C0B4B3EF98
tensor_layer_norm.comp.spv          2AD2BEC9A188C6839050DB399DC53C316AE95EE111511193D3E6EE8943F1CB1C
```

The modules declare `SignedZeroInfNanPreserve` but intentionally do not declare
`RoundingModeRTE`: the norm kernels perform no floating-point conversions, and
on the tested RTX 5090 driver (Vulkan 1.4.341) adding RTE to either complete
reduction module makes pipeline creation terminate with `0x80000003`. Isolated
shared/barrier-only and inverse-square-root/FMA-only probes both accepted every
mode; bisecting the complete RMS module showed SignedZero/Inf/NaN-only and
Denorm-only variants create successfully while RTE-only reproduces the fault.
There is no runtime SPIR-V mutation.

Exact VAE normalization is consequently a separate, fail-closed capability:
`exact_fp32_vae_normalization()` currently requires a compatible NVIDIA Vulkan
device and the queried signed-zero/Inf/NaN and RTE properties. The arithmetic
contract covers zero and finite-normal input, affine values, epsilon,
intermediates and results. NaNs and subnormal arithmetic are excluded;
host-known subnormal epsilon is rejected before recording. Other tensor
primitives remain available on non-NVIDIA Vulkan devices.
