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
tensor_norm.comp                    F82042E4D335DA14E442ABD2DDE6DF877B832BCCC8C14033D2AA22F7ED03C325
tensor_rms_norm.comp.spv            160B9D0F6260BC912379C26171DD8D0C60AD6E4D9B81DA57A413B831E513229C
tensor_layer_norm.comp.spv          516ECC94D93CD829F6F6B208ACFB3FF647861DAAE9DB3017B31729528D8724E5
```

The modules declare `SignedZeroInfNanPreserve` but intentionally do not declare
`RoundingModeRTE`: the norm kernels perform no floating-point conversions, and
on the tested RTX 5090 driver (Vulkan 1.4.341) adding RTE to either complete
reduction module makes pipeline creation terminate with `0x80000003`. Isolated
shared/barrier-only and inverse-square-root/FMA-only probes both accepted every
mode; bisecting the complete RMS module showed SignedZero/Inf/NaN-only and
Denorm-only variants create successfully while RTE-only reproduces the fault.
There is no runtime SPIR-V mutation.

CUDA `rsqrtf` and SPIR-V `InverseSqrt` differ by one ULP on a real AdaLN row,
and Vulkan's default division also differs from CUDA RNE on a rounding boundary.
All CUDA and Vulkan norm kernels therefore share a specified integer path:
binary32 `/dim` and normal epsilon addition use integer RNE, reciprocal square
root normalizes the mantissa to [1,4), performs six Q30 integer Newton steps
with ties-even shifts, and constructs the final binary32 value with RNE. One
lane computes it per row and broadcasts it. Products are proven below 2^62.
The path maps +/-0 to signed infinity, +infinity to +0, and negative/NaN to
canonical `0x7fc00000`; positive subnormals are normalized exactly. The private
addition helper is only defined for the nonnegative quotient plus normal
epsilon contract and a finite normal result. Dimensions above 2^24 and
subnormal epsilon are rejected.

Exact normalization is a separate, fail-closed capability:
`exact_normalization()` currently recognizes only the measured tuple
RTX 5090 (`deviceID 0x2b85`) with NVIDIA driver 610.88 (`driverVersion
0x98960000`), the queried signed-zero/Inf/NaN property, and an explicitly
enabled `shaderInt64` feature. The old fp32-VAE-named gate remains an alias. The
arithmetic contract covers zero and finite-normal input, affine values,
epsilon, intermediates and results. NaNs and subnormal arithmetic are excluded;
host-known subnormal epsilon is rejected before recording. Unknown devices and
drivers fail before recording a norm; other tensor primitives remain usable.

## Shared transformer normalization shaders

`tensor_shared_norm.comp` adds BF16 RMSNorm (a 32x8 narrow head-128 path and a
256-lane wide/scalar path), BF16 affine LayerNorm, and BF16/fp32 RMSNorm+AdaLN.
The wide reduction preserves CUDA's pack-major eight-value FMA order; the
narrow reduction preserves its 16-active-lane XOR tree. AdaLN rounds `1+scale`
separately and uses one final FMA. All five cached modules use the deterministic
normalization path above and are generated with glslang 16.5.0 plus
`--preserve-only` from `tools/add_spirv_float_controls.py`:

```text
glslang -V --target-env vulkan1.2 -S comp -D<VARIANT>=1 src/vulkan/tensor_shared_norm.comp -o <variant>.raw.spv
python tools/add_spirv_float_controls.py --preserve-only <variant>.raw.spv src/vulkan/tensor_<variant-lower>.comp.spv
```

Variants are `BF16_RMS_BLOCK`, `BF16_RMS_NARROW`, `BF16_LAYER`, `BF16_MOD`,
and `FP32_MOD`. CMake pins:

```text
tensor_shared_norm.comp             6435B32E3C0272EA0F20695CA38E993E01BBE8B16688258C2C79BC5845CFA972
tensor_bf16_rms_block.comp.spv       5784286FC645492FE7F9B68E292F47E1D8B6EFD5FF3BD213BEF003559F5EB49F
tensor_bf16_rms_narrow.comp.spv      5473B833FF74234C38817535F57B33DE46534AA678E5DCFE0F562A0B4F1779A2
tensor_bf16_layer.comp.spv           5F33AB2D86663D2611D64CDE7A62F27B13471256C808D067B80CF2BAEAC5C96A
tensor_bf16_mod.comp.spv             7C6634DF12153142FC0C17D9A7EC89275E9BFDE45EC7C0249E8353B7685F16AE
tensor_fp32_mod.comp.spv             C744EE45ACF2E34C017A1D1EC83BE5BC83042BA29AAC95C70AE864D55198F5EC
```

The deterministic integer normalization cost was measured on the validated RTX
5090/driver tuple against the pre-deterministic production implementation at
`330e34e`. CUDA medians are seven runs of 10 warmups plus 100 production-kernel
launches; Vulkan values use the same production APIs in 32-op batches after five
warmups (30 timed batches). Times are microseconds per complete operator:

| operator/shape | CUDA native | CUDA deterministic | Vulkan native | Vulkan deterministic |
|---|---:|---:|---:|---:|
| BF16 RMS 32768x128 | 8.683 | 18.137 | 11.855 | 31.712 |
| BF16 RMS 2048x5120 | 13.541 | 16.504 | 15.582 | 21.037 |
| BF16 RMS 2048x5376 | 14.244 | 16.148 | 15.992 | 22.021 |
| BF16 Layer 2048x1152 | 8.206 | 11.764 | 11.612 | 16.419 |
| BF16 Layer 1024x4608 | 14.556 | 18.214 | 15.274 | 16.086 |
| BF16 AdaLN 2048x5376 | 14.353 | 17.582 | 20.546 | 26.276 |
| fp32 AdaLN 2048x5376 | 30.795 | 31.142 | 29.687 | 33.575 |

These are isolated operator measurements, not an end-to-end generation claim.
The integer reciprocal square root runs once per logical row and is broadcast;
it is never recomputed by every lane.

The shipped 50-block DiT invokes two wide modulated norms and two head-128 norms
per block, plus its final fp32 modulated norm: 201 normalization launches per
denoising step. Applying the measured CUDA median deltas gives an estimated
1.27 ms per step, or about 37 ms over 29 steps. This is an arithmetic-kernel
impact estimate only; it excludes future Vulkan graph submission and every
other inference stage, and therefore is not presented as an end-to-end speedup
or regression measurement.

## Keyframe-VAE GroupNorm+SiLU shader

`tensor_group_norm.comp` implements the fused GroupNorm+SiLU operation used by
the keyframe encoder: contiguous fp32 CHW input/output, fp16 channel affine,
32 groups, and one 256-lane workgroup per group. It preserves the CUDA
strided accumulation and binary reduction tree, including
`max(0, E[x^2] - mean^2)`, and reuses the deterministic normalization helpers
above. The cached four-binding pipeline supports in-place input/output without
a host boundary or per-dispatch allocation. Dimensions whose group population
exceeds 2^24 fail before recording.

The production module is generated with Khronos glslang 16.5.0 and the same
float-control transformer used by the other tensor shaders:

```text
glslang -V --target-env vulkan1.2 -S comp src/vulkan/tensor_group_norm.comp -o tensor_group_norm.raw.spv
python tools/add_spirv_float_controls.py tensor_group_norm.raw.spv src/vulkan/tensor_group_norm.comp.spv tensor_group_norm.denorm.spv
```

CMake pins the exact source and executed module:

```text
tensor_group_norm.comp              988FAEC12F82A744A49CBB5E46AAE35B5563F408A7809E72652E4B67ED1457CA
tensor_group_norm.comp.spv          57AED42788D1D8CA9A3F6EDE8D6130B2C32FDA2D62C2261FE00F1D3FF2E98D33
```

Native CUDA `expf` and Vulkan `Exp`/division differed by one ULP in every
initial exact GroupNorm fixture. Both backends now use the same specified SiLU:
non-positive range reduction, a fixed degree-10 FMA polynomial, an
integer-constructed power of two, and an explicitly ordered sigmoid division.
The CUDA production result intentionally changed. The contract preserves
signed zero, maps +infinity to +infinity, -infinity to -0, and every NaN to
canonical `0x7fc00000`. Subnormal inputs and correctly rounded subnormal
results become signed zero. Inputs at or below -87 also become signed zero;
this deliberately discards a still-normal true result whose largest magnitude
at the cutoff is 1.432e-36. Over the continuous tested range (-87, 87), the
CUDA implementation differs from a double reference by at most two binary32
ULPs (maximum relative error 2.018e-7). The Vulkan module declares RTE and is
available only through the existing fail-closed exact-normalization device,
driver, float-control, and shaderInt64 gate.

Performance was measured on the validated RTX 5090/610.88 tuple with nonzero
normal input and affine values, three warmups, 10 launches per sample, and four
samples. CUDA compares the pre-deterministic production implementation at
`330e34e` with the current complete deterministic GroupNorm+SiLU; Vulkan uses
the identical current shader/reduction and changes only native versus
deterministic SiLU. Upload and readback are outside the timed region. The
largest device-resident operator footprint is 2.0 GiB input, 2.0 GiB output,
and 512 bytes of affine data with no GroupNorm scratch; the benchmark's
persistent 2.0 GiB upload buffer is setup infrastructure, not operator memory.
Times are medians in milliseconds:

| CHW shape | CUDA previous | CUDA deterministic | Vulkan native SiLU | Vulkan deterministic |
|---|---:|---:|---:|---:|
| 128x2048x2048 | 52.676 | 58.245 | 34.919 | 41.881 |
| 256x1024x1024 | 26.358 | 29.154 | 18.298 | 21.198 |
| 1024x128x128 | 1.577 | 1.754 | 1.084 | 1.269 |

Weighting those measurements by the shipped 25-call encoder topology (four
calls per two-block level across six levels, plus the final norm) gives a
GroupNorm-only estimate of 345.9 to 382.6 ms on CUDA (+36.7 ms, 10.6%) and
233.0 to 276.2 ms on Vulkan (+43.1 ms, 18.5%). Unmeasured intermediate level
sizes are scaled by the exact per-group element count from the adjacent real
shape. This is a concrete graph-weighted kernel estimate, not an end-to-end
encoder benchmark; convolution and every remaining neural stage are excluded.
