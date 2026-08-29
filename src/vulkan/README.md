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

## Video-VAE pointwise shaders

Three fixed-operation modules cover the remaining pointwise seams in the ViT
decoder without a second full-tensor pass: in-place fp32 layer-scale residual,
fused biased fp32 SwiGLU, and channel-major fp32 latent denormalization. The
production ViT shape is 36 blocks at D2048/I8192: every block records two
residual fusions and one SwiGLU. Existing `transpose_2d`, `heads_to_tokens`,
`split_qkv`, and `depth_to_space` primitives already cover its layout changes;
there is deliberately no duplicate pointwise layout API. Latent denormalization
is a future device-residency seam: the current CUDA decode pipeline still
performs that step on the host before invoking the Video-VAE.

All APIs require contiguous, distinct fp32 allocations. Residual is
`canon(fma(canon(canon(y)+canon(bias)),canon(scale),canon(x)))` and updates x in
place. Denormalization is `canon(fma(canon(z),canon(std),canon(mean)))`.
SwiGLU separately canonicalizes both bias adds, evaluates the accepted
deterministic SiLU polynomial, and canonicalizes the final explicit-RN
multiplication. At every boundary a binary32 subnormal becomes signed zero and
every NaN becomes `0x7fc00000`; signed zero, normals, and infinity retain their
canonical bits until arithmetic combines them. The SiLU cutoff `x<=-87` maps to
signed zero. CUDA's historical nullable-bias launch ABI remains available and
preserves its no-add branch, while the Vulkan production API requires bias.

The three persistent pipelines share a four-binding layout but declare only
the accesses used by their fixed operation. They use one dispatch, zero
operator scratch, no host scan/copy, and no steady-state allocation. Recording
validates type, rank, exact dimensions, nonzero extents, uint32 shader indices,
checked products/dispatch, context and allocation identity before access-state
mutation. DeviceTensor exposes whole allocations rather than offset subviews,
so overlap validation is allocation-identity rejection. Tests cover 3x67
tails, normal cancellation to a subnormal, output
underflow, signed zeros, Inf/NaN combinations, the -87 ULP neighbors, CUDA null
bias, rejection rollback, mixed 32/33-op batches, two in-flight submissions
plus third-slot reuse, wrapper drop, and stable pool/descriptor high-water. The
65k-plus dense SiLU corpus measures maximum 2 ULP, absolute error 1.249e-6 and
relative error 2.018e-7 against double precision.

Pointwise capability is deliberately separate from normalization even though
the first qualified tuple is the same RTX 5090 (`2b85`), NVIDIA 610.88
(`98960000`). It additionally requires shaderInt64, fp32 RTE, and
signed-zero/Inf/NaN preservation. This prevents a future normalization tuple
from silently enabling independently compiled pointwise modules.

Compiler isolation on that tuple found a driver compiler breakpoint
`0x80000003`: the combined dynamic-operation raw and preserve-only probes
created, while their RTE-only/normal variants failed; the specialized residual
normal module created, but the specialized SwiGLU normal module failed. The
committed residual therefore carries RTE plus preserve modes. SwiGLU is
preserve-only plus Int64 and defines both positive denominator addition and
division with live integer RNE arithmetic. Denormalization is preserve-only.
There is no runtime SPIR-V mutation and no dead capability-perturbing code.

The modules use Khronos glslang 16.5.0 and the repository float-control tool:

```text
glslang -V --target-env vulkan1.2 -S comp -DVAE_RESIDUAL=1 src/vulkan/tensor_vae_pointwise.comp -o residual.raw.spv
python tools/add_spirv_float_controls.py residual.raw.spv src/vulkan/tensor_vae_residual.comp.spv residual.denorm.spv
glslang -V --target-env vulkan1.2 -S comp -DVAE_SWIGLU=1 src/vulkan/tensor_vae_pointwise.comp -o swiglu.raw.spv
python tools/add_spirv_float_controls.py --preserve-only swiglu.raw.spv src/vulkan/tensor_vae_swiglu.comp.spv
glslang -V --target-env vulkan1.2 -S comp -DVAE_DENORM=1 src/vulkan/tensor_vae_pointwise.comp -o denorm.raw.spv
python tools/add_spirv_float_controls.py --preserve-only denorm.raw.spv src/vulkan/tensor_vae_denorm.comp.spv
nvcc --fatbin -std=c++17 -ccbin <MSVC-14.44> --generate-code=arch=compute_120a,code=[compute_120a,sm_120a] -Iinclude src/cuda/vae_kernels.cu -o vae_kernels.fatbin
```

```text
tensor_vae_pointwise.comp                 C0D3CA5A113AB83C08A12D2F06A63D127D2858FE72D96DF8E58F4F6AF559B83D
tensor_vae_residual.comp.spv              8B64D04CB5B579F787148FB546D77AFB1ED6B1A1782BDE8D6573B28EA9408C92
tensor_vae_swiglu.comp.spv                E4F51A55B55D7888FC264C1FA2F547BBC8930D9D7DB5F627D0356220B92CB453
tensor_vae_denorm.comp.spv                E295EDCAD9058D8581007B4776F0BC7DB95A10C99AA9DF4FB149947AAE9519F2
src/cuda/vae_kernels.cu                   48D1B0281E6D4A1B5AF915BA394E964701FE6C0CBBFFC03B09F4BF00AD40FE8C
include/vidfab/cuda/deterministic_math.cuh F84F74E46D0EA32E5F2A2B6FF5E87F16C71A86B97D1379C96768C022D84F1913
vae_kernels.fatbin                        17AA0AD0176D273716E2403E3F8B6599BD1D8BD9C62FEC55A06108BE8F5A3F02
```

A real-weight audit used `minimax_h3_video_vae_fp16.safetensors` SHA-256
`7C1F131492E7EDDACAAC9069A61B81BDD39DE5CC96561E677C5EAB1CDCE5E522`,
seed123 synthetic latents, and a 256x256/22-frame decode. The first real block
was R1797/D2048/I8192. Its 29,442,048 raw and biased FFN values had zero
subnormals/nonfinites; minima were 4.191e-9/6.985e-9 and maxima
33.4604/33.5927, with zero gate values at or below -87. The 14,721,024 SwiGLU
outputs had zero subnormals/nonfinites, minimum nonzero 5.015e-13 and maximum
64.1191. Residual x/y/biased/output each had 3,680,256 values and zero
subnormals/nonfinites; maxima were 0.5391/20.8282/20.8779/0.8486 and final
minimum nonzero was 4.961e-9.

Device-resident Release timing on the pinned tuple excludes upload/download
and averages eight launches. Direct benchmark tensors occupy 197.0 MiB;
425.2 MiB reserved includes persistent benchmark upload/readback staging, not
operator scratch:

| R1797 real decoder shape | shipped CUDA | exact CUDA | exact Vulkan |
|---|---:|---:|---:|
| biased SwiGLU I8192 | 0.092 ms | 0.136 ms | 0.139 ms |
| 36 SwiGLU calls | 3.3 ms | 4.9 ms | 5.0 ms |
| layer residual D2048 | - | 0.012 ms | 0.034 ms |
| latent denorm C24/V1792 | - | 0.005 ms | 0.007 ms |

The shipped-to-exact SwiGLU rebaseline changed 4,213,488 of 14,721,024 words
on the deterministic benchmark corpus with maximum absolute delta 1.788139e-7.

## Exact video-VAE ViT block stage

`vulkan::ExactViTBlockStage` composes one complete pre-norm decoder block from
the exact fp32 VAE norms, fp16 preparation/GEMM, fused split-QKV/norm/RoPE,
blocked D64 attention, BF16 conversion, layer-scale residual and biased SwiGLU
primitives. Its production `record` entry point updates an external fp32 token
tensor in place inside the caller's `TensorBatch`. `ExactViTBlockScratch` owns
the activation arena and prepared slots separately from immutable block
weights, so a future 36-block graph can share one scratch object and record all
blocks without a host boundary, per-block allocation, or per-block submission.
The host `forward` method is only a parity convenience.

The common loader pins the shipped names
`decoder.transformer_blocks.{i}.{norm1.weight,norm2.weight,scale1,scale2,
attn.to_qkv.weight,attn.to_qkv.bias,attn.to_out.weight,attn.to_out.bias,
ff.w1.weight,ff.w1.bias,ff.w2.weight,ff.w2.bias}`. Matrices must be rank-2 fp16
`[out,in]`; vectors must be rank-1 fp16/fp32. It keeps matrices compressed at
fp16 and widens only vectors. The real block-0 audit found 330,659 fp16
subnormal matrix words. CUDA WMMA and Vulkan cooperative matrices did not
share their treatment (the first final result differed by two ULP), so exact
mode canonicalizes those checkpoint words to signed zero and raw weight views
with uncanonicalized fp16 subnormals fail before allocation/upload.

The block forces the shared ascending-K scalar fp32-FMA GEMM order. This is an
intentional exact-mode performance rebaseline: even after weight
canonicalization, real values through cooperative matrices differed at the
final block boundary. The scalar mode is explicit in `DenseGemmPlanDesc` and
does not change existing cooperative plans. One exact block records 20 bounded
operators. `TensorContextOptions::max_batch_operators` makes graph capacity an
explicit bounded choice (default 32, maximum 4096); a 36-block graph will use
at least 720 and conventionally reserve 1024. Stage validation checks all
tensor shapes/identity, scratch compatibility and remaining capacity before it
records the first operation.

Release qualification used CUDA 13.0.48, MSVC 14.44.35207, Vulkan 1.4.341,
RTX 5090 / NVIDIA 610.88 and
`minimax_h3_video_vae_fp16.safetensors` SHA-256
`7C1F131492E7EDDACAAC9069A61B81BDD39DE5CC96561E677C5EAB1CDCE5E522`.
The opt-in replay hashes the mapped checkpoint and asserts this complete
SHA-256 before reading weights. It scans the four raw block-0 matrix
`TensorView`s and asserts exactly 330,659 fp16 subnormal words, then asserts
zero fp16 matrix subnormals and zero fp32 vector subnormals after typed load.
Layer 0 was loaded through the typed name/shape/dtype path and evaluated at the
real R1797/D2048/I8192 shape using deterministic finite-normal input and
identity rotary tables. CUDA and Vulkan matched all 3,680,256 final fp32 words;
the replay asserts FNV64 `c8a7ac3241effbb7`. The CUDA comparison convenience measured 169.062
ms including its host boundary; Vulkan record-to-completion measured 64.741 ms
with upload/download excluded. Vulkan direct stage accounting was 128.1 MiB of
persistent weights and 507.8 MiB peak including one reusable scratch arena,
tokens and rotary tables; operator scratch is bounded by the prepared slots
and there is no quadratic attention buffer. This is one block, not the wired
36-block decoder.

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

## RoPE shaders and canonical H3 tables

`tensor_rope.comp` contains two distinct in-place BF16 operators: H3 rotates
channels 0..95 as half-split pairs and preserves channels 96..127 byte-for-byte,
while GPT-NeoX rotates the complete even-width head (128 for Qwen text and 72
for Qwen vision). `tensor_vae_rope.comp` is the separate fp32 video-VAE
operation: split interleaved QKV, add bias, reproduce the 32-lane head-width-64
RMS reduction, rotate only the leading 48 channels, and write head-major Q/K/V.
Its seven bindings remain owned by the bounded job until timeline completion.

H3 tables have one canonical implementation, `build_h3_rope_tables`, on the
host. It performs double `pow`, rounds inverse frequency to fp32, converts each
position to fp32 before fp32 multiplication, then serializes host `cos`/`sin`
bits in row-major interleaved T/H/W order. CUDA and Vulkan upload those bytes
unchanged; no shader has a second trig builder. The small-fixture FNV-1a hash
is pinned to `dfd06ee912173e0f`.

Both BF16 backends canonicalize subnormal BF16 operands, fp32 table operands,
rounded products, internal high-half FMA products, and rounded BF16 results to
signed zero. Exact-zero operands retain the FMA path for signed-zero addition.
NaNs canonicalize on BF16 output. Tests enumerate all 254 signed nonzero BF16
subnormal patterns, minimum-normal neighbors, cancellation, result underflow,
and an underflowed internal product with a normal FMA base.

The production modules use Khronos glslang 16.5.0. BF16 RoPE uses the normal
and denorm float-control variants; the barrier/int64 VAE module uses the
validated preserve-only norm policy:

```text
glslang -V --target-env vulkan1.2 -S comp src/vulkan/tensor_rope.comp -o tensor_rope.raw.spv
python tools/add_spirv_float_controls.py tensor_rope.raw.spv src/vulkan/tensor_rope.comp.spv src/vulkan/tensor_rope_denorm.comp.spv
glslang -V --target-env vulkan1.2 -S comp src/vulkan/tensor_vae_rope.comp -o tensor_vae_rope.raw.spv
python tools/add_spirv_float_controls.py --preserve-only tensor_vae_rope.raw.spv src/vulkan/tensor_vae_rope.comp.spv
```

```text
tensor_rope.comp                    B98E96DEBC46548AC25065F4BA75037EBDA95B92F43303EDC625E50E4FDC6347
tensor_rope.comp.spv                FB36A5238C64DFFD756D65471E0BB492B4FE59910BB676F1D8A4109715391A90
tensor_rope_denorm.comp.spv         E15674264A7CF898C286CC4E9110F8D7A5918732682120010EE02272D73E1734
tensor_vae_rope.comp                57F4761647D8F8F1775CD8EDCE0156FF5A89B6E19ADF99B51330D78056B9C8FD
tensor_vae_rope.comp.spv            E51AE5F500A65B87B5FD1AF440E0B60DB375A75DC7021620FAE40379812A8196
```

An instrumented real-checkpoint audit used seed 424242, 256x256, 22 aligned
frames, one denoiser evaluation, and 526 packed rows. Checkpoints were
`qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` (SHA-256
`33E69E3EDAB846D52949BAFDB00378BD3F5A93F78124FC83D5EF109DC4A1FCBB`)
and `MiniMax_H3_FL2VA_pruned_nvfp4.safetensors` (SHA-256
`6AB7F0C48141E7919B32F925CA3DEF22E06A6AEBEB9E0B6F5A0BE0FE8409976F`).
Actual conditioner and DiT RoPE input/table/product/result subnormal counts
were all zero. Minimum nonzero magnitudes were `0x32300000` (1.02445483e-8),
`0x3488a34f` (2.54507967e-7), `0x2b60b915` (7.98376393e-13), and
`0x32370000` (1.06520019e-8), respectively. Audit hooks were removed.

Steady-state RTX 5090/610.88 Release measurements used five warmups and 20
iterations, excluding upload/readback. H3 rows37710/heads56/dim128 uses a
27.620 MiB table pair built once in 15.746 ms and uploaded once in 7.143 ms
to CUDA or 3.279 ms to Vulkan. Local size 64 measured 1.037 ms per Vulkan
apply versus 1.249 ms at 128, so 64 is retained; CUDA measured 0.563 ms.
Qwen rows190/dim128 Q56+K8 combined measured 0.0149 ms CUDA and 0.0475 ms
Vulkan. Fused VAE seq1797/heads32 measured 0.0412 ms CUDA and 0.1282 ms
Vulkan. Fifty DiT blocks perform Q and K RoPE: 103.7 ms Vulkan versus 56.3 ms
CUDA per denoiser evaluation, a 47.4 ms delta or about 1.37 seconds over 29
evaluations. These are operator measurements, not an unwired-backend claim.

## Persistent linear-weight preparation

`vidfab::vulkan::LinearWeight` retains immutable checkpoint metadata and only
the compressed/native bytes and auxiliaries on device. It supports F32, F16,
BF16, E4M3 FP8, per-output I8, NVFP4 and bitsandbytes NF4. Dense BF16/FP16 is a
caller-owned prepared tensor, so an orchestrator can reuse one bounded buffer
for the active weight/chunk rather than expanding every resident quantized
matrix. Upload allocates every component first and copies them in one timeline
submission; temporary host-visible buffers are released after exact completion.
`resident_bytes()` is the logical persistent payload; allocator rounding remains
visible through `TensorContext::reserved_bytes()`.

The implementation preserves the NVFP4 high-even nibble convention, 128x4
block-scale swizzle and `(block_scale * global_scale)` then E2M1 multiply. NF4
is deliberately restricted to the shipped 64/256 double-quant block contract;
its nested scale is fused multiply-add, followed by the quant-map multiply.
FP8 input-scale presence/value and `full_precision_matrix_mult` remain immutable
metadata for the later native-GEMM planner. AWQ BF16/fp32 pre-scale and regular-
H4 ConvRot are separate batched device operations. This slice prepares weights
only: it does not claim or select a Vulkan GEMM.

The checked modules are generated with Khronos glslang 16.5.0:

```text
glslang -V --target-env vulkan1.2 -S comp src/vulkan/tensor_weight.comp -o tensor_weight.raw.spv
python tools/add_spirv_float_controls.py tensor_weight.raw.spv src/vulkan/tensor_weight.comp.spv src/vulkan/tensor_weight_denorm.comp.spv

tensor_weight.comp                    55FADB68982ABCDB0079C8D9B0A84E77F64D268E63E7227F2A2BC770ED60307F
tensor_weight.comp.spv                DA6454D969F13F02F6571EAA5850E7CD757E9AB0DF68E005557B277C4D73F30E
tensor_weight_denorm.comp.spv         D67D8F272B694181A350760C29C4ADACCAFF692B043E6FA8A619D0A8CB807DAE
```

Exact tests cover all 256 E4M3 patterns, all 16 E2M1 values, I8 extremes and
odd row tails, every 65,536-bit F16/BF16 same- and cross-format input, a 3x5
NVFP4 scale-tile grid, NF4 high-even/nested-boundary/odd tails, both activation
types, 32/33-operation bounds, two submitted slots and wrapper-drop retention.
On RTX 5090/610.88 Release, device-resident steady-state measurements excluded
upload/readback. A real 384x5376 NVFP4 slab from
`MiniMax_H3_FL2VA_pruned_nvfp4.safetensors` (SHA-256
`6AB7F0C48141E7919B32F925CA3DEF22E06A6AEBEB9E0B6F5A0BE0FE8409976F`)
materialized in 0.051 ms Vulkan versus 0.012 ms CUDA; one-time Vulkan upload
was 1.37 ms, persistent payload 1.11 MiB and prepared BF16 3.94 MiB. The
largest NF4 tensor selected from `video_vae_nf4.safetensors` (SHA-256
`6D0CB4FF02EBB74CC6BCA40018E6EFAE5082CCD7EB066FA1263098C6DBF8F6F1`),
flattened 16384x2048, materialized FP16 in 0.225 ms Vulkan versus 0.037 ms CUDA;
upload was 30.03 ms, persistent payload 16.51 MiB and prepared FP16 64 MiB.
Both full prepared payloads matched CUDA byte-for-byte. These are preparation
costs, not GEMM or end-to-end model timings.

## Deterministic dense GEMM

`DenseGemmPlan` implements the row-major NT contractions used by dense model
weights. BF16 activation/weight produces BF16, with an explicit BF16 rounding
boundary before the optional fp32/BF16 bias and a second BF16 round. Video-VAE
fp32 activations are narrowed once per chunk into a caller-owned
`PreparedF16Activation`; its batch-scoped view can feed distinct FP16-weight
plans without repeating conversion. FP16 GEMM and FP32 SGEMM produce FP32.
Every scalar edge path uses the same ascending-K fused multiply-add order:
Vulkan uses a fixed 16x16 shared-load tile and the CUDA reference loads directly.
Full BF16/FP16 tiles use the fastest exact KHR cooperative-matrix path on the
pinned RTX 5090/610.88 tuple; CUDA tests use matching WMMA tile/K-call order.
No path calls CUDA from the Vulkan backend. Exact tests cover all three scalar
modes at M3/N11/K19 with row offsets and tails, BF16 fp32 and BF16 bias, a
K5376 cooperative contraction, cancellation, signed zero, infinity, and output
padding. The arithmetic contract covers zero, finite-normal, and infinity
operands/intermediates/results; NaN payload identity and subnormal arithmetic
require the corresponding explicit capability and are not otherwise claimed.

The cooperative-matrix result order is implementation-defined, so it fails
closed unless vendor `10de`, device `2b85`, raw driver `98960000`, subgroup 32,
driver UUID `8690f1c80a3f54999bf6ea2aee515602`, and the required nonsaturating
16x16x16 BF16/F32 or FP16/F32 tuple all match. BF16 and FP16 gates are
independent. An experimental `VK_NV_cooperative_matrix2` workgroup path was
exact but slower (about 0.19 ms for the 64x5376x5376 probe) and is neither
embedded nor created at runtime. The retained KHR path measured about 0.16 ms
versus 0.026--0.030 ms cuBLAS for BF16 M64/N5376/K5376.

Release measurements below are batch-divided steady device work on the same
tuple; uploads are excluded. The FP16 preparation number is a separate exact
submit/wait and is paid once per changed activation chunk. The current
`vit_decoder` call site has fanout one, so the honest comparison includes that
preparation for every projection. Root acceptance explicitly permits these
fastest exact native paths despite their current cuBLAS ratio.

| Mode and shape | CUDA cuBLAS | Vulkan | Ratio |
|---|---:|---:|---:|
| BF16 M64 N5376 K5376 | 0.026 ms | 0.16 ms | 6.2x |
| FP16 VAE GEMM only, M64 N6144 K2048 | 0.023 ms | 0.076 ms | 3.30x |
| FP16 VAE narrow/prepare + GEMM, fanout one | 0.026 ms | 0.111 ms | 4.27x |
| FP32 SGEMM M64 N2048 K2048 | 0.045 ms | 0.254 ms | 5.6x |

The stable component medians were 0.003 ms CUDA narrowing and 0.035 ms Vulkan
preparation for M64/K2048. Plans, prepared slots, descriptors and command
resources are persistent and bounded; tests cover distinct exact outputs in two
outstanding slots, third-slot backpressure, stale/discarded/superseded views,
true in-flight wrapper-drop retention and release, 32/33 operations, stable
descriptor and memory high-water, row offsets, multi-K tiles and non-tile tails.

The HLSL modules use DXC 1.9.2607 and the cooperative modules use Khronos
glslang 16.5.0. Float-control variants are produced by the repository helper:

```text
dxc -spirv -T cs_6_6 -E main -fspv-target-env=vulkan1.3 -fvk-use-dx-layout src/vulkan/tensor_gemm.hlsl -Fo tensor_gemm.raw.spv
python tools/add_spirv_float_controls.py tensor_gemm.raw.spv src/vulkan/tensor_gemm.comp.spv src/vulkan/tensor_gemm_denorm.comp.spv
dxc -spirv -T cs_6_6 -E main -fspv-target-env=vulkan1.3 -fvk-use-dx-layout src/vulkan/tensor_gemm_prepare.hlsl -Fo src/vulkan/tensor_gemm_prepare.comp.spv
glslang -V --target-env vulkan1.3 -S comp src/vulkan/tensor_gemm_coop.comp -o tensor_gemm_coop.raw.spv
python tools/add_spirv_float_controls.py tensor_gemm_coop.raw.spv src/vulkan/tensor_gemm_coop.comp.spv src/vulkan/tensor_gemm_coop_denorm.comp.spv
glslang -V --target-env vulkan1.3 -S comp src/vulkan/tensor_gemm_coop_f16.comp -o tensor_gemm_coop_f16.raw.spv
python tools/add_spirv_float_controls.py tensor_gemm_coop_f16.raw.spv src/vulkan/tensor_gemm_coop_f16.comp.spv src/vulkan/tensor_gemm_coop_f16_denorm.comp.spv

# CUDA 13.0.48, MSVC 14.44.35207; reproducible SM120a code artifact
nvcc --fatbin -std=c++17 --generate-code=arch=compute_120a,code=[compute_120a,sm_120a] -Iinclude src/cuda/deterministic_gemm.cu -o deterministic_gemm.fatbin
```

```text
tensor_gemm.hlsl                         86E608B81BA51DD0F827AA53A07D91CC8B2C0542C764DEC3D16C937AFCDB2F31
tensor_gemm.comp.spv                     F91563D2DE7553C84059A416812E2A4241C3F500206D57FA1655AF0E3D3061B1
tensor_gemm_denorm.comp.spv              D606A8E6EE510B8D32DE4341AF977D478121E0101CB8274C23640FEABE1C83F9
tensor_gemm_prepare.hlsl                 C8914CF9427FD5A904B7C503DE1F5F1B63E2BB9EC37DCA46AA6823B96ACC8063
tensor_gemm_prepare.comp.spv             D716981E9734554C9E18E9587959FA70770E768DF94F1AA88478192B4727AA59
tensor_gemm_coop.comp                    76589B0446B567F2FDAD655CC337EE96CDC2E22F909994815891F84A3E4C5691
tensor_gemm_coop.comp.spv                B9719969312C4962006CD0877DBDADB2AAB8E4C8232BE84D593F4267F7597CF9
tensor_gemm_coop_denorm.comp.spv         B5B80AC23262AAB3285AAC2F763910D56D6AAED256DA920274FD6D7778AC4EA9
tensor_gemm_coop_f16.comp                11B75EEFEFAEF0EFF8266BAE7D5E4CE0D1B6EBAA43C29F2D0CB9DA5B12E6203F
tensor_gemm_coop_f16.comp.spv            E76C1A114F177A4EAA48D69A1DF7C102C06E1109EA4512C04B4335E99FEA537D
tensor_gemm_coop_f16_denorm.comp.spv     9B51BE611B7F6477F6F5D4C82DABA6E1F900F29A06C9811F987126BDA9D23C9A
src/cuda/deterministic_gemm.cu           8504C1A85F511A49B16753BA476500F2A986CC564A036D10F7E0693A532AF7E4
include/vidfab/cuda/deterministic_gemm.cuh 3C422641C753992CFCC136BFB64327DD85841BC6B7047D1F2091E88686AB6670
deterministic_gemm.fatbin                E417A406011985FF0D8F518DE0532D7367FA148C2113A1435E6B1ED795DA4A2C
```

## Streamed NVFP4 execution

`StreamedNVFP4WeightCache` is the honest executable bridge between the g1
checkpoint representation and g2 GEMM. It owns one fixed-capacity BF16 device
slot, records the exact NVFP4 materialization once, and returns a nonreusable
batch/generation view consumed by any number of row chunks or fanout GEMMs
before the next weight overwrites the slot. Tensor access transitions emit the
required write-to-read and read-to-write barriers even across two queued
submissions; there is no queue/device idle. Descriptor sets are cached by the
bounded `(bind position, compatible storage-binding count)` pair, so mixed
materialize/GEMM sequences allocate only during warmup. The 32-operation batch
limit allows one materialization plus 31 chunk/fanout GEMMs and splits only at
a weight boundary.

This is deliberately **not** called native NVFP4 MMA. The measured RTX 5090,
610.88 driver exposes `VK_KHR_cooperative_matrix` revision 2 and
`VK_NV_cooperative_matrix2` revision 1, but exposes neither
`VK_EXT_shader_ocp_microscaling_types` (`shaderFloat4`/E2M1) nor
`VK_EXT_cooperative_matrix_maintenance1`; none of its legacy cooperative
tuples has an E2M1 operand. Decoding FP4 into a BF16 cooperative matrix would
still be BF16 execution. `native_nvfp4_gemm_available()` therefore returns
false and `require_native_nvfp4_gemm()` fails with an explicit diagnostic.
CUDA parity uses the same checkpoint NVFP4-to-BF16 boundary followed by the
deterministic g2 GEMM, not CUDA's opt-in block-scaled instruction.

The streamed raw-input path accepts the transform-free H3 transformer weights,
including their immutable `full_precision_matrix_mult` value. It rejects AWQ
pre-scale or ConvRot weights before recording; those require a future typed
prepared-activation view bound to the weight identity, rather than an unsafe
caller boolean. CUDA likewise keeps its native transformer path opt-in behind
`VIDFAB_NATIVE_NVFP4=1`; default generation does not require native FP4 MMA.

The real `blocks.0.attn.qkv_proj` 384x5376 slab, including high-even nibbles,
128x4 scale swizzle, global-scale multiply order, BF16 boundary, BF16 bias,
64-row cooperative chunk, two-row scalar tail, output offsets and sentinels,
matches CUDA byte-for-byte. Release total measurements were 0.188 ms CUDA and
0.546 ms Vulkan for materialization plus the 66x384x5376 GEMMs. The compressed
slab is 1.11 MiB and its single dense cache is 3.94 MiB. At the largest shipped
H3 matrix (28672x5376), the same one-slot policy is 294 MiB logical rather than
retaining dense copies for all model weights. Two-flight overwrite, true
wrapper drop, 32/33 operations and 50 sequential prepares keep allocator and
descriptor high-water stable and release every non-context allocation after
the exact completion token.

Additional pinned sources used by this composed path:

```text
src/cuda/nvfp4_gemm.cu                  1266CF8EB2F18E1E30A54C1472BB89DAB2EDA873C0F305FB4BA9B6339E3F1B82
src/cuda/linear.cu                      B92E7D44D184ECC4C7DA466855FEEFF30063FB4591770A44D43B55CD3A23137D
src/vulkan/tensor_weight.comp           55FADB68982ABCDB0079C8D9B0A84E77F64D268E63E7227F2A2BC770ED60307F
```

## Exact unmasked blocked attention

`BlockedAttentionPlan` is the first bounded attention slice: unmasked,
token-major BF16 Q/K/V with production head widths 64, 72, or 128. A persistent
`PreparedAttentionInputs` slot converts Q/K/V to three FP16 tensors in one
device pass. Its nonreusable batch/generation view may feed several query-row
ranges in the same command buffer, so K/V are not converted again when output
rows are chunked. Submitted jobs retain all resources through the exact
timeline token; no queue/device idle or per-dispatch Vulkan object allocation
is used.

CUDA and Vulkan share an explicit reference algorithm rather than claiming
equivalence to cuBLAS: ascending-D score FMA, FP16 score round, fixed 128-lane
max/sum trees, deterministic non-positive exp, FP16 probability round,
ascending-key PV FMA with online correction, integer-RNE final division, and
BF16 output round. The accepted scale values are the serialized fp32
`1/sqrt(D)` bits `3e000000`, `3df15bef`, and `3db504f3`. The exact input domain
requires every prepared FP16 Q/K/V value and every scaled score to be finite;
NaN/Inf input arithmetic and NaN payload identity are outside it. PV
accumulator subnormals after correction or FMA are a stated rebaseline: both
backends flush them to signed zero immediately. A two-tile score-jump fixture
constructs this path and asserts an exact `+0` result.

Preparation is independently exhaustive over all 65,536 BF16 bit patterns on
both backends. It uses RNE, preserves signed zero/Inf, and maps every NaN to
FP16 `0x7fff`. Exact final tests cover S1/D64, S17/D72, S129/D128, output-row
offsets, non-128 tails, one preparation feeding two row chunks, conversion
edges, failure recovery, stale/discarded/superseded views, 32/33 operations,
two outstanding submissions plus oldest-slot reuse, true in-flight wrapper
drop, and stable allocator/descriptor high-water. Alternating plans with
3/17/5/33 rows and then the reverse reuse the pool without growth.

The real-domain audit used `qwen3vl_32b_int8_convrot.safetensors` (SHA-256
`BC2CED0FBEA64757FA9ACDDCCFC0B3F4819D1DCF1DA6C124D690D368BE283923`)
and `reference2.png` (decoded as 1024x1024, SHA-256
`B759B58BADD00E3D4C897EF7D1749CE97A52284886181CE622E136F8E385F565`),
seed 1. Generation applies the Ref2VA reference-size contract before Qwen:
the square is resized to 2048x2048, then 16x16 patchification gives a 128x128
grid, so the first post-RoPE Qwen vision block had S16384/H16/D72 and
18,874,368 values in each Q/K/V stream. All had zero BF16 subnormals, zero
nonfinite values and zero nonfinite FP16 conversions. Minimum nonzero values
were Q=2^-22, K=2^-24 and V=2^-21; maxima were 7.8125, 7.59375 and 3.265625.
The conservative `D*max(|Q|)*max(|K|)/sqrt(D)` score bound was 503.399.

Release device-resident measurements on the pinned RTX 5090/610.88 tuple use
one preparation and two query-row dispatches; uploads/downloads are excluded.
S257 uses two warmups and five samples. S16384 uses two full real-shape
samples; each run downloads the final CUDA and Vulkan tensors and requires
byte equality:

| D72 shape | CUDA exact prepare | CUDA exact attention | Vulkan total | three-FP16 slot |
|---|---:|---:|---:|---:|
| S257, H16 | 0.008 ms | 0.328 ms | 0.988 ms | 1.69 MiB |
| S16384, H16 | 0.200 ms | 854.549 ms | 1174.155 ms | 108.00 MiB |

The table compares the shared exact path. A separate device-resident run of
the shipped CUDA cuBLAS `kBlocked` implementation on the same S16384 shape was
145.276 ms/call, or 3.922 s across Qwen vision's 27 blocks. In that same run,
CUDA exact was 854.549 ms (+0.200 ms preparation) and Vulkan exact was
1174.155 ms/call: the Vulkan exact path is 8.08x the current production CUDA
blocked call and accounts for about 31.7 s/reference image before the rest of
the vision tower. This is an explicitly accepted exact-mode performance
exception, not a claim of production-speed parity. Its scalar recurrence is
already parallelized over independent keys and head dimensions; a material
speedup requires a separately reviewed CUDA-WMMA/Vulkan-cooperative-matrix
semantic rebaseline and new exact goldens.

The maximum processor-admitted single-image S65536 shape would require a
432 MiB slot; it is per active invocation, not multiplied by the 27 vision
blocks. Quadratic projection from S16384 puts the Vulkan exact attention at
about 18.8 s/call, or 8.45 minutes for 27 blocks, versus about 62.8 s for the
production CUDA blocked path. This primitive is not yet wired into
Qwen/VAE/DiT. Causal GQA and exact H3 attention are separate plans described
below; Sage2 and SOL remain unsupported Vulkan modes and fail closed rather
than route to this unmasked implementation.

The modules use Khronos glslang 16.5.0. Blocked attention receives the
repository preserve-only transform; the integer-defined converter does not.
CUDA provenance is CUDA 13.0.48, MSVC 14.44.35207 and SM120a:

```text
glslang -V --target-env vulkan1.2 -S comp src/vulkan/tensor_attention_blocked.comp -o attention.raw.spv
python tools/add_spirv_float_controls.py --preserve-only attention.raw.spv src/vulkan/tensor_attention_blocked.comp.spv
glslang -V --target-env vulkan1.2 -S comp src/vulkan/tensor_attention_prepare.comp -o src/vulkan/tensor_attention_prepare.comp.spv
nvcc --fatbin -std=c++17 -ccbin <MSVC-14.44> --generate-code=arch=compute_120a,code=[compute_120a,sm_120a] -Iinclude src/cuda/deterministic_attention.cu -o deterministic_attention.fatbin
```

```text
tensor_attention_blocked.comp             C27A8133AD290D086E1AA0C03418DD87B5F3EDA924E69F0959AC3D9562F840B7
tensor_attention_blocked.comp.spv         9C1339B2FD44B9F453BD3974F720E635682130CE9F808411823C3657A97098F2
tensor_attention_prepare.comp             786295C4E33EEDC7F67317B9ECF6B1BDEA0108B319AE5B5E8D57B6B3CEA4677D
tensor_attention_prepare.comp.spv         56DC48503F296776CD1C105D0DC44D34E5F8768B35A8FEE6EC73EFAA9D3FF8F4
src/cuda/deterministic_attention.cu       143547BB7C6A421C59BDD95B695EE662C6D1B0CE58B63A16E9164226A2C6E82E
include/vidfab/cuda/deterministic_attention.cuh 61F8CFA242C7A581B2DC7FD1405993EB9CD3716DE090D6D44D6559B28BFC6BFF
deterministic_attention.fatbin            D8C01855993DA931125F2BA06D7683C79C04F0DAC23ECCBA985E5E72A01EC194
```

## Exact H3 full and frame-band attention

`H3AttentionPlan` consumes direct token-major BF16 Q/K/V and produces BF16 for
D64 or D128. Full attention visits every key. Banded attention uses an
immutable 4,720-byte table with at most two canonical 64-aligned ranges per
global 128-query tile. Touching/overlapping ranges merge, padded endpoints are
masked before arithmetic, and both ranges form one continuous ascending
64-key recurrence. Globally aligned 64-row workgroups mask writes outside the
requested row chunk, so record splitting and output offsets do not change bits.

The backend-neutral control spelling is `AttentionMode::kExact` / `--attention
exact`. CUDA transformer main blocks and the token refiner dispatch this
primitive directly; `kNone` remains the separate blocked reference and no
other mode is remapped. Vulkan accepts only exact as an attention choice, but
the complete Vulkan transformer/denoiser orchestrator is still absent, so the
CLI reports that missing graph after validating the choice and runs neither a
Vulkan attention pipeline nor a CUDA fallback.

CUDA and Vulkan use the same 1024-thread/32-subgroup cooperative contract:
guarded BF16 Q/K staging, ascending 16-channel BF16-QK/F32 cooperative tiles,
FP16-rounded probabilities and V, FP16-PV/F32 cooperative tiles, eight
contiguous eight-key softmax partials combined in order, deterministic exp and
division, and BF16 RNE output. Cooperative-matrix internal order is the pinned
WMMA/KHR tuple, not a scalar ascending-channel claim. There is no score tensor,
host boundary, per-dispatch allocation, operator scratch, scalar fallback, or
CUDA fallback. Direct Q/K/V/out are the persistent footprint.

The public exact domain requires finite BF16-to-FP16 V conversions, scaled
scores, positive finite-normal denominators, cooperative/scalar FMA results,
PV accumulators, and final numerators. Both backends canonicalize BF16-
subnormal inputs, scaled QK results, online-correction products, cooperative PV
tile outputs, corrected accumulator sums, and BF16-subnormal outputs to signed
zero. Tests cover D64/D128 adversarial signed-zero/exponent/cancellation and
near-rounding operands, single-key/uniform/two- and three-block correction
anchors, S129/S257 tails, offsets 1/15/16/63/64/65/127/128/129, untouched
write sentinels, full/wide identity, invalid recovery, 32/33 ops, moved/drop
retention, two-flight reuse, and 50 alternating full/banded high-water runs.
The real S37727/prefix431/37x1008/radius9 table pins FNV64
`32b19bc0895faa6a` with independent accepted-key probes.

The gate is deliberately empirical: RTX 5090 device 2b85, NVIDIA 610.88/raw
driver `98960000`, pinned driver UUID, subgroup32, required BF16/F16 cooperative
tuples and arithmetic/storage modes, local size1024, checked modules, pipeline
creation, and exact goldens. CUDA qualifies reusable RTX5090/SM120a model and
resource properties, CUDA 13.x driver API plus 13.0 runtime, and the pinned
compiled artifact; it deliberately does not pin a board-unique physical UUID.
CUDA requires 1024 threads and 99,328 physical shared bytes. Ptxas reports 64
registers/thread and zero spills. GLSL/SPIR-V declares
99,328 logical bytes across phase-disjoint arrays while Vulkan reports a
49,152-byte core limit; successful creation/execution implies pinned NVIDIA
lifetime lowering/overlay, not portable 99,328-byte Vulkan physical usage.

Device-resident Release timings on the pinned tuple (no upload/download) are:

| S37727,H56,D128 | CUDA exact | Vulkan exact | shipped CUDA fused |
|---|---:|---:|---:|
| full | 1666.832 ms | 2525.180 ms | 227.233 ms |
| radius9 band | 782.777 ms | 1260.531 ms | 110.525 ms |

Exact CUDA/Vulkan outputs and wide-band/full outputs are byte-identical. The
four direct tensors occupy 2,063.20 MiB and exact attention adds zero scratch.
At 50 blocks and 25 steps, the explicitly accepted fastest-native exception is
about 26.3 minutes of Vulkan band attention (52.6 minutes full), versus about
2.30 minutes shipped CUDA band attention. This is not production-speed parity.

A shipped NVFP4 layer0/step0 capture (seed12345, 384x384 reference, 22 frames)
was S9864/H56/D128/prefix8856. Capture SHA-256 is
`56C4E55931B83DCECB0596DCB51EB3C7EF5555722910ECE78D06AE87CA055A01`;
QKV FNV64 is `fc4780b4477f6eed`. Q/K/V maxima were 12.25/13.25/73.5 with zero
subnormals and nonfinites. Exact output FNV64 `a2fbdde25a6d3787` matched CUDA
and Vulkan byte-for-byte (117.464/171.984 ms). Shipped fused took 16.369 ms;
the intentional rebaseline changed 6,368/70,705,152 BF16 words, relative L2
`2.547692e-5`, max absolute delta `0.0625`.

The modules use Khronos glslang 16.5.0, Vulkan 1.3, and the repository normal
fp32-control transform (signed-zero/Inf/NaN plus RTE, no DenormPreserve). CUDA
uses CUDA 13.0.48, MSVC 14.44.35207 and SM120a:

```text
glslang -V --target-env vulkan1.3 -S comp src/vulkan/tensor_attention_h3.comp -o h3-full.raw.spv
glslang -V --target-env vulkan1.3 -S comp -DH3_BANDED=1 src/vulkan/tensor_attention_h3.comp -o h3-band.raw.spv
python tools/add_spirv_float_controls.py h3-full.raw.spv src/vulkan/tensor_attention_h3.comp.spv h3-full.denorm.spv
python tools/add_spirv_float_controls.py h3-band.raw.spv src/vulkan/tensor_attention_h3_banded.comp.spv h3-band.denorm.spv
nvcc --fatbin -std=c++17 -ccbin <MSVC-14.44> --generate-code=arch=compute_120a,code=[compute_120a,sm_120a] -Iinclude src/cuda/deterministic_attention.cu -o deterministic_attention.fatbin
```

```text
tensor_attention_h3.comp                  B9E51135B436DFF97CE6463F4731965E7466530DC1B1E69D3FD7444B3CB6BA2B
tensor_attention_h3.comp.spv              4FD87FBDBE7AE6EC6C40C67AAD6FD39A0B9EF05F6E3CC1A5ED688A5B17FBB37F
tensor_attention_h3_banded.comp.spv       9564933F40B33B8C6077CB073F6E0CB78AD40627CD0C973FF16897CAB1764EFB
src/cuda/deterministic_attention.cu       143547BB7C6A421C59BDD95B695EE662C6D1B0CE58B63A16E9164226A2C6E82E
include/vidfab/cuda/deterministic_attention.cuh 61F8CFA242C7A581B2DC7FD1405993EB9CD3716DE090D6D44D6559B28BFC6BFF
deterministic_attention.fatbin            D8C01855993DA931125F2BA06D7683C79C04F0DAC23ECCBA985E5E72A01EC194
```

## Exact causal GQA text attention

`CausalGQAAttentionPlan` is the bounded Qwen3-VL decoder contract, not a
generic mask mode: BF16 Q is `[L,64,128]`, BF16 K/V are `[L,8,128]`, query head
`h` maps to KV head `h/8`, and global row `r` reads only keys `0..r` even when
the output is recorded in row chunks. It dispatches one 128-lane workgroup per
query row/head and keeps only its online-softmax state, so there is no `L x L`
score tensor, auxiliary device workspace, host boundary, or per-call Vulkan
allocation. The four direct tensors are the complete device-resident
footprint.

CUDA and Vulkan share ascending-D nonfused score multiply/add, a fixed
128-lane max/sum tree, deterministic exp, BF16 probability rounding,
ascending-key PV multiply/add, integer-RNE division, and BF16 output rounding.
BF16-subnormal inputs and fp32-subnormal products/accumulators are deliberately
canonicalized to signed zero on both backends; later IEEE zero arithmetic may
combine signs. The exact value domain requires finite Q/K/V, finite scaled
scores, and finite PV accumulators/final numerators. The conservative caller
check `causal_keys * max(abs(V)) <= max_finite_fp32` is sufficient; recording
cannot scan device values. Tests cover the signed-subnormal boundary,
S1/127/128/129/257, the 128-key recurrence tail, all 64 heads/eight KV groups,
global row chunks, independent row-zero/future-sentinel causality, invalid
type/alias recovery, 32/33 operation poisoning, two in-flight submissions plus
oldest-slot reuse, submitted-wrapper retention, and stable allocator/descriptor
high-water.

The real-domain audit used
`qwen3vl_32b_int8_convrot.safetensors` (SHA-256
`BC2CED0FBEA64757FA9ACDDCCFC0B3F4819D1DCF1DA6C124D690D368BE283923`)
with one synthetic 256x256 image through the real vision tower and decoder
layer 0. Its post-insertion sequence was exactly L132: label rows 0-1,
vision-start row 2, 64 merged image-pad rows 3-66, vision-end row 67, and prompt
rows 68-131. The causal boundary therefore covers both the pad run and the
vision-end-to-prompt transition. Q had 1,081,344 values (min nonzero
3.56230885e-8, max 18.75), K/V each had 135,168 values (K min
3.96743417e-7/max 122.5; V min 4.76837158e-7/max 0.3359375); every stream had
zero subnormals and zero nonfinite values. The conservative scaled-score bound
was 25,986.1742 and the PV bound `L*max(abs(V))` was 44.34375, both finite.

This exact recurrence intentionally rebaselines the shipped cuBLAS attention.
On that real L132 activation, 2,618 of 1,081,344 BF16 output words changed;
relative L2 was 1.38034211e-4 and maximum absolute delta was 0.0009765625.
The shipped path measured 59.585 ms cold and 0.152 ms warm; exact CUDA measured
0.650 ms in the same audit. A separate device-resident constant-input benchmark
(one warmup, one measured launch; uploads/downloads excluded) gave:

| Causal shape | CUDA exact | Vulkan exact | direct Q/K/V/out | pool used / reserved | descriptors |
|---|---:|---:|---:|---:|---:|
| L132,H64/KV8,D128 | 0.209 ms | 0.321 ms | 4.64 MiB | 8.77 / 16.00 MiB | 1 |
| L8192,H64/KV8,D128 | 674.870 ms | 674.861 ms | 288.00 MiB | 544.00 / 544.00 MiB | 1 |

The L8192 reserved figure is the 288 MiB live tensor pool plus benchmark-grown
persistent upload and readback staging at the largest 128 MiB Q/output boundary
each (544 MiB total). Those staging buffers exist only because the benchmark
crosses a host boundary; a production device-only chain does not need them.
The attention operator itself has zero scratch. Fifty max-length decoder layers
would spend about 33.7 seconds in exact Vulkan attention. Sage2/SOL and the
shipped cuBLAS route are not silently selected by this plan.

The causal shader was built with Khronos glslang 16.5.0 and then the repository
normal fp32-control transform (signed-zero/Inf/NaN preservation plus RTE, with
no `DenormPreserve` execution mode). Subnormals are explicitly canonicalized,
so the capability gate does not claim or require denormal preservation. The
CUDA artifact uses CUDA 13.0.48,
MSVC 14.44.35207, and SM120a:

```text
glslang -V --target-env vulkan1.2 -S comp src/vulkan/tensor_attention_causal_gqa.comp -o causal.raw.spv
python tools/add_spirv_float_controls.py causal.raw.spv src/vulkan/tensor_attention_causal_gqa.comp.spv causal.denorm.spv
nvcc --fatbin -std=c++17 -ccbin <MSVC-14.44> --generate-code=arch=compute_120a,code=[compute_120a,sm_120a] -Iinclude src/cuda/deterministic_attention.cu -o deterministic_attention.fatbin
```

```text
tensor_attention_causal_gqa.comp          DD700E2FDC18ED483973B2E161AEA3F1F43E8F2DB18FC8796F800BE766A79930
tensor_attention_causal_gqa.comp.spv      9F8B4480179C01CC26A3E467D1F0915D606594388DAB8B5C0856770B2E7E3778
src/cuda/deterministic_attention.cu       143547BB7C6A421C59BDD95B695EE662C6D1B0CE58B63A16E9164226A2C6E82E
include/vidfab/cuda/deterministic_attention.cuh 61F8CFA242C7A581B2DC7FD1405993EB9CD3716DE090D6D44D6559B28BFC6BFF
deterministic_attention.fatbin            D8C01855993DA931125F2BA06D7683C79C04F0DAC23ECCBA985E5E72A01EC194
```
