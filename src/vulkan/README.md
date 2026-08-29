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
