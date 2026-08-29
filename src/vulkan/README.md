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
