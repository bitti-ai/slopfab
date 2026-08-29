# Vulkan inference gap and parity status

## What exists

Vulkan now owns both complete neural decoders used by the synthetic-latent
vertical slice: the 36-block video VAE and the 779-tensor audio VAE. Generation
selects them through `RunOptions::inference_backend`, independently of
`--output-accelerator`. `--inference-backend vulkan` is accepted only together
with `--synthetic-latents --attention exact`; every other attention mode is
rejected by name and conditioning/denoising remains fail-closed. No rejected or
accepted Vulkan request is remapped to CUDA.

The output converter is byte-exact against the canonical CPU conversion on the
tested RTX 5090. Its checked shader uses explicit operation order and SPIR-V
`NoContraction`, including adversarial luma/chroma half-step cases, packed tail
words, padded output strides, and multi-frame Y4M output.

This is exact decoder/generation-output parity, not full prompt-to-video parity.
The complete 50-main-block H3 transformer stack now has a device-resident
Vulkan graph and exact production-path CUDA replay at every block boundary,
but the two-block refiner, final layer and denoise loop have no Vulkan
orchestration. The text/vision
conditioners and reference-image encoder also remain CUDA-only.

## Measured implementation gap

The current CUDA neural implementation comprises:

| Area | Files | Lines |
|---|---:|---:|
| CUDA `.cu` runtime and kernels (`src/cuda`) | 19 | 8,393 |
| CUDA-facing headers (`include/vidfab/cuda`) | 16 | 1,794 |
| Host neural orchestration | 6 | 4,166 |
| Public neural-stage APIs | 5 | 978 |

The host count covers `src/vae/decode_pipeline.cpp`, `keyframe_cuda.cpp`,
`audio_decoder.cpp`, `src/dit/transformer.cpp`, `denoise.cpp`, and
`src/text/encoder.cpp`. These figures are physical source lines, recorded at
commit `1e1c300`; they quantify code to audit and de-entangle, not a prediction
that a Vulkan port needs the same line count.

Missing work by pipeline stage:

| Stage | CUDA implementation that has no Vulkan peer | Principal missing operations |
|---|---|---|
| Shared tensor/weights | `linear.cu` (1,080), `nf4_weight.cu` (73), `nvfp4_gemm.cu` (556), `nn_kernels.cu` (952), workspace/device code | native quantized GEMM, NN/batched attention GEMM, remaining activations, and residual/broadcast operations; tensor lifetime, conversion/layout, add/bias, normalization, GroupNorm+SiLU, used RoPE variants, dense NT GEMM, persistent seven-format weight preparation, AWQ pre-scale and ConvRot now have Vulkan primitives |
| Video VAE decode | Implemented by `vulkan::VideoVaeDecoder` | Exact 36-block graph and shared backend-neutral tile/stitch schedule are complete; shipped tensor-core mode remains CUDA-only |
| Audio VAE decode | Implemented by `vulkan::AudioDecoder` | All 779 tensors and 497 production operators are device-resident and exact; diagnostics add 13 in-batch boundary copies |
| Transformer and denoise | `dit_kernels.cu` (139), `transformer.cpp` (2,027), `denoise.cpp` (192), attention family (`attention.cu`, Sage and SOL: 2,282 lines) | the exact 50-block main stack is implemented with real every-boundary replay and a block-cache span seam; refiner/final layer, denoise/scheduler integration and non-exact attention modes remain CUDA-only |
| Qwen text/vision conditioner | `encoder_kernels.cu` (1,080), `encoder.cpp` (595), `qwen_vision*.cu` (332), keyframe CUDA path (547) | token embedding, causal decoder attention/MLP, vision patch/merge graph, deep-stack scatter, reference-image VAE encode; NeoX/mRoPE and exact unmasked D72 attention primitives exist but are not wired |

Checkpoint handling also remains CUDA-entangled. A Vulkan backend must preserve
the existing safetensors tensor names and metadata while supporting the shipped
fp16/bf16, fp8, int8 ConvRot, bitsandbytes NF4, and NVFP4/AWQ layouts. CUDA
handles and `cudaStream_t` appear in the present tensor/workspace and stage
interfaces, so backend selection cannot be added safely by switching only the
top-level `run_generate` call.

## Dependency-ordered implementation plan

1. Define backend-neutral device tensor, workspace, weight-cache, command, and
   stage interfaces. Implement Vulkan fp32/fp16/bf16 conversion, elementwise,
   normalization, RoPE, dense GEMM, and the shipped quantized GEMM formats.
   Add exact activation-dump comparisons for every primitive.
2. Port video and audio VAE graphs on those primitives. These form the first
   useful neural vertical slice because `--synthetic-latents` bypasses the
   conditioner and transformer. Require exact decoded fp32 RGB and PCM dumps,
   then exact Y4M/WAV output.
3. Extend the implemented exact 50-block main graph into the token refiner,
   final layer and denoise loop. Preserve its every-block boundary gate and
   wire the existing cache span through `record_layers`; port non-exact attention
   backends separately rather than silently substituting exact attention.
4. Port Qwen text/vision conditioning and reference-image encoding, retaining
   tokenizer and checkpoint behavior. Compare embeddings, deep-stack outputs,
   and packed conditioning rows.
5. Enable `--inference-backend vulkan`, retain the current fail-closed
   capability check, and run deterministic full-pipeline CUDA/Vulkan exact
   comparisons at every durable boundary: conditioner embeddings, denoiser
   latents, decoded fp32 RGB, decoded fp32 PCM, Y4M, and WAV. `compare-y4m` is
   the streaming raw-video check, not a substitute for the earlier activation
   and sample comparisons. Only this step can establish full pipeline parity.

Each increment depends on the one above. In particular, adding a Vulkan CLI
label before neural stage implementations would be a silent CUDA fallback, not
Vulkan support.

### Implemented primitive increment

The first dependency slice now has a backend-neutral contiguous tensor/view
contract and a persistent Vulkan batch path. The Vulkan implementation has
exact copies, fp32 add/add-bias over zero/normal/infinity operands and results,
fp32-to/from-fp16 and bf16 conversion, fp32
2-D transpose, trusted-index row gather/scatter, head-major fp32 to token-major
bf16, fp32 depth-to-space, and the fp32 RMSNorm/affine LayerNorm pair used by
the video VAE. It records bounded operations into one command buffer, retains
tensors through exact timeline completion, and reuses two bounded
descriptor/command slots.

FP32 arithmetic does not promise a NaN payload. Subnormal inputs/results are
available only when the queried float-control mode supports them; the public
`require_full_fp32_arithmetic_exactness()` gate fails closed otherwise.
Normalization is CUDA-bit-exact over its zero and finite-normal domain on the
separately gated RTX 5090/driver 610.88 tuple with shaderInt64 explicitly
enabled. CUDA and Vulkan use the same integer-RNE division/epsilon addition and
Q30 reciprocal-square-root algorithm, avoiding the observed one-ULP divergence
between CUDA `rsqrtf` and SPIR-V `InverseSqrt`. It covers the shipped 2048-wide
VAE norms, BF16 widths 128/1152/4608/5120/5376, affine LayerNorm, and BF16/fp32
RMSNorm+AdaLN without a host boundary. NaN tensor arithmetic is excluded,
subnormal epsilon and dimensions above 2^24 are rejected, and unknown or
shaderInt64-disabled devices fail before recording.

The same bounded batch now includes the keyframe encoder's exact fp32 CHW,
fp16-affine GroupNorm+SiLU primitive. CUDA and Vulkan share deterministic
integer division/epsilon/reciprocal-square-root and a fixed polynomial SiLU;
the latter deliberately canonicalizes NaNs and subnormal inputs/results and
maps values at or below -87 to signed zero. The production 32-group,
256-thread reduction tree is preserved exactly. This primitive is not yet
wired into a Vulkan keyframe encoder graph.

It also implements all three used rotary semantics: BF16 H3 partial-96 with a
raw 32-channel tail, BF16 full-width GPT-NeoX for Qwen text/vision, and the
video-VAE fused fp32 split-QKV, head64 RMSNorm and partial-48 rotation with
suffix bypass. The host canonical H3 builder supplies identical serialized
fp32 table bits to CUDA and Vulkan. Conditioner and DiT orchestration still
call CUDA; both VAE decoders now consume the Vulkan primitives directly.

Persistent linear-weight preparation now covers F32/F16/BF16, E4M3 FP8,
per-output I8, NVFP4 and NF4 without retaining dense copies of every quantized
matrix. Caller-owned BF16/FP16 prepared storage is reusable over chunks; one
batched upload retains immutable compressed bytes and all auxiliary metadata.
The device batch also provides BF16/fp32 AWQ pre-scale and regular-H4 ConvRot.
Real NVFP4 and video-VAE NF4 tensors match CUDA materialization byte-for-byte.
Dense NT execution now covers BF16/BF16 to BF16 with its separate bias round,
fp32 activations prepared once to FP16 for FP16-weight VAE linears, and FP32
SGEMM. CUDA and Vulkan use pinned matching WMMA/cooperative tiles or the same
ascending-K scalar edge order. Native NVFP4/NF4 execution, NN and batched GEMM
remain separate work because equivalent formulas cannot be assumed
bit-identical to cuBLAS or native quantized execution.

NVFP4 weights can now execute through one bounded streamed BF16 slot: g1
materializes the active checkpoint matrix and g2 reuses it across all row
chunks/fanout before the next weight overwrites it. CUDA exact comparison uses
the same dequantization boundary. This is not native FP4 MMA. The pinned
RTX 5090/610.88 Vulkan driver lacks the E2M1/microscaling extension and tuple,
so native NVFP4 requests fail closed. AWQ pre-scale/ConvRot weights also remain
outside this raw-input seam until a typed transformed-activation view binds the
transformation provenance to the weight.

Exact unmasked blocked attention now exists as a bounded device primitive for
BF16 D64/D72/D128. It prepares Q/K/V once into a persistent three-FP16 slot,
supports multiple query-row consumers in one batch, and mirrors a pinned CUDA
reference byte-for-byte. A real Qwen vision S16384/H16/D72 activation audit
proved finite prepared values and scaled scores; its slot is 108 MiB. This is
not orchestration: causal GQA, H3 banding/fusion, Sage2/SOL, and all attention
call-site wiring remain missing and may not silently route to the unmasked
primitive. The exact Vulkan kernel is also an accepted performance exception:
1.174 s/call and about 31.7 s for 27 Qwen vision blocks at S16384, versus
145.276 ms/call and 3.922 s/27 for the shipped CUDA cuBLAS blocked path.
S65536 projects to roughly 8.45 minutes/27 on Vulkan. A cooperative-matrix
semantic rebaseline remains a future optimization, not an implicit fallback.

Exact Qwen causal GQA now exists as a second, distinct primitive for
BF16 `[L,64,128]` Q and `[L,8,128]` K/V through L8192. It uses no quadratic
workspace and preserves global causality across recorded row chunks. A real
post-vision-insertion L132 checkpoint audit found zero subnormal/nonfinite
Q/K/V values, finite score/PV bounds, and measured the intentional exact-mode
rebaseline against shipped cuBLAS at relative L2 1.38034e-4 (max absolute
0.0009765625). Exact Vulkan measured 0.321 ms at L132 and 674.861 ms at L8192;
the latter has 288 MiB of direct tensors and no attention scratch. This is
still a primitive: text-encoder orchestration remains CUDA-owned.

Exact H3 full and frame-banded attention is part of the complete 50-main-block
Vulkan graph: typed projection loading, rank-8 AdaLN, normalization, RoPE,
attention, residuals and SwiGLU remain device-resident in one caller batch.
The production CUDA exact loop supplies a real S526 capture whose input,
metadata, 50 block-boundary digests and final residual all match Vulkan
exactly. A CUDA-disabled test independently loads and executes two shipped
NVFP4 layers. This is still not denoise orchestration: the refiner, final layer
and scheduler remain unavailable, so non-synthetic Vulkan generation
continues to fail before execution.

These operations correspond to launchers in `linear.cu`, `vae_kernels.cu`, and
`nn_kernels.cu`. Current CUDA uses include transformer checkpoint widening and
projection narrowing, video-VAE channel/token layout, attention head packing,
patch reconstruction, and packed-sequence row selection. Packed row indices
are generated as unique in-range host sequences by `packing.cpp` and
`ref2va.cpp`; a future Vulkan stage must pass those generated tensors rather
than arbitrary device data. The Vulkan shader also bounds-checks each index to
prevent an invalid device read or write.

This is a tested operator substrate with both VAE call sites wired. Native
quantized matrix execution and conditioner/DiT orchestration remain on the
missing list above. A Vulkan request that needs those stages fails before
weights or output files; the exact synthetic-latent decoder slice proceeds.

The full 36-block exact video-VAE transformer stack is now available through a
device-resident Vulkan graph. It streams all real checkpoint blocks through a
shared typed loader, retains fp16 matrices only on device, and records 720
operations into one caller-owned batch with one shared activation arena. CUDA
and Vulkan matched every block boundary and the final 3,680,256-word token
tensor exactly; the pinned final FNV64 is `50d92f167ac90922`. Vulkan allocator
used/reserved high-water was 5,120.5/5,123.1 MiB and remained stable on repeat.

The existing CUDA `ViTDecoder` can explicitly select the shared exact graph
implementation for its transformer body, including two bounded reusable
scratch shapes for ragged tiled windows; the shipped path remains the default.

The video-VAE decoder around that graph is now implemented as
`vulkan::VideoVaeDecoder`. It has native Vulkan latent de-normalization,
post-quant projection, CHW/token embedding, register/zero suffix assembly,
canonical RoPE, final affine norm/projection, and depth-to-space. CUDA and
Vulkan implement the same `VideoVaeWindowBackend` interface and consume one
shared host temporal-chunk/spatial-tile/stitch/cross-fade/pixel-de-normalize
schedule, which is the backend-selection seam intended for `RunOptions`.
That schedule is compiled into `vidfab_core` and has no CUDA profiler or link
dependency; a CUDA-disabled Vulkan build links and runs the decoder contract
test. There are no CUDA calls or resources in the Vulkan decoder. Construction
requires an explicit `ViTTransformerMode::kExact` configuration; `kShipped` is
rejected before allocation because Vulkan has no shipped-mode implementation.

One Vulkan window records `15 + 20*num_layers` operators (735 for the shipped
36-layer checkpoint). A configuration whose single document exceeds 4096
operators is rejected before context allocation; tests cover 1 layer (35), the
204-layer boundary (4095), and 205-layer rejection. Equal-shape groups split into at
most five documents per 4096-operator transaction, so a large tiled group
cannot overflow the command batch. Checkpoint matrices stay fp16 and
device-resident; the graph and decoder retain at most two shape-keyed scratch
sets. Peak accounting includes all three prepared-fp16 activation buffers in
each decoder shape; the real replay checks one- and two-shape accounted deltas
against Vulkan allocator-used deltas. The real-checkpoint test explicitly runs
six equal 7x1x1 windows as 5+1,
then switches through the full and ragged shapes without reloading weights.

Real-checkpoint provenance is
`weights/vae/minimax_h3_video_vae_fp16.safetensors`, SHA-256
`7C1F131492E7EDDACAAC9069A61B81BDD39DE5CC96561E677C5EAB1CDCE5E522`.
On an RTX 5090 the exact 7x16x16 final window
matched CUDA bit for bit at FNV64 `4d84e832e07db0a8`. A complete normalized
7x1x1 latent decode matched the CUDA final fp32 `PixelBuffer` bit for bit at
FNV64 `f455f77e718d9c21` (22 frames, 16x16), and Y4M written through the Vulkan
converter matched the canonical writer byte for byte. Measured Vulkan load and
7x16x16 forward times were 5.0 s and 2.4 s; persistent/peak device accounting
was 4625.0/5087.0 MiB, pooled used/reserved was 5215.0/5445.0 MiB, and the final
descriptor high-water was 3676.

This completes the exact video-VAE decoder component. Together with the audio
decoder it enables top-level synthetic-latent Vulkan generation, while the
text/vision conditioners and denoiser remain gated out.

## Current vertical-slice comparison

The opt-in `cuda_vulkan_exact_generate_vertical_slice` test writes one
deterministic fp32 init-latent archive, then invokes real `run_generate` at the
minimum 32x32/22-frame geometry through CUDA exact and Vulkan exact. It asserts
the video checkpoint SHA-256
`7C1F131492E7EDDACAAC9069A61B81BDD39DE5CC96561E677C5EAB1CDCE5E522`
and audio checkpoint SHA-256
`8E505D95DD1561D47ABD43D4238FD40D9BB1AE9E147ED0A4CBA778D76AE4DB48`.

The latent FNV64 is `5529904CB8C9DC9E`. Final pinned FNV64 digests are
`E2CA5273E36E9ED7` for 67,584 PixelBuffer floats, `5933499108CE9C79` for
59,200 interleaved PCM floats, `D55DBD1D534B8787` for Y4M bytes, and
`6B066C7CF430117D` for PCM16 WAV bytes. CUDA and Vulkan match bit for bit at
the float boundaries and byte for byte at both containers.

The audio graph test also compares dec-in projection, pre-convolution, all
seven post-stage averages, final activation, final convolution, clamp and
interleave. Production remains one 497-operator batch; diagnostic replay adds
13 device copies inside one 510-operator batch. Production `A=405` is pinned at
FNV64 `0B9084D3F1C6355A`.

The CUDA-off build proves decoder-library purity only. It executes the real
Vulkan audio decoder without a CUDA target, pins `A=3` at FNV64
`528F17A83D5EF7EE`, repeats without pool/descriptor growth, and unloads. The
top-level `run_generate` still lives in `vidfab_cuda`, so a CUDA-disabled CLI
does not yet expose this slice; this is not a decoder fallback.
