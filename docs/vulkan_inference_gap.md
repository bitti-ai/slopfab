# Vulkan inference gap and parity status

## What exists

Vulkan currently wires only the final planar fp32 RGB to BT.709 limited-range
YUV420 conversion into generation. A tested Vulkan neural-primitive substrate
now exists, including tensor/layout operations, normalization, RoPE, and
every required dense NT GEMM mode plus persistent preparation of every shipped
dense/quantized linear-weight format,
but generation, conditioning, denoising, and both neural decoders still execute
through CUDA. `--output-accelerator vulkan` names that narrow output stage.
`--inference-backend vulkan` validates attention before any prompt file,
checkpoint, or output is opened. Vulkan accepts only `--attention exact`;
`none`, `flash2`, `sage2`, `sol`, and `sol-experimental` are rejected by name
and are never remapped. An accepted exact selection then fails separately
because the full Vulkan neural orchestrator is still missing. It never routes
the request to CUDA under a Vulkan name.

The output converter is byte-exact against the canonical CPU conversion on the
tested RTX 5090. Its checked shader uses explicit operation order and SPIR-V
`NoContraction`, including adversarial luma/chroma half-step cases, packed tail
words, padded output strides, and multi-frame Y4M output.

This is not full Vulkan/CUDA pipeline parity. No complete Vulkan neural stage is
wired yet.

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
| Video VAE decode | `vae_kernels.cu` (503), `vit_decoder.cu` (684), `decode_pipeline.cpp` (420) | Conv3D/Conv2D, causal padding, upsample, residual blocks, spatial/temporal attention, tile scheduling and merge |
| Audio VAE decode | `audio_vae_kernels.cu` (452), `audio_decoder.cpp` (504) | weight-normalized Conv1D/transposed Conv1D, residual units, Snake activation, channel/layout transforms |
| Transformer and denoise | `dit_kernels.cu` (139), `transformer.cpp` (2,027), `denoise.cpp` (192), attention family (`attention.cu`, Sage and SOL: 2,282 lines) | multimodal projections, causal/banded/fused attention, residual paths, timestep conditioning, scheduler loop integration and caches; AdaLN, Q/K RMSNorm, H3 RoPE and exact unmasked blocked attention primitives exist but are not wired |
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
3. Port one transformer block, attention backends, timestep/AdaLN paths, and
   the denoise loop. Compare every block boundary before enabling the 50-block
   graph or its cache modes.
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
the video VAE. It records up to 32 operations into one command buffer, retains
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
fp32 table bits to CUDA and Vulkan. These are device primitives; conditioner,
DiT and VAE orchestration still calls CUDA.

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

Exact H3 full and frame-banded attention also exists as a bounded cooperative
primitive for BF16 D64/D128. Its typed range contract traverses two canonical
key ranges per global query tile, uses no quadratic score buffer, and is wired
into CUDA transformer main blocks and the token refiner by the explicit
`AttentionMode::kExact`. This completes the arithmetic/control seam, not the
Vulkan transformer: the Vulkan H3 plan is tested independently, while the
projection, normalization, residual, scheduling, and model-lifetime graph
around it remains unwired. Consequently `--inference-backend vulkan
--attention exact` reports the missing orchestrator instead of claiming a
generation route.

These operations correspond to launchers in `linear.cu`, `vae_kernels.cu`, and
`nn_kernels.cu`. Current CUDA uses include transformer checkpoint widening and
projection narrowing, video-VAE channel/token layout, attention head packing,
patch reconstruction, and packed-sequence row selection. Packed row indices
are generated as unique in-range host sequences by `packing.cpp` and
`ref2va.cpp`; a future Vulkan stage must pass those generated tensors rather
than arbitrary device data. The Vulkan shader also bounds-checks each index to
prevent an invalid device read or write.

This is a tested operator substrate, not a wired Vulkan model stage. Native
quantized matrix execution, NN/batched GEMM, remaining activations, causal/fused attention,
convolutions, primitive call-site wiring, and all four model-stage
orchestrators remain on the missing list above. Therefore
`--inference-backend vulkan` continues to fail before weights or output files.

## Current vertical-slice comparison

The intended control holds all neural work constant and changes only output
conversion:

```text
vidfab generate --synthetic-latents --seed 424242 --frames 6 \
  --resolution 32x32 --raw --vae <real-video-vae> \
  --dump-latents parity-latents.safetensors \
  --output-accelerator cpu --out parity-fixed-cpu.mp4
vidfab generate --synthetic-latents --seed 424242 --frames 6 \
  --resolution 32x32 --raw --vae <real-video-vae> \
  --init-latents parity-latents.safetensors \
  --output-accelerator vulkan --out parity-fixed-vulkan.mp4
vidfab compare-y4m parity-fixed-cpu.y4m parity-fixed-vulkan.y4m
```

Measured on 2026-08-28 from build commit
`c66ee3349c87d39b2a013b5fd71d96316c1726b2`, configured Release with CUDA 13.0,
`sm_120a`, `VIDFAB_ENABLE_CUDA=ON`, `VIDFAB_ENABLE_VULKAN=ON`, and
`VIDFAB_WITH_FFMPEG=OFF`, using Vulkan 1.4.341 and an RTX 5090. The real
checkpoints and inputs were:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `weights/vae/video_vae_nf4.safetensors` | 1,613,201,536 | `6D0CB4FF02EBB74CC6BCA40018E6EFAE5082CCD7EB066FA1263098C6DBF8F6F1` |
| `weights/vae/audio_vae_nf4.safetensors` | 284,004,112 | `759662130BA3618B7F196DA8F983F857A1F1EC6AF8110D657796D9792C0D64E5` |
| `parity-latents.safetensors` | 12,312 | `5FE9F6E048465ADE3695950531727FE850C0E9B23F871AEED7E698956C07FD83` |

The video VAE occupied 1.17 GiB on device. The requested six frames align to
22 model frames; the latent grid is 7x2x2. The second run read the exact dumped
fp32 video/audio rows instead of redrawing them.

`compare-y4m` returned 0: both files were 33,965 bytes with header
`YUV4MPEG2 W32 H32 F24:1 Ip A1:1 C420jpeg` and SHA-256
`A38ADCDACC22DED7CA58810CBCA4E19B723E963AB8D8B619CAECA621484C01CC`.
The independently written WAV files also matched at
`71F3A8AA31560D6206BDE640769AC568D00FB834A667027B46070457546C48FE`.

This result proves exact parity between CPU and Vulkan output conversion after
the same real CUDA VAE/audio pipeline. It does not compare CUDA neural
inference with Vulkan neural inference: every neural stage was CUDA on both
sides, because the Vulkan implementations enumerated above do not exist.
