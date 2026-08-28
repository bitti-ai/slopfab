# Vulkan inference gap and parity status

## What exists

Vulkan currently accelerates only the final planar fp32 RGB to BT.709
limited-range YUV420 conversion. Generation, conditioning, denoising, and both
neural decoders still execute through CUDA. `--output-accelerator vulkan` names
that narrow output stage. `--inference-backend vulkan` is rejected before any
prompt file, checkpoint, or output is opened; it never routes the request to
CUDA under a Vulkan name.

The output converter is byte-exact against the canonical CPU conversion on the
tested RTX 5090. Its checked shader uses explicit operation order and SPIR-V
`NoContraction`, including adversarial luma/chroma half-step cases, packed tail
words, padded output strides, and multi-frame Y4M output.

This is not full Vulkan/CUDA pipeline parity. There is no Vulkan implementation
of a neural stage to compare yet.

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
| Shared tensor/weights | `linear.cu` (1,080), `nf4_weight.cu` (73), `nvfp4_gemm.cu` (556), `nn_kernels.cu` (952), workspace/device code | device tensors and views, upload/cache lifetime, fp16/bf16/fp8 conversion, dense and NF4/NVFP4 GEMM, bias/activation, RMS/group/layer norm, RoPE, residual and broadcast elementwise operations |
| Video VAE decode | `vae_kernels.cu` (503), `vit_decoder.cu` (684), `decode_pipeline.cpp` (420) | Conv3D/Conv2D, causal padding, upsample, residual blocks, spatial/temporal attention, tile scheduling and merge |
| Audio VAE decode | `audio_vae_kernels.cu` (452), `audio_decoder.cpp` (504) | weight-normalized Conv1D/transposed Conv1D, residual units, Snake activation, channel/layout transforms |
| Transformer and denoise | `dit_kernels.cu` (139), `transformer.cpp` (2,027), `denoise.cpp` (192), attention family (`attention.cu`, Sage and SOL: 2,282 lines) | multimodal projections, AdaLN, Q/K normalization, 3-D RoPE, causal/banded attention, fused residual paths, timestep conditioning, scheduler loop integration and caches |
| Qwen text/vision conditioner | `encoder_kernels.cu` (1,080), `encoder.cpp` (595), `qwen_vision*.cu` (332), keyframe CUDA path (547) | token embedding, decoder attention/MLP, mRoPE, vision patch/merge graph, deep-stack scatter, reference-image VAE encode |

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
The VAE normalization reduction tree is CUDA-bit-exact over its zero and
finite-normal domain on the separately gated, measured RTX 5090/driver 610.88
tuple. Generic NVIDIA and float-control support are not accepted as proof of
`InverseSqrt` bit identity. It covers the shipped 2048-wide VAE norms and final
affine LayerNorm without a host boundary. NaN/subnormal arithmetic is excluded,
subnormal epsilon is rejected, and unknown driver tuples fail before recording.

These operations correspond to launchers in `linear.cu`, `vae_kernels.cu`, and
`nn_kernels.cu`. Current CUDA uses include transformer checkpoint widening and
projection narrowing, video-VAE channel/token layout, attention head packing,
patch reconstruction, and packed-sequence row selection. Packed row indices
are generated as unique in-range host sequences by `packing.cpp` and
`ref2va.cpp`; a future Vulkan stage must pass those generated tensors rather
than arbitrary device data. The Vulkan shader also bounds-checks each index to
prevent an invalid device read or write.

This is a tested operator substrate, not a wired Vulkan model stage. GEMM and
quantized weights, bf16/head/modulated normalization, group norm, other
reductions, RoPE, attention, convolutions, and all four model-stage
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
