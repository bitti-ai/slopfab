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
   Add activation-dump comparisons for every primitive.
2. Port video and audio VAE graphs on those primitives. These form the first
   useful neural vertical slice because `--synthetic-latents` bypasses the
   conditioner and transformer. Require exact decoded fp32 dumps where the
   arithmetic contract permits it, then exact Y4M/WAV output.
3. Port one transformer block, attention backends, timestep/AdaLN paths, and
   the denoise loop. Compare every block boundary before enabling the 50-block
   graph or its cache modes.
4. Port Qwen text/vision conditioning and reference-image encoding, retaining
   tokenizer and checkpoint behavior. Compare embeddings, deep-stack outputs,
   and packed conditioning rows.
5. Enable `--inference-backend vulkan`, retain the current fail-closed
   capability check, and run deterministic full-pipeline CUDA/Vulkan exact
   comparisons through the streaming `compare-y4m` command. Only this step can
   establish full pipeline parity.

Each increment depends on the one above. In particular, adding a Vulkan CLI
label before neural stage implementations would be a silent CUDA fallback, not
Vulkan support.

## Current vertical-slice comparison

The intended control holds all neural work constant and changes only output
conversion:

```text
vidfab generate --synthetic-latents --seed 424242 --frames 6 \
  --resolution 32x32 --raw --vae <real-video-vae> \
  --output-accelerator cpu --out <cpu.mp4>
vidfab generate --synthetic-latents --seed 424242 --frames 6 \
  --resolution 32x32 --raw --vae <real-video-vae> \
  --output-accelerator vulkan --out <vulkan.mp4>
vidfab compare-y4m <cpu.y4m> <vulkan.y4m>
```

The measured command, checkpoint, runtime, and result are recorded here once
the CUDA+Vulkan build has completed. This comparison proves the output backend
only; the neural path is CUDA on both sides by design.
