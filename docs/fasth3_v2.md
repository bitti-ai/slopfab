# FastH3 V2

Slopfab supports the pruned FastH3 8-Step V2 transformer on CUDA and Vulkan, including INT8 ConvRot weights. It loads the learned `to_gate_compress` projections and uses VSA-H3 attention: 4×4×4 video tiles, 80% video-tile sparsity, dense text/audio attention, and the gated pooled-attention branch.

The checkpoint selects the trained eight-evaluation Euler schedule automatically. Video/audio shifts are 10/3, with noise-clock rungs 999, 874, 749, 624, 500, 375, 250, and 125 followed by zero. A different `--steps` value does not override this trained schedule.

```powershell
build/Release/slopfab.exe generate --prompt "A steam train crosses a mountain bridge" --transformer "D:/Projects/weights/baf29933173ff891-fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors" --resolution 256x256 --frames 22 --out train.mp4
```

Use your existing MiniMax H3 text encoder, video VAE, and audio VAE. The CLI's model discovery supplies them when they are in the usual model folders; applications using the C API supply their paths as usual.

Add `--inference-backend vulkan` to run on Vulkan, or select the Vulkan inference backend in the C API. VSA-H3 is selected automatically from the checkpoint. Vulkan requires BF16 and FP16 cooperative-matrix support and 32-lane subgroups, as used by its Flash attention backend. Pooling, tile selection, sparse attention and gate fusion all stay on the GPU. Quantized transformer weights remain resident; automatic transformer offloading is available on CUDA.

Checkpoint identification requires the gate tensors plus `fasth3_8step_v2` in the filename (hyphens and case are also accepted), or `source`/`model_id` metadata identifying `FastVideo/FastVideo-FastH3-8-Step-V2`. V1 shares the gate layout but has a different schedule, so it is not inferred to be V2 from the gates alone.

This model is for text-to-video/audio. Reference conditioning, continuation, frame-banded attention, and step/block caching are not supported with it. The text refiner keeps dense attention. Geometry is limited to 4,096 tiles.

Reference: [FastVideo model card](https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2) and [published inference contract](https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2/blob/main/fastvideo_inference.json).

## FastH3 OmniRef compatibility

The `fasth3_omniref_8step_pruned_int8_convrot.safetensors` release is not
supported by the FastH3 V2 runner. Its metadata declares `pdd_steps: 32`, and
its final video/audio heads have 3072/1024 output rows instead of 96/32.
These are Parallel Decoding Distillation (PDD) heads, requiring output decoding
and sampling that the current runners do not implement. OmniRef also needs
reference-aware VSA layouts and its own inference recipe.

Version 1.32.1 detects the widened heads during plan validation and reports
`FastH3 OmniRef/PDD checkpoint (32 output heads) is not supported`, before
loading the transformer. Renaming the file to FastH3 V2 or supplying legacy
model metadata does not make the tensor layout compatible. For current
reference conditioning and image edits, use a supported MiniMax H3 Ref2VA
checkpoint.

See the [official OmniRef PDD example](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_fasth3_omniref_pdd.py)
for its separate model and sampling contract.
