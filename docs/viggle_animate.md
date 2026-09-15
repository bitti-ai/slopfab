# Viggle-Animate transformer checkpoints

The CUDA and Vulkan transformer loaders accept
`Viggle-Animate-pruned_rank8_int8_convrot.safetensors` directly. Select it
explicitly with `--transformer` or `SLOPFAB_MODEL_TRANSFORMER` in the C API.
Automatic H3 checkpoint discovery continues to select the general-purpose
FL2VA/Ref2VA models.

```sh
slopfab inspect weights/transformer/Viggle-Animate-pruned_rank8_int8_convrot.safetensors
```

The model is identified from `source=Viggle/Viggle-Animate` metadata, so
renaming the file preserves its identity. Without source metadata, the
`Viggle-Animate` filename marker identifies the family. Preserve the
`qkv_layout` metadata: it determines how attention weights are loaded.

## Checkpoint layout

The inspected checkpoint contains 50 main blocks and two token-refiner
blocks, with the usual H3 dimensions (5376 hidden channels, 56 heads of
128 channels, 5120 text features). Its AdaLN curve is F32 `[1025,8]`, and
the rank-eight AdaLN projections and biases remain F32.

`qkv_layout=interleaved` stores attention rows as
`[head, Q/K/V, channel]`. Both backends reorder these into `[Q; K; V]` when
loading, including the per-row F32 INT8 scales. The BF16 token-refiner
weights receive the same permutation. This preserves the quantized values
and ConvRot input transformation; it does not requantize the model.
Checkpoints without layout metadata retain the existing contiguous layout.
Unsupported layout names and interleaved packed NF4/NVFP4 weights are rejected.

## Animation workflow requirements

This change supports the transformer checkpoint. It does not add a dedicated
Viggle animation command or reproduce the upstream character-replacement
pipeline automatically. The generic H3 reference-generation defaults differ
from Viggle's evaluated recipe.

The [upstream inference script](https://huggingface.co/Viggle/Viggle-Animate/blob/main/inference/sample.py)
uses a frozen embedding with modality tags, a driving video followed by one
repainted image, and references resized to the target's short edge. It omits
Qwen and drops the driving video's reference soundtrack. Its distilled
configuration uses four sigma boundaries (three forward passes), Euler
sampling, and video flow shift 3.

The local checkpoint explicitly declares `distillation_lora_merged=false`.
Its rank-eight AdaLN compression is separate from the distillation adapter.
The [upstream model card](https://huggingface.co/Viggle/Viggle-Animate) requires
that adapter on the Viggle finetune for the three-pass recipe. The upstream
Diffusers adapter also needs conversion to slopfab's supported projection
names/layout before use; passing `--steps 4` alone does not produce the
distilled model. The upstream frozen `.pt` embedding, reference ordering and
geometry, and optional pinned target audio require separate pipeline work.

## Verification

Host tests cover metadata/filename detection, Ref2VA acceptance, invalid
layouts, and an INT8 projection with distinct per-row scales. The actual
downloaded checkpoint is checked across all 52 blocks, including Vulkan's
host-side archive validation. CUDA and Vulkan regression fixtures compare
forward results from equivalent contiguous and interleaved checkpoints;
the Vulkan fixture also checks F32 AdaLN storage.

The Release CLI, DLL, CUDA tests and Vulkan tests compile. In the implementation
environment, CUDA reports no visible device and Vulkan instance creation
fails with `VK_ERROR_INCOMPATIBLE_DRIVER`, so GPU execution and end-to-end
animation quality have not been verified.
