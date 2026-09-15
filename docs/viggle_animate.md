# Viggle-Animate transformer checkpoints

For CUDA denoising VRAM reductions and stage profiling, see
[Denoising memory](denoising_memory.md). Flash2 uses compact Q/output buffers
by default; projection/FFN scratch is bounded to 2,048 rows.
CUDA generation also automatically streams selected transformer blocks and
their adapters from CPU RAM when the estimated VRAM requirement exceeds the
available budget. See the linked document for budget controls and measurements.

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
that adapter on the Viggle finetune for the three-pass recipe. Slopfab now loads
`viggle_animate_distillation_bf16.safetensors` directly on CUDA and Vulkan.
Add these options to your generation command:

```sh
--transformer weights/transformer/Viggle-Animate-pruned_rank8_int8_convrot.safetensors --lora weights/loras/viggle_animate_distillation_bf16.safetensors --lora-strength 1
```

All 604 tensors are used across 302 projections. Diffusers' separate Q/K/V
adapters retain their per-projection head order, and the first feed-forward
projection's B rows are reordered from `[value; gate]` to H3's `[gate; value]`.
The `proj_in` and `proj_out` updates are merged once into the F32 video
endpoint weights at load time, without rewriting the checkpoint. The adapter's
embedded PEFT metadata supplies alpha 128 for its rank-128 factors.

Loading the adapter does not select a different schedule; passing `--steps 4`
alone does not reproduce the full recipe. The upstream frozen `.pt` embedding, reference ordering and
geometry, and optional pinned target audio require separate pipeline work.

## Verification

Host tests cover metadata/filename detection, Ref2VA acceptance, invalid
layouts, and an INT8 projection with distinct per-row scales. The actual
downloaded checkpoint is checked across all 52 blocks, including Vulkan's
host-side archive validation. CUDA and Vulkan regression fixtures compare
forward results from equivalent contiguous and interleaved checkpoints;
the Vulkan fixture also checks F32 AdaLN storage.

The LoRA host suite loads the actual 604-tensor adapter against the downloaded
checkpoint, validates all 302 targets, and merges both video endpoints.
Synthetic tests cover alpha/strength scaling, Q/K/V dimensions, SwiGLU order,
stacking, and malformed inputs. GPU regression cases check that each of the
eight projection types affects the forward pass on both backends.

The Release CLI, DLL, CUDA tests and Vulkan tests compile. In the implementation
environment, CUDA reports no visible device and Vulkan instance creation
fails with `VK_ERROR_INCOMPATIBLE_DRIVER`, so GPU execution and end-to-end
animation quality have not been verified.
