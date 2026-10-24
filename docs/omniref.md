# FastH3 OmniRef

Slopfab 1.33.0 supports the pruned Comfy-layout FastH3 OmniRef PDD32 checkpoint
on CUDA and Vulkan, including `fasth3_omniref_8step_pruned_int8_convrot.safetensors`.
Applications select it with the existing `slopfab_request_set_model_path` API
and use the usual H3 text encoder and video/audio VAEs. No new C API calls are
required. Reference images, videos and audio use the existing reference APIs.

The loader recognizes the 3072-row video and 1024-row audio output projections
alongside the rank-eight AdaLN table and compressed-attention gates. This
format stores head zero in full and heads 1–31 as offsets from head zero.
The loader fuses each group of four heads into one projection using the
modality's fine-grid integration weights. Unconverted FastVideo directory
checkpoints and legacy `slopfab.model` descriptors are not accepted.

The trained recipe is selected automatically: eight Euler evaluations,
PDD boundaries 0/4/8/12/16/20/24/28/32, video/audio shifts 12/3 and maximum
noise level 0.999. `steps` does not override this grid. Image-edit strength
selects a suffix of the grid. Custom sigma grids, independent audio steps,
other samplers and step/block/MotionCache reuse are rejected.

VSA uses 4×4×8 patch tiles (128 tokens), retaining 10% of tiles independently
within each reference-video region and the target-video region. Text,
reference images/audio and target audio remain dense; learned pooled-attention
gates are applied. References stay fixed while target rows follow the flow.
The text refiner uses dense attention. Frame-banded attention and output-head
LoRA adapters are unsupported. The total limit remains 4,096 tiles.

Vulkan requires the same BF16/FP16 cooperative matrices and 32-lane subgroups
as the FastH3 V2 backend. CUDA also supports automatic transformer offloading.

Contract sources: [official OmniRef inference recipe](https://huggingface.co/FastVideo/FastVideo-FastH3-Omni-8-Step-V1/blob/main/fastvideo_inference.json)
and [ComfyUI head representation](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/minimax/model.py).
