# VEDA attention

VEDA predicts which 128-token video tiles to attend to using a separate learned
predictor. Slopfab implements the [Veda-on-ComfyUI](https://github.com/veda-sparse/Veda-on-ComfyUI)
bundle, TripPool, residual projection, tile-plan selection, and sparsity-budget
rules in native C++. Python, PyTorch and Triton are not runtime dependencies.
The compatibility reference is upstream revision
[`67e4eb5`](https://github.com/veda-sparse/Veda-on-ComfyUI/tree/67e4eb5541b648555563144401ebaba832bd73ec).

Both CUDA and Vulkan use BF16 Q/K with FP32 score accumulation and FP16
probabilities/values in the sparse attention kernel. This differs from upstream's
Triton INT8 backend; output is approximate and is not expected to match it bit
for bit. No performance or perceptual-quality equivalence is claimed.

## Usage

Download a predictor from the upstream [T2VA / FL2VA release](https://huggingface.co/Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview)
or [R2VA release](https://huggingface.co/Veda-Sparse/Minimax-H3-R2VA-Veda-Preview)
and place it under `weights/veda/`. Slopfab never downloads predictors during generation.
The predictor must have the same layer count, head count and head dimension as
the transformer. FP32, BF16 and per-head scaled FP8 bundles are supported.

```sh
slopfab generate --prompt "A cat walking through a garden" \
  --attention veda \
  --veda-predictor weights/veda/minimax_h3_t2va_veda_8nfe_600step_preview_fp8.safetensors \
  --frames 124 --resolution 1344x768 --out video.mp4
```

Add `--inference-backend vulkan` for Vulkan. CUDA requires SM80 or newer;
Vulkan requires the same cooperative BF16/FP16 matrix, subgroup32 and shared
memory capabilities as Flash2. Unsupported hardware produces an error.

`--veda-sparsity trained` is the default and reads the generated-video budget
from the predictor. `--veda-sparsity 90%` skips approximately 90% of ideal
video tile pairs; `--veda-sparsity 32` keeps 32 key tiles per video query tile.
`0%` retains all tiles. Padding affects the percentage budget, fractional
budgets are spread across query tiles, and each tile always retains itself.
Text and audio attention stays dense in both directions.

Reference image/video/audio tokens also retain dense attention in both
directions. Sparse reference routing and a separate reference budget are not
implemented. Use a predictor trained for the corresponding task.

The per-layer/head tile shapes come from the bundle. Other video geometries
select the nearest aspect ratio and duration, considering transposed plans,
then the least padding. Verbose output identifies when a nearest trained plan
is used. A plan may have at most 4096 tiles including global tiles; larger
requests fail explicitly. Predictors do not change the sampling schedule or
automatically install a Turbo LoRA. Use a schedule and adapters appropriate
for the predictor; upstream's T2VA preview was trained with eight Turbo steps.

VEDA cannot be combined with frame-banded attention or FastH3 V2's compressed
VSA attention. Missing, malformed, or mismatched predictors fail explicitly.
The text refiner always uses dense Flash2.

## Library APIs

With C API 1.29:

```c
slopfab_request_set_attention(request, "veda");
slopfab_request_set_veda(request, "predictor.safetensors", "trained");
/* Select CUDA or Vulkan with slopfab_request_set_inference_backend. */
```

Check return codes as for other request setters. Strings are copied. Passing
an empty predictor path clears the VEDA settings; select another attention
mode before generating. A null budget means `trained`.

C++ callers set `RunOptions::attention_mode = AttentionMode::kVeda` and
`RunOptions::veda.predictor_path`. `VedaOptions` also supports explicit dense
layer and denoising-step lists through `full_attention_layers` and
`full_attention_steps` (zero-based). Low-level CUDA callers prepare a
`dit::VedaRequest` and pass it to `Transformer::set_veda` before loading, so
automatic offloading includes the predictor's device scratch in its budget.

CUDA streams one predictor layer at a time and caches converted BF16 weights
on the host. Vulkan keeps BF16 predictor weights alongside each resident
transformer block. Both bound routing scratch to one head at a time and reuse
the geometry maps and temporary buffers.

## Verification

`SLOPFAB_TEST_FILTER=veda` selects the host contract tests in `slopfab_tests`,
the CUDA numerical tests in `slopfab_kernel_tests`, and the Vulkan numerical
tests in `slopfab_vulkan_tests`. GPU tests compare learned routing and sparse
attention against an independent FP64 reference, including ragged tiles,
different head/layer plans, dense global rows, no global rows, fractional
budgets and the all-tiles budget.

Real-checkpoint smoke runs also completed on an RTX 5090 under Windows with
the released T2VA FP8 predictor and the NVFP4 FL2VA transformer, on both CUDA
and Vulkan: 128x128, 22 frames, one denoiser evaluation, video/audio decoding
and latent export. These small runs validate integration, not production
speed or video quality.
