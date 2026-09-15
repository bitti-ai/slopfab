# H3 LoRA adapters

`generate` accepts local safetensors adapters on CUDA and Vulkan. Each
`--lora-strength` applies to the preceding `--lora`; the default is 1.0.
Repeat the pair to combine adapters. Zero disables an adapter and negative
strengths subtract its update.

```powershell
slopfab generate --prompt "A cat playing piano" --lora weights/loras/style.safetensors --lora-strength 0.7
```

Adapters update H3's `attn.qkv_proj`, `attn.out_proj`, `mlp.fc1`, and `mlp.fc2`
in main transformer blocks and text-refiner blocks. Supported key suffixes are
`lora_A.weight` / `lora_B.weight`, PEFT's `lora_A.default.weight` /
`lora_B.default.weight`, and `lora_down.weight` / `lora_up.weight`. Names can
start with `diffusion_model.`, `model.diffusion_model.`, `base_model.model.`,
or the projection name directly. Each projection may provide an `alpha`
scalar; absent alpha uses embedded `lora_adapter_metadata.lora_alpha`, or the
rank when metadata is absent. External PEFT configuration files
are not read, so adapters relying on an external non-default alpha must first
embed those scalars in their safetensors file.

Viggle's Diffusers-format adapter is also supported directly: `proj_in`,
`proj_out`, and the six attention/MLP projections under `transformer_blocks.N`.
Separate Q/K/V factors are applied to their respective base projection slices.
Diffusers SwiGLU B rows are reordered from `[value; gate]` to `[gate; value]`.
The video input/output updates are merged into the floating-point weights at
load time on both backends. See [Viggle usage and workflow limits](viggle_animate.md).

The update is `strength * alpha / rank * B @ A`. For block projections, the implementation folds
scaling into B and uploads both factors as BF16; intermediate projections and
the residual sum round to BF16. Multiple adapters are combined by concatenating
their low-rank factors. FP8, INT8+ConvRot, native NVFP4 and the loader's supported
NF4 base weights retain their original storage. LoRA activations use the
original input before the base weight's ConvRot transform. AWQ activation
scales, other target modules, DoRA, RSLoRA, per-target alpha patterns, LyCORIS, convolutional adapters, malformed
pairs and incompatible dimensions are rejected explicitly.

The base checkpoint is never rewritten. LoRAs are uploaded once per model
load and released with the transformer. CUDA uses scratch for at most 256
rows; Vulkan shares full-sequence scratch between projections and layers.
For TaoMate, adapter device storage is about 1.29 GiB (Q/K/V share the base
projection but currently upload separate copies of A). Vulkan adds roughly
2.95 GiB of LoRA activation scratch at the default 37,727-row geometry; smaller
canvases use proportionally less. Zero-strength adapters allocate no GPU data.

## TaoMate H3, three evaluations

Download the [ComfyUI-format adapter](https://huggingface.co/CZMartin22/TaoMate-H3-3step-ComfyUI)
from the supplied URL:

```powershell
New-Item -ItemType Directory -Force weights/loras
curl.exe -L --fail "https://huggingface.co/CZMartin22/TaoMate-H3-3step-ComfyUI/resolve/main/TaoMate-H3-3step-ComfyUI.safetensors" -o weights/loras/TaoMate-H3-3step-ComfyUI.safetensors
```

Use FL2VA base weights and select the schedule explicitly:

```powershell
slopfab generate --prompt "A person smiles and says hello" --transformer weights/transformer/MiniMax_H3_FL2VA_pruned_nvfp4.safetensors --lora weights/loras/TaoMate-H3-3step-ComfyUI.safetensors --schedule taomate-3step --out output/taomate.mp4
```

Add `--inference-backend vulkan` to use Vulkan. Ordinary model discovery still
supplies the conditioner and VAEs. Strength 1 uses the released rank-128,
alpha-128 adapter. `--schedule taomate-3step` selects states `[0,16,33,49]`
from the 50-point teacher schedule, with video shift 12 and audio shift 3,
and performs exactly three Euler evaluations. Omit `--steps`: slopfab's
ordinary step count includes the terminal sigma, so `--steps 3` would mean
only two evaluations. Step and block caching and AB2 are rejected with this
preset. Other adapters retain the normal schedule unless explicitly changed.

The retained states follow [TaoMate's published schedule](https://github.com/TaoLiveAIGC/TaoMate-H3/blob/main/src/taomate_h3/denoise_schedule.py).
This supports the adapter and its three-interval schedule within slopfab's
existing generation pipeline. It does **not** reproduce the upstream runtime's
causal streaming chunks, clean KV cache, or separate audio-guidance preparation.
The upstream project's latency and quality measurements therefore do not apply
to this implementation; see [the TaoMate runtime](https://github.com/TaoLiveAIGC/TaoMate-H3).

## C and C++ APIs

C ABI 1.7 adds:

```c
slopfab_request_add_lora(request, "TaoMate-H3-3step-ComfyUI.safetensors", 1.0f);
slopfab_request_set_schedule(request, SLOPFAB_SCHEDULE_TAOMATE_3STEP);
/* To restore ordinary generation: */
slopfab_request_clear_loras(request);
slopfab_request_set_schedule(request, SLOPFAB_SCHEDULE_DEFAULT);
```

C++ callers set `GenerateRequest::loras` and `GenerateRequest::schedule`.
`resolve_plan` exposes the actual three model evaluations, and both generation
backends consume the resolved sigma grids directly. File validation happens
before conditioning or transformer upload. A dry run checks request arithmetic
but does not load or validate adapter tensor contents.

## Verification

The `lora` test filter covers adapter naming, alpha/rank scaling, negative and
zero strengths, composition, malformed archives, the released adapter's 208
projection shapes, custom-grid validation, and the three-step plan. GPU tests
compare split Q/K/V updates with an independent CPU contraction, including a
real TaoMate projection, partial tiles and non-tensor-core ranks. A reduced
real-checkpoint test runs both text refiners and one main block through all
three Euler steps on CUDA and Vulkan, checks that enabling the adapter changes
the result, and verifies repeatability. Cross-backend boundaries are compared
on the supported exact-attention tuple; Flash2 differences are reported
separately because it has no bit-identity contract. This is an integration check,
not a full 50-block generated-video quality assessment.
