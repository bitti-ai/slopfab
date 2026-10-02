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
`lora_B.default.weight`, `lora_down.weight` / `lora_up.weight`, and
`lora.down.weight` / `lora.up.weight`. Names can
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

## Turbo adapters with AdaLN updates

Adapters may also update `blocks.N.adaln_proj.linear` and
`final_layer.adaln_proj.linear`. Rank-8 AdaLN factors are merged into the
floating-point pruned weights at model load on CUDA and Vulkan.

Some Turbo adapters, including the LightX2V v4 step-600 DARE-TIES file, retain
the original 2,688-dimensional AdaLN input. Pruned checkpoints have only eight
input coordinates and omit the original timestep embedder. Supply a matching
`h3_silu_temb_grid.safetensors` beside the LoRA, or use an adapter that already
contains `slopfab.silu_t_emb_grid`. An embedded grid takes precedence over the
companion. slopfab does not download grids or rewrite adapters during inference.

To embed a local companion for later use without that file, prepare the adapter
explicitly:

```sh
slopfab prepare-lora --adapter adapter.safetensors --width 2688
```

Preparation requires write access to the LoRA and enough space beside it for a
temporary full copy. slopfab validates the grid and atomically replaces the
LoRA, preserving its existing tensor bytes and metadata and adding about
5.26 MiB for the standard grid. The companion is left in place and can be
removed once every adapter that needs it has embedded its copy. Later loads
work with a read-only adapter. The base checkpoint is never changed.

The runtime fits the adapter's timestep input to the loaded base's
`adaln_t_table`, preserving the
constant offset as an AdaLN bias update. Both weight and bias updates respect
adapter strengths and stacking. Nothing is silently dropped. The fit uses F64
QR and checks the low-rank activation curve at all 1,025 grid positions; errors
above 1% are rejected. Invalid adapters are not rewritten. Disabled adapters
do not require or embed a grid.

This is an approximate conversion. The LightX2V v4 step-600 DARE-TIES adapter
loaded all 259 projections against `minimax_h3_fl2va_pruned_int8_convrot`, with
a worst relative activation fit error of 0.1629% across its 51 AdaLN targets.
That measures the fitted adapter activation, not final video quality or the
error of the full model. The loader prints this measurement. Use a grid derived
from the adapter's original timestep embedder; matching tensor dimensions alone
cannot prove it came from the correct model.

This support does not select a new sampling schedule or enable full, unpruned
AdaLN execution. To request eight evaluations on the ordinary schedule, use
`--steps 9`.

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

## DMAD H3, four evaluations

The released rank-128 DMAD adapters load directly, including the mixed
Diffusers attention/MLP keys and both token refiners. Use the local full-critic
adapter with FL2VA:

```powershell
New-Item -ItemType Directory -Force output | Out-Null
build/Release/slopfab.exe generate --prompt "A polar bear plays the violin in the snow." --text-encoder weights/text_encoder/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors --transformer weights/transformer/MiniMax_H3_FL2VA_pruned_nvfp4.safetensors --lora weights/loras/dmad_minimax_h3_4step_full_critic.safetensors --schedule dmad-4step --seed 42 --out output/dmad.mp4
```

`--schedule dmad-4step` selects four evaluations, video shift 12, audio shift 2,
and the upstream re-noising rule: `x0 = x + sigma * velocity`, then
`x_next = (1 - sigma_next) * x0 + sigma_next * fresh_noise`. The terminal update
returns x0. The base sigma grid is `[1, .75, .5, .25, 0]`. Omit `--steps`, or use
`--steps 5`; other explicit counts are rejected. This preset selects re-noising
even with the ordinary Euler default. AB2, approximate caches and conditioning
workflows are rejected. Ordinary `--lora` loading does not select the preset.

Add `--inference-backend vulkan` for Vulkan. CUDA uses the host scheduler;
Vulkan also updates re-noised rows on the host, transferring latents and
velocities each step. Both use the same deterministic noise streams, separate
for every step and modality. These streams do not reproduce PyTorch's RNG.
The default geometry is 1344x768, 124 frames at 24 fps; no CFG is needed.

For an Euler comparison, omit the preset, use `--steps 5`, and supply
`--sampling-settings` with `{"version":1,"video_sigma_shift":12,"audio_sigma_shift":2}`.
`--sampler renoise` also selects re-noising with an ordinary custom step count
and shifts. Fixed grids outside the DMAD preset retain their Euler requirement.

DMAD was trained on the T2VA transformer. FL2VA is supported here as requested,
but the upstream quality results do not establish quality for this base swap
or quantized weights. See the [DMAD release](https://huggingface.co/ZhengmingYu/DMAD)
and [reference sampler](https://github.com/Yzmblog/DMAD/blob/main/dmad_h3/sampling.py).
The adapter has no AdaLN targets and needs no timestep companion grid.

Validation used the full-critic adapter with the FL2VA NVFP4 checkpoint. A full
CUDA/Sage2 run generated 124 frames at 1344x768 plus stereo audio, with about
51 seconds for four denoising evaluations on an RTX 5090. The reduced real-model
test exercises both text refiners and one main block on CUDA and Vulkan,
including generated audio and repeatability. Flash2 cross-backend differences
are reported rather than treated as an exact arithmetic contract. A constant
velocity Vulkan fixture separately checks the re-noising formula at every
video/audio boundary. These checks do not establish upstream quality parity.

C++ uses `GenerateRequest::schedule = sampler::ScheduleKind::kDmad4Step`.
C ABI 1.16 adds `SLOPFAB_SCHEDULE_DMAD_4STEP` to `slopfab_request_set_schedule`;
the preset selects re-noising on both backends without another sampler setter.

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
projection shapes, custom-grid validation, and the three-step plan. Embedding
tests check preservation of tensor bytes and metadata, repeat loads without a
companion, unchanged files after invalid fits, and cleanup after replacement
fails. Set `SLOPFAB_TEST_LORA_PATH` and `SLOPFAB_TEST_LORA_BASE` to also check a
local Turbo adapter's complete tensor hashes and reload behavior; this test
embeds the grid on first use, just like the runtime. GPU tests
compare split Q/K/V updates with an independent CPU contraction, including a
real TaoMate projection, partial tiles and non-tensor-core ranks. A reduced
real-checkpoint test runs both text refiners and one main block through all
three Euler steps on CUDA and Vulkan, checks that enabling the adapter changes
the result, and verifies repeatability. Cross-backend boundaries are compared
on the supported exact-attention tuple; Flash2 differences are reported
separately because it has no bit-identity contract. This is an integration check,
not a full 50-block generated-video quality assessment.
