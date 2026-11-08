# H3 LoRA adapters

`generate` accepts local safetensors adapters on CUDA and Vulkan. Each
`--lora-strength` applies to the preceding `--lora`; the default is 1.0.
Repeat the pair to combine adapters. Zero disables an adapter and negative
strengths subtract its update.

[H3 Character Swap V1.5](character-swap.md) loads directly with a Ref2VA base,
one source video and one character image. Adapters declaring
`ss_base_model_version=minimax_h3_ref2va` reject bases without reference support
before loading or sampling.

```powershell
slopfab video generate --prompt "A cat playing piano" --lora weights/loras/style.safetensors --lora-strength 0.7
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
load time on both backends. See [Viggle Remake](viggle_remake.md) for the
rank-64 swap/edit adapters and merged INT8 alternatives. A merged Remake
transformer already includes its mode adapter; do not apply it again.

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
slopfab models prepare-lora --adapter adapter.safetensors --width 2688
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
slopfab video generate --prompt "A person smiles and says hello" --transformer weights/transformer/MiniMax_H3_FL2VA_pruned_nvfp4.safetensors --lora weights/loras/TaoMate-H3-3step-ComfyUI.safetensors --schedule taomate-3step --out output/taomate.mp4
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
build/Release/slopfab.exe video generate --prompt "A polar bear plays the violin in the snow." --text-encoder weights/text_encoder/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors --transformer weights/transformer/MiniMax_H3_FL2VA_pruned_nvfp4.safetensors --lora weights/loras/dmad_minimax_h3_4step_full_critic.safetensors --sampling-settings settings/dmad.json --seed 42 --out output/dmad.mp4
```

[settings/dmad.json](../settings/dmad.json) contains the complete sampling recipe:

```json
{
  "version": 1,
  "sampler": "renoise",
  "video_sigma_shift": 12,
  "audio_sigma_shift": 2,
  "base_sigmas": [1, 0.75, 0.5, 0.25, 0]
}
```

No named schedule or separate sampler flag is needed. The five sigma points
produce four evaluations; a fixed grid determines the count regardless of
`--steps`. Re-noising predicts `x0 = x + sigma * velocity`, then updates
`x_next = (1 - sigma_next) * x0 + sigma_next * fresh_noise`. The terminal
update returns x0. Approximate caches and AB2 with fixed grids are rejected.
The same JSON can be embedded as `slopfab.sampling` metadata in an adapter.
Ordinary loading of the released DMAD files does not infer these settings
from their filenames or descriptive upstream metadata.

Add `--inference-backend vulkan` for Vulkan. CUDA uses the host scheduler;
Vulkan also updates re-noised rows on the host, transferring latents and
velocities each step. Both use the same deterministic noise streams, separate
for every step and modality. These streams do not reproduce PyTorch's RNG.
The default geometry is 1344x768, 124 frames at 24 fps; no CFG is needed.

For an Euler comparison, add `--sampler euler` to the same command. An explicit
sampler flag overrides the JSON sampler regardless of command-line order.
The previous `--schedule dmad-4step` spelling remains a compatibility alias
that supplies exactly the settings above; explicit settings can override it.

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
The settings-only path was also run with the same FL2VA checkpoint, prompt and
seed; its saved video and audio latent tensors were byte-identical to the
earlier named-schedule run.

C++ callers set `GenerateRequest::sampling` using `parse_sampling_settings()`.
C callers pass the JSON to `slopfab_request_set_sampling_settings()`.
`RunOptions::sampler` is an optional override; unset inherits the resolved
settings, with Euler as the fallback when no settings choose a sampler.

## Hyperflow H3, eight evaluations

[Hyperflow](https://github.com/Video-Rebirth/hyperflow) loads through `--lora`.
Its metadata selects the trained nine-point sigma grid, video/audio shifts
12/3, and eight Euler evaluations automatically. Omit `--steps` and
`--sampling-settings`.

Hyperflow adapts **two** time embedders and blends their outputs before SiLU
and the full AdaLN projections. Pruned checkpoints have removed these weights;
an ordinary timestep companion grid cannot recover this calculation. Prepare
a separate modulation companion once (Python with torch, numpy and safetensors):

```powershell
python tools/prepare_hyperflow.py --adapter weights/loras/minimax_h3_hyperflow_8step_v1.0.safetensors --variant fl2va
```

This reads about 26 GB of original model tensor ranges from the official model
at the revision recorded in the adapter, and writes a roughly 333 MiB
`minimax_h3_hyperflow_8step_v1.0.hyperflow-fl2va.safetensors` beside the adapter.
It never changes the adapter or base checkpoint. For offline preparation,
pass `--source /path/to/original/diffusers/transformer`, containing the original
shards and index. Inference requires only the quantized base, adapter and
prepared companion, with no Python or network access.
Remote preparation retains completed projections in a neighboring `.parts`
directory so interrupted runs can resume. This cache is not needed for inference.

```powershell
build/Release/slopfab.exe video generate --prompt "A polar bear plays the violin in the snow." --transformer weights/transformer/MiniMax_H3_FL2VA_pruned_nvfp4.safetensors --lora weights/loras/minimax_h3_hyperflow_8step_v1.0.safetensors --seed 42 --out output/hyperflow.mp4
```

Add `--inference-backend vulkan --audio-vae weights/vae/minimax_h3_audio_vae_fp32.safetensors`
for Vulkan; the Vulkan audio decoder requires the floating-point checkpoint,
so select it explicitly if discovery chooses `audio_vae_nf4.safetensors`.
The FL2VA base is used for text-to-video here; the CLI's existing reference-input
restrictions still apply.
Ref2VA requires preparation with `--variant ref2va` (and its original
`transformer_ref` directory when using `--source`), then a pruned Ref2VA base.
Video conditioning rows use `(t,r)=(0.999,0.999)` and audio references use
`(1,1)`; generated rows use the current and next modality-specific times.
Pairs remain distinct even when their starting times coincide.

Companions contain the full modulation results for every trained pair, including
the two FP32 time-embedder LoRAs and the original BF16 AdaLN projection rounding.
Runtime selects these results through one-hot codes; it does not fit them to
the pruned rank-eight basis. The companion adds about 333 MiB of device weights.
Base attention and MLP weights retain their quantization, so this does not
establish output parity with the upstream unquantized pipeline.

The companion is checked against the adapter's SHA-256, revision, variant and
strength. To change `--lora-strength`, prepare again with the same `--strength`;
zero disables the adapter without requiring its companion. Only one active
Hyperflow adapter is supported. Attention/MLP style adapters may be stacked;
AdaLN projection adapters cannot. Alternate grids, shifts, samplers, independent
audio steps, approximate caches, Animate, continuation and image editing are
rejected. Missing or mismatched companions fail before model upload.

Validation covers metadata and schedule rejection, pair-aware row selection,
companion identity and strength checks, and a separate Python time-embedding
oracle. The released adapter runs through both text refiners and one main block
for all eight evaluations on CUDA and Vulkan, with repeatability and finite
audio/video checks. Flash2 backend differences are reported, not assumed to be
bit-identical. A full 50-block CUDA/Sage2 run with the FL2VA NVFP4 base generated
124 frames at 1344x768 and stereo audio. A full-stack Vulkan/Flash2 smoke run
generated 22 frames at 256x256 and stereo audio using the FP32 audio VAE.
Ref2VA has not been validated with a full generated clip.

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
