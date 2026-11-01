# Sampling settings

Models and LoRAs can supply the sampler, default step counts, video/audio sigma shifts and fixed sampling grids as data. These settings work with the existing H3 flow scheduler on CUDA and Vulkan; a different transformer graph or velocity convention still needs a model-family implementation.

For example, save this as `sampling.json`:

```json
{
  "version": 1,
  "video_sigma_shift": 6,
  "audio_sigma_shift": 3,
  "base_sigmas": [1, 0.75, 0.5, 0.25, 0]
}
```

```sh
slopfab generate --prompt "A moonlit forest" --sampling-settings sampling.json --dry-run
```

The example specifies four evaluations. It illustrates the format; choose values appropriate to the model's training recipe. `base_sigmas` contains **unshifted** sigma points, including terminal zero. Each modality applies its shift as `shift * base / (1 + (shift - 1) * base)` in float32. Do not put already-shifted sigmas here.

Only `version` is required. Omitting a field inherits it. `sampler` accepts `"euler"`, `"ab2"`, `"renoise"`, or `"er_sde"`; absent a selection, Euler is used. Shifts must be finite and positive. A grid must have at least two finite points, start in `(0,1]`, decrease strictly, and end at zero. Shifted grids must also remain valid in float32. Unsupported versions, unknown fields, incorrect types and invalid grids are errors. JSON syntax uses the repository's existing parser.

Omit `base_sigmas` to inherit the selected grid, or use the ordinary linspace if no lower layer supplies a grid. `default_steps` is an integer from 1 to 1,000,000 specifying denoising updates. Explicit `--steps` takes precedence over this default and resolves conflicting adapter step defaults. A fixed grid determines its own length and takes precedence over `--steps`. Fixed grids support Euler, re-noising or ER-SDE and disable approximate step, block and MotionCache reuse. AB2 is rejected with fixed grids. These restrictions also apply when the grid comes from metadata; FastH3 and OmniRef still require Euler.

CLI, C++ and C API now inherit the same recipe defaults: ordinarily 50 updates, or 4 for Animate. The CLI previously forced 15. `GenerateRequest::num_inference_steps = 0` means inherit; positive explicit counts are retained when toggling Animate. H3 step counts now follow ComfyUI: `--steps 4` requests four updates and creates five sigma boundaries. This also applies to `default_steps`, `audio_steps`, and the C++/C API. To reproduce an older generated H3 grid, subtract one from its former step setting (for example, old `--steps 30` becomes `--steps 29`). Explicit `base_sigmas` and named distilled grids retain their trained trajectories.

## ER-SDE and beta

```sh
slopfab generate --model krea2 --sampler er_sde --schedule beta --steps 8 --prompt "A red apple" --out apple.ppm
slopfab generate --model krea2 --inference-backend vulkan --vulkan-arithmetic portable --sampler er_sde --schedule beta --steps 8 --prompt "A red apple" --out apple-vulkan.ppm
```

`--scheduler` is an alias for `--schedule`. ER-SDE and beta are independent:
either can be selected without the other. Both support H3 on CUDA/Vulkan,
Krea 2 on CUDA/Vulkan, and MOSS/Fibo on CUDA. Music retains its own flow solver.

ER-SDE implements the three-stage extended reverse-time SDE solver with one
model evaluation per update, unit noise strength and the reference noise-scaling
function. It ramps through first, second and third order, keeps separate histories
for each modality, and uses fresh seeded noise except at the terminal denoise.
It follows [ComfyUI's flow ER-SDE implementation](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/k_diffusion/sampling.py),
with stable double-precision scalar coefficients. Native noise uses slopfab's
counter-based RNG, so seeds are repeatable but do not match PyTorch images.
Sigma 1 is offset by 0.0001 in unshifted time before model evaluation to avoid
an infinite noise/data ratio. Default Euler arithmetic is unchanged.

Vulkan uses the same host ER-SDE update as CUDA, downloading/updating/uploading
the latent rows between GPU predictions, as its existing re-noising path does.
ER-SDE rejects approximate step/block/MotionCache reuse, H3 image editing and
recipes requiring Euler. FastH3 and OmniRef retain their trained grids.

Beta follows [ComfyUI's beta quantile scheduler](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/samplers.py):
descending beta quantiles select nearest-even rounded training-grid indices;
consecutive duplicates are removed and terminal zero is appended. The training
grid has 1,000 points for H3/MOSS and 10,000 for Krea/Fibo. Each model's shift
is applied to the selected training times. Defaults are alpha=beta=0.6.
H3 and image-model `--steps` count requested updates. Quantization may reduce the realized count.

The JSON/C API equivalent, with optional shape overrides, is:

```json
{"version":1,"sampler":"er_sde","scheduler":"beta","beta_alpha":0.6,"beta_beta":0.6}
```

`scheduler` accepts `default` or `beta`; `beta_alpha` and `beta_beta` must be
finite values in `[0.01,100]`. These fields follow the normal metadata and
override precedence. Beta cannot be combined with `base_sigmas`, including a
fixed grid inherited from a model or LoRA. Independent audio counts use beta
too when selected. C callers can also select `SLOPFAB_SCHEDULE_BETA` through
`slopfab_request_set_schedule`, then set the sampler through sampling JSON.

CPU tests compare ER-SDE trajectories with upstream and beta indices with SciPy.
Regenerate those reference values with `tools/sampler_reference.py --source PATH_TO_COMFYUI`.

## Independent audio steps (experimental)

Use `--audio-steps` to give audio its own step count, for example when a LoRA
uses a short video schedule:

```sh
slopfab generate --prompt "A person playing a piano" --steps 4 --audio-steps 25 --out piano.mp4
```

Like `--steps`, this counts denoising updates: the example
runs 4 video updates and 25 audio updates (28 joint transformer calls). Audio uses the selected generated schedule with
the resolved `audio_sigma_shift`. Video retains its selected grid, including a
fixed grid from a LoRA, `base_sigmas`, or a named schedule. Without this option,
audio continues to inherit the video count and base grid exactly as before.

Both CUDA and Vulkan interleave updates by relative progress through each
modality's step count. Each transformer call sees the current video and audio
noise levels; only modalities due for an update advance. The other latents
remain unchanged. Both terminal updates happen together. With V video updates
and A audio updates, there are `V + A - gcd(V, A)` transformer calls. Counts that
are multiples of one another need only `max(V, A)` calls. `--dry-run` and C API
progress report the total calls, and the plan description also lists both
modality update counts.

Extra audio updates still run the shared transformer, so they cost generation
time. Video's update grid stays unchanged, but the additional audio states can
change its predictions and final output. More audio steps are experimental and
do not guarantee better quality, especially with distilled adapters. Compare
results with the option unset using the same seed.

`audio_steps` also works in version-1 sampling JSON and model/LoRA metadata.
It accepts integers from 2 to 1,000,000 and follows the usual field precedence
and adapter conflict rules. Explicit `--audio-steps` wins over the JSON file
regardless of argument order. A fixed `base_sigmas` grid still governs video;
`audio_steps` overrides that grid for audio. Even an explicit equal count uses
the generated audio grid, so omit the option to preserve a fixed audio recipe.

Independent counts support Euler, re-noising and ER-SDE on both backends, and AB2 on
CUDA when no fixed grid is selected. Different counts cannot use step, block,
or MotionCache reuse. `audio_steps` is rejected for still images, preserved
target audio, and FastH3 V2, which requires its trained audio grid.

In C++, set `request.sampling.audio_steps = 25`. C API 1.21 adds a setter
without changing any public struct layouts:

```c
slopfab_request_set_steps(request, 4);
slopfab_request_set_audio_steps(request, 25);
/* Clear the explicit audio override and inherit metadata/defaults again. */
slopfab_request_set_audio_steps(request, 0);
```

The setter changes the `audio_steps` field in the request's sampling overrides.
`slopfab_request_set_sampling_settings` replaces those overrides, including
this field, so call the dedicated setter after loading JSON if it should win.
Invalid setter values leave the previous setting intact.

## Model and adapter metadata

Store the same JSON object as a **string** under the SafeTensors metadata key `slopfab.sampling`:

```json
{
  "__metadata__": {
    "slopfab.sampling": "{\"version\":1,\"video_sigma_shift\":6,\"base_sigmas\":[1,0.5,0]}"
  }
}
```

This is a metadata fragment, not a complete checkpoint. Add it when producing the checkpoint or adapter. Planning only reads existing headers; it does not rewrite weights or infer new recipes from an adapter filename.

Resolution applies:

1. Existing H3 defaults and legacy model/task compatibility presets.
2. Transformer metadata.
3. Metadata from LoRAs with nonzero strength, including negative strength.
4. The existing named `--schedule` compatibility preset, if selected.
5. Explicit `--sampling-settings` fields or C++/C API overrides.
6. An explicit CLI `--sampler` overrides the sampler field, regardless of argument order.

Two active adapters may agree or supply different fields. Conflicting values for the same field are rejected unless that field has an explicit request override (including a named schedule's grid). Adapter order cannot silently select the recipe. Disabled adapters contribute no defaults. Missing model/adapter files still allow geometry-only planning with defaults; execution retains its existing file checks. Malformed metadata in an existing active file fails planning.

FastH3 V2 retains its structural and trained-recipe requirements: effective settings must match its existing nine-point grid, video/audio shifts of 10/3, and Euler sampler. Settings cannot grant reference support, introduce missing VSA gate weights, or bypass those requirements. Existing Animate conditioning and TaoMate adapter requirements also remain.

DMAD needs no named schedule: `--sampling-settings settings/dmad.json` supplies
`"sampler":"renoise"`, the five-point base grid `[1,.75,.5,.25,0]`, and shifts
12/2. The existing `--schedule dmad-4step` option only supplies those same
settings as a compatibility alias. See [DMAD usage](loras.md#dmad-h3-four-evaluations).

`--dry-run` reports the effective sampler, shifts, evaluation count, whether the grid is fixed, and settings sources. Source entries show contributing layers, rather than provenance for each individual field.

## Library interfaces

In C++, set `GenerateRequest::sampling` directly or use `parse_sampling_settings()`. The resolved `GeneratePlan` contains the sampler and both modality grids, and both runners consume those settings. `RunOptions::sampler` is optional: unset inherits the plan, while an explicit value overrides it. Put sampler overrides in `GenerateRequest::sampling` when resolving conflicting adapter metadata during planning. Frontends with their own `RunOptions` can call `validate_sampling_sampler(plan, options.sampler)` during validation; `run_generate()` checks it again before loading models.

The C API adds:

```c
slopfab_request_set_sampling_settings(request,
    "{\"version\":1,\"video_sigma_shift\":6,\"base_sigmas\":[1,0.5,0]}");
```

The setter replaces the request's explicit overrides. Invalid input leaves existing overrides intact. Pass a null JSON pointer to clear overrides and inherit defaults again. Existing ABI struct layouts and schedule enum values are unchanged.

## Further architecture work

See [conditioning settings and sessions](conditioning_settings.md), [model contracts](model_contracts.md), [backend contracts](backend_contracts.md) and the updated [architecture review](architecture_review.md) for the accompanying implementation and remaining family-specific constraints.
