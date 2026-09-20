# Sampling settings

Models and LoRAs can supply video/audio sigma shifts and fixed sampling grids as data. These settings work with the existing H3 flow scheduler on CUDA and Vulkan; a different transformer graph or velocity convention still needs a model-family implementation.

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

Only `version` is required. Omitting a field inherits it. Shifts must be finite and positive. A grid must have at least two finite points, start in `(0,1]`, decrease strictly, and end at zero. Shifted grids must also remain valid in float32. Unsupported versions, unknown fields, incorrect types and invalid grids are errors. JSON syntax uses the repository's existing parser.

Omit `base_sigmas` to inherit the selected grid, or use the ordinary `--steps` linspace if no lower layer supplies a grid. A fixed grid determines its own length and takes precedence over `--steps`. Fixed grids currently require Euler and disable approximate step, block and MotionCache reuse. This restriction also applies when the grid comes from metadata.

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

Two active adapters may agree or supply different fields. Conflicting values for the same field are rejected unless that field has an explicit request override (including a named schedule's grid). Adapter order cannot silently select the recipe. Disabled adapters contribute no defaults. Missing model/adapter files still allow geometry-only planning with defaults; execution retains its existing file checks. Malformed metadata in an existing active file fails planning.

FastH3 V2 retains its structural and trained-recipe requirements: effective settings must match its existing nine-point grid and video/audio shifts of 10/3. Settings cannot grant reference support, introduce missing VSA gate weights, or bypass those requirements. Existing Animate and TaoMate restrictions also remain.

`--dry-run` reports the effective shifts, evaluation count, whether the grid is fixed, and settings sources. Source entries show contributing layers, rather than provenance for each individual field.

## Library interfaces

In C++, set `GenerateRequest::sampling` directly or use `parse_sampling_settings()`. The resolved `GeneratePlan` contains both modality grids, and both runners use those exact grids. Frontends with their own `RunOptions` can call `validate_sampling_sampler(plan, options.sampler)` during validation; `run_generate()` checks it again before loading models.

The C API adds:

```c
slopfab_request_set_sampling_settings(request,
    "{\"version\":1,\"video_sigma_shift\":6,\"base_sigmas\":[1,0.5,0]}");
```

The setter replaces the request's explicit overrides. Invalid input leaves existing overrides intact. Pass a null JSON pointer to clear overrides and inherit defaults again. Existing ABI struct layouts and schedule enum values are unchanged.

## Further architecture work

This change makes sampling recipes extensible without a new model-named scheduler branch. Conditioning policy, geometry, checkpoint schemas, backend capabilities, session ownership, and the large source files remain separate follow-up work described in the [architecture review](architecture_review.md).
