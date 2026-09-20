# Conditioning settings and sessions

Models and LoRAs can supply conditioning policy without adding a model-named runner path. Pass request overrides with `--conditioning-settings FILE`:

```json
{
  "version": 1,
  "image_short_edge": 1024,
  "media_short_edge": 768,
  "include_reference_audio": false,
  "require_euler": true,
  "allow_caches": false
}
```

Store the same JSON as a string under SafeTensors metadata key `slopfab.conditioning`. Resolution applies H3 compatibility presets, transformer metadata, enabled LoRA metadata, and explicit request fields, in that order. Conflicting adapter fields require explicit overrides. Zero-strength adapters contribute no defaults. Unknown fields, versions and incorrect types fail; omitted fields inherit. JSON syntax uses the repository parser.

| Field | Ordinary default | Meaning |
|---|---|---|
| `image_short_edge` | 2048 | Reference image preprocessing size |
| `media_short_edge` | 768 | Decoded reference media preprocessing size |
| `media_max_pixels` | 1032192 | Reference media area limit |
| `references_at_target_canvas` | false | Use target short edge and uncapped reference area |
| `canvas_from_reference_video` | false | Derive an unspecified canvas from the leading video |
| `include_reference_audio` | true | Include available soundtrack as reference conditioning |
| `require_prompt_embedding` | false | Require an explicit prompt embedding |
| `fixed_prompt_tokens` | 0 | Required embedding length; zero accepts any supported length |
| `video_first` | false | Use the supported video-before-image reference layout |
| `pin_target_audio` | false | Preserve the driving soundtrack in target audio rows |
| `max_frames` | 0 | Maximum aligned output frames; zero means no recipe limit |
| `require_euler` | false | Restrict the recipe to Euler |
| `allow_caches` | true | Allow supported approximate denoising caches |

Sizes must be positive; token and frame limits must be nonnegative. Settings select existing implementations and cannot grant unsupported transformer capabilities. `video_first` currently requires one video, one image, an explicit prompt embedding and no continuation/refmods. `video_first` and `pin_target_audio` require `include_reference_audio=false`; pinning also requires a soundtrack and excludes continuation, still images and initial latents. Positive `fixed_prompt_tokens` requires `require_prompt_embedding=true`.

`--animate` remains a compatibility preset resolved into these same fields. Metadata and explicit overrides can change its defaults while structural restrictions remain enforced. Preprocessing settings contribute to cache identity; sampling-only changes do not invalidate prompt or reference caches.

The CLI now leaves an unspecified canvas to the shared planner, so model geometry and `canvas_from_reference_video` take effect. Its former forced 864x480 canvas is available with `--resolution 864x480`; ordinary H3 planning uses a 768 short edge capped to the trained area. See the sampling guide for the accompanying unified step default.

In C++, set `GenerateRequest::conditioning` or call `parse_conditioning_settings()`. Use `resolve_plan()` and `validate_generation_options()` for device-free validation. CLI, C API and execution share that validation. The C API accepts the same JSON:

```c
slopfab_request_set_conditioning_settings(request,
    "{\"version\":1,\"image_short_edge\":1024}");
```

Invalid input preserves old overrides. A null JSON pointer clears them.

## Sessions

`GenerationSession` owns reusable conditioning, tokenizer and reference caches:

```cpp
slopfab::GenerationSession session;
slopfab::RunOptions options;
options.reuse_models = true;
auto plan = slopfab::resolve_plan(request);
auto result = slopfab::run_generate(session, request, plan, options);
session.clear();
```

The legacy overload uses a default session. Separate sessions isolate caches. GPU execution remains serialized because dispatch/profiling retain process state; simultaneous execution returns a busy result. Transformer and VAE weights retain their staged lifetimes.

C API 1.12 adds `slopfab_session_create`, `slopfab_session_destroy`, `slopfab_session_clear` and `slopfab_request_set_session`. Requests and active generations retain shared ownership, so destroying a handle does not invalidate attached requests. Passing a null session restores the default session. Existing caller-allocated ABI structs are unchanged.

## Adapter preparation

Inference reads adapter assets without downloading or rewriting the adapter. Prepare a missing H3 AdaLN grid explicitly:

```sh
slopfab prepare-lora --adapter adapter.safetensors --width 2688 --download
```

Omit `--download` to permit local assets only. See [model contracts](model_contracts.md) for asset metadata/API, [sampling settings](sampling_settings.md) for step/grid defaults, and [backend contracts](backend_contracts.md) for supported execution contracts.

C API 1.13 exposes the same operation from the DLL:

```c
int status = slopfab_prepare_lora_grid("adapter.safetensors", 2688, 0);
if (status != SLOPFAB_OK) {
    fprintf(stderr, "%s\n", slopfab_last_error());
}
```

The final argument is `allow_download`: zero uses local assets only; nonzero
allows the verified legacy download. This call is synchronous and performs no
GPU work. It validates an already embedded grid or embeds a companion using
atomic replacement, requiring write access and space for an adapter copy.
Close readers of that adapter before preparation and prepare it before starting
generation. Calling it again after successful preparation does not rewrite the
adapter. Invalid path/width arguments return `SLOPFAB_ERR_INVALID_ARGUMENT`;
file and grid errors are reported through the usual status and `slopfab_last_error()`.
