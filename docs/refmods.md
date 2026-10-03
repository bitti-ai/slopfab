# H3 refmod references

Refmods are pre-encoded reference latents created by
[ComfyUI-MiniMaxH3Mod](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod).
The [creation guide](https://huggingface.co/datasets/malcolmrey/various/raw/main/h3-center/docs/MINIMAX_H3_REFMOD_CREATION_GUIDE.md)
describes encoding and stacking reference images into these files.
Slopfab loads standalone image, video, and audio refmods and mixed version-5
bundles on CUDA and Vulkan. Since ABI **1.18.0**, it also encodes raw images,
video and audio into standalone RefMods or mixed bundles. It can package
existing refmods and export standalone text embeddings.
Using media refmods for generation requires **Ref2VA transformer weights**; the pruned FL2VA
checkpoint has a different timestep/AdaLN contract.

Add these options to a normal generation command:

```powershell
slopfab generate --prompt "A person walking through a sunlit garden" `
  --refmod references/person.safetensors --refmod-strength 1 --refmod-copies 2 `
  --refmod references/style.safetensors --refmod-strength 0.7
```

The CLI selects Ref2VA when an enabled reference is present. Each strength or
copies option applies to the preceding `--refmod`. Defaults are strength `1`
and one copy. Strength must be between `0` and `1`; zero omits the reference
entirely. Copies must be between `1` and `10`. Copies increase the packed
sequence length and GPU memory requirement. An 8192-token refmod with two
copies adds 16384 tokens; there is no artificial 8192-token aggregate limit.

Refmods follow ordinary image/video/audio references in the packed sequence.
Each copy has its own temporal position. Their source images are unnecessary,
and loading does not run the reference VAE encoder or add Qwen vision tokens.
The VAEs are still needed to decode generated outputs. Ordinary prompt encoding
and `--prompt-embedding` both work for refmod-only conditioning. Descriptions
and trigger words in the metadata are not inserted into the prompt automatically.

Strength blends the stored latent with a spatially blurred version (temporally
blurred for audio), matching the upstream flat-strength operation. Video frames
are not blurred together. Slopfab then applies its normal fixed-reference
conditioning: visual timestep/noise augmentation `0.999`, clean audio timestep
`1.0`. Each visual copy uses Slopfab's deterministic reference noise stream.
This does not promise bit-identical output to ComfyUI's different RNG packing.

## DLL and C++

Refmod attachment was introduced in DLL ABI 1.8. ABI **1.17.0** also accepts
version-5 bundles through the same call:

```c
int status = slopfab_request_add_refmod(request, "person.safetensors", 1.0f, 2);
/* Check status, and use slopfab_last_error() on failure. */
slopfab_request_clear_refmods(request);
```

`add_refmod` validates and copies every member immediately, preserving member
order. Strength and copies apply to all members in that file. The request and its
queued generation own an immutable snapshot; the file can subsequently be
changed or removed. Failed additions leave the existing list intact.
Clearing refmods leaves ordinary references and LoRAs attached.

```cpp
request.refmods.push_back({slopfab::RefMod::load("person.safetensors"), 1.0f, 2});
// Either a standalone file or a mixed bundle:
for (const auto& mod : slopfab::RefModBundle::load("hero.safetensors").members)
  request.refmods.push_back({mod, 1.0f, 1});
```

## Encode raw media

`encode-refmod` runs the H3 VAE encoders without a text encoder or diffusion
transformer. Supply images, video files, audio files, or a mixture:

```powershell
slopfab encode-refmod --reference-image portrait.png `
  --video-vae weights/vae/minimax_h3_video_vae_fp16.safetensors `
  --output portrait.safetensors

slopfab encode-refmod --reference-audio voice.wav `
  --audio-vae weights/vae/minimax_h3_audio_vae_fp32.safetensors `
  --output voice.safetensors

slopfab encode-refmod --reference-image portrait.png --reference-video motion.mp4 `
  --reference-audio voice.wav --name character --description "Appearance, motion and voice" `
  --output character.safetensors
```

Each reference option is repeatable. Omitted VAE paths use the CLI's local model
discovery. Images/videos need floating-point video VAE encoder weights; audio
needs the audio VAE encoder. CLI video/audio file decoding requires an FFmpeg-enabled
build and `ffmpeg`/`ffprobe` executables. Image files use the platform image decoder
when FFmpeg is disabled. Select `--inference-backend cuda|vulkan`; Vulkan also
accepts `--vulkan-arithmetic portable|exact`.

A single image, silent video, or audio clip produces a version-4 standalone file
with a `latent` tensor. Multiple assets produce a version-5 bundle. A video's
soundtrack becomes a separate audio member immediately after its visual member,
so a video with audio produces a bundle even when it is the only input file.
Images come first, followed by video/audio inputs in attachment order. This uses
the [upstream tensor and container format](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod/blob/main/BUNDLE_FORMAT.md).

Preprocessing follows Slopfab's native reference path:

- Images retain their aspect ratio, resize to a 768-pixel short edge with an
  area cap of `768*1344`, and align dimensions to multiples of 32. `--short-edge`
  accepts multiples of 32 from 32 to 768 for smaller exports.
- Videos/audio retain the existing 2–15 second input contract. Video is sampled
  at 24 fps and its encoding length rounds **down** to `17*n+5` frames, discarding
  the remaining tail. For example, 2 seconds becomes 39 encoded frames and 12
  latent frames. Audio retains its available duration independently.
- Mono/stereo PCM is resampled to planar stereo at 32 kHz; mono is duplicated.
  Audio encoding pads to a multiple of 800 samples per channel.
- Visual encoding uses deterministic seed-42 posterior sampling; audio uses
  the posterior mean. The saved F32 latents are normalized and contain no
  generation-time conditioning noise. Metadata records geometry, backend,
  VAE identity, description and media preparation details.

At most 9 images, 3 videos, 3 standalone audio clips, and 12 total input
references are accepted. Videos and standalone audio each have a 15-second
aggregate limit. The parent output directory must exist. Saving replaces the
destination atomically after all encodes succeed. There is no pooling,
compression, refinement or training in this export operation.

### Library integration for raw media

The Windows DLL and Linux SO expose the same synchronous call:

```c
int slopfab_export_refmod(const slopfab_request* request,
                         const char* output_path, const char* name,
                         const char* description, int32_t short_edge);
```

Use existing reference setters to supply the raw assets. Image input is a file
path; video/audio input is decoded host memory, so the library needs no FFmpeg
for those inputs. Slopus can use its own media decoder and pass frames/PCM:

```c
slopfab_request* request = slopfab_request_create();
/* Check every returned status; errors are available through slopfab_last_error(). */
slopfab_request_set_model_path(request, SLOPFAB_MODEL_VIDEO_VAE, video_vae_path);
slopfab_request_set_model_path(request, SLOPFAB_MODEL_AUDIO_VAE, audio_vae_path);
slopfab_request_add_reference_image(request, "portrait.png");

slopfab_reference_video* video = NULL;
slopfab_reference_video_create(clip_duration_seconds, &video);
/* Repeat for each frame, starting at timestamp 0, with increasing timestamps. */
slopfab_reference_video_append_rgb24(video, pixels, pixel_buffer_bytes,
                                    width, height, row_stride_bytes, timestamp_seconds);
slopfab_request_add_reference_video(request, video);
slopfab_reference_video_destroy(video); /* The request owns its snapshot. */

slopfab_request_add_reference_audio_f32(request, pcm, float_count, channels, sample_rate);
int status = slopfab_export_refmod(request, "character.safetensors",
                                  "character", "Appearance, motion and voice", 0);
slopfab_request_destroy(request);
```

Use only the setters needed for the desired asset. Audio-only export works
without an image/video, a prompt or a generation plan. `short_edge=0` selects
768; otherwise the same 32..768 range applies. Name/description may be NULL.
An attached video soundtrack uses `slopfab_reference_video_set_audio_f32` before
attaching the video. RGBA frames are accepted by the existing RGBA8 setter.

The export uses VAE paths, backend, Vulkan arithmetic, and raw references from
the request. It ignores generation dimensions/seed, prompt, text embeddings,
transformer, LoRAs, and already-attached RefMods. It does not modify the request.
The call returns `SLOPFAB_ERR_BUSY` during generation or another export; do not
mutate or destroy the request until it returns. Image files must remain readable
until export completes. Existing generation/session ownership rules still apply.

Load the result with `slopfab_request_add_refmod` or `generate --refmod`.
The C++ equivalent is `export_refmod(RefModExportRequest, output_path)` from
`slopfab/refmod_export.h`, linked through `slopfab_generation`.

## Package mixed assets

Package pre-encoded image, video and audio references into one upstream-compatible
version-5 file. This operation uses no model weights or GPU:

```powershell
slopfab bundle-refmods --refmod face.safetensors --refmod motion.safetensors `
  --refmod voice.safetensors --name hero --description "Appearance, motion and voice" `
  --output hero.safetensors
```

Inputs may also be bundles: they are flattened in input/member order, up to 256
members. Each member keeps its own dimensions, dtype, description, configuration
and custom metadata. The first input bundle's container metadata is retained;
`--name` and `--description` override it. Without an inherited name, the default
is the output filename stem. The parent output directory must exist. Saving
replaces the output atomically only after every member has been validated.

Load a bundle with the ordinary `generate --refmod hero.safetensors` option.
Following `--refmod-strength` and `--refmod-copies` options apply to every member
of that file. A later `--refmod` starts a new slot. Bundle descriptions stay as
metadata and are not inserted into the prompt. Saved configuration is preserved
for other consumers but Slopfab does not execute retention/curve settings.
Bundling does not establish voice-to-character assignments or audiovisual timing.

The DLL exposes the same packaging operation in ABI 1.17:

```c
const char* inputs[] = {"face.safetensors", "motion.safetensors", "voice.safetensors"};
int status = slopfab_save_refmod_bundle(inputs, 3, "hero.safetensors",
                                       "hero", "Appearance, motion and voice");
/* Check status and slopfab_last_error(). Name/description may be NULL. */
```

Raw-media encoding is provided by `encode-refmod`; pooling and refinement are
not part of either command.

## Standalone text embeddings

Encode text once and save its complete H3 conditioning without loading a
diffusion transformer or either VAE:

```powershell
slopfab encode-text --prompt "A red bird lands on a branch." `
  --text-encoder weights/text_encoder/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors `
  --output prompt.safetensors

slopfab generate --prompt-embedding prompt.safetensors --refmod hero.safetensors
```

`--prompt-file` reads a UTF-8 file instead of `--prompt` (an initial BOM is
removed). `--tokenizer` overrides the embedded tokenizer; omitting
`--text-encoder` uses the normal discovery under `weights/text_encoder`.
The output directory must already exist.

CUDA is the default when compiled in; select Vulkan with
`--inference-backend vulkan`. CUDA supports `--residency streaming|resident|auto`
(default streaming) and `--arithmetic shipped|exact` (default shipped). Vulkan
uses its exact encoder graph; `--vulkan-arithmetic portable|exact` selects
feature-compatible or qualified-device execution. There is no CPU encoder.

The archive stores F32 `prompt_embedding [L,5120]` and I32
`text_token_tags [L]`, all text tags for this text-only operation. Metadata
records the source prompt, conditioner contract, checkpoint path/stat identity,
the loaded tokenizer JSON's SHA-256, backend and arithmetic. These are provenance
fields; loading does not require the source files. Exports are atomic and can be
read through the existing `--prompt-embedding` / DLL prompt-embedding option.

Saved embeddings replace the complete prompt conditioning. They do not merge
with a new prompt, and are not upstream RefMod members of kind `text`. Keep the
text embedding as a companion file to the compatible media bundle. Descriptions
are not encoded implicitly, and this command does not present media to Qwen.

The synchronous DLL export uses a request's prompt, text-encoder/tokenizer paths,
inference backend and arithmetic settings. It ignores other request fields,
including references. Set an explicit text-encoder path, then call:

```c
int status = slopfab_export_prompt_embedding(request, "prompt.safetensors");
/* Check status and slopfab_last_error(). No generation handle is needed. */
```

It returns `SLOPFAB_ERR_BUSY` if a generation or another export is active. CUDA
weights are streamed. The C++ equivalent is `text::export_prompt_embedding` with
`text::TextExportRequest` from `slopfab/text/export.h`.

## File compatibility

Supported files contain a tensor named `latent`, stored as F16, BF16, or F32:

| Kind | Shape | Packed tokens |
| --- | --- | --- |
| Image | `[1,24,1,H,W]` | `(H/2)*(W/2)` |
| Video or stacked images | `[1,24,T,H,W]` | `T*(H/2)*(W/2)` |
| Audio | `[1,32,2,T]` | `2*T` |

Visual H/W must be positive and even. Latents are already normalized, as stored
by the upstream H3 VAE; they are not normalized a second time. The loader reads
the JSON string in safetensors metadata `refmod_meta` (or `audio_refmod_meta`),
falling back to a same-stem `.json` sidecar only when embedded metadata is absent.
Metadata dimensions, when present, must agree with the tensor. Malformed
metadata, incompatible shapes, and non-finite latent values are rejected.

Standalone encode/pooled/trained refmods all use the same loading path. Version-5
bundles use `kind: "bundle"`, an ordered `members` array in `refmod_meta`, and
matching `ref_0`, `ref_1`, ... tensors. Nested bundles, missing tensors, invalid
members and unsupported bundle versions are rejected before attachment.
Optional embedded retention/curve configuration is not applied. This feature
does not optimize or train refmods. Compatible H3 LoRAs can be attached
separately; a refmod itself is a latent reference, not a weight adapter.

## Verification

Real-VAE export checks exercise standalone image/video/audio, mono resampling,
video soundtracks, mixed member equality and reload on either GPU backend:

```powershell
python tools/refmod_export_smoke.py dist/windows/slopfab.dll . cuda build/export-smoke
python tools/refmod_export_smoke.py dist/windows/slopfab.dll . vulkan
python tools/refmod_generation_smoke.py dist/windows/slopfab.dll . cuda embedding refmod=build/export-smoke/mixed.safetensors
```

On Linux, use `python3` and `dist/linux/libslopfab.so` with the same arguments.

CPU tests cover metadata and dtype loading, visual/audio packing, strength
blurring, disabled references, copy positions, deterministic noise, and prompt
cache reuse. DLL tests cover exports, input validation, transactional additions,
and file-independent ownership. The full generation smoke tool uses small
synthetic refmod tensors with real model weights:

```powershell
python tools/refmod_generation_smoke.py build/Release/slopfab.dll . cuda
python tools/refmod_generation_smoke.py build/Release/slopfab.dll . vulkan embedding
python tools/refmod_generation_smoke.py build/Release/slopfab.dll . cuda export bundle
python tools/refmod_generation_smoke.py build/Release/slopfab.dll . vulkan export bundle
```

The `export bundle` runs create real text embeddings and a mixed bundle through
the DLL, then generate with the text-encoder path cleared. Host tests also cover
archive round trips, unknown metadata, integer tags and failed writes. The
`conditioning_assets_cli` CTest (when Python is available) checks byte-preserving
packaging, bundle flattening and per-file strength/copy scoping without models.
These tests
check execution and finite decoded outputs; they do not measure identity/style
fidelity or establish parity with ComfyUI.
