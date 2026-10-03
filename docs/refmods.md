# H3 refmod references

Refmods are pre-encoded reference latents created by
[ComfyUI-MiniMaxH3Mod](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod).
The [creation guide](https://huggingface.co/datasets/malcolmrey/various/raw/main/h3-center/docs/MINIMAX_H3_REFMOD_CREATION_GUIDE.md)
describes encoding and stacking reference images into these files.
Slopfab loads standalone image, video, and audio refmods and mixed version-5
bundles on CUDA and Vulkan. It can package existing refmods into bundles and
export standalone text embeddings.
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

Raw-media encoding, pooling and refinement are not part of this command.

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
does not encode raw media, optimize, or train refmods. Compatible H3 LoRAs can be attached
separately; a refmod itself is a latent reference, not a weight adapter.

## Verification

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
