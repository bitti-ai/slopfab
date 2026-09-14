# H3 refmod references

Refmods are pre-encoded reference latents created by
[ComfyUI-MiniMaxH3Mod](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod).
The [creation guide](https://huggingface.co/datasets/malcolmrey/various/raw/main/h3-center/docs/MINIMAX_H3_REFMOD_CREATION_GUIDE.md)
describes encoding and stacking reference images into these files.
Slopfab loads standalone image, video, and audio refmods on CUDA and Vulkan.
They require **Ref2VA transformer weights** in Slopfab; the pruned FL2VA
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

The DLL ABI is **1.8.0**, with two new exports:

```c
int status = slopfab_request_add_refmod(request, "person.safetensors", 1.0f, 2);
/* Check status, and use slopfab_last_error() on failure. */
slopfab_request_clear_refmods(request);
```

`add_refmod` validates and copies the file immediately. The request and its
queued generation own an immutable snapshot; the file can subsequently be
changed or removed. Failed additions leave the existing list intact.
Clearing refmods leaves ordinary references and LoRAs attached.

```cpp
request.refmods.push_back({slopfab::RefMod::load("person.safetensors"), 1.0f, 2});
```

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

Standalone encode/pooled/trained refmods all use the same loading path. Bundles
are rejected with an explicit error; export their individual members first.
Optional embedded retention/curve configuration is not applied. This feature
does not create, optimize, or train refmods. Compatible H3 LoRAs can be attached
separately; a refmod itself is a latent reference, not a weight adapter.

## Verification

CPU tests cover metadata and dtype loading, visual/audio packing, strength
blurring, disabled references, copy positions, deterministic noise, and prompt
cache reuse. DLL tests cover exports, input validation, transactional additions,
and file-independent ownership. The full generation smoke tool uses small
synthetic refmod tensors with real model weights:

```powershell
python tools/refmod_generation_smoke.py build-vulkan-fast-cuda128/Release/slopfab.dll . cuda
python tools/refmod_generation_smoke.py build-vulkan-fast-cuda128/Release/slopfab.dll . vulkan embedding
```

The second invocation also exercises captured prompt conditioning. These tests
check execution and finite decoded outputs; they do not measure identity/style
fidelity or establish parity with ComfyUI.
