# ClipProj for H3 on lower-VRAM hardware

Slopfab can use a Qwen3-VL-4B or 8B encoder with a matching
[NicoLab28 ClipProj checkpoint](https://huggingface.co/NicoLab28/ClipProj-MiniMax-H3)
to produce H3's 5120-wide text conditioning. Enable this with
`--clip-projection FILE`. The ordinary 32B conditioner remains the default.

The smaller encoder streams one decoder layer at a time and gathers only the
requested token embeddings. It does not upload the vocabulary table or visual
tower. After encoding, its GPU allocations are released before the projection
runs on the CPU. Projection weights are converted in small output tiles to bound
temporary RAM use. This trades CPU work and host RAM for a smaller VRAM peak.
The diffusion transformer and VAEs keep their existing memory requirements;
lower conditioning memory does not imply that the entire video pipeline fits
on the same small card.

## Files and commands

The download mod lists smaller encoders and matching projections in
`slopfab download minimaxh3-clipproj`. For the 4B INT8 ConvRot / v3.1 ridge pair:

```sh
slopfab download minimaxh3-clipproj text-4b-int8
slopfab download minimaxh3-clipproj projection-4b-v3.1
```

Files download into `weights/` beside the executable. The same group includes
`text-4b-fp8`, `text-8b-int8`, `text-8b-fp8`, `projection-8b-v3.1`, and
`-mlp` variants of both projections. Choose one encoder and one matching-size
projection; downloading them does not enable ClipProj automatically.

Supply both an encoder and a projection of the same size, for example:

- `qwen3vl_4b_int8_convrot.safetensors`
- `mmh3-4b-ClipProj-v3.1.safetensors`

ComfyUI-format BF16, FP16, FP32, INT8 ConvRot and scalar FP8-scaled student
linears are supported. Embeddings must be floating point or INT8 ConvRot.
INT4 and NVFP4 **student** weights are unsupported. A 4B projection requires
2560 input features; an 8B projection requires 4096. Both must output 5120.
No model downloads happen during generation.

Generate directly, using explicit paths or your configured H3 model alias for
the transformer and VAEs:

```sh
slopfab generate video --model h3 --prompt "A red ball on a wooden table." \
  --text-encoder /models/qwen3vl_4b_int8_convrot.safetensors \
  --clip-projection /models/mmh3-4b-ClipProj-v3.1.safetensors \
  --out ball.y4m
```

The line continuations above are for Linux shells; use one line in PowerShell.
For a still image, use `generate image --model h3`. Add `--agent` for JSON
responses and generation progress. `--dry-run --check-ready` validates the
projection layout and encoder dimensions without loading GPU weights.

To encode once and reuse the result:

```sh
slopfab encode-text --prompt "A red ball on a wooden table." \
  --text-encoder /models/qwen3vl_4b_int8_convrot.safetensors \
  --clip-projection /models/mmh3-4b-ClipProj-v3.1.safetensors \
  --output ball-prompt.safetensors
slopfab generate video --prompt-embedding ball-prompt.safetensors --out ball.y4m
```

The exported archive includes modality tags, encoder/projection file identities
and tokenizer provenance. `encode-text` does not load or discover transformers
or VAEs and shares the CLI's GPU lock with generation. A busy GPU returns
`GPU_BUSY` (exit 7); retry when the other job finishes.

## Settings and agent requests

Model aliases accept `clip_projection` beside `text_encoder`:

```json
{
  "schema_version": 1,
  "weight_paths": ["/models"],
  "models": {
    "h3-small": {
      "family": "h3",
      "text_encoder": "/models/qwen3vl_4b_int8_convrot.safetensors",
      "clip_projection": "/models/mmh3-4b-ClipProj-v3.1.safetensors"
    }
  }
}
```

Use `slopfab generate video --model h3-small ...`. You can set these fields with
`slopfab settings set models.h3-small JSON`. Include transformer/VAEs in the
alias if your weight directories contain ambiguous matches.

When a projection is selected and no encoder is supplied, discovery searches
configured weight paths for the matching `qwen3vl_4b` or `qwen3vl_8b` filename.
Multiple candidates require an explicit choice. Projection selection is always
explicit; it is not enabled merely because a file exists in the weights folder.

JSON requests can select an alias or pass the options directly:

```json
{
  "schema_version": 1,
  "task": "video",
  "model": "h3",
  "prompt": "A red ball on a wooden table.",
  "options": {
    "text-encoder": "/models/qwen3vl_4b_int8_convrot.safetensors",
    "clip-projection": "/models/mmh3-4b-ClipProj-v3.1.safetensors"
  }
}
```

Changing the projection path or file identity invalidates cached conditioning.

## Supported scope

- CUDA and Vulkan text-only H3 conditioning. Vulkan requires
  `--vulkan-arithmetic portable`.
- Linear ridge, ridge plus GELU MLP, and residual-only MLP files, including
  v3/v3.1 files without a `W` tensor. Optional `sink_out` replaces the first row.
- Projection metadata `tap` selects a raw hidden state before that decoder
  layer: tap 24 executes layers 0 through 23. No chat template or final norm is
  added. The projection contract follows
  [upstream ClipProj](https://github.com/nicolab28/ComfyUI-ClipProj/tree/c01ba8fb8f41b4f2094dbd0b185cdc238fb6134c).
- Image/video/native-audio reference conditioning requires the ordinary 32B
  encoder. Unsupported reference requests fail explicitly. Pre-encoded RefMods
  do not require a student visual tower.
- Exact conditioner arithmetic and resident student loading are unsupported.
  Do not combine ClipProj with `--prompt-embedding`; that archive already
  contains the final conditioning.

The C ABI is unchanged. Existing DLL requests continue using the original path;
the optional C++ request field defaults to empty.

## Validation

Host tests cover matrix orientation, normalization, the sink, FP16 MLP arithmetic,
residual-only files, malformed files and cache identity. CLI tests cover flags,
JSON requests, aliases, readiness checks and standalone export.

A real Windows CUDA encode used the local 4B INT8 ConvRot encoder and upstream
v3.1 ridge file. Against an independent PyTorch calculation, non-sink tokens
measured **0.84% relative RMS error and 0.99997 cosine similarity**. This measures
implementation agreement with the small encoder/projection, not visual quality
relative to the 32B model. The 8B and FP8 paths still need real-checkpoint
qualification. No end-to-end video-quality claim is made by these checks.

The development reference tool uses local files and requires PyTorch,
`tokenizers` and `safetensors`:

```sh
python tools/clipproj_reference.py --encoder /models/qwen3vl_4b_int8_convrot.safetensors \
  --projection /models/mmh3-4b-ClipProj-v3.1.safetensors \
  --tokenizer ref/text_encoder/tokenizer.json --native ball-prompt.safetensors
```

The `--prompt` must match the exported native archive; its default is the red
ball example above.
