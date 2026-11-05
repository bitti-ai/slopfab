---
name: download
description: List model, LoRA, upscaler and Whisper URLs and download selected files.
version: "1.1"
command: download
script: main.js
---
# Model downloads

Run `slopfab download` to list model groups, then `slopfab download MODEL` to
list their files without network access. The catalog includes Slopus's bundled
download options from `src/lib/settings.ts`, `h3Components.ts`, `loras.ts` and
`upscalers.ts`, plus Whisper:

| Model group | Options |
| --- | --- |
| `minimaxh3` | First/Last Frame, References, FastH3 Omni, Singularity v1/v2, Animate, shared text encoder, video/audio VAEs and Sol H3 VAE |
| `minimaxh3-loras` | TaoMate, Turbo, Viggle Animate, LightX2V Turbo, DMAD, Pika First/Last and Reference, Hyperflow |
| `krea2` | Turbo and Raw, text encoder and image VAE |
| `moss-image` | SFT transformer, text encoder, image VAE and tokenizer |
| `fibo2` | Edit Turbo and Base, text encoder and image VAE |
| `yue2` | INT8 and BF16 transformers |
| `minimax-music3` | INT8, FP16 and FP32 transformers, text encoder and audio VAE |
| `realesrgan` | RealESRGAN x4plus |
| `seedvr2` | 3B FP16 model and VAE |
| `latent-upscale` | MiniMax H3 latent upscaler |
| `taeh3` | Generation preview VAE |
| `whisper` | Large v3 Turbo safetensors |

Select checkpoints appropriate to the model, backend, available memory
and intended task; this catalog is a set of options, not a requirement to fetch
every file. Review the source repository's model license before use.

Run `slopfab download MODEL FILE_ID DESTINATION` to download one selected file.
The destination must not exist. Model files are large; downloads allow up to two
hours and do not resume partial transfers. Register the containing weights root
with `slopfab settings set weight_paths PATH` after placing files in the desired
`transformer/`, `text_encoder/` and `vae/` folders.

File IDs select variants explicitly. `minVramGb` records Slopus's per-file
minimum VRAM guidance in GB; zero means unrestricted, and omitted values have
no supplied tier. The mod does not detect GPUs or automatically select variants.
LoRA `stepOverride` and `samplingPreset` fields describe Slopus defaults;
configure generation and any required companion grids separately. Group notes
identify required companion files, including Animate conditioning and LoRAs.

For speech transcription, download the standard Hugging Face Whisper checkpoint:

```sh
slopfab download whisper large-v3-turbo weights/whisper/whisper-large-v3-turbo.safetensors
slopfab transcribe speech.wav --model weights/whisper/whisper-large-v3-turbo.safetensors --out transcript.srt
```

Whisper's tokenizer and mel assets are embedded in Slopfab; no companion downloads
are required. The checkpoint is from
[OpenAI's Whisper Large v3 Turbo repository](https://huggingface.co/openai/whisper-large-v3-turbo).

Add `--agent` for the normal agent response. Edit `catalog.json` to add more
models or URLs. This example does not automatically alter Slopfab settings.
