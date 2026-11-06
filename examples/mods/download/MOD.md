---
name: download
description: List model, LoRA, upscaler and Whisper URLs and download selected files.
version: "1.3"
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

Run `slopfab download MODEL FILE_ID` to download one selected file into `weights/`
beside the running Slopfab executable, regardless of the working directory.
The filename comes from the URL unless the catalog supplies `filename` (used for
the MOSS tokenizer and Whisper checkpoint). Listings report `weightsDirectory`.
An optional `DESTINATION` overrides this default; relative overrides use the
working directory. The destination must not exist. Downloads use Slopfab's native HTTP downloader
with four parallel connections, two retries per failed request, and a two-hour
total timeout. Interrupted transfers keep their completed and partial ranges in
`DESTINATION.download/`. Run the same command with the same destination to resume.
Keep that directory intact until completion; it is removed after the final file
is assembled and published. A small `DESTINATION.download.lock` file remains to
coordinate concurrent invocations; its presence does not mean a download is active.

Resume requires byte-range support and a matching strong ETag. Changed remote
files restart safely. Servers without reliable range support or a strong ETag
use a single streamed download, which restarts after interruption. The final
destination appears only after the complete transfer; assembly temporarily needs
space for both the parts and the final file. No external download program is used.

Register the absolute `weights/` path beside the executable with
`slopfab settings set weight_paths PATH` so generation can discover the downloads.

File IDs select variants explicitly. `minVramGb` records Slopus's per-file
minimum VRAM guidance in GB; zero means unrestricted, and omitted values have
no supplied tier. The mod does not detect GPUs or automatically select variants.
LoRA `stepOverride` and `samplingPreset` fields describe Slopus defaults;
configure generation and any required companion grids separately. Group notes
identify required companion files, including Animate conditioning and LoRAs.

For speech transcription, download the standard Hugging Face Whisper checkpoint:

```sh
slopfab download whisper large-v3-turbo
# When running from the executable's folder:
slopfab transcribe speech.wav --model weights/whisper-large-v3-turbo.safetensors --out transcript.srt
```

Whisper's tokenizer and mel assets are embedded in Slopfab; no companion downloads
are required. The checkpoint is from
[OpenAI's Whisper Large v3 Turbo repository](https://huggingface.co/openai/whisper-large-v3-turbo).

Add `--agent` for the normal agent response. Edit `catalog.json` to add more
models or URLs. This example does not automatically alter Slopfab settings.
