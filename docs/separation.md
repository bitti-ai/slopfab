# Vocal and stem separation

`slopfab separate` runs standard four-stem HTDemucs directly from safetensors.
Inference is native C++ on the CPU, including when the executable also has CUDA
or Vulkan enabled. Python and PyTorch are not runtime dependencies. GPU inference,
six-stem models, HDemucs and ensembles such as `htdemucs_ft` are not supported.

```sh
# Four files: drums.wav, bass.wav, other.wav, vocals.wav
slopfab separate song.wav --model weights/htdemucs.safetensors --out output/stems

# Two files: vocals.wav and instrumental.wav
slopfab separate song.wav --model weights/htdemucs.safetensors --two-stems vocals --out output/vocals
```

The default output directory is `output/<input-name>-stems`. Existing outputs
are protected; pass `--overwrite` to replace them. `--two-stems` also accepts
`drums`, `bass` and `other`, writing the selected stem and `no_<stem>.wav`.
The companion track is the input minus the selected stem, so the pair reconstructs
the input. Four-stem estimates are the model's predictions and need not sum
exactly to the input.

Output is stereo 44,100 Hz float32 WAV. Samples are neither clipped nor separately
normalized, so stems can exceed ±1. Native input supports mono/stereo PCM16/24/32
or float32 WAV at 44,100 Hz; mono is duplicated to stereo. Other sample rates and
compressed audio/video use external FFmpeg, optionally selected with `--ffmpeg`.
The result has the same frame count as the decoded 44,100 Hz input.

`--segment` defaults to the trained 7.8-second window. Values from 0.05 to 7.8
seconds are accepted; shorter windows reduce CPU work and memory but can reduce
separation quality. `--overlap` defaults to 0.25 and accepts 0 through 0.9.
Chunks are combined with triangular weights, retaining context for the final
partial chunk. Inference is deterministic and does not use random shift averaging.
CPU processing can be slow for full songs. `--backend cpu` and `--backend auto`
both select the CPU implementation.

Add `--agent` for progress events and a result with `artifacts`, stem names,
sample rate, channel count and frame count. The artifacts work with the existing
SSH result download workflow. Cancellation is checked between network stages
and chunks. Separation is a CLI/C++ feature; it is not exposed through the C ABI.

## Weights

Use the standard upstream `htdemucs` checkpoint (signature `955717e8`), with
its original PyTorch `state_dict` tensor names, converted to safetensors. F32,
F16 and BF16 storage are supported; calculations use F32. A directory passed
to `--model` resolves to `model.safetensors`. There are no automatic downloads
or pickle loads in the executable.

For a reproducible conversion in a Python development environment:

```sh
python -m pip install torch torchaudio demucs==4.0.1 safetensors
python tools/demucs_reference.py export --out weights/htdemucs.safetensors
```

This explicit export downloads and loads the official upstream checkpoint and
writes a single safetensors file. Keep the original tensor names and source
order: `drums`, `bass`, `other`, `vocals`. Shape-compatible custom models must
also use the standard architecture settings; tensor shapes do not encode
attention head count, positional encoding or source semantics. Missing, extra,
incorrectly shaped and non-finite tensors are rejected before inference.

The native implementation follows the MIT-licensed
[upstream HTDemucs architecture](https://github.com/facebookresearch/demucs/blob/main/demucs/htdemucs.py).
The attribution and license are in `third_party/demucs`.

## Verification

`ctest -R "^(demucs|separate_cli)$"` covers spectral transforms, input validation,
all three weight storage types, short/mono/silent inputs, stem reconstruction,
output protection and agent responses without downloading weights.

```sh
python tools/demucs_reference.py verify --exe build/Release/slopfab_demucs_tests.exe
python tools/demucs_reference.py verify --real --exe build/Release/slopfab_demucs_tests.exe
```

The first command compares a small synthetic HTDemucs network against PyTorch;
the second uses the released checkpoint. Both compare the STFT, inverse STFT,
complete network and overlapping chunks, using a short inference window to
keep the checks practical on CPU.
