# Still decoder temporal-context diagnostic

Seven-token VAE decoding is now the default for still images. Denoising still
produces one latent; the decoder repeats each spatial tile across seven temporal
positions and stitches only the first retained frame. The old one-token decoder
is removed, with no public ABI or application-call changes.

`slopfab_stillprobe` checks the default still decoder against the normal video
decoder on **the same generated latent**, using CUDA and the same VAE checkpoint.
It exits unsuccessfully if their float RGB outputs differ. The original
one-token/seven-token investigation is preserved under Local results below.

Build:

```powershell
cmake -S . -B build -DSLOPFAB_BUILD_DEV_TOOLS=ON
cmake --build build --config Release --target slopfab_stillprobe slopfab_tests
```

Generate one still latent and run the comparison:

```powershell
./build/Release/slopfab_stillprobe.exe `
  --vae weights/vae/minimax_h3_video_vae_fp16.safetensors `
  --transformer weights/transformer/MiniMax_H3_FL2VA_pruned_nvfp4.safetensors `
  --text-encoder weights/text_encoder/qwen3vl_32b_int8_convrot.safetensors `
  --tokenizer ref/text_encoder/tokenizer.json `
  --prompt "A studio portrait against a plain beige background" `
  --width 768 --height 768 --steps 50 --seed 1234 `
  --out output/still-probe
```

The output directory must not already exist. Model paths are explicit; no models
are downloaded. Generation uses the library's Euler/Flash2 defaults, not the CLI's
preview defaults. It captures the packed denoiser output and intentionally stops
generation before the VAE stage. The probe then loads the VAE once and runs:

1. **still-default**: `decode_still_image` with seven repeated temporal positions
   and default spatial tiling, assembling only one output frame.
2. **video-reference**: repeat each channel's normalized spatial latent seven times,
   run the normal video decoder, and extract its first retained frame (temporal
   phase 3). Spatial tiling, latent normalization and weights remain the same.

The seven-token input has no real motion or independently generated temporal
context. It is not a normal video sample. Both paths evaluate the seven-token
VAE window, but the default still decoder skips assembling and normalizing the
discarded full frames. It does not rerun denoising or generate target audio.

Repeat the decode experiment without any more denoising:

```powershell
./build/Release/slopfab_stillprobe.exe `
  --vae weights/vae/minimax_h3_video_vae_fp16.safetensors `
  --latents output/still-probe/input.safetensors `
  --width 768 --height 768 --out output/still-probe-redecode
```

Saved inputs must contain F32 `video_rows` of shape
`[(height / 32) * (width / 32), 96]` from a one-latent still run. Geometry is
supplied explicitly. A PNG cannot recover the latent used to generate it.

## Outputs and interpretation

- `input.safetensors`: packed denoiser output when the probe generates a sample.
- `normalized-latent.safetensors`: the unpatchified input, `[24,1,H/16,W/16]`.
- `still-default`, `video-reference`: lossless PPM images
  and F32 RGB safetensors. `*-third.ppm` files are Lanczos-downscaled previews
  when both dimensions are divisible by three.
- `row-luma.csv`: row-average luminance, in 0..255 units, for the whole image
  and the outer 5% of columns on each side.
- `report.txt`: paths, generation settings, decode timings, RGB differences and
  the amplitude of the 16-pixel row-luminance fundamental. The latter uses a
  second-difference detrend and compensates for that filter's gain.

The edge strips are **not** automatically segmented background. Scene content,
boundaries and texture contribute to the metric; it is not an aesthetic score.
Inspect the images and use multiple seeds before drawing a general conclusion.
Compare the original-resolution float outputs, not just resized 8-bit previews.

The two current outputs must match exactly. A difference indicates a regression
in temporal repetition, phase selection, spatial stitching or normalization.

## Local results, 2026-09-08 (before changing the default)

Ran on an RTX 5090, driver 610.88, CUDA 12.8-built executable. Used the FP16 VAE,
NVFP4 FL2VA transformer and INT8 ConvRot conditioner named in the example,
768x768, 50 sigma points (49 evaluations), Euler/Flash2, and seeds 1234 and 42.
These are newly generated portraits similar to the reported example, not a
reconstruction of the user's original latent. The prompt was:

> integrated_multimodal_description: A realistic studio portrait of a rugged
> middle-aged cowboy wearing a brown leather wide-brimmed hat, tan worn jacket
> and dark red bandana. Head and upper torso centered, facing the camera,
> serious expression. Plain uniform light beige background, soft even studio
> lighting, crisp detailed fabric and natural skin.

Measured 16px row-band amplitude (whole image / edge strips), in luma levels:

| Seed | One token, tiled | Seven repeated tokens, tiled | One token, untiled |
| --- | --- | --- | --- |
| 1234 | 1.57992 / 0.59940 | 0.52756 / 0.06486 | 2.59184 / 1.00568 |
| 42 | 1.89506 / 1.14858 | 0.38355 / 0.23692 | 2.70423 / 1.29478 |

Seven-token decoding reduced the edge-strip fundamental by approximately 89%
and 79%, respectively. Visual inspection of both downscaled comparisons showed
substantially less banding. Untiled one-token decoding produced stronger patch
artifacts, especially around the subject. This strongly implicates the shortened
one-token decoder path, rather than spatial tile seams alone. It does not prove
the exact learned/numerical mechanism or establish quality across other prompts.
Seven-token decoding also changed overall brightness and color; it is not merely
a stripe-removal filter.

The seed-1234 decode-only replay took 0.219 s for one-token decoding and 1.200 s
for seven-token decoding, excluding checkpoint load and image writing. The first
run was slower (0.287 s / 3.482 s); these are diagnostic timings, not controlled
benchmarks. The seven-token path runs the full 22-frame decode and discards all
but the first frame. At the time of this investigation no production default
had been changed; the subsequent implementation adopts seven-token decoding
without assembling the discarded full frames.

Local artifacts (ignored by Git) are in:

- `output/still-diagnostic-seed1234/`
- `output/still-diagnostic-seed42/`
- `output/still-diagnostic-seed1234-redecode/`

Each seed directory also contains `comparison.png`: left is one-token tiled,
middle is seven-token tiled, right is one-token untiled. PNGs were losslessly
converted from the probe's downscaled PPM previews with the local FFmpeg binary.

Host validation: 12,440 checks passed, including latent channel/time repetition,
planar first-frame extraction, and equality of still/video spatial blending and
phase selection under a temporally invariant fake backend. The latter covers
multiple tile rows and columns with channel- and pixel-dependent output.

## Default implementation validation

The production still decoder now repeats each denormalized spatial tile seven
times in `[batch,channel,time,height,width]` order. It evaluates the same VAE
window as the reference but stitches only phase 3 of the first latent position
for each RGB channel. It does not allocate the assembled 22-frame video.

Both saved 768x768 inputs were replayed through the new default. The resulting
F32 image archives matched the original diagnostic's `seven-token.safetensors`
files byte-for-byte, and matched the current full-video reference with zero
RGB error. Outputs are in `output/still-default-seed1234/` and
`output/still-default-seed42/`. The new default took 1.190 s and 1.152 s,
respectively, excluding load and writing.

After this change, 13,203 host checks and 122 C API checks passed. Tests cover
seven-position repetition with nonuniform spatial values and distinct channel
statistics, output phase/channel indexing, and exact tiled reference agreement.
The Release DLL was rebuilt at
`build/Release/slopfab.dll`. No opt-out for one-token decoding
was introduced. The scheduling change is shared by CUDA and Vulkan; real-weight
image comparisons here were run on CUDA.
