# H3 continuation drift experiment

`tools/lcs_drift.py` is an opt-in, real-weight experiment using Python, NumPy,
Pillow and FFmpeg. It saves cumulative latent archives, videos, logs and exact
generation commands. No weights are downloaded. Allow several minutes and enough
disk space for the raw videos. The normal build/test suite does not launch it.

The fixture uses a fixed camera, steady light, a close-up adult portrait, and a
small head turn returning toward the original pose. Compare contact sheets before
interpreting pixel statistics: motion, blinking and changing pose are confounders.
The test measures every decoded frame, rather than asserting that a prompt always
produces a particular amount of degradation.

## Reproduce

From the repository root in PowerShell (use a fresh output directory):

```powershell
$model = 'D:/Projects/weights/fcffec69b0bbb0c9-Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors'
$run = 'output/lcs-drift-new'
New-Item -ItemType Directory -Force $run
build/Release/slopfab.exe encode-text --prompt-file tests/data/lcs_drift_prompt.txt `
  --text-encoder weights/text_encoder/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors `
  --output "$run/prompt.safetensors"
python tools/lcs_drift.py generate --transformer $model --embedding "$run/prompt.safetensors" `
  --output $run --name baseline --windows 5
python tools/lcs_drift.py analyze "$run/baseline-04.mp4" --output "$run/baseline-metrics"
python tools/lcs_drift.py calibrate --output "$run/color.safetensors"
python tools/lcs_drift.py fit --calibration "$run/color.safetensors" `
  --early "$run/baseline-00.safetensors" --late "$run/baseline-04.safetensors" `
  --output "$run/drift.safetensors"

# Hold the continuation source and seed constant; neither result enters fitting.
python tools/lcs_drift.py generate --transformer $model --embedding "$run/prompt.safetensors" `
  --output $run --source "$run/baseline-04.safetensors" --name holdout-baseline `
  --seed 424247 --windows 1
python tools/lcs_drift.py generate --transformer $model --embedding "$run/prompt.safetensors" `
  --output $run --source "$run/baseline-04.safetensors" --name holdout-corrected `
  --seed 424247 --windows 1 --lcs "$run/drift.safetensors" `
  --lcs-settings-file "$run/drift.settings.json"
python tools/lcs_drift.py analyze "$run/holdout-baseline-00.mp4" --output "$run/holdout-baseline-metrics"
python tools/lcs_drift.py analyze "$run/holdout-corrected-00.mp4" --output "$run/holdout-corrected-metrics"
```

Defaults: 384x384, 24 fps, 73 initial frames, four 68-frame extensions (345
frames, 14.375 seconds), 22-frame overlap, 25 Euler grid points / 24 updates,
Flash2 attention, seed 424242 incremented for each window. No LoRA, step cache,
Keystone, locked overlap or reference-image conditioning is used. Video VAE is
`weights/vae/minimax_h3_video_vae_fp16.safetensors`; audio VAE is
`weights/vae/minimax_h3_audio_vae_fp32.safetensors`. The text embedding is real
Qwen3-VL-32B conditioning, not a synthetic smoke-test embedding.

If the native build emits Y4M/WAV because FFmpeg support is disabled, the harness
muxes an MP4 with external FFmpeg. For numeric comparisons, prefer the Y4M files
when available; lossy MP4 compression changes the texture metric slightly.

`analyze` writes a CSV row for every frame, a contact sheet, and first/last
22-frame summaries. It measures the center half of the image in each dimension:
luminance percentiles, pixels above 0.8 luminance, the 99th-minus-50th percentile,
and RMS of a discrete Laplacian. Luminance here is weighted display RGB, not linear
scene radiance. These are appearance proxies, not a trained material classifier.

`fit` uses the first clip's six latent slices after its boundary slice and the
last clip's final six slices. It subtracts their spatial/temporal means, removes
the calibrated color projection, then normalizes the remaining direction. The
reference is the early mean projection, with tolerance twice the standard
deviation of those early frame means (minimum 0.01). It saves the fit values and
input paths alongside the archive. This is distribution matching; it does not
assume corresponding patches follow the same facial feature.

The fit also writes `.settings.json`: highlight strength 0.5, knee equal to the
early samples' 90th-percentile calibrated lightness, and start update 16. It
compresses the highlight tail while preserving the other latent coordinates.
This fitted shoulder is independent of the color-orthogonal mean-shift direction.
Omitting `--lcs-settings-file` selects the initial mean-shift experiment instead:
drift strength 0.5, cap 1 raw VAE unit, and updates 16–23.
The archived latents are corrected before decoding and feed subsequent
continuations. It is not a pixel-space filter.

## Checks

```powershell
$env:SLOPFAB_TEST_FILTER = 'lcs'
build/Release/slopfab_tests.exe
Remove-Item Env:SLOPFAB_TEST_FILTER
python tests/test_lcs_drift.py
```

Native tests check normalization, one-sided thresholds, correction caps, masks,
step/audio exclusion, serialization and color/residual preservation. Python tests
recover a known shift with a deliberately misleading color change and boundary
slice, and check that the metrics distinguish exposure from texture.
