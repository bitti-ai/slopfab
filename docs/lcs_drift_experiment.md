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
python tools/lcs_drift.py compare --baseline "$run/holdout-baseline-metrics.csv" `
  --candidate "$run/holdout-corrected-metrics.csv" --start-frame 345 --output "$run/paired"
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
The optional `compare` command requires Matplotlib and writes paired per-frame
values/deltas plus SVG curves; use the new-section start frame to exclude the
retained source from aggregate interpretation.

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
python tests/test_lcs_cli.py build/Release/slopfab.exe
```

Native tests check normalization, one-sided thresholds, correction caps, masks,
step/audio exclusion, serialization and color/residual preservation. Python tests
recover a known shift with a deliberately misleading color change and boundary
slice, and check that the metrics distinguish exposure from texture.

## Measured run, 2026-10-31

Artifacts are under `output/lcs-drift/` (ignored by Git). The transformer is the
exact Singularity INT8 path above. Generation used CUDA on an RTX 5090. The
original is `baseline-00`; `continuation-00` through `continuation-03` extend it
to 345 frames. The matched-source comparison extends `continuation-03` with seed
424247, producing 413 frames. All numbers below use uncompressed Y4M, the fixed
center crop, and averages of 22 decoded frames. The correction was fitted before
generating this extra continuation, but its strengths were explored on that
continuation, so this is a development experiment rather than an unbiased test
of a finished method.

| Section / intervention | Pixels above 0.8 | Mean luminance | P99 minus median |
| --- | ---: | ---: | ---: |
| Original first 22 frames | 0.799% | 0.4641 | 0.2980 |
| Baseline 345-frame clip, last 22 | 3.324% | 0.4596 | 0.3604 |
| Extra continuation, no correction | 5.685% | 0.4577 | 0.4044 |
| Residual direction, strength 0.5 / cap 1 / start 16 | 5.674% | 0.4585 | 0.4032 |
| Residual direction, strength 1 / cap 10 / start 8 | 5.553% | 0.4591 | 0.3985 |
| Highlight shoulder, strength 0.5 / start 16 | 5.488% | 0.4568 | 0.3988 |

The fitted knee was 0.780911 in LCS lightness. The raw residual shift norm was
5.863734; its reference was -5.481375 with tolerance 3.949553. The first
mean-shift approach had little visual effect. The shoulder reduced the highlight
coverage proxy by about 3.5% relative to the paired baseline, with about 0.2%
lower mean luminance. This is a modest change, not evidence that plastic skin or
lost texture has been fixed. The texture proxy actually rises in the extra
continuation; freckles, facial edges, and pose changes make it unsuitable as a
standalone detail-quality score.

The contact sheets show broader cheek highlights in later frames, including
frames where the head returns toward its initial angle. Pose and blinking are
not identical. `holdout-comparison.mp4` shows the extra 68 frames, baseline left
and shoulder right. `paired-shoulder.csv` and `.svg` contain frame comparisons.
Every matched-source variant preserved the entire source video and audio latent
prefix exactly (`prefix-checks.json`).

A separate prevention trial starts from the original 73-frame archive and applies
strength 1, the same knee, and start update 0 to **all five** subsequent windows.
It is saved as `stabilized-00` through `stabilized-04`; the name identifies the
trial, not a quality guarantee. `chain-comparison.mp4` shows the uncorrected chain
on the left and this trial on the right. `paired-chain.csv` / `.svg` cover every
new frame.

| Ending at frame count | Baseline highlight coverage | Strong shoulder coverage |
| --- | ---: | ---: |
| 141 | 3.239% | 2.187% |
| 209 | 1.135% | 1.687% |
| 345 | 3.324% | 4.476% |
| 413 | 5.685% | 3.932% |

These are again the last 22 frames in each window. The strong intervention
changes the later head motion, so the same crop covers different portions of
face, hair and background. At 413 frames its crop mean is 0.3001 versus 0.4577
for baseline, and P99-minus-median is 0.6443 versus 0.4044. The lower highlight
coverage at that endpoint therefore **does not establish better material quality**.
The prevention trial is inconsistent and does not support using maximum strength
throughout sampling as a general fix. The conservative fitted settings remain an
opt-in experiment; validation on additional subjects, seeds, resolutions and
pose-aligned skin regions is still needed.

Reproduce that separate trial with a new name:

```powershell
@{highlight_strength=1; highlight_knee=0.7809110144; start_step=0} |
  ConvertTo-Json | Set-Content "$run/strong.json"
python tools/lcs_drift.py generate --transformer $model --embedding "$run/prompt.safetensors" `
  --output $run --source "$run/baseline-00.safetensors" --name strong-chain `
  --seed 424243 --windows 5 --lcs "$run/drift.safetensors" `
  --lcs-settings-file "$run/strong.json"
```

Validation for the implementation: 23,119 native LCS checks passed, the three
Python experiment tests and the model-free CLI checks passed, and the existing
sampler/continuation CLI and C API suites passed. The wider native synthetic suite
also passed before the highlight-shoulder addition (9,736,217 checks; one FFmpeg
fixture skipped). The final shoulder addition was covered by the focused native
LCS suite rather than claiming another full-suite run.
The final CUDA DLL smoke test with the requested Singularity checkpoint also
passed native calibration, callback ordering, cancellation, and non-finite-output
rejection (`output/lcs-drift/dll-test.log`). Vulkan quality was not evaluated.
