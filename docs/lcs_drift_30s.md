# Persistent color and edge correction: 30-second H3 experiment

This extends the [initial drift experiment](lcs_drift_experiment.md) with two
independent native controls: `color_stability` and `edge_stability`. Both are off
by default and work on completed new latents, before saving and VAE decoding.
The saved correction becomes the next continuation's input. They do not edit the
retained source or audio, or change the current window's sampling trajectory.
Later trajectories can still differ because the continuation source has changed.

## Recipe

The experiment uses the user's
`D:/Projects/weights/fcffec69b0bbb0c9-Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors`,
the existing portrait prompt, CUDA / RTX 5090, 384x384, 24 fps, Flash2,
25 Euler grid points (24 updates), and seeds starting at 424242. The original
73-frame clip is followed by ten 68-frame extensions with 22-frame overlap.
This produces 753 frames / 31.375 seconds on H3's temporal grid. Comparison media
and the final 30-second metrics use **only the first 720 frames**.

The reference is fitted only to the approved original 73-frame clip. It stays
fixed throughout the chain; neither baseline drift nor candidate output is used
to update it. Settings tested are:

```json
{"color_stability":0.75,"edge_stability":0.75,"color_max_shift":0.1}
```

No highlight shoulder, residual drift direction, brightness/contrast adjustment,
LoRA, Keystone, overlap lock, or sampling cache is enabled. `start_step` and
`end_step` do not govern these two output controls.

Fit details and invariants are in [native LCS documentation](lcs.md#persistent-color-and-edge-reference).
The original chroma distribution is recorded per temporal phase to avoid
mistaking H3's periodic latent statistics for a color shift. The edge limit is a
one-sided cap on positive local lightness contrast. Both edits preserve the
color-orthogonal residual; the color control also preserves calibrated lightness.
Decoded appearance is not guaranteed to preserve those properties exactly.

## Reproduce

From the repository root, using a fresh output directory:

```powershell
$model = 'D:/Projects/weights/fcffec69b0bbb0c9-Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors'
$run = 'output/lcs-drift-30s-new'
New-Item -ItemType Directory -Force $run
build/Release/slopfab.exe encode-text --prompt-file tests/data/lcs_drift_prompt.txt `
  --text-encoder weights/text_encoder/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors `
  --output "$run/prompt.safetensors"
python tools/lcs_drift.py generate --transformer $model --embedding "$run/prompt.safetensors" `
  --output $run --name baseline --windows 11
python tools/lcs_drift.py calibrate --output "$run/color.safetensors"
python tools/lcs_drift.py fit-stability --calibration "$run/color.safetensors" `
  --reference "$run/baseline-00.safetensors" --output "$run/stability.safetensors"
python tools/lcs_drift.py generate --transformer $model --embedding "$run/prompt.safetensors" `
  --output $run --source "$run/baseline-00.safetensors" --name corrected `
  --seed 424243 --windows 10 --lcs "$run/stability.safetensors" `
  --lcs-settings-file "$run/stability.settings.json"
python tools/lcs_drift.py analyze "$run/baseline-10.y4m" --max-frames 720 `
  --output "$run/baseline-metrics" --stride 48
python tools/lcs_drift.py analyze "$run/corrected-09.y4m" --max-frames 720 `
  --output "$run/corrected-metrics" --stride 48
python tools/lcs_drift.py compare --baseline "$run/baseline-metrics.csv" `
  --candidate "$run/corrected-metrics.csv" --start-frame 73 --output "$run/paired"
ffmpeg -v error -n -i "$run/baseline-10.y4m" -i "$run/corrected-09.y4m" `
  -filter_complex '[0:v][1:v]hstack=inputs=2[v]' -map '[v]' -frames:v 720 `
  -c:v libx264 -crf 16 -pix_fmt yuv420p "$run/comparison-30s.mp4"
```

Use `.mp4` instead of `.y4m` for builds that emit MP4 directly; numeric texture
and edge comparisons are more reliable on uncompressed Y4M. The analysis scripts
need NumPy and Pillow; paired SVG plotting additionally needs Matplotlib.

## Measurements and interpretation

Every frame has a CSV row. In addition to the earlier luminance and texture
proxies, the script measures median CIELAB `a*` / `b*` in the center crop's
nonneutral midtones (display luminance 0.15–0.8 and RGB range above 0.04). The
chroma distance is Euclidean distance from the original first 22 frames' mean
median `a*` / `b*`. This is a skin-color **proxy** for this portrait, not semantic
skin segmentation or full perceptual Delta E. Selection coverage is reported
because pose, shadow, hair and background can change the selected population.

The bright-edge proxy is positive luminance excess over an 11x11 box mean. Both
its 99th percentile and fraction above 0.08 are recorded. It includes legitimate
boundaries, highlights and texture as well as halos. It does not measure halo
width directly. Contact sheets and the full video must accompany the numbers.
Summaries include the last 120 frames (five seconds at the fixture's 24 fps), as
well as the first/last 22 frames, to expose sensitivity to the endpoint pose.

## Implementation checks

Focused native tests cover output normalization, caps, retained-prefix
preservation, temporal phase alignment across nonmultiple-of-five joins, masks,
color/lightness separation, residual preservation, and serialization. CLI checks
cover missing-reference rejection and ranges. Python tests check phase-wise
reference fitting, color-shift and bright-edge metrics, and experiment manifest
preservation. Existing continuation, sampler and C API suites also pass.

On the first real corrected extension, retained source video was bit-exact,
audio matched the uncorrected same-seed extension exactly, and the largest change
outside the LCS color subspace was 4.51e-7 in raw VAE units. This verifies that the
new pass is operating on the intended latent coordinates; it is not a claim of
perceptual invariance.
