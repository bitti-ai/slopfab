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
build/Release/slopfab.exe conditioning encode-text --prompt-file tests/data/lcs_drift_prompt.txt `
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
python tools/lcs_drift.py compare-colors --baseline "$run/baseline-10.y4m" `
  --candidate "$run/corrected-09.y4m" --max-frames 720 --output "$run/common-colors"
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
`compare-colors` additionally evaluates both videos on the **intersection** of
their midtone masks at each frame. This controls the differing selected-pixel
populations. It does not align head motion or establish semantic correspondence.

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

## Recorded results, 2026-10-31

Artifacts are in `output/lcs-drift-30s/`. The baseline reuses the unchanged
413-frame archive from the earlier experiment, then `baseline-00` through
`baseline-04` extend it to 753 frames. The corrected chain starts from the same
original 73 frames: `corrected-first-00`, then `corrected-00` through
`corrected-08`, also ending at 753 frames. Window seeds match baseline. The fitted
settings above were fixed before running the chain; no later output was used in
the fit. Logs and command manifests preserve every generation command.

`comparison-30s.mp4` contains exactly 720 frames at 24 fps (30.000 seconds),
baseline left and corrected right. `paired-30s.csv` / `.svg` show every new frame.
`common-color-30s.csv` / `.json` contain the common-mask color check.

The last five seconds (frames 600–719) average as follows:

| Measurement | Baseline | Corrected | Change |
| --- | ---: | ---: | ---: |
| Chroma distance, shared midtone pixels | 7.971 | 5.637 | **29.3% lower** |
| Chroma distance, independent midtone masks | 8.064 | 6.044 | 25.1% lower |
| Bright-edge fraction, positive local contrast >0.08 | 1.192% | 0.884% | **25.9% lower** |
| Positive local contrast, 99th percentile | 0.0848 | 0.0764 | 10.0% lower |
| Broad highlight fraction, luminance >0.8 | 7.064% | 7.606% | 7.7% higher |
| Crop mean luminance | 0.3061 | 0.3469 | 13.3% higher |
| Laplacian RMS texture proxy | 0.0340 | 0.0325 | 4.5% lower |
| Independent midtone mask coverage | 66.2% | 77.4% | Different populations |

The shared mask covers 60.2% of the center crop for both videos. It confirms that
the measured color improvement is not solely due to selecting different pixels.
The final 22-frame shared-mask color distances are 8.746 versus 5.897, consistent
with the five-second result. At the earlier 17-second checkpoint, independent
color distance was 3.441 versus 2.868 and bright-edge fraction 2.478% versus
2.032%, with similar crop luminance and coverage.

This is a useful improvement on the tested color/edge proxies, **not a complete
material-quality fix**. Skin still shifts from the starting color and becomes
shinier; broad highlight coverage did not improve. The changed source latents
also change later motion, exposure and the population in a fixed crop. The
edge proxy includes legitimate texture/boundaries, so its reduction does not
prove that only halos were removed. The unchanged color-orthogonal latent
residual does not guarantee unchanged decoded texture. The controls remain
opt-in and should be checked on additional scenes, checkpoints and resolutions.

All ten corrected hand-offs preserved the preceding video and audio latent
prefixes exactly (`prefix-checks.json`). Validation passed: 45,328 focused native
LCS checks; 9,758,717 full native synthetic checks with one unavailable FFmpeg
fixture skipped; five Python experiment tests; and the LCS CLI, continuation CLI,
sampler CLI and C API suites. A same-video common-mask comparison returned exact
agreement. CUDA was evaluated; Vulkan uses the shared CPU correction but its
generation quality was not tested here.
