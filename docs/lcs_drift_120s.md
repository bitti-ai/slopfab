# Two-minute chained H3 drift experiment

This extends both arms of the [30-second experiment](lcs_drift_30s.md). The
checkpoint, prompt embedding, VAE, sampler, resolution, overlap, seed sequence
and fixed original-clip reference remain unchanged. No correction is refitted
or strengthened during the test.

The starting archives each contain 753 frames (31.375 seconds). Another 32
68-frame windows per arm yield 2,929 frames (122.042 seconds). Comparison videos
and headline measurements use exactly the first **2,880 frames / 120 seconds**.
There are 42 continuation hand-offs since the original 73-frame clip.

## Completed results, 2026-10-31

**The current LCS does not prevent long-run degradation.** It retains more of
the starting chroma, but both clips develop severe patterned facial texture and
plastic-looking highlights. In this run, the corrected clip has higher measured
bright-edge, highlight and texture values than baseline by the end. The useful
30-second edge result does not hold over two minutes.

Artifacts in `output/lcs-drift-120s/`:

- [120-second comparison](../output/lcs-drift-120s/comparison-120s.mp4): baseline
  left, LCS right; exactly 2,880 frames at 24 fps, verified with FFprobe.
- [Timeline figure](../output/lcs-drift-120s/timeline.png), also available as SVG.
- [Final frame](../output/lcs-drift-120s/final-frame.png) and contact sheets
  `baseline-metrics.jpg` / `corrected-metrics.jpg` sampled every ten seconds.
- Per-frame `baseline-metrics.csv`, `corrected-metrics.csv`, `paired.csv` / `.svg`,
  and `common-colors.csv`; `interval-summary.csv` / `.json` hold interval means.
- `manifest.json`, `progress.json`, `decoded.json`, `validation.json` and
  `comparison-probe.json` preserve the recipe and completion checks.

The final five seconds (frames 2,760-2,879) average as follows:

| Measurement | Baseline | LCS | Relative change |
| --- | ---: | ---: | ---: |
| Shared-mask chroma distance from start | 16.780 | 10.455 | 37.7% lower |
| Independent-mask chroma distance | 16.750 | 10.573 | 36.9% lower |
| Bright-edge fraction, local contrast >0.08 | 5.581% | 7.642% | 36.9% higher |
| Positive local contrast, 99th percentile | 0.1572 | 0.1916 | 21.9% higher |
| Broad highlight fraction, luminance >0.8 | 4.271% | 10.224% | 139.4% higher |
| Laplacian RMS texture proxy | 0.0584 | 0.0954 | 63.2% higher |
| Crop mean luminance | 0.3643 | 0.4724 | 29.7% higher |
| Independent midtone-mask coverage | 79.1% | 84.5% | Different populations |

Shared-mask coverage is 66.0% of the center crop. For context, the original first
22 frames have 1.029% bright-edge coverage and 0.796% broad highlight coverage.
Both arms drift substantially from that starting appearance. The LCS color
distance itself also grows from about 5.6 at 30 seconds to 10.5 at 120 seconds;
the relative advantage does not mean that color is stable.

The table below uses the five seconds preceding each listed time. Each cell is
**baseline / LCS**, measured from the final complete decodes.

| Clip time | Shared-mask chroma distance | Bright-edge coverage | Broad highlights |
| --- | ---: | ---: | ---: |
| 15 s | 2.815 / 2.270 | 1.442% / 1.188% | 4.558% / 4.226% |
| 30 s | 7.967 / 5.632 | 1.192% / 0.884% | 7.049% / 7.588% |
| 45 s | 10.283 / 5.414 | 1.653% / 2.003% | 4.982% / 13.055% |
| 60 s | 12.952 / 7.903 | 2.004% / 3.222% | 7.013% / 16.812% |
| 75 s | 14.318 / 9.347 | 2.936% / 5.063% | 3.996% / 26.054% |
| 90 s | 15.948 / 10.337 | 4.222% / 4.568% | 5.559% / 9.936% |
| 105 s | 16.662 / 11.290 | 4.499% / 6.101% | 2.569% / 9.153% |
| 120 s | 16.780 / 10.455 | 5.581% / 7.642% | 4.271% / 10.224% |

Color drift grows and then flattens somewhat late in the clip. Bright-edge
coverage generally increases, with motion-related fluctuations. Highlights
oscillate markedly rather than growing monotonically. The visual failure is
more than an exposure shift: initially fine freckles grow into large, strongly
contrasted dots, with glossy cheek/nose patches and altered lip/eyelid color.

This is one scene and one matched seed sequence, with diverging poses and
lighting. The fixed-crop metrics cannot isolate material changes from motion,
and the comparison does not separate the color and edge controls' effects.
It establishes a long-duration failure case for the current combined settings,
not a general causal claim about either control. No settings were adjusted after
seeing the intermediate results. A subsequent diagnostic should separate color
and edge controls and address the growing texture pattern, rather than treating
chroma distance alone as the success criterion.

All 64 additional latent windows completed 24 denoising evaluations each,
contained finite values, and preserved retained video and stereo audio prefixes
exactly. Both final archives contain 2,929 frames; all eight milestone decodes
succeeded. Watched model/reference/settings inputs remained unchanged. Sampling
and archive validation took about 46 minutes, plus decoding and analysis.
The new CPU hand-off validation test passed, as did the pre-run baseline/LCS
latent parity checks described below. Native production LCS code was unchanged
for this experiment.

## Reproduce

Use the existing 30-second artifacts, or first reproduce that experiment.
`tools/lcs_long_run.py` uses the local DLL, NumPy, and `tools/lcs_drift.py`:

```powershell
python tools/lcs_long_run.py `
  --transformer 'D:/Projects/weights/fcffec69b0bbb0c9-Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors' `
  --embedding output/lcs-drift/prompt.safetensors `
  --baseline-source output/lcs-drift-30s/baseline-04.safetensors `
  --corrected-source output/lcs-drift-30s/corrected-08.safetensors `
  --calibration output/lcs-drift-30s/stability.safetensors `
  --settings output/lcs-drift-30s/stability.settings.json `
  --output output/lcs-drift-120s --pairs 32 --seed 424253 --decode-every 8
```

Use a fresh output folder. Each pair uses the same seed in baseline and corrected
requests. Settings are copied into `manifest.json`; input file size/mtime changes
abort the experiment. Progress and decode manifests are written atomically.
Saved archives can be used as sources for a resumed run in a fresh folder, using
the next seed and remaining pair count.

Recipe: CUDA / RTX 5090, 384x384, 24 fps, Flash2, 25 Euler grid points / 24 model
evaluations, 22-frame continuation overlap, no LoRA/cache/Keystone/locked overlap.
The corrected arm uses `color_stability=0.75`, `edge_stability=0.75`, and
`color_max_shift=0.1`. The reference was fitted only to the original 73 frames.

## Deferred decoding

Repeatedly decoding an ever-longer prefix adds substantial work without changing
the next generation. The harness therefore completes all denoising updates,
output correction, joining, and atomic latent saving, then uses the public
progress callback to cancel at the **start of video decoding**. It accepts only
that specific cancellation, with all expected denoise progress indices observed,
a finite saved archive, the expected frame count, and exact retained video and
audio latent prefixes. Earlier cancellation or any other failure aborts the run.
The cancellation status is recorded explicitly; it is not presented as a
successful decoded generation.

Every eight pairs, both complete archives are decoded through the standard
`slopfab video decode` command. This produces checkpoints near 54, 77, 99 and 122
seconds. The final video is decoded from the complete cumulative latents, with
normal VAE temporal context; independently decoded windows are not spliced.
Comparison videos are silent, while the saved archives retain generated audio.

Before the long run, both baseline and corrected 141-frame continuations were
repeated and compared with the earlier normal-generation archives. Video and
audio latents were bit-exact. Standalone Y4M decoding was very close but not
bit-exact to the earlier generation decoder output: maximum byte difference 2,
mean absolute byte difference about 0.018. Both arms of this experiment use the
same standalone decoder. Full verification is recorded in
`output/lcs-drift-120s-verified/parity.json`.

## Analysis

```powershell
python tools/lcs_drift.py analyze output/lcs-drift-120s/baseline-31.y4m `
  --max-frames 2880 --stride 240 --output output/lcs-drift-120s/baseline-metrics
python tools/lcs_drift.py analyze output/lcs-drift-120s/corrected-31.y4m `
  --max-frames 2880 --stride 240 --output output/lcs-drift-120s/corrected-metrics
python tools/lcs_drift.py compare --baseline output/lcs-drift-120s/baseline-metrics.csv `
  --candidate output/lcs-drift-120s/corrected-metrics.csv --start-frame 73 `
  --output output/lcs-drift-120s/paired
python tools/lcs_drift.py compare-colors --baseline output/lcs-drift-120s/baseline-31.y4m `
  --candidate output/lcs-drift-120s/corrected-31.y4m --max-frames 2880 `
  --output output/lcs-drift-120s/common-colors
```

The same fixed-crop midtone chroma, positive local-contrast, luminance, highlight,
and texture proxies are used as in the 30-second test. Metrics do not recognize
skin semantically or align head motion. Visual inspection and selected-pixel
coverage are necessary, especially if appearance or framing changes dramatically.
Interval summaries should accompany endpoint measurements to show whether drift
is gradual, oscillating or accelerating.

CPU hand-off validation checks can be run with
`python tests/test_lcs_long_run.py`. They reject changed video/audio prefixes,
unexpected frame counts and non-finite saved latents.

## Milestone observations

At 54.04 seconds (`baseline-07` and `corrected-07`), the last five seconds show
shared-mask chroma distance of 12.569 versus 7.048 (43.9% lower with correction).
The bright-edge fraction is 2.237% versus 2.425% (8.4% higher), so the earlier
edge benefit does not persist consistently. Broad highlight coverage is 7.348%
versus 9.113%. Both contact sheets show stronger freckles and glossy facial
surfaces relative to the original. The corrected clip also changes its head
pose/framing; color-mask coverage differs substantially. These are observations
from this scene, not evidence of universally stable skin rendering.

At 76.71 seconds (`baseline-15` and `corrected-15`), shared-mask chroma distance
is 14.184 versus 9.423 (33.6% lower), but bright-edge coverage is 3.174% versus
4.671% (47.2% higher), and broad highlight coverage is 6.740% versus 21.129%.
Texture RMS is 0.0384 versus 0.0916. Contact sheets show growing speckled facial
texture in both arms and stronger glossy patches in the corrected arm. The
current correction is not maintaining stable material appearance over long
chains, even while the measured chroma remains closer to the starting clip.

At 99.38 seconds (`baseline-23` and `corrected-23`), shared-mask chroma distance
is 16.340 versus 10.897 (33.3% lower). Bright-edge coverage is 3.712% versus
5.986% (61.3% higher); broad highlights cover 2.431% versus 16.708% of the crop.
Both clips have severe speckled texture and glossy facial surfaces. Baseline
color continues losing warmth; correction retains more color but does not
prevent the long-run texture/material failure.
