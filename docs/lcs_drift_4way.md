# Four-way two-minute H3 continuation comparison

This adds two chains to the [unanchored two-minute test](lcs_drift_120s.md):

| Quadrant | Keystone | Capstone | Locked overlap | LCS |
| --- | --- | --- | --- | --- |
| Top left: baseline | Off | Off | Off | Off |
| Top right: LCS | Off | Off | Off | Color + edge |
| Bottom left: anchored | On | On | On | Off |
| Bottom right: anchored + LCS | On | On | On | Color + edge |

## Completed results

**The continuation anchors provide the dominant improvement in this scene.**
Both anchored two-minute clips retain much more of the original skin color and
texture, without the severe growing dot pattern and plastic-looking surfaces of
the unanchored clips. Adding LCS reduces average chroma drift further, although
the extra benefit varies with time and is small at the final endpoint.

- [Full four-way video](../output/lcs-drift-120s-anchors/comparison-4way-120s.mp4):
  exactly 120 seconds / 2,880 frames at 24 fps, with the layout above.
- [Baseline versus anchors](../output/lcs-drift-120s-anchors/baseline-vs-anchors-120s.mp4):
  baseline left, Keystone + Capstone + locked continuation right, without LCS
  in either arm. Reuses the existing decodes; verified 120 seconds / 2,880 frames
  at 24 fps, 768x416 with labels above both full images.
- [Timeline figure](../output/lcs-drift-120s-anchors/fourway-timeline.png) and
  [final frame](../output/lcs-drift-120s-anchors/final-frame-4way.png).
- Measurements, source manifests, checkpoints and verification records are in
  `output/lcs-drift-120s-anchors/`. `fourway-metrics.json` summarizes the endpoint;
  `fourway-intervals.csv` includes 30/60/90/120-second checkpoints.

Final five-second means (frames 2,760-2,879):

| Arm | Chroma distance from start | Bright-edge coverage | Broad highlights | Texture RMS |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 16.750 | 5.581% | 4.271% | 0.0584 |
| LCS | 10.573 | 7.642% | 10.224% | 0.0954 |
| Anchors | 1.792 | 1.042% | 2.003% | 0.0383 |
| Anchors + LCS | 1.806 | 0.989% | 2.330% | 0.0382 |

This four-way table uses each clip's independent midtone mask. Coverage is
79.1%, 84.5%, 86.6%, and 86.4%, respectively. For the anchored pair, the shared-mask
check gives final chroma distances 1.702 / 1.688, with 83.4% common coverage.
That is essentially equal final color retention. Adding LCS lowers final
bright-edge coverage by 5.1%, while broad highlight coverage rises by 0.327
percentage points (16.3% relative).

Across **all generated frames in the 120-second export**, excluding the shared
original 73 frames, anchored shared-mask chroma distance averages 2.332 without
LCS versus 1.904 with LCS: **18.4% lower**. Independent-mask distance agrees
(2.456 / 2.009). Bright-edge coverage averages 0.995% / 0.966%, and broad
highlights average 1.990% / 2.025%. Thus LCS has a modest average benefit on
these proxies, while the anchors account for the much larger visual difference.
`whole-continuation-summary.json` records these 2,807-frame averages.

The anchored clips have small fluctuations rather than cumulative growth on the
tested proxies. This is one portrait, one seed sequence and one resolution.
The clips differ in head pose, framing and motion; these fixed-crop measurements
do not track faces or establish semantic correspondence. Anchors may constrain
motion and composition, and this experiment does not quantify that tradeoff.
The three continuation options were enabled together, so it does not isolate
their individual contributions.

The two new arms completed all 84 windows / 2,016 denoising evaluations, with
finite saved latents, exact retained video/audio prefixes, and correct source
links for all handoffs. Every LCS request used the fixed correction settings,
and each next LCS request loaded its own saved corrected archive. Both final
archives contain 2,929 frames. All ten milestone decodes succeeded; watched
checkpoint/reference/settings files remained unchanged. `validation.json`
records these checks, and the export's `.command.json` / `.probe.json` record
the four input paths, labels, frame count and duration. GPU generation is
finished. Production LCS code was unchanged during this comparison.

## Controlled settings

The existing LCS is kept identical in both LCS arms: `color_stability=0.75`,
`edge_stability=0.75`, `color_max_shift=0.1`. No reference is refitted. The two
new arms start from the same original 73-frame archive, not from already drifted
30-second outputs. Their 42 seeds are 424243 through 424284, matching the older
chains. Each extension adds 68 frames with 22-frame overlap. Final archives have
2,929 frames; comparisons use the first 2,880 frames / exactly 120 seconds.

The fixed Keystone/Capstone source is the original 73-frame clip. Capstone uses
the default four-video-update delay. Locked overlap is explicitly enabled in
each request. The native Keystone constraint already incorporates the overlap
constraint, substitutes the first locked video slice with the original anchor,
and retains the advancing source for the remaining motion/audio context. Rear
Capstone is model-call context, not a frame appended to the saved timeline.

## Verify correction before the long chain

The native `lcs_stability_covers_42_saved_keystone_continuations` test checks the
same temporal geometry for 42 joins, with Keystone, Capstone, locked context,
output correction, save/reload, and reuse of the corrected source. All 840 new
latent slices, including each extension's first and last slice, change under
the controlled drift fixture. Retained source frames and audio are preserved;
the fixed original anchor is not modified. Sampling step limits do not truncate
this output correction. Existing global-phase and mask tests also pass.

The production path applies the output correction after joining and before
archive saving, the completed-latent hook, and decoding. The five-slice seam ramp
is 20%, 40%, 60%, 80%, 100%; the rest of each extension uses full configured
strength. The original 73 frames remain the shared approved reference. Every
subsequent segment is corrected once, then preserved in later continuations.

The real run additionally gates continuation on `--verify-first-lcs`: the first
pair has identical source/seed/anchor settings, and must show finite changes in
every new video latent slice with identical audio and retained prefixes. Later
pairs use their own saved archives; each source path is recorded in progress.

Preflight passed: 48,229 LCS checks, 3,529 Keystone checks (including the new
chain test), 38 Capstone checks, and two Python harness tests. The first real
pair passed too: all 20 new latent slices changed, with exact retained-prefix
and audio agreement. Per-slice maximum changes are in
`output/lcs-drift-120s-anchors/first-correction-check.json`.

## Reproduce

```powershell
python tools/lcs_long_run.py `
  --transformer 'D:/Projects/weights/fcffec69b0bbb0c9-Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors' `
  --embedding output/lcs-drift/prompt.safetensors `
  --baseline-source output/lcs-drift/baseline-00.safetensors `
  --corrected-source output/lcs-drift/baseline-00.safetensors `
  --keystone-source output/lcs-drift/baseline-00.safetensors `
  --capstone-lock-after 4 --lock-overlap --verify-first-lcs `
  --calibration output/lcs-drift-30s/stability.safetensors `
  --settings output/lcs-drift-30s/stability.settings.json `
  --output output/lcs-drift-120s-anchors --pairs 42 --seed 424243 --decode-every 10
```

Inside this new directory, `baseline-*` denotes anchored without LCS and
`corrected-*` denotes anchored with LCS. The previous two clips remain in
`output/lcs-drift-120s/`. Generation uses the same checkpoint, prompt embedding,
384x384 canvas, 24 fps, Flash2, 25 Euler grid points / 24 updates, and CUDA backend
as before. Deferred decoding and retained-prefix checks follow the earlier test.

Export the four complete decodes with labels above each image:

```powershell
python tools/lcs_fourway.py `
  --baseline output/lcs-drift-120s/baseline-31.y4m `
  --lcs output/lcs-drift-120s/corrected-31.y4m `
  --anchored output/lcs-drift-120s-anchors/baseline-41.y4m `
  --anchored-lcs output/lcs-drift-120s-anchors/corrected-41.y4m `
  --output output/lcs-drift-120s-anchors/comparison-4way-120s.mp4
```

The export is silent, 768x832, and includes all four full 384x384 images with
32-pixel label bars. The helper checks input length, geometry, and frame rate,
then verifies the output frame count and duration and saves its command/probe.

## Intermediate observations

The first 30-second four-way preview is
`output/lcs-drift-120s-anchors/comparison-4way-30s.mp4` (720 frames, 24 fps).
Both anchored arms retain much more of the original skin texture and color than
the unanchored arms. Head pose also differs, so fixed-crop numbers are proxies.
Over seconds 25-30, anchored without/with LCS has independent-mask chroma distance
1.556 / 1.360, bright-edge coverage 0.919% / 0.901%, and broad highlights
1.708% / 2.009%. The anchored-pair shared-mask check gives chroma distances
1.468 / 1.291 (12.1% lower with LCS), with 82.5% common crop coverage.
These are early results; they do not establish two-minute stability.

At 59.71 seconds, both anchored contact sheets still retain natural-looking
texture without the unanchored clips' growing dot pattern. In the last five
seconds, anchored without/with LCS has independent chroma distance 2.192 / 2.996,
bright-edge coverage 0.978% / 0.832%, and highlights 1.585% / 1.801%. Shared-mask
chroma distance is 1.942 / 2.889, with 71.7% coverage: the extra color benefit
from LCS is not consistent at this checkpoint. Unanchored without/with LCS at
the same frames has independent chroma distance 12.664 / 7.949, bright edges
1.997% / 3.146%, and highlights 6.996% / 15.533%. Poses differ, so these proxy
comparisons accompany, rather than replace, visual inspection.

At 88.04 seconds, both anchored contact sheets remain close to the initial
appearance, without the severe dot-pattern growth of the unanchored chains.
Anchored without/with LCS has shared-mask chroma distance 1.671 / 1.529 (8.5%
lower), bright-edge coverage 1.006% / 0.915% (9.0% lower), and highlights
1.919% / 1.931%. Common-mask coverage is 84.1%. The incremental LCS effect is
modest relative to the effect of enabling the continuation anchors.
