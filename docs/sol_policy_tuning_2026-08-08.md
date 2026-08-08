# Experimental Sol policy tuning — 2026-08-08

This tuning used only the predeclared detail, motion, and dialogue scenarios at
seeds 11 and 29. Identity, scene-change, and music remained untouched holdouts.
The acceptance thresholds are those in `sol_quality_matrix_2026-08-08.md`.

On the representative 37,715-token step-10/layer-2 capture:

| beta | Sol ms | dense ms | ratio | exact routes | rel-L2 | cosine |
|---:|---:|---:|---:|---:|---:|---:|
| 1.00 | 164.112 | 232.117 | 1.414x | 19.0% | .0416846 | .9991555 |
| 0.50 | 251.031 | 231.292 | 0.921x | 33.6% | .0247923 | .9997003 |
| 0.25 | 304.143 | 233.253 | 0.767x | 42.2% | .0184496 | .9998337 |

Lower beta is slower than dense, so only beta 1 was cadence-tuned. Activating
every second eligible step and layer (120 of the released policy's 432 sparse
main-block calls) passed latent/video gates broadly, but failed motion seed 11
decoded audio cosine (.97881) and dialogue seed 11 audio latent cosine (.9822)
and decoded audio cosine (.87569). Every-third cadence fixed motion (audio
cosine .99234) and dialogue latent cosine (.9913), but dialogue decoded audio
still failed at .90644.

The minimum nonzero policy—only step 10, layer 2—passed the hardest dialogue
seed numerically: video/audio latent cosine and correlation .9999/.9991,
decoded video correlation .99980 and PSNR 49.43 dB, decoded audio cosine
.98823. One sparse call among 950 main-block calls saves about 68 ms of roughly
220 s representative long-sequence attention work, approximately 1.0003x.
That is below normal timing dispersion and is not a meaningful speedup.

Therefore no cadence/beta policy simultaneously provides material acceleration
and the declared quality. No holdout was consumed and the experimental path
remains disabled. Artifacts are under `build-sol/quality-tuning/`.

## Error-aware routing follow-up

The experimental route score was extended with CTA-local query norm times the
preprocessed within-block K residual RMS, optionally weighted by within-block V
dispersion. Both weights default to zero. A capture-only grid (before another
prompt run) selected `error_k=.02,error_v=.10`: on step 0/layer 0 and
step 10/layer 2 it routed 22.4%/19.7% exact, ran at 1.259x/1.370x, and changed
rel-L2 from about .1442 to .1304 and .04168 to .04036. This was the best
matched-density point across `error_k=.02,.05,.10` and V weights `.02,.05,.10`.

The frozen candidate then failed the hardest allowed tuning case, dialogue seed
11: video/audio latent cosine .9877/.9429 and decoded audio cosine .75965.
Testing stopped immediately; the other tuning rows and every holdout remained
untouched. Default-zero routing still gives 1.404x on the representative active
capture, all outputs finite, and six pipeline checks pass. Error-aware controls
remain experimental diagnostics, not a promoted policy.
