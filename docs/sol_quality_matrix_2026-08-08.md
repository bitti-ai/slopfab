# Sol pipeline final quality matrix — 2026-08-08

This matrix and its acceptance criteria were committed before its generation
runs. Every row compares `--attention sol` with `VIDFAB_SOL_PIPELINE=1` against
`--attention flash2`, using the same executable, prompt, seed, step schedule,
22-frame 1:1 geometry, tokenizer, and weights. Pipeline remains opt-in.

## Predeclared acceptance criteria

A scenario passes only when all of these hold:

- every tensor scanned by a diagnostic run is finite;
- flattened final video and audio latent cosine and Pearson correlation are at
  least 0.99;
- decoded RGB video SSIM is at least 0.94, Pearson correlation at least 0.98,
  and PSNR at least 28 dB;
- decoded stereo audio cosine is at least 0.98;
- side-by-side visual inspection and level-matched listening reveal no material
  semantic, identity, temporal-coherence, text/detail, dialogue/lip-sync, or
  music/audio failure.

Finite checks and numerical thresholds are conjunctive. A subjective material
failure fails the row even if its aggregate metrics pass. A scenario without
both matched artifacts or without inspectable audio/video is incomplete, not a
pass. The overall gate passes only if all six scenarios pass at both seeds.

## Fixed matrix

Seeds are 11 and 29. All prompts use the model's
`integrated_multimodal_description:` prefix.

| ID | Stress | Prompt |
|---|---|---|
| `detail` | static text/fine detail | a locked camera close-up of a red ceramic teapot on a wooden table; a crisp white label reading SOL TEST 2026 remains fully legible; warm studio light; quiet room tone |
| `motion` | rapid motion | a motocross rider rapidly jumps across frame from left to right while the camera pans quickly; flying dirt and spinning wheels; loud engine rev and landing impact |
| `identity` | long identity consistency | the same elderly woman with a silver braid and green scarf walks toward camera, turns around, then faces camera again; her face, braid, scarf, and clothing remain consistent; footsteps and park ambience |
| `scene` | scene transition | a match cut from a snowy mountain cabin at dawn to the same cabin interior beside a fireplace at night; clear intentional scene change; wind transitions to crackling fire |
| `dialogue` | dialogue/lip sync | close-up of a man clearly saying one two three four while facing camera; natural synchronized mouth movements; clean spoken dialogue and no music |
| `music` | music/audio | a drummer and acoustic guitarist perform a short upbeat duet on stage; visible strikes and strums align with crisp percussion and guitar; no speech |

## Common invocation

The exact executable Git commit and SHA-256, expanded commands, artifact hashes,
metrics, and verdicts are appended after execution. Common model paths are the
absolute paths recorded in `sol_quality_gate_2026-08-08.md`. Runs use
`--frames 22 --aspect 1:1 --steps 20`; no cache, banding, or unrelated lossy
option is enabled. One representative pair additionally uses
`VIDFAB_TENSOR_DIAG=1` to establish finiteness; timing runs keep diagnostics off.

## Results

All 24 generation commands completed successfully. Artifacts, individual logs,
and full SHA-256 manifests are under `build-sol/quality-matrix/`; in particular,
`metrics.tsv` is the raw metric record and `sha256.tsv` contains hashes for the
executable, tools, all MP4 files, and all latent dumps. These large artifacts
are intentionally outside Git.

The executable SHA-256 was
`42F2BB9BB0BEE4EFE9B786BA1C4417A246587DC925CA888DBBA2C365D92DD7AD`.
It was linked from the guarded ragged-read implementation at `97fae35`; later
CLI-only source changes were deliberately not linked until the matrix ended.

For each scenario and seed, the exact dense suffix was:

```text
--seed SEED --frames 22 --aspect 1:1 --steps 20 COMMON_MODELS
--attention flash2 --dump-latents build-sol\quality-matrix\ID-sSEED-dense.safetensors
--out build-sol\quality-matrix\ID-sSEED-dense.mp4
```

The pipeline command changed only the artifact suffix to `pipeline`, set
`VIDFAB_SOL_PIPELINE=1`, and used `--attention sol`. `COMMON_MODELS` expands to
the five absolute tokenizer/model paths in `sol_quality_gate_2026-08-08.md`.
The prompts are exactly the strings in the fixed matrix above. Logs record the
expanded seed, geometry, steps, output, token count, backend, and schedule.

Metrics below are pipeline versus dense. `vLat` and `aLat` are final-latent
cosine/correlation. `RGB` is decoded RGB Pearson correlation; PSNR is RGB dB.
`aCos` is decoded float32 stereo cosine. PASS requires every predeclared
threshold, so a dash in the last column means a numeric failure already makes
the subjective gate moot rather than silently passing it.

| Scenario | Seed | vLat cos/corr | aLat cos/corr | SSIM | RGB corr | PSNR | aCos | Numeric verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| detail | 11 | .9943/.9943 | .9946/.9945 | .95335 | .98287 | 30.34 | .99528 | PASS |
| detail | 29 | .9930/.9930 | 1.0000/1.0000 | .93370 | .98688 | 30.02 | .98712 | FAIL SSIM |
| motion | 11 | .9853/.9851 | .9938/.9938 | .89803 | .96315 | 25.11 | .95669 | FAIL |
| motion | 29 | .9801/.9792 | .9232/.9229 | .86316 | .95367 | 24.48 | .85094 | FAIL |
| identity | 11 | .9820/.9814 | .9989/.9989 | .90934 | .97961 | 30.73 | .95367 | FAIL |
| identity | 29 | .9790/.9784 | .9990/.9989 | .88562 | .97819 | 28.35 | .99183 | FAIL |
| scene | 11 | .9853/.9853 | .9971/.9971 | .95350 | .99380 | 32.83 | .99868 | FAIL latent |
| scene | 29 | .9839/.9838 | .9976/.9976 | .94580 | .99052 | 31.20 | .99744 | FAIL latent |
| dialogue | 11 | .9871/.9871 | .9417/.9417 | .94410 | .98739 | 31.50 | .76556 | FAIL |
| dialogue | 29 | .9903/.9903 | .9779/.9778 | .94747 | .98758 | 31.10 | .94078 | FAIL audio |
| music | 11 | .9672/.9666 | .9807/.9806 | .85284 | .95764 | 24.40 | .97041 | FAIL |
| music | 29 | .9650/.9648 | .9761/.9761 | .86189 | .97209 | 25.14 | .93592 | FAIL |

The representative `detail`, seed 11 pipeline run was repeated with
`VIDFAB_TENSOR_DIAG=1`. Its 5,766-line
`detail-s11-pipeline-diag.log` contains zero lines with a nonzero `nonfinite`
count. The repeat exited successfully.

`vidfab_chunkprobe stats` computes final-latent metrics directly from the FP32
safetensors. `tools/media_compare.py` streams FFmpeg-decoded `rgb24` video and
`f32le` audio and accumulates dot products, centered covariance, absolute and
squared error in double precision. FFmpeg's `ssim` filter supplies SSIM.

Side-by-side inspection found the aggregate metrics representative: the detail
pair retained the same composition but showed fine-label/detail changes, while
the motion, identity, and music failures showed materially larger spatial or
temporal differences. A controlled level-matched listening session was not
available in this non-interactive environment, so subjective audio is marked
incomplete rather than inferred from metrics.

## Verdict

**FAIL.** Only one of twelve rows clears all numeric thresholds, and subjective
listening is incomplete. The experimental pipeline must remain disabled by
default.
