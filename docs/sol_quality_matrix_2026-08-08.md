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

Pending execution.
