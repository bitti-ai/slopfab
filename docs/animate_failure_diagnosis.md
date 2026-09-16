# Animate flat-output investigation, 2026-09-16

Correcting one checkpoint metadata field resolves the reproduced flat-texture
failure. The same render now contains the replacement character and driving scene
with changing poses. No inference code change was required for this correction.

## Confirmed checkpoint defect

The local `weights/transformer/Viggle-Animate-pruned_rank8_int8_convrot.safetensors`
declares `qkv_layout=interleaved`, but its attention weights are already in
contiguous `[Q; K; V]` order. SlopFab trusts this declaration and performs an
additional permutation, mixing Q, K and V rows and head assignments. This affects
both the main transformer and the text refiner, before denoising begins.

This is a defect in this checkpoint's metadata, not evidence that all Viggle
checkpoints should ignore their layout metadata. Genuine interleaved files still
need the loader's permutation. The earlier synthetic layout tests validated that
permutation but did not establish whether the real file's declaration was true.

## Direct comparison with original weights

The audit reads eight rows from head 1 of each Q/K/V projection in main blocks
0, 25 and 49 and both text-refiner blocks: 15 comparisons. Original tensors come
from [Viggle's transformer](https://huggingface.co/Viggle/Viggle-Animate/tree/16e05b96cf715035544f0330d1d29b123340a2ba/transformer),
pinned to revision `16e05b96cf715035544f0330d1d29b123340a2ba`. HTTP range reads
download only headers and sampled tensor rows. The audit dequantizes local INT8
rows using their own scales and reverses the regular-Hadamard ConvRot transform.

| Sample group | Contiguous interpretation | Declared interleaved interpretation |
| --- | --- | --- |
| Main blocks, 9 projections | Cosine 0.999956-0.999964; relative L2 error 0.85-0.95% | Cosine -0.0574 to -0.0016; error 109-248% |
| Text refiner, 6 projections | Exact equality to original BF16 values | Cosine -0.0096 to 0.0104; error 127-228% |

These are sampled comparisons, not a full checkpoint integrity check. Additional
spot checks found exact matches for the video input/output and text input
projections. The existing SwiGLU half swap also agrees with the original model:
the local first FFN projection stores gate first, while Diffusers stores value
first. There is no basis from these checks to change that adapter mapping.

Reproduce (requires PyTorch, NumPy and safetensors):

```powershell
python tools/verify_viggle_qkv.py weights/transformer/Viggle-Animate-pruned_rank8_int8_convrot.safetensors --report out/qkv-original.json
```

The verifier exits 2 for a declared/observed mismatch, 1 for an inconclusive
sample or error, and 0 when the declaration agrees with all sampled rows. Report
paths must be new; it does not overwrite reports or modify checkpoint files.

## Controlled correction

A separate local file was created at:

```
weights/transformer/Viggle-Animate-pruned_rank8_int8_convrot-contiguous.safetensors
```

Only `qkv_layout` changes to `contiguous`. The header occupies the same byte count;
all tensor offsets, tensor payloads and other metadata remain unchanged. The
original file is preserved. The tensor payload SHA-256 was verified independently
on the corrected copy:

```
ef315015ab185818bb4e5b42d7783ffb0b841ee08ce42d39d66f116d549e4376
```

The corrected copy passes the same weight audit. No loader code or inference
settings changed for the comparison: same DLL, 704x1248 canvas, 124 frames,
seed 42, four sigma boundaries, LoRA strength 1, frozen prompt, repainted frame,
driving video and pinned target audio.

```powershell
python tools/animate_generation_smoke.py build-release-cuda128-vs/Release/slopfab.dll 'C:\Users\NN\Downloads\Work_it._Shuffle_cuttingshapes_720p60.mp4' output/animate-comparison/repainted-reference.png output/animate-comparison/animate-contiguous.mp4 --transformer weights/transformer/Viggle-Animate-pruned_rank8_int8_convrot-contiguous.safetensors
```

Choose a new output stem to repeat the render. Artifacts and logs are local and
are not committed. The original failed render is `animate-pinned.mp4`; its log is
`run.log`. The corrected comparison uses `animate-contiguous.mp4` and
`contiguous-run.log`. Weight-audit JSON and checkpoint-copy details are under
`out/animate-upstream/`.

## Comparison result

The corrected CUDA render completed in 249.8 seconds, including three forward
passes totaling 181.9 seconds. It contains 124 frames at 704x1248 and 24 fps.
Inspection of frames 0, 60 and 123 shows the replacement character in the driving
scene with different poses; the original run showed a nearly uniform texture at
all three positions.

| Measurement | Original metadata | Corrected metadata |
| --- | --- | --- |
| Video latent mean | 0.0554 | 0.1050 |
| Video latent standard deviation | 0.2372 | 1.0089 |
| Output content in inspected frames | Flat brown texture | Character and scene |
| Pinned target audio | Unchanged through denoising | Unchanged through denoising |

Both runs have identical target audio latents and identical decoded WAV files.
The original VAE round-trip diagnostic already reconstructed the reference scene;
the metadata-only comparison now identifies the attention permutation as the
cause of this particular flat-output failure. This does not establish numerical
parity with a full BF16 upstream run or validate every frame's motion quality.

Use the corrected checkpoint copy explicitly in the host. The DLL is unchanged
by this investigation, so replacing the DLL alone cannot repair the bad header.
