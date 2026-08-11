# Caching and precomputation plan

A survey of every value in the pipeline that is recomputed when it could be
cached, precomputed, memoised or hoisted. Seven read-only agents covered
disjoint areas; this file is the triage. Findings that survived are grouped
into the four workstreams below. Findings that did not survive are recorded
too, with the measurement that killed them, so nobody re-treads them.

## Measurement discipline

Every number here is either **measured** on an idle RTX 5090 or explicitly
marked **not measured**. Two rules were learned the hard way and are binding
on anything added to this file:

**Measure with the GPU idle.** The same 22-frame run measured 0.98 s/step on
an idle card and 3.66 s/step while other agents were building and running
probes — 3.7x. README's own warning (up to 14x on dependent chains) is not an
exaggeration. Check `nvidia-smi --query-compute-apps` before believing a
number.

**State the geometry.** Attention is O(S^2), the GEMMs are O(S), and weight
dequantisation is O(1) in S. A win measured at the quick geometry mis-ranks
against the default. Dequantisation is 8.07% of a step at 4123 rows and about
0.6% at 37710; sage2 attention goes the other way, 10.01% to 61.5%.

### Baseline — 22 frames, 1:1, 30 steps, seed 11, sage2, nvfp4 pair, idle card

    total            35.65 s
    denoise          28.4 s   (0.98 s/step, 29 evaluations, 4123 packed rows)
    transformer load  3.7 s   (warm page cache)
    video vae         2.78 s
    audio vae         0.31 s
    output/mux        0.35 s

Per denoise step (991.9 ms, device timeline accounts for 99.98%):

| phase | ms/step | %step |
|---|---|---|
| `mlp.fc1` | 287.116 | 28.94% |
| `attn.qkv_proj` | 221.285 | 22.31% |
| `mlp.fc2` | 153.230 | 15.45% |
| `attn.sage2` | 99.337 | 10.01% |
| `attn.out_proj` | 76.328 | 7.69% |
| `mlp.dequant` | 47.651 | 4.80% |
| `attn.dequant` | 32.448 | 3.27% |
| `attn.qknorm_rope` | 29.381 | 2.96% |
| `mlp.swiglu` | 17.942 | 1.81% |
| **GPU idle in step** | **4.173** | **0.42%** |

The four cuBLAS GEMMs are 74.4% of a step. The host is never blocked
(`forward: sync wait` 0.009 ms) and the card is idle 0.42% of a step, so
**host-side caching inside the denoise loop is capped at 0.42%.** Wins have to
come from device work, or from outside the loop.

Video VAE stage (2780 ms wall) is the opposite shape — **55.86% host-only**,
with `vae weight load` alone at 1190.8 ms (42.83% of the stage), paid after
denoising ends while the card sits idle.

## Result — measured

All four workstreams merged. Measured end to end at the quick geometry, both
arms built from the same tree state and run back to back on an idle card at the
same 450 W cap with a warm page cache:

| stage | before (`7593a50`) | after | delta |
|---|---|---|---|
| prompt encode | 3.17 s | 3.10 s | -0.07 |
| transformer load | 2.45 s | 2.41 s | -0.04 |
| denoise (29 steps) | 29.9 s (1.03 s/step) | 29.8 s (1.03 s/step) | -0.1 |
| video VAE | 2.55 s | 2.42 s | -0.13 |
| audio VAE | 0.30 s | 0.30 s | 0 |
| output / mux | 0.79 s | **0.24 s** | **-0.55** |
| **total** | **36.01 s** | **35.27 s** | **-0.74 s (2.1%)** |

**2.1% is the honest warm-cache number and it is small.** The campaign's largest
wins are cold-start wins — the text encoder prefetch (28.90 s -> 10.26 s) and the
vision tower's ranged prefetch (77.4 s of demand faulting) — and a warm page
cache hides all of them. Dropping the standby list needs a privileged tool that
was not available, so **the cold-start figures in this document come from the
survey agents and have not been reproduced end to end on an idle card.** They
are reported as measured-by-survey, not measured-here.

The one clearly reproduced win is **output/mux, 0.79 s -> 0.24 s (-70%)**, from
the parallel RGB->YUV conversion and the ffmpeg probe order. The video VAE gains
5%. The denoise loop is unchanged, which is expected: it was already 99.98%
device-bound with the host 0.42% idle, so there was never host-side headroom
there.

Suites on the merged tree, idle card, 31.7 GB free: **`unit` 11036 checks / 0
failures**, **`kernels` 1262 checks / 0 failures / 11 DEFERRED**. The F16 AdaLN
device-widen is confirmed bit-identical by arena hash on the fp8 checkpoint:
`21045398272 bytes, 730 records, fnv1a 190cdce19da29c2d`, identical before and
after.

### The sage2 regression, caught only by measuring

`attn.sage2` went from 97.34 ms/step to 181.62 ms/step — **86% slower** — when
the quantiser templating landed. Nothing else in the step moved by more than 1%,
and restoring that one file recovered the whole 84 ms, taking the run from
38.29 s back to 35.27 s.

The change was **bit-identical** (the frozen output digests pass either way, and
still pass after the revert, which is what proves it) and ptxas put both versions
at 6 blocks/SM, so neither correctness review nor register-pressure analysis had
any way to see it. It was reviewed twice and shipped, and it was caught by the
first wall-clock measurement taken after the merge.

Reverted, along with the shared-memory opt-in deletion that rode with it. That
deletion was correct on its own terms — 32 KB at D=128 is under the 48 KB
threshold and `static_assert`s prove it — but it is not worth carrying alone for
a call that was already a no-op.

## Accepted work

### Workstream `agent/cache-load` — startup and load path

The largest measured wins in the whole campaign, and none of them touch
arithmetic.

1. **`Encoder::load` never calls `prefetch()`** — `src/cuda/encoder_kernels.cu:809`.
   Every other loader does (`src/dit/transformer.cpp:1006`,
   `src/cuda/vit_decoder.cu:370`, `src/vae/audio_decoder.cpp:240`). Measured on
   the 15.687 GB nvfp4 conditioner, cold: **28.90 s -> 10.26 s** (0.54 ->
   1.53 GB/s). One line. Advisory hint, correctness-neutral.
2. **The 1.19 GB Qwen vision tower is demand-faulted a page at a time** —
   `src/cuda/qwen_vision_encoder.cu:63-69`. Measured **77.374 s cold** vs
   0.446 s warm: 15.4 MB/s, roughly 290k serialised 4 KB faults. The tensors
   are one contiguous forward-walked extent, so a ranged
   `SafeTensors::prefetch(begin, bytes)` overload fixes it. The existing
   whole-file `prefetch()` is the wrong tool — it would pull all 27 GB.
3. **The keyframe encoder path has neither prefetch nor page-locking** —
   `src/vae/keyframe_encoder.cpp`, `src/cuda/keyframe_encoder.cu`. It missed
   both conventions. Reference-image runs only. Not measured.
4. **`tokenizer.json` is parsed into a `std::map`-backed JSON tree, then walked
   once and discarded** — `src/text/tokenizer.cpp:422-487`. Measured
   **147-166 ms** per process (parse 71-85, structure build 69), ~40 MB of
   transient allocation across ~303000 80-byte nodes. Direct scan of the
   retained bytes with reserved maps measures **35-44 ms**.
5. **AdaLN projections are widened F16->F32 on the host and shipped at double
   width** — `src/dit/transformer.cpp:1185-1194`, `src/core/tensor_convert.cpp:19-22`.
   43.6 M F16 elements, single-threaded, and `to_f32` memsets 155 MB before
   overwriting every byte. The weights are F16 on disk and F16->F32 is exact,
   so uploading verbatim and widening on device is bit-identical and saves
   ~87 MB of VRAM. `cuda::launch_widen_f16` already exists.
6. **Qwen RoPE builders call `std::pow` in the innermost loop** —
   `src/text/qwen_vision.cpp:140-141` and `:160`. `inv` depends only on `j`
   (18 and 64 distinct values). Measured **12.01 ms** for a 2048^2 vision
   table and **12.30 ms** for an L=8192 decoder table, ~90% of it in `pow`.
   `src/cuda/vit_decoder.cu:266-270` already does this correctly.
7. ~~**The ConvRot activation rotation is computed 5x per decoder layer where 2
   would do**~~ — `src/cuda/encoder_kernels.cu:597-599` and `:622-623`.
   **Implemented, validated bit-identical, then reverted on VRAM grounds.**
   Hoisting the rotation needs an `L x hidden` bf16 workspace reserved for
   `kI8ConvRot` — **84 MB at the 8192-token bound** — to buy ~0.06% of an
   encode. That is a bad trade on a 32 GB card, and it lands on the live path:
   `generate.cmd` ships `qwen3vl_32b_int8_convrot.safetensors` as the default
   text encoder. It was also the riskiest item on the sheet
   (`docs/text_encoder_spec.md` 5.3 and 9: a skipped or doubled rotation is
   silent, well-scaled noise). Recorded so nobody re-derives it.

### Workstream `agent/cache-kernels` — device-side reuse

1. **sage2 re-issues `cudaFuncSetAttribute` and `cudaGetDevice` on every
   call** — `src/cuda/sage_attention.cu:188`, `:254`. 100 driver round-trips
   per step on the default backend. `src/cuda/attention.cu:1111-1122` already
   has the `static thread_local` pattern to mirror.
2. **`quant_qk`/`quant_v` divide by a runtime `dim` and read Q/K/V twice** —
   `src/cuda/sage_attention.cu:105-141`, `:146-175`. `launch_official` is
   already templated on `D`, so the divisions can become a shift and a mask;
   the amax pass and the quantise pass can share registers.
3. ~~**nvfp4 scale gather is a per-thread 1-byte scattered read**~~ —
   `src/cuda/linear.cu:211-240`, `:133-137`. **Proposed, redesigned,
   implemented, measured 24% SLOWER, and reverted. Struck — do not re-attempt.**
   See the rejected section below for the measurement and the reason.
4. **NF4 dequant does two runtime 64-bit divides per thread and moves 2
   elements** — `src/cuda/linear.cu:242-278`, where the f8/i8 siblings at
   `:151`/`:175` move 8 with a 16-byte store. Both block sizes are powers of
   two. The 16-entry LUT must **not** go in `__constant__` — `:92-99` explains
   why (the nibble is the index, so a warp serialises the broadcast).
5. **The Qwen vision encoder allocates ~15 device buffers per image inside
   `encode()`** — `src/cuda/qwen_vision_encoder.cu:83-85`. Each `cudaFree`
   synchronises the whole device.

### Workstream `agent/cache-vae-output` — decode and output

1. **`blend_axis` re-materialises the whole tile when only the overlap slab
   changes** — `src/vae/decode_pipeline.cpp:87-107`, call sites `:324-340`.
   For `i >= overlap` the body is a pure copy. Measured **~480 ms per decode,
   9.10% of the stage**, 45 full-tile passes per chunk.
2. **Decoded tiles cross host memory twice** — `src/cuda/vit_decoder.cu:559`,
   `:568-572`: DMA into one reused pinned buffer, then `memcpy` into the
   hoisted tile vector. Measured **~181 ms per decode (3.43%)**, 1.2 GiB of
   host-to-host copy.
3. **`chunk split` copies what `tile stitch` could have written directly** —
   `src/vae/decode_pipeline.cpp:369-377` against `:342-355`. Measured
   **~79 ms per decode**.
4. **Tile geometry and the shape-group map are rebuilt per temporal chunk** —
   `src/vae/decode_pipeline.cpp:257-269`, from values already hoisted at
   `:207-214`.
5. **`z_batch` is allocated and zero-filled per shape group per chunk** —
   `src/vae/decode_pipeline.cpp:279`, 14 MiB value-initialised then fully
   overwritten. The site commits e0d4aa3 and 994dff1 missed.
6. **Lanczos resample weights are rebuilt per output pixel** —
   `src/core/image.cpp:73-117`. A verbatim copy of the function measured
   **321.4 ms -> 40.7 ms** for 1920x1080 -> 1280x768. Store raw weights and the
   row sum separately; folding normalisation in moved 11400 bytes by +-1.
7. **ffmpeg discovery probes 41 majors downward** — `src/video/mux.cpp:91-92`,
   `:127-145`, while the required majors are pinned at `:164-166`. Measured
   **31.9 ms** of failed `LoadLibraryA` calls.
8. **RGB->YUV 4:2:0 is single-threaded scalar host work per frame** —
   `src/video/y4m.cpp:51-87` and its byte-identical twin
   `src/video/mux.cpp:680-725`. The two must stay bit-identical to each other;
   `tests/test_output.cpp` asserts it.

### Workstream `agent/cache-orchestration` — whole-pipeline scheduling

1. **The video VAE checkpoint read starts cold after denoising ends** —
   `src/generate.cpp:534`. Measured **1190.8 ms, 42.83% of the VAE stage**,
   with the card idle. The denoise loop runs for minutes and touches no disk.
   `SafeTensors::prefetch()` on a worker thread during denoising hides it.
2. **Reference-image decode and encode sit outside the reuse cache** —
   `src/generate.cpp:128-146`, `:207-243`, both before the `reuse_models`
   check at `:261`. `encode_reference_image` is seed-independent; only
   `video_noise` + `scale_noise` at `:227-234` depend on the seed, so the
   clean rows cache and the noise stays per-generation.
3. **The tokenizer is rebuilt on every conditioning-cache miss** —
   `src/generate.cpp:270-272`, a local inside the `else` branch. ~150 ms per
   generation on a prompt change.
   **Correction, after review: no shipped CLI path reaches this.** `req.prompt`
   is assigned once at `src/main.cpp:1166` and the `--count` loop mutates only
   `seed` and `out_path`, so `conditioning_cache_key` is constant across a
   counted run, the conditioning cache always hits from generation 2, and the
   `else` branch containing the tokenizer reuse is entered only on generation 1
   — where the cache is empty by construction. The change is kept because it is
   small and correct and serves a library caller that varies the prompt, but it
   is **dormant for the CLI as it ships** and the ~150 ms is not realised by any
   command a user can type. Realising it would need a prompt-sweep CLI, which is
   `src/main.cpp` and was out of scope.
4. **`conditioning_cache_key` keys reference images by path, not content** —
   `src/generate.cpp:99-110`. Overwrite an image in place between two runs and
   the second silently reuses the first image's conditioning. This is a
   correctness bug in an existing cache and must be fixed before anything
   leans harder on that cache.
5. **The flow schedule is built twice with the shift constants re-hardcoded** —
   `src/pipeline.cpp:74-82` and again at `src/generate.cpp:372-377`, with
   `12.0f` and `3.0f` written out a second time. Microseconds, but a live
   drift hazard between the grid that is printed and the grid that is
   integrated.
6. **CMake re-downloads the licence from HuggingFace on every configure** —
   `CMakeLists.txt:165-176`, no existence guard, no `EXPECTED_HASH`, and
   `FATAL_ERROR` on failure, so every configure needs network.

## Rejected, with the measurement that killed it

**A sidecar precomputed cache file for the weights. Not justified — do not
build it.** The campaign explicitly authorised one; the evidence says no.

- The transformer arena is a near-verbatim copy of the checkpoint: 12615830272
  bytes of arena against 12528636800 of file. Everything except ~180 MB of
  converted records is `Store::kVerbatim` and DMA'd straight out of the
  mapping. A sidecar would be a second 12.6 GB file whose read costs the same
  I/O, to save ~0.2 s of host conversion.
- **There is no weight repacking to persist.** `src/cuda/nvfp4_gemm.cu:20-25`
  states it: the checkpoint's bytes are already the mma's B fragments, "no
  transpose, no repacking".
- A blob-layout sidecar to replace scattered reads with one sequential read is
  the one shape that would have been defensible, and it is measured dead:
  prefetched mapped reads reach **1.73 GB/s**, *above* the 1.35 GB/s unbuffered
  sequential control. There is no headroom to recover.
- **Content hashing to key such a file is itself disqualified**: a single
  `generate` touches ~34 GB of checkpoints, about **25 s of hashing**, more
  than the cold load it would protect.

The one genuine sidecar candidate was the tokenizer: a 4.49 MB flat blob loads
in 24-27 ms against 147-166 ms today. But the in-process direct scan
(accepted, `cache-load` item 4) reaches 35-44 ms with **no file format, no
staleness class and no invalidation test to get wrong**. The sidecar's
marginal gain over it is ~15 ms. That does not justify a keyed, versioned,
checksummed on-disk format plus `cache build`/`cache inspect` tooling, so it
is not being built. If one is ever wanted, the design work is done and is
recorded in the campaign notes: magic + format version + per-source key of
(size, mtime, header length, FNV-1a of the header bytes) + independently
checksummed sections, in `%LOCALAPPDATA%\Vidfab\cache\`, temp-file-plus-atomic-
rename, and silent fallback on any mismatch.

**Restructuring the nvfp4 dequant grid to stop wasting scale sectors. Measured
24% slower and reverted — the most instructive failure of the campaign.**

The observation was real: for a fixed output row, four consecutive `k` are
contiguous and then the address jumps 512 B, so a warp touches four scattered
32-byte sectors to consume 16 bytes, and each 32-byte sector holds scales for
eight different output rows that the row-wise grid had placed in eight
different blocks. Regrouping to one block per 512-byte scale tile takes the
scale traffic from 128 sector fetches per 8192 elements to 16. That arithmetic
is correct, and the resulting kernel is provably correct: an independent host
enumerator over ten shapes confirmed every element written exactly once with an
identical scale byte, no out-of-bounds stage, no bank conflicts, and no
occupancy change (30 -> 33 registers, still 6 blocks/SM).

It is also 24% slower, measured against the shipped `launch_dequant_nvfp4` in
each tree, alternating M/B/B/M on an idle card with intra-arm variance below
0.1%:

| shape | row-wise (master) | tile-wise | delta |
|---|---|---|---|
| `attn.qkv_proj` 3072x4096 | 0.0331 ms | 0.0332 ms | +0.4% |
| `attn.out_proj` 4096x4096 | 0.0529 ms | 0.0536 ms | +1.6% |
| `mlp.fc1` 14336x4096 | 0.2420 ms | 0.2990 ms | **+23.6%** |
| `mlp.fc2` 4096x14336 | 0.2430 ms | 0.3011 ms | **+23.8%** |

**The analysis optimised the wrong stream.** Per 8192 elements the kernel reads
4096 B of packed data, reads 512 B of scales, and *writes* 16384 B of bf16. The
write is 78% of the traffic; the scales are 2.4%. The row-wise grid emits one
contiguous 4096-byte store per block (`gy = min(out_features, 65535)` equals
`out_features` at every real shape, so the stride loop runs once). The tile-wise
grid emits 128 separate 128-byte stores strided `in_features*2` — 8 KB at fc1,
28 KB at fc2. Every store stays sector-aligned, which is why the small shapes
are neutral, but across 117 MB of output the write locality is gone. Meanwhile
the 128 scale sectors it recovers were nearly all L2 hits: the fc1 scale array
is 3.7 MB on a card with a large L2.

Two hypotheses died here. Staging the gather through shared memory does not
work either — the four scattered sectors are dictated by the swizzle layout,
not by how the load is issued, so restaging the same row changes nothing.

**The lesson worth keeping.** A sector-count argument is not a bandwidth
argument. Before restructuring a kernel's grid, account for *all* the traffic
it moves and check which stream dominates; the one being optimised here was
2.4% of it, and was already absorbed by L2. This was caught only because the
reviewer measured against the shipped kernel instead of re-deriving the
arithmetic, which had been checked twice and was right both times.

**A sidecar of validated ffmpeg struct offsets.** There is no offset probing to
persist. `src/video/ffmpeg_abi.h:173-204` is a compile-time `constexpr Layout`
and `validate_layout` (`src/video/mux.cpp:180-345`) is a once-per-process
safety *check*, not a probe. Caching it would save microseconds and trade away
the only thing standing between a moved field and memory corruption.

**CUDA graph capture of the denoise step.** The code is unusually well shaped
for it — one sync per step at `src/dit/transformer.cpp:1821`, stable arena
pointers, fixed shapes, and the only host branches are diagnostics that are off
by default. But the prize is bounded by `GPU idle in step`, measured at
**0.42%** of a step (1.51% as a 3-step upper bound), falling further at the
default geometry. Capture would freeze arena pointers into the exec graph and
silently break `VIDFAB_TENSOR_DIAG` and `VIDFAB_SOL_CAPTURE`. Not worth it.

**Caching dequantised weights across steps.** The largest genuinely
step-invariant GPU work in the loop, and it does not fit: 19.27 G parameters at
bf16 is **38.5 GB against a 32 GB card**. Even fully cached it saves ~25-36 ms
of a 991.9 ms step. Partial caching of the ~14 GB of headroom buys ~18 of 50
blocks, about 11 ms/step, in exchange for the entire VRAM margin.

**Sorting the weight upload into file offset order.** Measured dead three
separate ways. Name order does 453 backward seeks over 1.63 TB cumulative and
*is* slower unbuffered (0.50 vs 0.83 GB/s) — but the loaders read a mapping
that `prefetch()` has already been issued against, and `PrefetchVirtualMemory`
issues one whole-file async read in file order regardless of how the consumer
walks it. Measured with prefetch on: name order **1.73 GB/s**, file order
1.56 GB/s. Sorting optimises something no longer on the critical path.

**cuBLASLt algorithm/heuristic caching.** README records it was measured and
buys nothing; the four linears already run at 97.6-101% of the clock-corrected
roofline. The project does not use cuBLASLt at all.

**The Ref2VA full-AdaLN modulation recompute.** Would be a 13 GB VRAM and
~40-50 ms/step item — `src/dit/transformer.cpp:756-772` re-multiplies a
`[96768, 2688]` weight per block per step to produce a pure function of the
timestep. It is **dead code for every shipped checkpoint**: all four files in
`weights/transformer/` carry `adaln_t_table` with `adaln_proj.linear.weight` at
`[96768, 8]`, the rank-8 pruned table, so `detect_transformer_architecture`
never returns `kRef2VAFullAdaLN`. Recorded in case such a checkpoint ever
ships.

**Moving the Euler/AB2 update and the latent round trip onto the device.**
Measured `latent_h2d` 0.105 ms + `velocity_d2h` 0.244 ms + host
`scheduler_step` 0.241 ms = 0.59 ms of a 991.9 ms step, 0.06%. The current
design keeps a much simpler correctness argument, including the deliberate
separation of `sigma_from_timestep` from the grid ratio and the AB2 form at
`src/sampler/scheduler.cpp:165`.

**Memoising `build_row_timesteps`.** Provably at most three distinct S-length
vectors exist across a run, so it is genuinely precomputable — and it measures
0.009 ms/step of host work, entirely hidden. Not worth touching code whose
`torch.unique(sorted=True)` ordering is load-bearing for the AdaLN table row.

**Per-block cross-step attention-output reuse.** Unbounded error; no per-tensor
tolerance describes it, and it is strictly dominated by the existing
`StepCache`, which skips the whole forward pass for the same accounting.

**Composing `StepCache` with the AB2 sampler.** Already documented as unsound
in two places — AB2's `v_{n-1}` would be a reused velocity. Do not "fix" this
by caching more.

## Found along the way, unrelated to caching

**The golden tokenizer suite has never run in this repository.**
`tests/test_tokenizer.cpp:31` hard-codes `ref/FL2VA/text_encoder/tokenizer.json`.
That path does not exist — the tokenizer ships at `ref/text_encoder/tokenizer.json`
— so both `tokenizer_golden_ids` and `tokenizer_round_trip` take the
"not present; skipping" branch. On master, `VIDFAB_TEST_FILTER=tokenizer`
reports **`0 checks, 0 failures`**. With the file placed at the expected path it
reports **28 checks, 0 failures**, so the tests themselves are fine and have
simply never been exercised.

This surfaced because the campaign rewrote `Tokenizer::load_json`, and the
suite that would have caught a regression was the one silently disabled. The
test now tries both paths.

**It is not one test — the pass signal is decoupled from coverage across the
whole suite.** A sweep found **21 silently-skipping sites in 9 files**:
`test_transformer.cu` (5), `test_encoder.cu` (6), `test_audio_vae.cu` (2),
`test_output.cpp` (2), `test_tokenizer.cpp` (2), and one each in
`test_adaln.cpp`, `test_image.cpp`, `test_kernels.cu`, `test_nn_kernels.cu`.
Each prints a line and returns green, contributing zero checks.

Coverage also varies with **VRAM**, not just fixtures: the resident encoder
cases need ~23 GB, so a card with less headroom silently runs fewer checks.
Observed `kernels` totals during this campaign were 1121, 1133, 1158, 1191 and
1206 — all reported as passing runs. **1191 is the figure on a confirmed-idle
card with all fixtures present; anything lower means the environment, not the
code, was different.**

**And a passing golden suite is not proof the golden suite tests what you
think.** A negative control deliberately removed the tokenizer's longest-first
added-token sort — the ordering that makes `<|im_start|>` win over a shorter
prefix — and **all 8 golden cases still passed**, including
`<|im_start|>system<|im_end|>`. Only the newly added synthetic-document test
caught it. The golden ids were pinned against a real reference and are worth
keeping, but they were never sufficient, and for months they were not running
at all.

The cheap fix is to make skips visible: have the harness count them and print
`N skipped` beside the check and failure totals, the way `DEFERRED` already is.
Deferred to after this campaign's merges, because it touches a file every
workstream is editing.

Also checked and already correct, so left alone: cuBLAS handle and stream
lifetime, the workspace bump allocator and its high-water sizing, profiler
event pooling, RoPE tables in the transformer and the video VAE, the
token refiner running once per request, per-block dequant hoisting (6251e07),
pinned staging buffers, the safetensors header parse (~3-5 ms), BPE encode
(~0.6 us/token), and the added-token pre-scan (26 tokens, not 151000).
