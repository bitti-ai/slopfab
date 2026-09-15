# CUDA denoising memory

The CUDA transformer defaults to 2,048-row projection/FFN chunks. Flash2 also
uses compact query and attention-output buffers. These changes apply to all
supported transformer checkpoints, including Animate, without filename checks.

## Progress and measurement

The CUDA and Vulkan generation paths notify `kDenoising` **before** starting
their first denoiser evaluation. That entry event has `step=-1` and a populated
`total_steps`. Subsequent events keep their existing zero-based completed-step
index. Cancellation at entry stops before the first evaluation.

Set `SLOPFAB_PROFILE=1` before starting the application to print immediate,
flushed CUDA memory/timing boundaries for:

- `transformer.weights`: checkpoint upload and adapter loading;
- `transformer.text`: text projection and token refiner;
- `transformer.sequence`: geometry and denoising workspace allocation;
- `denoising.forward`: each actual transformer evaluation.

Each boundary reports device-wide used/free GiB. An exception prints a
`failed` boundary, so the last log remains useful when allocation fails.
These are samples, **not exact allocation peaks**, and include other GPU
processes. Sampling adds no explicit synchronization. Sequence preparation
also prints token counts, chunk size, Q/output and K/V sizes, and old/new
workspace reservations before allocation.

## What changes

Projection, FFN and final normalization scratch is bounded by the row chunk,
including inside the text refiner. Sequence preparation resizes its scratch
arena before allocating the sequence buffers, releasing an oversized refiner
or previous-request reservation. The arena stays fixed during denoising.

For Flash2, each block first builds full K/V from the original hidden state.
It then projects, normalizes and rotates one Q chunk, attends against all K/V,
and immediately projects the attention output back into that chunk's hidden
rows. Later queries read disjoint hidden rows, so the original computation is
preserved. Frame bands use each chunk's global query offset.

At 150,000 tokens and 7,168 attention channels, compact Q/output saves
**3.951 GiB** compared with full Q/output. The 8,192-to-2,048 row-chunk change
saves roughly another **1 GiB** of scratch for the production INT8 ConvRot
architecture. These are allocation estimates, not a promise that every
10-second geometry fits: weights, adapters, K/V, hidden state and optional
block-cache buffers still occupy VRAM.

## Controls and backend coverage

- Flash2 defaults to compact queries in both the refiner and main transformer.
- Exact, Sage2, Sol and blocked main attention retain full Q/output buffers.
  Their row-wise scratch still uses the smaller chunks.
- Full-tensor diagnostic capture retains full Q/output for its capture contract.
- Vulkan receives the progress fix; these allocator/kernel changes are CUDA-only.
- `SLOPFAB_DIT_FULL_QUERIES=1` restores full Q/output for comparisons while
  retaining the smaller row scratch.
- C++ callers can use `Transformer::set_row_chunk(rows)` and
  `set_query_chunking(enabled)` before preparation. Compact queries require a
  row chunk divisible by Flash2's query-tile size (currently 128); other sizes
  retain full Q/output. `workspace_bytes()` reports the actual reservation.

## Automatic block offloading

CUDA generation plans weight residency before uploading the transformer. If
the weights and estimated working memory fit, all blocks stay resident.
Otherwise it selects the shortest suffix of main transformer blocks that fits
the budget, accounting for **both** alternating GPU transfer slots. Streaming
one or two similarly sized blocks usually provides no net saving because the
slots occupy that space too.

The estimate reserves the larger of text preparation and denoising memory,
including the requested block cache, LoRA scratch, conservative ConvRot scratch,
and **1 GiB of safety headroom**. Available space comes from current device-wide
free memory. This is a load-time estimate, not a hard allocator limit; another
process allocating VRAM afterward can still cause an OOM.

Selected attention/FFN weights, norms, quantization metadata and LoRA factors
are converted once into owned, pinned CPU storage. They are never first loaded
as a complete GPU model. Input/output projections, the token refiner, and AdaLN
timestep-conditioning weights remain resident; AdaLN is evaluated across all
blocks before the block loop. No checkpoint is rewritten or requantized.

A separate CUDA stream prefetches the next selected block while the current
block computes. Events prevent transfer slots from being overwritten before
their previous users finish. Logical adapter keys remain distinct even when
blocks share a GPU address. Streaming also works across repeated evaluations
and block-cache skips; CPU staging outlives the original checkpoint mapping.

### Budget controls

Automatic selection is enabled for the normal CUDA generator. Optional process
environment variables must be set before launching the application:

| Variable | Behavior |
| --- | --- |
| `SLOPFAB_DIT_VRAM_GIB=22` | Plan for at most 22 GiB total device usage, including existing allocations. |
| `SLOPFAB_DIT_OFFLOAD_BLOCKS=12` | Stream the last 12 blocks, still checking the memory budget. |
| `SLOPFAB_DIT_OFFLOAD_BLOCKS=0` | Require fully resident weights; fail early if they exceed the budget. |

An `offload` log line reports the selected half-open block range, GPU weight
and adapter bytes (including transfer slots), pinned CPU bytes and working-memory
reserve. `SLOPFAB_PROFILE=1` also reports `offload.wait` in the step timeline.
If even the fully streamed plan cannot fit, reduce the sequence geometry or
raise the configured budget; streaming cannot remove activation memory.

Low-level C++ callers pass `TransformerLoadOptions` to `Transformer::load`,
including the intended `layout` and whether block caching will be enabled.
Set attention mode and chunk options before loading so the estimate matches
execution. Reload to replan residency for a different geometry. The low-level
default without a layout preserves resident loading. `offloaded_blocks()` and
`offloaded_host_bytes()` expose the selected residency. Full-tensor diagnostic
capture requires resident weights and rejects a streamed plan explicitly.
This feature currently applies to CUDA; Vulkan residency is unchanged.

## Validation

Focused CUDA tests compare compact and full-buffer attention bit for bit,
covering 64/128-wide heads, ragged sequence tails, frame-band offsets and
output-buffer guards. Transformer tests cover refiner output, RoPE, AdaLN,
repeated forwards, workspace shrinking and every Animate LoRA projection type.
Row-chunk comparisons allow BF16 rounding differences caused by GEMM shape.

`tools/reference_generation_smoke.py DLL REPOSITORY_ROOT` additionally asserts
the denoising entry event and completed-step sequence during a real DLL run.
An RTX 5090 comparison using the local Viggle INT8 ConvRot checkpoint and
distillation adapter produced **identical decoded video/audio bytes** with
4,874 packed rows and 2,048-row scratch in both runs:

| Query buffers | Workspace | Sampled denoising usage | One forward |
| --- | ---: | ---: | ---: |
| Full | 1.033 GiB | 23.928 GiB | 1.541 s |
| Compact | 0.957 GiB | 23.854 GiB | 1.569 s |

Timings are single-run observations. This small run exercises the integration;
it does not establish memory requirements or throughput for a user's full clip.
The existing CPU-reference comparison retains 11 documented deferred checks; the
exact-attention integration test skipped on the installed driver/runtime tuple.

### Offloading validation

Planner tests include transfer-buffer costs, uneven block sizes and budgets
that cannot fit. CUDA comparisons use six blocks with adapters on every
attention/FFN projection, stream one/five/all blocks, repeat evaluations and
sequence preparation, exercise block-cache skips, release source checkpoints
before execution and verify cleanup. Outputs match resident execution bit for bit.

With `SLOPFAB_DIT_VRAM_GIB=22`, the same real Animate/LoRA smoke case above
selected blocks **[41,50)**: 3.455 GiB of pinned CPU storage and two GPU slots
totaling 0.768 GiB. Sampled denoising usage dropped from **23.854 to 21.110 GiB**.
Decoded video and audio bytes were identical. One forward took 1.664 seconds
versus the earlier resident observation of 1.569 seconds; these single-run
measurements do not predict full-clip throughput. Regression checks passed:
21,289 CPU, 209 C API and the selected CUDA suites, with the same existing
deferred checks and exact-mode skip described above.
