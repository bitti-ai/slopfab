# Decode performance and repeated DLL generations

`videoDecode` includes loading the video VAE, decoding its windows, composing
tiles and producing the full planar float video. `audioDecode` includes loading
the audio VAE and generating PCM. The `transformerLoad` progress interval includes
both checkpoint loading and text/sequence preparation before denoising starts.
The C++ result separates these as `seconds_transformer_load` and `seconds_prepare`.

## Decode timing breakdown (C API 1.27)

After successful completion, call `slopfab_generation_video_decode_timings` to
read six wall-clock durations without changing the existing `slopfab_output`
layout. The getter returns `NOT_READY` during generation, the terminal error on
failure, and leaves the destination unchanged on either path. Timings survive
`slopfab_generation_release_samples` and remain owned by the generation handle.

| Field | Measured work |
| --- | --- |
| `seconds_prepare` | Allocate and unpatchify the latent volume |
| `seconds_upscale` | Optional latent upscaling; zero when disabled |
| `seconds_model_open` | Open/map the VAE file and read normalization statistics |
| `seconds_weight_load` | Create the decoder/device and load VAE weights |
| `seconds_compute` | Decode, including workspace allocation, transfers and tile assembly |
| `seconds_cleanup` | Remaining time: decoder destruction, temporary buffers, file mapping teardown and instrumentation overhead |

These fields sum to `seconds_video_decode`. Since 1.27 that total also includes
temporary latent-buffer and checkpoint-mapping destruction. Measurements use
the monotonic host clock and add no GPU synchronization. They are phase wall
times, not isolated kernel timings. C++ callers receive the same breakdown in
`RunResult::video_decode_timings`.

## Output ownership

Each completed `slopfab_generation` owns its output until the application destroys
it. The borrowed pointers from `slopfab_generation_output` do not transfer
ownership. A float RGB video alone occupies `frames * width * height * 3 * 4`
bytes; 107 frames at 1344x768 occupy about 1264 MiB. Retaining a list of generation
handles also retains all those videos, in addition to any copies made by the host.

Destroy the handle after encoding/copying the samples if it is no longer needed.
For continuation, the next request takes shared ownership of the source latents:
after a successful `slopfab_request_set_continuation_generation`, the source
generation can be destroyed before starting the next generation.

C API 1.25 adds `slopfab_generation_release_samples` for applications that need
to keep a completed handle for later latent saving or continuation:

```c
/* Enable retain_latents on the request before starting, when needed. */
if (slopfab_generation_wait(generation, -1) == SLOPFAB_OK) {
    slopfab_output output;
    if (slopfab_generation_output(generation, &output) == SLOPFAB_OK) {
        /* Finish consuming output.video and output.audio here. */
        slopfab_generation_release_samples(generation);
    }
}
/* Retained latents remain usable; destroy the handle when finished with them. */
slopfab_generation_destroy(generation);
```

Releasing samples invalidates all borrowed video/audio pointers. Serialize it
against readers and destruction. It is idempotent after completion and returns
`NOT_READY` while generation is running. Subsequent output/frame access on a
successful released generation returns `INVALID_REQUEST`; generation status and
retained latents are unchanged. Failed/cancelled generations retain their status.

Completed workers also release their snapshots of reference media, continuation
inputs and sessions. The application's request and explicit session still own
their respective data until cleared or destroyed.

Continuation outputs contain the joined clip. Progressively longer continuations
therefore require increasing output memory and decode work even when old handles
are destroyed. Sample release does not make a growing clip constant-size.

## Implementation changes

- Video tiles compose directly into the final planar buffer. Removing the second
  full video allocation saves approximately one output video's worth of peak
  host memory. Tile registrations and buffers are released before normalization.
- Audio decoder staging reserves its destination once, directly appends plain
  FP32 weights, folds weight normalization in place, and prefetches only decoder
  tensors. The ordinary FP32 checkpoint has about 248 MiB of decoder tensors
  versus 577 MiB for the whole file. CUDA and Vulkan reuse an idle activation
  buffer, reducing their working arena by one buffer (19.8 MiB at 405 tokens).
- CUDA transformer uploads pack small converted tensors into bounded pinned
  staging slots, avoiding waits after every second small tensor. Slots allocate
  lazily; the limit remains 64 MiB. Weights are not kept resident between runs.

These changes preserve inference precision, decoder arithmetic and blending.

## Reproducing the memory check

From the repository root, with real video/audio VAE checkpoints under `weights/vae`:

```powershell
python tools/decode_repeat_probe.py build/Release/slopfab.dll --runs 10
python tools/decode_repeat_probe.py build/Release/slopfab.dll --runs 10 --lifetime release
```

The probe skips conditioning/denoising, feeds fixed seeded latents to both VAEs,
and prints JSON with decode timings, output hashes, and Windows process memory
before/after cleanup. `--width`, `--height`, `--frames` and `--backend` select the
workload. The `retain` lifetime intentionally keeps complete results to reproduce
ownership-related growth; use a small geometry for that comparison. The `release`
mode also checks that output access is refused and saved latents remain identical.
Process memory includes driver allocations and excludes dedicated GPU memory.
Synthetic inputs exercise allocation and decoding, not perceptual quality.

A standalone host scheduler comparison at 107x768x1344, four runs per version,
reduced peak process commit from 3244 to 1978 MiB with identical full output hashes.
Median host scheduling time was 1.734 versus 1.491 seconds; this excludes GPU work
and is not an end-to-end speed claim.

On an RTX 5090, three repeated CUDA DLL decodes of fixed synthetic latents at
107x768x1344 gave these local medians (seconds):

| Measurement | Before | After |
| --- | ---: | ---: |
| Video decode, including model load | 13.707 | 13.025 |
| Audio decode, including model load | 0.488 | 0.261 |
| Whole decode-only request | 15.411 | 14.141 |

Video and audio SHA-256 hashes matched across both versions and all repetitions.
Peak process commit fell from 15428 to 14175 MiB. Post-destruction private commit
stayed near 1156 MiB before and 1154 MiB after; these figures include CUDA driver
state. This short test did not reproduce the reported progressive slowdown/OOM.
Timing is indicative, with ordinary background activity and no controlled cold
file-cache experiment; synthetic decoding also excludes conditioning/transformer
and application-side copies/encoding.

Ten 56x512x512 runs retaining handles/latents but releasing samples also passed:
all sample hashes matched the baseline, double release succeeded, output/frame
access was refused after release, and latent archives were byte-identical before
and after release. Each release freed about 168 MiB of samples. Video decode
remained around 3.2-3.4 seconds; destroying all retained handles returned private
commit to about 1100 MiB, the first run's post-release level.

A bounded upload replay (6.25 GiB, reusing a 128 MiB source/destination) verified
identical bytes and reduced registered-transfer completion from 165.77 to
158.27 ms. CPU issue time fell from 160.06 to 6.41 ms. This measures staging,
not full transformer loading; conversion, storage, LoRAs and preparation can
change the result in a generation.
