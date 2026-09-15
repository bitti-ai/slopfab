# Reference video and audio implementation

Status: **CUDA and Vulkan generation support reference videos, their soundtracks, and
standalone audio alongside an image or video reference.** The CLI decodes files
with FFmpeg; DLL callers supply decoded frames and PCM without FFmpeg.
Use a Ref2VA transformer, a video VAE with encoder weights, and, for audio
references, an audio VAE with floating-point encoder weights (the shipped FP32
audio VAE works; NF4 audio encoder weights are not supported). Vulkan video
references require FP16/BF16 encoder weights (the shipped FP16 video VAE
works). Vulkan executes both encoder graphs on its selected device, without
CUDA fallback. The existing top-level executable/DLL build still includes CUDA;
the Vulkan reference encoder and its probe also build with CUDA disabled.

## Implemented input boundary

`ReferenceMedia` in `include/slopfab/reference_media.h` contains decoded RGB24
frames with clip-relative timestamps and optional interleaved float PCM. The
audio-only form represents a standalone reference. There is no file decoder,
FFmpeg, or accelerator dependency in this type.

C ABI 1.6 adds an opaque `slopfab_reference_video` builder:

```c
slopfab_reference_video* video = NULL;
/* Check every returned status; omitted here for brevity. */
slopfab_reference_video_create(2.0, &video);
for (int i = 0; i < 48; ++i) {
    /* Obtain the host-decoded frame for i, with its dimensions and stride. */
    slopfab_reference_video_append_rgba8(
        video, pixels, pixel_bytes, width, height, stride_bytes, i / 24.0);
}
/* Optional soundtrack, beginning at clip time zero. */
slopfab_reference_video_set_audio_f32(
    video, pcm, pcm_float_count, channels, sample_rate, 0.0);
slopfab_request_add_reference_video(request, video);
slopfab_reference_video_destroy(video);

/* A separate sound reference, also supplied as decoded PCM. */
slopfab_request_add_reference_audio_f32(
    request, other_pcm, other_float_count, other_channels, other_sample_rate);
```

Each append/set copies the caller's data before returning. Attaching a video
copies its metadata and shares immutable payloads, so later builder edits,
destruction and buffer reuse cannot change an attached reference. Failed
mutations leave the previous contents intact. As with request setters, callers
serialize access to mutable handles.

The video duration is explicit, including the last frame's display time.
Timestamps start at zero and strictly increase within that duration. Input
frames have constant dimensions; RGB24 and RGBA8 permit positive row padding,
and alpha is ignored. PCM is finite float in [-1,1], mono or stereo, at a
positive native sample rate. Soundtrack start offsets preserve synchronization.
Only standalone audio references must independently meet the two-second minimum.

Validation enforces 2–15 second clips, up to 9 images, 3 videos, 3 standalone
audio references, and 12 references overall. Videos and standalone audio each
have a 15-second aggregate limit. Attached soundtracks do not consume standalone
audio slots. Audio-only requests require an image or video before plan resolution.

Legacy image paths precede decoded video/audio references; video/audio references
retain insertion order. General arbitrary interleaving with images is pending.
Cache keys include pixel/PCM hashes, dimensions, timestamps, duration, PCM format
and offsets, plus the target duration used for preprocessing. Pixel and PCM
hashes are computed when copying input, so computing
cache keys does not reread all frame buffers. Media reference encoding keys also
include the audio VAE checkpoint identity.

## CLI adapter

With `SLOPFAB_WITH_FFMPEG=ON`, the CLI accepts:

```text
slopfab generate --prompt "A scene inspired by the references" --reference-video clip.mp4 --reference-audio sound.wav
```

This decodes the clip, includes its first audio stream when present, adds the
separate audio reference, and generates using CUDA by default. Select Vulkan with
`--inference-backend vulkan`. Supply the usual model and
output options as needed. Add `--dry-run` to inspect the plan; its packed row
count includes video and audio conditions. Reference videos require a video
target of at least 22 frames; still-image output with a reference video is rejected.

`src/cli/reference_decode.cpp` belongs to the executable, never the DLL. It
launches `ffprobe`/`ffmpeg` directly, without a shell. Install both command-line
tools beside the executable or on PATH; repository builds also search
`external/ffmpeg/bin` beneath the current directory. The existing muxer DLLs alone
do not provide these tools. FFmpeg-disabled builds exclude this implementation.

The adapter samples video at 24 fps, preserves native pixel dimensions, and
preserves mono/stereo audio at its native sample rate. Audio stream start offsets
are kept; audio preceding the first video frame is trimmed. Codec overshoots are
clamped to the public PCM range. Inputs must have a known 2–15 second duration;
trim longer files first. Current bounds are 8192 pixels per axis, 2 GiB decoded
video and 256 MiB PCM. Rotation metadata is not applied. File-decoded inputs
then use the same model preprocessing as host-provided inputs.

## Generation path

The following contracts were checked against the primary Diffusers implementation
at revision `c419dac0152186060246c93a095bc1bfaea342b3`:

1. Videos are sampled at 24 fps, resized with Lanczos to their own aspect-derived
   canvas, and truncated to the target duration. VAE input snaps down to
   `17*n+5` frames. Independent 17-frame chunks use causal temporal convolutions,
   with final-frame repetition and removal of only the last three combined
   latent frames. The sampled posterior uses seed 42, rounds through FP16, and
   is normalized and patchified into 96-channel condition rows.
2. Audio becomes 32 kHz stereo using Hann-windowed sinc resampling and mono
   duplication. Soundtrack offsets become leading silence; samples beyond the
   target duration are trimmed. Right-padding to a multiple of 800 samples
   yields 40 Hz latents. The convolutional encoder, causal attention and mean
   projection produce the normalized posterior **mean**, without sampling,
   packed into 32-channel rows in stereo channel order.
3. Qwen receives video frames sampled at 2 fps, merged in temporal pairs with
   timestamped video-pad blocks. Soundtracks add an audio label before their
   video label; waveform samples enter the audio VAE. Labels are numbered per
   modality. The combined conditioning prompt is limited to 32,768 tokens,
   shared across text, images, video pairs, timestamps, and modality labels.
4. CUDA or Vulkan Ref2VA receives both condition-row arrays and their temporal geometry.
   Audio conditions stay clean at timestep 1.0; visual conditions use the
   existing 0.999 noise rule. Prompt cache identity includes the decoded media
   and target duration. With model reuse enabled, video/audio VAE outputs are
   cached in host RAM before seed-dependent noise is applied. A hit skips VAE
   loading/encoding, waveform resampling and all but Qwen's 2 fps frame resize.
   The single retained media entry includes both VAE file identities, decoded
   media content/timing, target duration, backend and video precision. It is
   published only after every reference encode succeeds. CLI counted runs and
   DLL `slopfab_request_set_reuse_models` enable reuse; clearing reused models
   also releases this cache.

Vulkan video activations use separate frame buffers, so causal convolutions
read their two preceding frames without exceeding the device's per-buffer size
limit. Convolutions use shared 16x16 FP32 tiles; normalizations, Snake activation,
audio attention and projections also execute on Vulkan. These encoders are
numerically checked against CPU PyTorch; bitwise equality to CUDA is not promised.

Remaining work includes NF4 reference encoder support,
longer-run perceptual quality comparisons.

### CUDA reference performance

Generation uses FP16 video encoder activations and im2col scratch, with cuBLAS
FP32 accumulation and FP32 normalization statistics. FP16 checkpoint weights
feed GEMM directly, avoiding the former per-convolution FP32 conversion.
Audio encoding remains FP32. Set `SLOPFAB_REFERENCE_FP32=1` before launching the
CLI or DLL host to use the original video arithmetic. Precision is part of the
media cache key. The low-level encoder/probe defaults to FP32 for authority
comparisons; its explicit mixed-precision switch does not affect images.

Measured on an idle RTX 5090, 768x768 reference, 22 frames (two 17-frame chunks),
FP16 video checkpoint, warm encoder, second of two identical runs. Both modes
use the new pools and `SLOPFAB_PROFILE=1`; encoding includes pixel preparation
and latent packing but excludes model loading:

| Measurement | FP32 | FP16 |
|---|---:|---:|
| Reference video encode | 12.677 s | 4.840 s |
| Peak live activation storage | 14.344 GiB | 7.172 GiB |
| Reserved activation + scratch pools | 14.555 GiB | 7.277 GiB |
| Sampled total device memory peak | 16.54 GiB | 9.26 GiB |

The resulting 387,072 condition values differ by 0.000585 relative RMS and
0.04735 maximum absolute error. The original 32x32 CPU golden gives FP16 errors
of 0.000420 relative RMS for moments and 0.000544 for condition rows. The FP32
golden tolerances remain unchanged; the separate FP16 gate is 0.005 relative
RMS and 0.1 maximum absolute error. These numerical checks do not establish
perceptual equivalence for every reference/checkpoint.

The CUDA DLL smoke test with video, soundtrack and standalone audio reached
conditioning at 6.8 s on the first generation and about 0.1 s on cache hits.
Repeated same-seed decoded video and audio were byte-identical; changing the
seed changed the video. These are reference-stage savings, not a claim about
total generation time.

Sources: [reference normalization](https://github.com/huggingface/diffusers/blob/c419dac0152186060246c93a095bc1bfaea342b3/src/diffusers/modular_pipelines/minimax_h3/before_encoder.py),
[conditioning](https://github.com/huggingface/diffusers/blob/c419dac0152186060246c93a095bc1bfaea342b3/src/diffusers/modular_pipelines/minimax_h3/encoders.py),
[audio encoder](https://github.com/huggingface/diffusers/blob/c419dac0152186060246c93a095bc1bfaea342b3/src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3_audio.py).
The [official model specifications](https://github.com/MiniMax-AI/MiniMax-H3/blob/d21241f0a4b3acbb34c97dae47fa417b7065e438/README.md)
provide the modality count and duration limits.

## Verification

Verbose generation logs separate media preprocessing, video/audio VAE loading,
and each reference encode, including input geometry and output row counts.
`SLOPFAB_PROFILE=1` additionally samples CUDA device memory while convolution
inputs, outputs and scratch coexist. These logs include baseline/peak device
usage and minimum free memory; they include other processes and may miss
unsampled peaks. Vulkan verbose logs report encoder tensor peak and reserved
pool bytes. Neither figure should be compared to an idle-card baseline without
checking for other GPU workloads.

CUDA reference encoders retain separate activation and scratch buffer pools,
and one cuBLAS handle, across chunks/references within the stage. Pool logs
report logical activation/scratch peaks, current reserved bytes and allocation
counts. Reuse removes repeated allocation stalls; reserved memory can remain
above the live tensor size until the encoder is destroyed before conditioning.
The 32x32 golden video probe reaches 20 pool allocations in its first chunk
and performs no additional allocations in its next two chunks, with unchanged
FP32 golden errors.

`test_reference_media.cpp` covers pixel strides, transactional failures, overflow,
timing, snapshot ownership, PCM validation, reference limits, target-dependent
geometry/cache identities, temporal patchification and paired Qwen pixels.
The DLL tests cover the exports, handle lifetime and acceptance of Vulkan media
requests. These checks run with `SLOPFAB_WITH_FFMPEG=OFF`.

`reference_media_decode` creates lossless media fixtures and runs the exact CLI
decoder through a separate test executable, avoiding interactive application
startup. It checks RGB values, frame timestamps, native-rate mono PCM, waveform
gain, identical standalone/embedded audio, and a delayed soundtrack's offset.
No model checkpoint is loaded by these tests.

`tools/reference_encoder_goldens.py` computes independent CPU PyTorch outputs
from the real checkpoints; the CUDA `slopfab_referenceprobe` dev tool compares
resampling, temporal video moments, chunked posterior/condition rows, and audio
means. Generate and check a fixture with:

```text
python tools/reference_encoder_goldens.py VIDEO_VAE AUDIO_VAE golden.safetensors
slopfab_referenceprobe VIDEO_VAE AUDIO_VAE golden.safetensors
```

Add `--fp16` to validate CUDA mixed precision. To reproduce the performance
comparison, run the following from a scratch directory (each writes its
condition rows to `reference-bench-fp32.bin` or `reference-bench-fp16.bin`):

```text
slopfab_referenceprobe VIDEO_VAE AUDIO_VAE golden.safetensors --bench 768 768 22
slopfab_referenceprobe VIDEO_VAE AUDIO_VAE golden.safetensors --bench 768 768 22 --fp16
```

`--bench` checks finiteness and exact repeatability over two encodes. Enable
`SLOPFAB_PROFILE=1` for per-chunk timings and memory/pool counters. The focused
`reference_` CUDA kernel tests independently check causal/reflected convolution
boundaries, both strides, FP16/FP32 arithmetic, normalization and residual adds.
Add `--reuse` to `tools/reference_generation_smoke.py DLL REPOSITORY_ROOT` to
check three serial generations, cache hits, same-seed equality, changed-seed
output and release of input handles while the final worker is active.

The FP16 video/FP32 audio checkpoint run measured relative RMS errors of
`2.01e-6` for video moments, `8.33e-7` for audio means, and `5.58e-8` for
44.1-to-32 kHz resampling. Chunking, posterior sampling and FP16 rounding together
measured `5.46e-5` relative RMS (maximum absolute difference `0.00403`).

`tools/reference_generation_smoke.py DLL REPOSITORY_ROOT` runs an FFmpeg-free
DLL generation with video, an attached soundtrack and standalone audio. On an
RTX 5090 it produced 22 finite 32x32 RGB frames and 29,600 stereo sample frames
at 32 kHz, including request/builder destruction while the worker was active.
This is a one-evaluation integration smoke test, not a perceptual quality test.

`slopfab_referenceprobe_vulkan` accepts the same checkpoint/fixture arguments
and builds with `SLOPFAB_ENABLE_CUDA=OFF`. Its measured relative RMS errors are
`8.77e-7` for video moments, `1.17e-6` for audio means and `2.40e-5` for chunked
posterior/condition rows. Add `vulkan` to the DLL smoke script arguments to test
the complete Vulkan generation path. The Vulkan run produced the same output
geometry (22 RGB frames and 29,600 stereo sample frames) in 88.8 seconds on the
RTX 5090, including 32 seconds of reference preparation and 8.7 seconds for one
Flash2 denoiser evaluation. A separate 15-second audio fixture exercised all
600 encoder attention positions, with `7.02e-6` relative RMS error against CPU
PyTorch. Both fixtures also passed in the CUDA-disabled probe build.

Rebuild the checked-in shader with
`python tools/build_vulkan_reference.py --glslang PATH_TO_GLSLANG`;
CMake verifies its source and SPIR-V hashes.
