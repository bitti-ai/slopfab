# Lip synchronization analysis

```sh
slopfab video syncnet clip.mp4 --model weights/syncnet_v2.model --face-model weights/sfd_face.pth --out output/project-syncnet.json
```

Native C++ SyncNet evaluates each automatically detected face track. S3FD locates
faces; overlap matching, gap interpolation and median smoothing produce face
crops. Histogram scene changes split tracks. This is geometric tracking, not
identity recognition; crossings, cuts and occlusions can fragment or mix tracks.
All eligible tracks are reported separately, including non-speaking faces.

`--cropped` skips detection when the video already contains one face crop. It
cannot be combined with `--face-model`. `--backend auto` prefers Vulkan and falls
back to the CPU when Vulkan cannot initialize. CPU inference is a slower reference
path. Neither Python nor LibTorch is required. External FFmpeg and ffprobe decode
video and audio; override them with `--ffmpeg` and `--ffprobe`.

## Weights

The downloadable [SyncNet v2 .model](https://huggingface.co/lithiumice/models_hub/resolve/a1004b6b826279d04634cbbf8fdd1879e7503fc9/syncnet_v2.model)
must have SHA256
`961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442`.
This restriction is unconditional: no alternate checkpoint, conversion or hash
override is accepted. Slopfab hashes the complete bounded buffer before reading
compiled tensor byte ranges from that same buffer. It does not execute pickle.

The [upstream S3FD checkpoint](https://huggingface.co/lithiumice/syncnet/resolve/16abbb51f33a0f34acd9add2ad0741bce6496531/sfd_face.pth)
must have SHA256
`d54a87c2b7543b64729c9a25eafd188da15fd3f6e02f0ecec76ae1b30d86c491`.
Alternatively, `--face-model FILE.safetensors` accepts the same S3FD architecture
with standard upstream state-dict names and FP32 tensors, without hash pinning.
Tensor names, shapes, sizes and finite values are validated. The Hugging Face
repositories checked during implementation supplied `.pth`, not safetensors.

The bundled download catalog lists both pinned URLs:

```sh
slopfab download syncnet syncnet-v2
slopfab download syncnet s3fd
```

## Scores and timing

The JSON report defaults to `output/<input-name>-syncnet.json`. Use a project's
name in `--out` when analyzing project output. Existing reports require
`--overwrite`; publication is atomic. `--agent` emits progress and a final result.

Each track includes its start/end, detected boxes, window count, `lse_d`, `lse_c`,
`zero_shift_distance`, `offset_frames`, `audio_shift_ms`, and distances for every
tested shift. Coordinates refer to the reported analysis dimensions.

- **LSE-D:** minimum mean audio/video embedding distance across candidate offsets;
  lower indicates a closer match.
- **LSE-C:** median candidate distance minus the minimum; higher indicates a more
  distinct best offset. It is not a probability or accuracy percentage.
- **Audio shift:** negative milliseconds advance audio, positive milliseconds delay
  it. For example, `-160` recommends moving audio 160 ms earlier.
- **Zero-shift distance:** the score at the input's current alignment. The minimum
  LSE-D alone can hide a constant delay because it searches for the best offset.
- **Search boundary:** the optimum reached the search limit; increase `--vshift`
  and analyze a sufficiently long track before trusting the offset.

The pipeline uses 25 fps, 16 kHz mono PCM, 13 MFCC coefficients, five video frames
and 20 MFCC frames per window, matching the upstream model. Distances use upstream
zero-padded shifted embeddings, without embedding normalization. Input audio/video
timestamps share the video stream's origin; resampling preserves their relative
delay. Output scores are uncalibrated and unreliable for silence, non-speech,
small/occluded mouths or unsuitable face crops. Entirely silent audio fails.

## Limits

`--start` selects seconds relative to the video stream origin; `--duration` caps
analysis at 30 seconds by default (maximum 600). The report states the duration
actually analyzed. Decoded video is temporarily spooled to disk, limited to
8 GiB and a longest side of 1280 pixels, then deleted on completion or error.
Select shorter segments if disk space is limited.

Detection runs every five frames at a longest side of 320 pixels by default.
Use `--detect-size 640` for smaller faces, `--detect-every 1` for fast motion, and
`--face-threshold` to adjust the default 0.9 cutoff. Up to 32 detections per frame
are retained; matches require IoU above 0.5 and gaps no longer than ten frames.
Tracks shorter than `--min-track` (default 1.5 seconds) or `2*vshift+6` frames are
skipped. The default search is ±15 frames (±600 ms), in 40 ms increments.

Implementation references: [SyncNet model and pipeline](https://github.com/joonson/syncnet_python),
[python_speech_features MFCC](https://github.com/jameslyons/python_speech_features).
The detector and scorer use native FP32 operations, with bounded im2col and Vulkan
matrix products. `tools/syncnet_manifest.py` reproduces the pinned byte tables;
`tools/check_syncnet_reference.py` checks optional real-model reference outputs.
