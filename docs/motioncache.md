# MotionCache

MotionCache can skip H3 transformer evaluations by reusing the previous computed
video/audio residual when estimated changes are small. It is approximate and
disabled by default. Enable it with ordinary Euler sampling on CUDA or Vulkan:

```sh
slopfab generate --prompt "A cat playing a piano" --frames 22 --resolution 512x512 --seed 11 --motion-cache --out cached.mp4
```

Compare with the same request without `--motion-cache`, especially for faces,
hands, fast motion and synchronized audio. Lower the threshold if detail or
motion becomes unstable. A threshold of zero disables reuse.

| Option | Default | Range |
| --- | --- | --- |
| `--motion-cache-threshold` | 0.15 | 0..1 |
| `--motion-cache-strength` | 1 | 0..4 |
| `--motion-cache-warmup` | 4 | 2..20 |
| `--motion-cache-max-skips` | 2 | 1..10 |
| `--motion-cache-start` | 0.15 | 0..1, below end |
| `--motion-cache-end` | 0.95 | 0..1, above start |
| `--motion-cache-subsample` | 8 | 1..32 |
| `--motion-cache-verbose` | off | Log each decision |

Tuning flags alone do not enable caching; include `--motion-cache`. `--dry-run`
prints the active configuration. Run statistics report actual computed and
reused calls. These counts are not a measurement of total generation speed.

The C++ request exposes `GenerateRequest::motion_cache`. The C API exposes
`slopfab_request_set_motion_cache`; its parameters and validation ranges are in
[`capi.h`](../include/slopfab/capi.h). Each trajectory starts with an empty cache,
including batch variations and runs that follow cancellation. Reference anchors
are excluded from cache estimation and reuse. Video-only stills use the video
score alone.

MotionCache requires Euler and cannot be combined with the existing step or
block caches, FastH3 V2, TaoMate's three-step schedule or Animate. Short schedules
may finish without any reuse. The last evaluation always runs, even if the
active range extends to 100%.

This implementation adapts the algorithm from
[Mozer/ComfyUI-MiniMax-H3-MotionCache-FastVAE](https://github.com/Mozer/ComfyUI-MiniMax-H3-MotionCache-FastVAE),
under its [MIT license](../third_party/motioncache/LICENSE). It ports MotionCache;
the repository's separate batched Fast VAE Decode node is not part of this feature.

The estimator samples spatial video coordinates and audio time coordinates.
Temporal differences in the predicted clean video produce normalized motion
weights. Changes between computed inputs/outputs estimate a change rate for each
modality; the larger normalized video/audio score accumulates until the next
computed call. Range percentages map through the video scheduler's flow shift,
not directly to sigma or rounded step indices. Native velocities have the
opposite sign to ComfyUI outputs: the stored residual is `velocity + input`,
and reuse returns `residual - current_input`.

CUDA already passes target rows through the host denoising loop. Vulkan uses
the same host estimator, downloading target latents and computed velocities and
uploading reused velocities while enabled. This adds synchronization and transfer
cost; benchmark your workload. Disabled runs retain the existing device-only
Vulkan trajectory. The cache retains full target residuals plus subsampled
history in host memory. No new model weights or runtime Python dependency are
required.
