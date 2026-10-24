# Native LCS and sampling callbacks

C API 1.31 adds native H3 Latent Color Subspace effects and a writable sampling
callback on CUDA and Vulkan. Both operate on denoised predictions before the
sampler updates the latent. They require no Python or ComfyUI runtime.

The built-in effects are color steering (Type I, Type II and sigma interpolation),
contrast, brightness, saturation, temperature, optional masks, experimental
sharpness control, and self/smoothing color anchors. Applications can combine
these with their own callback or use the callback alone.

This is an H3 implementation of the subspace approach described by
[The Latent Color Subspace](https://github.com/ExplainableML/LCS) and
[ComfyUI-LCS](https://github.com/facok/ComfyUI-LCS). It does not execute ComfyUI
nodes or import workflows. The color geometry is calibrated to H3's 96-value
patches. It uses clean VAE calibration directly, rather than assuming the
FLUX timestep statistics apply to H3. The anchor and sharpness algorithms are
native variants; results are not promised to match the Python plugin.
Reference-image/automatic anchors, the plugin's preview UI and per-batch color
lists are not built-in controls. Separate requests can use different colors;
the sampling callback can implement additional effects and observation.

## Calibrate once per VAE

Set the request's video VAE path and inference backend, then call
`slopfab_lcs_calibrate`. Only the VAE encoder is required; no transformer or text
encoder is loaded. The call is synchronous, so GUI applications should run it
on their own worker thread. It shares the library's generation lock and returns
`SLOPFAB_ERR_BUSY` if another generation or export is active.

```c
slopfab_request* request = slopfab_request_create();
/* Check each returned status before continuing. */
int status = slopfab_request_set_model_path(
    request, SLOPFAB_MODEL_VIDEO_VAE, "weights/vae/minimax_h3_video_vae_fp16.safetensors");
if (status == SLOPFAB_OK)
    status = slopfab_request_set_inference_backend(request, SLOPFAB_INFERENCE_CUDA);
if (status == SLOPFAB_OK)
    status = slopfab_lcs_calibrate(request, "h3-lcs.safetensors", 64, 256, NULL, NULL);
/* On failure, inspect slopfab_last_error(). */
```

The encoder processes deterministic solid colors, eight anchor colors and four
sharp/blurred grayscale pairs. Calibration uses posterior means without sampling
noise. A small CPU eigensolver fits the color basis and a color-orthogonal,
DC-free sharpness direction. Image size must be a multiple of 32 in 64–512;
the color count is 8–4096. Use at least 64 colors and size 256 for initial
experiments; 8 colors at size 64 is useful for smoke testing.

The output is written atomically after successful calibration. Existing output
survives cancellation or failure. A progress callback receives completed and
total encoder calls (including an initial `0`); return `0` to continue, `1` to
cancel, or any other value to report failure. It runs on the calling thread and
must not throw. It can run for minutes depending on the VAE and GPU.

The archive contains the raw-space basis `[96,3]`, mean `[96]`, color anchors
`[8,3]`, model normalization `[24]`, sharpness direction `[96]`, explained color
variance, and a SHA-256 fingerprint of the entire VAE file. Generation checks
that fingerprint. Recalibrate after changing or quantizing the VAE. The same
archive can be used by both backends with that VAE; neither generation nor
calibration downloads files. ComfyUI-LCS calibration files are not this archive
format and cannot be loaded directly.

## Configure native effects

```c
int status = slopfab_request_set_lcs(request, "h3-lcs.safetensors",
    "{\"mode\":\"interpolated\",\"color\":[1.0,0.35,0.1],"
    "\"strength\":0.35,\"start_step\":0,\"end_step\":2,"
    "\"contrast\":1.1,\"saturation\":0.95}");
```

All fields are optional. Unknown keys, invalid values and malformed JSON are
rejected. Unspecified fields use neutral defaults; setting a calibration file
alone applies no effects. Settings replace the previous settings, rather than
merging. Files are read immediately and copied into immutable request data, so
later deletion or modification does not change queued generations. Invalid calls
leave the previous settings intact.

| Key | Default | Values / meaning |
| --- | --- | --- |
| `mode` | `"interpolated"` | `"type_i"`, `"type_ii"`, `"interpolated"` |
| `color` | `[1,0,0]` | Target RGB, three numbers in `[0,1]` |
| `strength` | `0` | Color steering, `[0,2]` |
| `contrast` | `1` | Lightness contrast around midpoint, `[0,4]` |
| `brightness` | `0` | Offset along black-to-white axis, `[-1,1]` |
| `saturation` | `1` | Chroma multiplier, `[0,4]` |
| `temperature` | `0` | Cool-to-warm chroma shift, `[-1,1]` |
| `sharpness` | `0` | Signed calibrated direction strength, `[-5,5]` |
| `anchor` | `"none"` | `"none"`, `"self"`, `"smooth"` |
| `anchor_strength` | `0` | Anchor blend, `[0,1]` |
| `start_step` | `0` | First video update index, inclusive |
| `end_step` | `2147483647` | Last video update index, inclusive |

Type I translates each frame's color coordinates toward the target while
preserving their differences. Type II interpolates each patch's hue, saturation
and lightness using calibrated bicone geometry. Interpolated mode favors Type I
at high sigma and Type II at low sigma. Tone controls operate in this same color
subspace. The orthogonal raw-latent residual is preserved by color/tone/anchor
edits, up to floating-point rounding; this is not a guarantee that decoded
image structure or perceived detail never changes.

The self anchor blends toward the previous active step's unmodified color
coordinates. Its first active step establishes history. Smooth anchoring blends
with the four spatial neighbors in each frame; it never crosses frame edges.
Each generation gets fresh anchor state. Sharpness adds the calibrated leading
sharp-minus-blurred direction; its perceptual effect and strength require testing
on the chosen model and schedule. Use small strengths initially.

Native effects only edit target video predictions. Fixed reference rows and
audio are excluded. The mask is a copied F32 array in `[0,1]`, supplied with
`slopfab_request_set_lcs_mask`. It contains either one spatial patch plane
`(latent_height/2)*(latent_width/2)` repeated over frames, or one value per target
video patch across all frames. Use the resolved plan's latent dimensions.
For continuation/bridging this is the sampling window, not the final joined
clip. Downsample a pixel mask in the application before supplying it. A zero
mask value preserves that patch exactly. Pass `NULL, 0` to clear the mask.

`slopfab_request_set_lcs(request, NULL, NULL)` disables native LCS and clears its
mask, while preserving the application callback.

## Implement a custom sampling effect

```c
static int SLOPFAB_CALL effect(const slopfab_sampling_step* step, void* userdata) {
    (void)userdata;
    if (step->modality != 0 || step->step > 2)
        return 0;
    /* Illustrative arbitrary latent edit, not calibrated color correction. */
    for (uint64_t i = 0; i < step->count; ++i)
        step->denoised[i] *= 0.99f;
    return 0;
}

/* Called before slopfab_generation_start. NULL clears the callback. */
int status = slopfab_request_set_sampling_callback(request, effect, NULL);
```

The callback runs on the generation worker thread once for each advancing
modality, after native LCS and before its outer sampler update. Video precedes
audio when both advance. Unequal schedules call it only for the modality being
updated; `step` and `total_steps` refer to that modality, and `evaluation` refers
to the joint schedule. Inner outpaint Langevin iterations do not call it. Pinned
audio and conditioning rows are excluded. Image-edit and locked-overlap
constraints still run after the sampler, preserving their normal behavior.

| Field | Meaning |
| --- | --- |
| `struct_size` | Size of the supplied C structure |
| `modality` | `0` video, `1` audio |
| `sigma`, `sigma_next` | Current prediction noise level and next scheduled sigma |
| `latent` | Borrowed, read-only current model-space latent |
| `denoised` | Borrowed, writable model-space denoised prediction |
| `count` | Number of floats in either buffer |
| `row_width` | `96` for H3 video; `32` for H3 audio |
| `latent_frames/height/width` | Video latent geometry; all zero for audio |

Video layout is `[frame, y/2, x/2, channel, dy, dx]` with 24 channels and 2×2
patches. Audio layout is `[stereo channel, latent time, feature]` with 32 features.
The relation is `denoised = latent + sigma * native_velocity`. These buffers use
normalized model coordinates, not raw VAE coordinates or decoded pixels. Native
LCS applies the VAE normalization internally. The sampler consumes edited
predictions through velocity deltas, so Euler, CUDA AB2 and re-noising retain
their existing integration rules. A read-only callback leaves velocity bits
unchanged. Cached base predictions remain unmodified by the callback.

Only buffer contents may be edited. Metadata and pointers must not be changed,
freed or retained after return. Return `0` to continue, `1` to cancel before the
current update, or another value to fail the generation. Non-finite prediction
values or overflowing velocities fail the generation; the error is available
through `slopfab_generation_error`. Callbacks must not throw or wait for/destroy
their own generation. `slopfab_generation_cancel` is allowed.

Request snapshots copy callback and userdata pointers, not the application-owned
objects behind them. Keep both alive through completion, even if the request is
destroyed or reused. Managed bindings must retain their delegate and userdata
objects and use the header's calling convention.

The implementation uses host F32 arithmetic for built-in effects and callbacks.
Vulkan synchronizes and transfers target latent/prediction buffers when enabled;
this has a performance cost. Disabled effects retain the ordinary device path.
Both interfaces currently support H3; MOSS and synthetic-latent requests reject
these options. This feature does not add support for other ComfyUI model families.

## Validation

Synthetic tests cover calibration serialization, invalid settings, normalization,
color-residual preservation, mask boundaries, per-generation history, no-op
callbacks, edited trajectories and cancellation on CUDA and Vulkan. The optional
real-VAE calibration test runs with:

```powershell
$env:SLOPFAB_TEST_FILTER = 'lcs_real_vae_calibration'
$env:SLOPFAB_LCS_TEST_VAE = 'D:/path/to/minimax_h3_video_vae_fp16.safetensors'
$env:SLOPFAB_LCS_TEST_BACKEND = 'cuda' # or vulkan
& build/Release/slopfab_generation_tests.exe
```

Calibration variance and synthetic trajectories verify the implementation, but
do not establish visual-quality parity with ComfyUI-LCS. Evaluate output quality
on the checkpoints, resolutions and schedules your application uses.

`tests/test_lcs_dll.py` additionally exercises the exported ABI with a real H3
transformer. It verifies native calibration, native-before-callback ordering,
cancellation and rejection of non-finite callback output. It uses a synthetic
prompt embedding and cancels before decoding, so it is not a quality test:

```powershell
python tests/test_lcs_dll.py --dll build/Release/slopfab.dll `
  --vae weights/vae/minimax_h3_video_vae_fp16.safetensors `
  --transformer weights/transformer/fl2va_pruned_fp8_scaled.safetensors
```
