# Saved latents and video continuation

Save normalized FP32 video and stereo audio latents while generating:

```sh
slopfab generate --prompt "A camera follows a cyclist along a forest road" --frames 124 --out first.mp4 --save-latents first.safetensors
```

Continue from the archive:

```sh
slopfab generate --continue-from first.safetensors --overlap-frames 22 --frames 119 --prompt "The cyclist keeps riding as the road curves to the right" --out extended.mp4 --save-latents extended.safetensors
```

With `--continue-from`, `--frames` requests **new** frames, rounded up to a
multiple of 17. In this example the model samples a 141-frame window containing
22 frames of hidden overlap and 119 new frames. The output and saved archive
contain the full 243-frame joined clip. Continue again from `extended.safetensors`.
`--count` creates independent branches from the same source and numbers both
media and latent output paths; it does not automatically chain the branches.

The canvas is inherited from the archive. An explicit `--resolution` must match;
omit `--aspect`. The overlap must be exactly `17*k+5` frames, at least 5, and no
longer than the source. The default is 22. Use the same H3 VAE and compatible H3
weights across the chain. This works with the FL2VA transformer, or Ref2VA when
also using ordinary subject/style references. Both CUDA and Vulkan use the same
continuation planning and latent joining code.

The sampler starts from fresh noise. The source tail becomes fixed visual/audio
conditioning at the target window's origin, with H3's normal visual condition
noise augmentation. The sampled overlap is discarded, the suffix is appended to
the original latents, and the cumulative clip is decoded with context across the
join. Source latent values are preserved exactly. Boundary pixels can change
because the VAE decodes overlapping temporal windows; this is not a promise of
bit-identical previously decoded frames or a perfectly seamless motion transition.

Video slices respect H3's temporal grid. Audio boundaries are computed from the
cumulative 24 fps timeline against the 40 Hz latent clock, rather than rounding
each extension independently. The sampling window stays bounded, but stored
latents and full output decoding grow with the cumulative clip length.

## DLL (C API 1.9)

For automatic saving, configure the request before generation:

```c
slopfab_request_set_save_latents(req, "first.safetensors");
```

For a later process or a saved project, load an archive into a new request:

```c
slopfab_request_set_continuation_file(next_req, "first.safetensors", 22);
slopfab_request_set_frames(next_req, 119); /* new frames */
slopfab_request_set_save_latents(next_req, "extended.safetensors");
```

The setter loads an owning snapshot immediately. The file may then be moved,
deleted, or overwritten. A failed attachment leaves the previous source intact.

To hand off latents without disk I/O, enable retention **before starting** the
first generation. Once it succeeds:

```c
/* Before slopfab_generation_start: */
slopfab_request_set_retain_latents(req, 1);

/* After first_gen has completed successfully: */
slopfab_request_set_continuation_generation(next_req, first_gen, 22);
slopfab_request_set_frames(next_req, 119);
slopfab_request_set_retain_latents(next_req, 1); /* if extending this one later */
slopfab_generation_save_latents(first_gen, "first.safetensors"); /* optional */
slopfab_generation_destroy(first_gen); /* next_req retains shared ownership */
```

Check every returned status in application code. Retention defaults off. Access
to retained latents requires a successful, completed generation; running and
failed generations return their corresponding status. Clear continuation with
`slopfab_request_clear_continuation`. Passing NULL or an empty path to
`slopfab_request_set_save_latents` disables automatic saving.

The existing `slopfab_output` remains unchanged: it contains the **full joined**
video and audio. `slopfab_plan.aligned_frames` and `duration_seconds` describe that
full output. Its latent dimensions and sequence counts describe the bounded
sampling window (including the hidden overlap), for estimating inference costs.
`slopfab_describe_plan` reports both lengths explicitly. No FFmpeg is needed by
these DLL operations.

## Archive format and limitations

`--save-latents` writes a safetensors archive with `video_rows [V,96]` and
`audio_rows [2*A,32]`, both F32. Metadata identifies `h3-av-v1`, dimensions, frame
count, 24 fps, whether sampling ran, and the most recent model/VAE paths. Paths
are provenance, not embedded weights or cryptographic model identities. Use the
same VAE normalization when reusing an archive. Saves use a temporary sibling
file and replace the destination after writing succeeds. Create the destination
directory first.

Saving happens after sampling/joining and before decoding, so a later decoder
failure can still leave a usable saved archive. Latents are also saved when the
audio decoder is omitted, since the model still generates audio latents.
Still-image and diagnostic synthetic runs can be saved, but cannot be used as
video-continuation sources.

`--dump-latents` remains the diagnostic dump of the **sampling window**, without
continuation metadata. Old dumps cannot be passed to `--continue-from` because
their canvas/timeline cannot be recovered reliably. `--init-latents` still
replaces initial noise and cannot be combined with continuation.

At 1344x768, a roughly 15-second clip occupies about 41.6 MB in this format.

## Validation

Host tests cover archive roundtrips and rejection, owning snapshots, guide
coordinates/timesteps, stereo tail selection, overlap removal, and thirty
successive extensions without cumulative audio rounding drift. The real-weight
DLL smoke test covers automatic and explicit saving, an in-memory handoff, a
file handoff after deleting the source file, and finite joined video/audio:

```sh
python tools/continuation_smoke.py build/Release/slopfab.dll . cuda
python tools/continuation_smoke.py build/Release/slopfab.dll . vulkan
```

The CUDA and Vulkan smoke runs both passed with 22, 39, and 56 joined frames.
This deliberately uses a tiny canvas and one model evaluation to validate
plumbing. It does not measure continuation quality at normal generation settings.
The temporal-guide approach follows the native H3 guide formulation described
in [ComfyUI's H3 model](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/minimax/model.py)
and the [guided continuation workflow](https://github.com/ttulttul/ComfyUI-Minimax-H3-Continuation).
