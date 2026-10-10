# Bounding-box image editing

Image editing generates new content inside a rectangle while preserving every
decoded source RGB pixel outside it. CUDA and Vulkan are supported. The output
is one still image with the source dimensions, even when those dimensions are
not multiples of 32.

```sh
slopfab generate --edit-image scene.png --edit-box 120,80,240,160 --prompt "A red ceramic vase on the table" --edit-strength 0.8 --edit-feather 8 --seed 17 --out edited.ppm
```

`--edit-box` is `x,y,width,height` in original-image pixels. Its right and
bottom edges are exclusive, and the whole nonempty box must fit inside the
image. The source is decoded once, including on `--dry-run`. Inputs use the
same image decoders as reference images. Source dimensions are limited to
8192 pixels per axis.

`--edit-strength` is in `(0,1]` (default 1). It retains the last
`ceil(strength * evaluations)` evaluations of the resolved schedule, at least
one. Lower strength starts from a less noisy source and usually retains more
of the original content. It is a fraction of evaluations, not a direct sigma
value; H3's shifted schedule makes the relationship nonlinear. Existing
trained schedule points are retained without interpolation.

`--edit-feather` blends inward from the rectangle edges, in source pixels
(default 0). No blending changes pixels outside the box. PPM output is lossless
RGB; applications using the C API receive planar RGB floats directly. An
external conversion to a lossy format may change preserved pixels.

The source is edge-padded to H3's 32-pixel canvas alignment, encoded using the
existing keyframe VAE, and used to initialize the target latents. At each step,
unmasked cells are restored to the source at the resulting sigma. Euler reuses
the initial noise field; re-noising uses the same fresh noise as the sampler
at each step. Latent cells overlapping the box are editable;
final pixel compositing enforces the exact rectangle and removes padding.

Image editing requires a video VAE containing encoder weights and Euler or
re-noising sampling (including DMAD), without step, block or MotionCache reuse.
Both samplers also support outpainting and reduced edit strength. Editing cannot
be combined with initial-latent overrides, video continuation or animation. The source supplies
target latents and does not automatically become a Ref2VA reference. Explicit
references remain available with a compatible Ref2VA checkpoint. Vulkan's
editing path transfers target latents to and from the host after each step.

The CLI writes `.ppm` for edits, with a timestamped filename when `--out` is
omitted. `--resolution`, if supplied, must equal the padded source dimensions;
`--aspect` is rejected. Saved/dumped latents describe the padded generation
before final compositing and do not contain the original source or edit mask.

## C API

Set the models and prompt as for ordinary generation, then attach an edit:

```c
int status = slopfab_request_set_image_edit_rgb24(
    request, pixels, buffer_bytes, image_width, image_height, row_stride_bytes,
    120, 80, 240, 160, 0.8f, 8);
/* Check status, then call slopfab_generation_start as usual. */
```

Alternatively, `slopfab_request_set_image_edit_path(request, path, x, y,
width, height, strength, feather)` decodes a file immediately. Both setters
validate the complete edit, copy its source into an immutable snapshot and
enable still-image mode. Failed calls leave the previous edit intact. Caller
buffers and request handles can be released after their respective copying
operations, following the normal asynchronous generation contract.

`slopfab_request_clear_image_edit` removes the edit and retains still-image
mode. `slopfab_plan.canvas_width/height` report the padded internal canvas;
`slopfab_output.width/height` report original source dimensions. Outside the
box, output floats equal the decoded source bytes divided by `255.0f`.

C++ callers use `GenerateRequest::image_edit` with an immutable `RGBImage`
snapshot and set `still_image = true`. Use a `.ppm` output path or an
`on_samples` callback. Keep the snapshot immutable while any request uses it.

## Outpainting (C API 1.20)

Place the original on a larger canvas, attach that canvas with an image-edit
setter using the original's rectangle and zero feather, then call
`slopfab_request_set_image_edit_invert_mask(request, 1)`. The rectangle now
identifies the source original. All surrounding space is denoised together,
with the original latent context restored at each outer step on CUDA
and Vulkan. Do not split the border into separate edits or submit an enlarged
copy of the original as a reference.

The preserved crop is supplied to Qwen as `Source scene` for visual semantic
context, without adding a resized DiT reference anchor. Explicit user images
keep their existing `<Picture N>` numbering. Outpainting bypasses the prompt
cache so changing the source with the same instruction cannot reuse stale
visual conditioning.

Only latent cells fully inside the original rectangle are locked, allowing
unaligned boundary cells to generate the seam. Pixel compositing preserves
the interior of the original, with the edge blend described below. The source box must contain
at least one complete 16x16 cell and leave room for new pixels. Normal edit
setters reset inversion; failed inversion calls preserve the previous request.
In C++, set `ImageEdit::invert_mask = true`.

Version 1.27.1 also supplies the fully preserved 32-pixel patches as a
positioned keyframe. It reuses the source VAE encoding, places those rows at
the corresponding target RoPE coordinates and time, and keeps them clean
throughout denoising. This is separate from latent locking and Qwen's visual
context; an ordinary resized reference cannot supply this spatial anchor.
Both CUDA and Vulkan use the same packed geometry. Tiny legacy boxes without
a complete patch retain latent locking only; interactive hosts should require
a complete patch of context. Compressed-attention models reject outpainting.

### Edge blend (C API 1.28)

Outpainting now expands the generated mask into the source box by max pooling,
Gaussian-blurs that mask, and composites `original * (1-mask) + generated * mask`.
`ImageEdit::blend_overlap` defaults to 9; C callers can set it after enabling
inversion with `slopfab_request_set_outpaint_blend_overlap(request, 9)`.
Accepted kernel sizes are odd integers from 1 to 51. At 1 the entire original
box is preserved exactly. Larger values blend up to `blend_overlap - 1` pixels
inward on edges facing generated space, including corners; the deeper interior
is still copied exactly. Attaching a new source restores the default.

The Gaussian uses sigma `(blend_overlap - 1) / 4`, following
[LanPaint's mask blend](https://github.com/scraed/LanPaint/blob/master/src/LanPaint/nodes.py).
For our rectangular mask, dilation and separable convolution reduce to two
one-dimensional profiles. We replicate canvas edges instead of zero-padding
the convolution so the new border stays fully generated and source edges
touching the canvas do not acquire a spurious seam. Decode alignment padding
does not enter the mask. Ordinary inpainting retains its existing inward feather.

Hosts should pass the full original rectangle and use this blend instead of
applying a second feather or color correction. Final blending is independent
of the generation sampler and does not add model evaluations.

### Langevin refinement (C API 1.28)

Outpainting also runs five extra model evaluations at each schedule point by
default. Set `ImageEdit::langevin_steps`, or call
`slopfab_request_set_outpaint_langevin_steps(request, steps)` after enabling
inversion, to select 0 through 100 iterations. Zero restores the previous
sampling path. The setting resets to 5 when a new source is attached.

CUDA and Vulkan use the same flow-to-variance-preserving conversion and
overdamped Langevin dynamics, based on
[LanPaint's sampler](https://github.com/scraed/LanPaint/blob/master/src/LanPaint/lanpaint.py).
At a fixed noise level, each inner iteration reevaluates the model and combines
its clean-image estimate with the known source and fresh Gaussian noise. The
step size scales with remaining noise variance (base 0.2, guidance lambda 5,
known-region beta 1). Later iterations use two half-steps with a refreshed
force correction. H3 uses its single distilled prediction for both guidance
terms; this does not implement ComfyUI's separate CFG/Prompt First mode,
early stopping, or tail step-size pinning. The native seeded noise stream is
deterministic but does not reproduce PyTorch seeds.

Only target image latents are refined. Positioned keyframes stay clean and
unchanged, and known target latents are restored at each outer step boundary.
Both Euler and re-noising (including DMAD) support refinement. Ordinary
inpainting does not enable it. Progress still reports the outer schedule step,
with repeated callbacks during refinement to permit cancellation between model
calls; `steps_computed` counts all actual calls. Five inner iterations mean six
model evaluations per outer step. Final edge blending remains independently
configurable and adds no model evaluations.
