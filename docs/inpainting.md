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
unmasked cells are restored to the source at the resulting sigma using the
same noise field throughout. Latent cells overlapping the box are editable;
final pixel compositing enforces the exact rectangle and removes padding.

Image editing requires a video VAE containing encoder weights and Euler
sampling without step, block or MotionCache reuse. It cannot be combined with
initial-latent overrides, video continuation or animation. The source supplies
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
