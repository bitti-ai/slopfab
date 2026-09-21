# Extend and Bridge from imported videos

C API 1.14 adds `slopfab_request_set_video_transition(request, mode)`:

- `0`: ordinary generation.
- `1`: Extend, using the final 22 frames of reference video 1.
- `2`: Bridge, using the final 22 frames of reference video 1 and the opening 22 frames of reference video 2.

Attach decoded frames with the existing reference-video API and provide a text prompt describing the new action. Sources must each contain at least 22 frames at 24 fps. Existing reference duration limits still apply. These modes require a reference-capable transformer, exactly one/two videos, and no image references, refmods, Animate, saved-latent continuation, synthetic generation or initial-latent overrides.

Planning validates source ranges and reports only the new output duration; it does not execute a VAE. During generation each boundary is resized to the target canvas and VAE-encoded into seven normalized video latent frames. The existing seed-independent media cache retains those encodings, with transition mode, canvas and source identity included in its key.

The start guide's temporal origin precedes the target by the seven-latent boundary duration. The end guide begins immediately after the target timeline. Both share the target's spatial coordinates, and neither advances the target's origin. Ordinary reference packing and saved-latent continuation retain their prior behavior. This provides temporal latent guidance rather than an exact pixel-copy constraint; visual seam quality depends on the model and prompt.

Only the newly generated segment is returned and saved to a latent archive. Source soundtracks are omitted from guides; audio is generated from the prompt. Hosts assemble source/generated/source clips as needed. No FFmpeg functions are used in this path.
