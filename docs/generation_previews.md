# TAEH3 generation previews

C API 1.46 adds `slopfab_request_set_preview(request, path, callback, userdata)`.
Supply the [TAEH3 checkpoint](https://github.com/madebyollin/taehv/blob/62f7591f59dfbb4c3c02b7a621d180a9eeaba26c/safetensors/taeh3.safetensors)
and copy the callback's packed RGB8 frames before returning. The callback and
userdata must live until the generation handle is destroyed. A null callback
or empty path disables previews. The standard progress/output structs are unchanged.

The CUDA and Vulkan denoisers decode clean video predictions before scheduler
updates, including re-noising. Preview decoding does not modify diffusion or
the final VAE selection. Missing or invalid preview weights disable previews
for that run while final generation continues. Cancellation is checked between
preview frames and residual blocks.

The decoder follows the upstream base TAEH3 architecture with spatial and
temporal upsampling disabled. It retains the final pixel shuffle, uses the
last temporal-growth channels, and resets temporal memory between previews.
Input is normalized diffusion-space CTHW without the full VAE's scale/shift.
Large latent grids and long sequences are sampled down to at most 128 pixels
per latent axis and 127 latent frames. RGB output is at most 256 pixels per
axis. `fps` spans the sampled segment; retained continuation source footage
is not included. Previews have no audio.

For parity checks, use the pinned upstream `taehv.py` and checkpoint:

```sh
python tools/taeh3_reference.py /path/to/taehv.py /path/to/taeh3.safetensors reference.safetensors 37 26 46
slopfab_taeh3_probe /path/to/taeh3.safetensors reference.safetensors cuda
slopfab_taeh3_probe /path/to/taeh3.safetensors reference.safetensors vulkan
```

The probe compares RGB8 output, checks cancellation and checks repeatability
after resetting temporal memory. PyTorch is used only to make test fixtures.
