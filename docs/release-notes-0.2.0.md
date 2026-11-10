# slopfab 0.2.0

Changes since **0.1.0**:

- **MiniMax-H3-Image:** native six-step image generation on CUDA and Vulkan, with up to 16 reference pictures, prompt editing, rectangular and pixel-mask inpainting, and experimental outpainting. Canvas sizes now support up to 20 Mi-pixels (20,971,520 pixels), including 5120x4096. Outpainting preserves the source interior but can produce visible seams.
- **P-H3 Image:** native eight-step image generation on CUDA and Vulkan using the matching Qwen3-VL text connector and image VAE.
- **Viggle Remake:** replaces Animate in the CLI and download catalog. Supports character replacement and scene editing from a source video and painted reference frames, with source-audio retention through the audio VAE.
- **H3 Regenerate 2K:** adds `slopfab video regenerate` for draft-video refinement up to 2560x1440, with `up3`, `up4`, and `hf8` modes on CUDA and Vulkan.
- **Image-editing CLI:** adds `--edit-mask` for grayscale masks and `--outpaint` for canvas expansion by pixel margins, with edge-blending controls. Black mask pixels preserve the source, white pixels select generated content, and gray pixels blend the two.
- **Automatic updates:** regular commands check for and install updates in the background once per UTC day. Global settings store `last_update_check`; disable automatic updates with `slopfab settings set auto_update false`.
- **Model downloads:** adds P-H3 Image, MiniMax-H3-Image, Viggle Remake, and H3 Regenerate 2K entries to the download mod.
- **C API:** advances from 1.50.0 to 1.52.0, adding Remake and Regenerate 2K request setters, the new image model families, and support for up to 16 MiniMax-H3-Image reference pictures.
