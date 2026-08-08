# Experimental Sol pipeline quality gate — 2026-08-08

Pipeline remains disabled by default. Runs used commit
`ae10d87b3d7827cbc7c4e55f27a86cd791be9911` and executable SHA-256
`64A2BFDA1EE660896FAF2F0D91F91AE2EBD3231CE7A2D6E991FF8F948CE808EB`.

Common arguments:

```text
generate
--prompt "integrated_multimodal_description: a red ceramic teapot on a wooden table, warm studio light"
--frames 22 --aspect 1:1 --steps 20 --seed 11
--tokenizer D:\Projects\vidfab\ref\text_encoder\tokenizer.json
--text-encoder D:\Projects\vidfab\weights\text_encoder\qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
--transformer D:\Projects\vidfab\weights\transformer\MiniMax_H3_FL2VA_pruned_nvfp4.safetensors
--vae D:\Projects\vidfab\weights\vae\minimax_h3_video_vae_fp16.safetensors
--audio-vae D:\Projects\vidfab\weights\vae\minimax_h3_audio_vae_fp32.safetensors
```

Dense appended `--attention flash2 --dump-latents build-sol\dense-final-latents.safetensors
--out build-sol\dense-final.mp4`. Pipeline set `VIDFAB_SOL_PIPELINE=1` and appended
`--attention sol --dump-latents build-sol\pipeline-final-latents.safetensors
--out build-sol\pipeline-final.mp4`. Both set `VIDFAB_TENSOR_DIAG=1`; every scanned
attention, projection, residual, AdaLN and MLP tensor was finite.

| Artifact | SHA-256 |
|---|---|
| `dense-final-latents.safetensors` | `3935D801695FD91BC5F41306153BFE8C415BDCE2E5642CED9B5901373E26C398` |
| `pipeline-final-latents.safetensors` | `78825EB450D3AE7BB5E13EE1910647117916E339BB6B106111D2DC657807F9EB` |
| `dense-final.mp4` | `5BF8D5E24B8FDF8FB1403AB9BC62A7E5A33E11F8E54001C12F8DD3BF7F874E93` |
| `pipeline-final.mp4` | `59D914444928A1607A24D4151D58EBFC21A7084EA4E696030E6F10D6E4D23B3F` |

Dense denoising was 20.7 s; pipeline denoising was 20.2 s. Final FP32 latent
metrics (pipeline versus dense): video rel-L2 0.0897082, cosine 0.9959682,
correlation 0.9959693, MAE 0.0711768, max 2.07160; audio rel-L2 0.0887819,
cosine 0.9960579, correlation 0.9960093, MAE 0.0326419, max 0.476962.

Decoded 768x768 RGB (21 encoded frames): rel-L2 0.0731024, cosine 0.9974416,
correlation 0.9933166, MAE 4.44229/255, RMSE 7.72542, PSNR 30.3724 dB;
FFmpeg SSIM all 0.964223 (Y 0.952858, U 0.990323, V 0.983584). Decoded stereo
audio (59,392 samples): rel-L2 0.0795319, cosine/correlation 0.9987108,
MAE 0.00980154, RMSE 0.0139272, max 0.151330.

The representative step-10/layer-2 capture (`seq=37715`, `heads=56`, beta 1)
measured 164.112 ms pipeline versus 232.117 ms dense over 10 idle iterations
(1.414x), with 19% exact routes, `REG=191`, `STACK=0`, and all outputs finite.
