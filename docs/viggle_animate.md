# Viggle-Animate transformer checkpoints

For CUDA denoising VRAM reductions and stage profiling, see
[Denoising memory](denoising_memory.md). Flash2 uses compact Q/output buffers
by default; projection/FFN scratch is bounded to 2,048 rows.
CUDA generation also automatically streams selected transformer blocks and
their adapters from CPU RAM when the estimated VRAM requirement exceeds the
available budget. See the linked document for budget controls and measurements.

The CUDA and Vulkan transformer loaders accept
`Viggle-Animate-pruned_rank8_int8_convrot.safetensors` directly. Select it
explicitly with `--transformer` or `SLOPFAB_MODEL_TRANSFORMER` in the C API.
Automatic H3 checkpoint discovery continues to select the general-purpose
FL2VA/Ref2VA models.

```sh
slopfab inspect weights/transformer/Viggle-Animate-pruned_rank8_int8_convrot.safetensors
```

The model is identified from `source=Viggle/Viggle-Animate` metadata, so
renaming the file preserves its identity. Without source metadata, the
`Viggle-Animate` filename marker identifies the family. Preserve the
`qkv_layout` metadata: it determines how attention weights are loaded.

## Checkpoint layout

The inspected checkpoint contains 50 main blocks and two token-refiner
blocks, with the usual H3 dimensions (5376 hidden channels, 56 heads of
128 channels, 5120 text features). Its AdaLN curve is F32 `[1025,8]`, and
the rank-eight AdaLN projections and biases remain F32.

`qkv_layout=interleaved` stores attention rows as
`[head, Q/K/V, channel]`. Both backends reorder these into `[Q; K; V]` when
loading, including the per-row F32 INT8 scales. The BF16 token-refiner
weights receive the same permutation. This preserves the quantized values
and ConvRot input transformation; it does not requantize the model.
Checkpoints without layout metadata retain the existing contiguous layout.
Unsupported layout names and interleaved packed NF4/NVFP4 weights are rejected.

## Animation workflow requirements

Use `--animate` (C API 1.10: `slopfab_request_set_animate`) to select the
fixed-conditioning recipe. It requires exactly one driving video and one
repainted scene frame. Generic H3 reference generation retains its own defaults.

The [upstream inference script](https://huggingface.co/Viggle/Viggle-Animate/blob/main/inference/sample.py)
uses a frozen embedding with modality tags, a driving video followed by one
repainted image, and references resized to the target's short edge. It omits
Qwen and drops the driving video's reference soundtrack. Its distilled
configuration uses four sigma boundaries (three forward passes), Euler
sampling, and video flow shift 3.

The local checkpoint explicitly declares `distillation_lora_merged=false`.
Its rank-eight AdaLN compression is separate from the distillation adapter.
The [upstream model card](https://huggingface.co/Viggle/Viggle-Animate) requires
that adapter on the Viggle finetune for the three-pass recipe. Slopfab now loads
`viggle_animate_distillation_bf16.safetensors` directly on CUDA and Vulkan.
Add these options to your generation command:

```sh
--transformer weights/transformer/Viggle-Animate-pruned_rank8_int8_convrot.safetensors --lora weights/loras/viggle_animate_distillation_bf16.safetensors --lora-strength 1
```

All 604 tensors are used across 302 projections. Diffusers' separate Q/K/V
adapters retain their per-projection head order, and the first feed-forward
projection's B rows are reordered from `[value; gate]` to H3's `[gate; value]`.
The `proj_in` and `proj_out` updates are merged once into the F32 video
endpoint weights at load time, without rewriting the checkpoint. The adapter's
embedded PEFT metadata supplies alpha 128 for its rank-128 factors.

Selecting a local Viggle-Animate transformer automatically sets video flow
shift 3 in the CLI and C API, using the same metadata/filename detection as
the loaders. With `--steps 4 --sampler euler`, its sigma boundaries are
`1 -> 0.857143 -> 0.600000 -> 0` (three forward passes). Audio shift remains 3;
general-purpose H3 video models retain shift 12. Planning inspects only the
checkpoint header; without a local checkpoint, it uses the H3 defaults.

Loading the adapter alone does not select the step count. Animate mode defaults
to four boundaries and requires Euler without step/block caching, continuation,
refmods, or supplied initial latents.

## Frozen conditioning and references

Download `assets/fixed_embed_fwd_anyframe.pt` from the upstream repository, then
convert it once (requires PyTorch and NumPy):

```sh
python tools/convert_viggle_embedding.py fixed_embed_fwd_anyframe.pt weights/conditioning/viggle_animate.safetensors
```

The converter preserves the 362 x 5120 BF16 values exactly in F32 and copies the
modality tags. SlopFab reads `prompt_embedding` and `text_token_tags` directly;
it never tokenizes the prompt or references, and never loads Qwen in this mode.
The generic embedding API also accepts video/image references when explicit
I32/I64 `text_token_tags [L]` are included.

Animate packs the driving video first, then the repainted image, including their
latent payloads, modality indices, noise streams, and shared rotary clock. Both
references use the target's short edge; video resizing also uses the target's
pixel budget. With no explicit canvas, the driving video's aspect determines
the canvas, with its short edge rounded down to a multiple of 32. Use
`--resolution WxH` to choose the same canvas as an upstream comparison.

The reference image should be a frame of the driving scene with the performer
repainted as the desired character. Preserve pose, framing, background, and
lighting. A standalone portrait on white is not an equivalent input.

## Soundtrack preservation

`--preserve-driving-audio` encodes the driving soundtrack once, pads/truncates
it to the aligned output duration plus one video frame, and takes the required
normalized posterior-mean latents for each stereo channel. The target audio
rows remain clean (`t=1`) on every forward pass and are never scheduler-updated,
on CUDA and Vulkan. They remain target rows and are decoded normally. A runtime
check verifies they are bitwise unchanged after denoising.

The driving soundtrack is never a reference in Animate mode. Without this flag,
output audio is generated. Preservation follows upstream's audio-VAE round trip;
it does not promise sample-identical PCM or establish the cause of earlier
corrupted audio.

```sh
slopfab generate --animate --prompt-embedding weights/conditioning/viggle_animate.safetensors --reference-video driving.mp4 --reference-image repainted.png --resolution 704x1248 --frames 124 --preserve-driving-audio --transformer weights/transformer/Viggle-Animate-pruned_rank8_int8_convrot.safetensors --lora weights/loras/viggle_animate_distillation_bf16.safetensors --video-vae weights/vae/minimax_h3_video_vae_fp16.safetensors --audio-vae weights/vae/minimax_h3_audio_vae_fp32.safetensors --out animated.mp4
```

Reference clips must be 2-15 seconds; trim longer sources first. The C API takes
decoded frames and PCM as before. Call `slopfab_request_set_animate(req, 1, 1)`
to select four boundaries, Euler, and preserved driving audio, then set the
embedding path, transformer, LoRA, references, and any canvas/frame overrides.
Pass zero as the third argument for generated audio. Existing hosts must call
this new setter to opt in; attaching PCM alone retains generic reference behavior.

## Verification

Host tests cover metadata/filename detection, Ref2VA acceptance, invalid
layouts, and an INT8 projection with distinct per-row scales. The actual
downloaded checkpoint is checked across all 52 blocks, including Vulkan's
host-side archive validation. CUDA and Vulkan regression fixtures compare
forward results from equivalent contiguous and interleaved checkpoints;
the Vulkan fixture also checks F32 AdaLN storage.

The LoRA host suite loads the actual 604-tensor adapter against the downloaded
checkpoint, validates all 302 targets, and merges both video endpoints.
Synthetic tests cover alpha/strength scaling, Q/K/V dimensions, SwiGLU order,
stacking, and malformed inputs. GPU regression cases check that each of the
eight projection types affects the forward pass on both backends.

The Release CLI and DLL build. Host and C API tests cover fixed embedding tags,
Animate geometry, ordering, cache separation, soundtrack padding, and stereo
latent cropping. CUDA loop tests verify clean audio timesteps and unchanged
samples at every boundary; an independent Vulkan fixture verifies three pinned
audio boundaries while video advances. Broader Vulkan stage tests currently
fail on operator-count and AWQ/LoRA checks unrelated to this recipe.

### Local comparison, 2026-09-16

The CUDA DLL rendered 124 frames at 704x1248 using the supplied shuffle video,
a repainted first frame, the shipped 362-token embedding, the INT8 ConvRot
transformer and distillation adapter. Intentionally nonexistent Qwen/tokenizer
paths confirmed that neither was loaded. The three forward passes completed,
and the runtime check confirmed bitwise-unchanged target audio latents.
Decoded audio correlated with the source resampled to 32 kHz at 0.975/0.977
for the left/right channels over the first 5.167 seconds.

**Visual validation failed:** the output is a nearly uniform brown texture,
not an animated performer. A separate encode/decode of the repainted frame
with the same video VAE reconstructs the person and scene correctly; its still
and repeated-latent video decode paths agree exactly. This narrows investigation
to the conditioning/transformer path, but does not establish the cause or rule
out every temporal decoding issue. The recipe changes are not evidence of
successful end-to-end Animate quality.

`tools/animate_generation_smoke.py` reproduces the DLL comparison without Qwen
and saves video, decoded audio and normalized latents. Its runtime checks do
not assess visual quality. `slopfab_stillprobe --reference-image FILE` bypasses
denoising for the VAE round-trip diagnostic. Local comparison artifacts and
logs are under `output/animate-comparison/`; they are not committed.
