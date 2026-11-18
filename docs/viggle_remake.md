# Viggle Remake

[Viggle Remake](https://huggingface.co/Viggle/Viggle-Remake) replaces Animate
in the CLI and download catalog. `--remake swap` replaces characters;
`--remake edit` changes objects, backgrounds, weather or lighting. Both use
the native H3 CUDA/Vulkan pipeline, one source video and one to four painted
frames, with a six-section H3 prompt. Optional timeline pins anchor painted
frames at specific positions in the output. Without pins, references determine
appearance and the model chooses the composition.
Use a short edge of 480 or 736 for visual work; very small test canvases can
produce abstract textures despite completing inference successfully.

## Weights

The local `weights/viggle_remake_swap_pruned_int8_convrot.safetensors` already
contains the swap LoRA. Use it alone. The separate
`weights/viggle_remake_swap_lora_r64.safetensors` is an **alternative** for an
unmerged **FL2VA** H3 transformer, not an extra adapter for the merged file.
The equivalent edit downloads are in `slopfab download viggle-remake`.
Both paths share the H3 text encoder and video/audio VAEs. No Animate fixed
embedding, Animate distillation adapter or Turbo adapter is needed.

Merged exports have no identity metadata. Keep their distribution filenames
or add `source=Viggle/Viggle-Remake` and `mode=swap`/`edit` metadata when
renaming them. Their QKV layout is contiguous; the old Animate checkpoint's
layout workaround does not apply. The rank-64 Diffusers adapter loads directly
on both backends using its embedded alpha 64 and strength 1.

## Generate

Prepare a 24 fps Y4M source on the desired canvas (both dimensions multiples
of 32), and extract its soundtrack to WAV. Supply `--driving-audio` to retain
it; omitting that option with a silent Y4M source produces silent audio.
Audio goes through the audio VAE, so it is not sample-identical passthrough.

```sh
slopfab video generate --remake swap --transformer weights/viggle_remake_swap_pruned_int8_convrot.safetensors --text-encoder weights/text_encoder/qwen3vl_32b_int8_convrot.safetensors --vae weights/vae/minimax_h3_video_vae_fp16.safetensors --audio-vae weights/vae/minimax_h3_audio_vae_fp32.safetensors --reference-video source.y4m --driving-audio source.wav --reference-image painted.png --prompt-file prompt.txt --frames 124 --resolution 832x480 --out output/PROJECT_NAME.y4m
```

The command writes Y4M video and a WAV soundtrack; use external FFmpeg to
encode/mux the final MP4.

### Timeline pins

Add `--pin FRAME` once per `--reference-image`, in the same image order, or
omit pins entirely. Indices are zero-based source/output frames and must be
less than `--frames`. For example, two painted frames can be anchored with:

```sh
--reference-image painted-start.png --reference-image painted-cut.png --pin 0 --pin 96
```

Each pinned image also remains an ordinary appearance reference. A second
copy is encoded at the target canvas and placed at that frame's rotary time.
The first keyframe is stretched to the canvas; subsequent keyframes are
scaled to cover it and center-cropped, matching upstream's fitting rules.

`--keyframe-until 0.9` is the default. Timeline keyframes participate while
the current **video sigma is at least 0.9**, then are removed from the model
input. This holds them for the first four evaluations of either published
six-step grid. With `--steps N`, release still follows sigma, not a fraction
of the step count. Values range from 0 (keep throughout) to 1 (only at sigma
1). Pins guide composition; they do not replace output pixels or clamp target
latents. Use them when shot timing or a painted frame's composition matters.

JSON requests use `options.pin: [0, 96]` and `options["keyframe-until"]: 0.9`.
The C++ fields are `remake_pins` and `remake_keyframe_until`. C API 1.53 adds
`slopfab_request_set_remake_pins(request, frames, count, keyframe_until)`;
passing count zero clears the pins. Both CUDA and Vulkan support release.
VEDA attention is incompatible with the changing keyframe sequence; use the
default attention mode for pinned requests.

For the adapter alternative, replace `--transformer` with an unmerged pruned
H3 FL2VA file and add `--lora weights/viggle_remake_swap_lora_r64.safetensors`.
The CLI rejects adapters on a merged Remake file, mismatched merged modes,
and Ref2VA bases for the separate adapter workflow. For Vulkan add
`--inference-backend vulkan --vulkan-arithmetic portable`.

Frames must be `17*n+5`, from 90 through 345, and the source must cover that
duration. The input source determines the default canvas; an explicit
resolution controls the target/reference short edge. Pre-crop inputs to the
target canvas for an upstream comparison. Longer videos need separate windows.

Write the six sections `subject_definitions`, `summary`, `retention_analysis`,
`detailed_description`, `overall_soundscape`, and `non_diegetic_music`, in
that order. Describe the result, reference the source as `<Video 1>` and the
painted frames as `<Picture 1>` through `<Picture 4>`, and keep timestamps
within the clip. See upstream's [prompt guide](https://huggingface.co/Viggle/Viggle-Remake/blob/main/PROMPTING.md).

Remake tokenizes the source/timestamp/image labels and prompt as separate
text segments. Qwen receives no image/video vision tokens; the video VAE
encodes the actual references. The 32B H3 encoder matches the upstream
recipe. ClipProj is also accepted as an approximate text-only alternative;
matching upstream quality with a smaller encoder is not established. Cached
embeddings must use the same labels and contain only text modality tags.

## Sampling and library API

Both modes default to six Euler updates, video/audio shifts 12/3 and CFG 1.
Set `--steps N` (for example, `--steps 12`) to change the number of updates.
JSON requests use `options.steps`; library requests use `num_inference_steps`
or `sampling.default_steps`, and the C API uses `slopfab_request_set_steps`.
The explicit step count takes precedence over `sampling.default_steps`.
At six steps, the published mode-specific video sigma grids are preserved:

| Mode | Video sigmas, including terminal zero |
|---|---|
| swap | `1, .990476, .965243, .923077, .8, .631579, 0` |
| edit | `1, .983607, .96, .923077, .857143, .705882, 0` |

Other step counts use a uniform unshifted grid from 1 to 0 with the same
video/audio shifts 12/3. These are custom schedules, not the published
six-step recipe. The CLI still rejects alternate samplers, grids, shifts,
caches and other editing modes.

The catalog also records the corresponding **unshifted** `base_sigmas`;
these are not interchangeable with the video sigma list. Source audio stays
clean throughout denoising. Reference order is source video, then painted
images, on both backends.

C API 1.51 adds `slopfab_request_set_remake(request, "swap")` (or `"edit"`;
`""` disables it). Attach decoded video/PCM and painted image paths using
the existing reference setters. The older Animate setter remains available
for ABI compatibility; its frozen-conditioning recipe is separate.

## Validation

The local swap INT8 checkpoint rendered 90 frames at 832x480 with two upstream
painted references and a shortened six-section prompt. Beginning, middle and
final frames show the pineapple mascot following the source performance.
CUDA and Vulkan also completed six-step 160x96 execution tests; these tiny
canvases are not quality tests. Both runs verified that the pinned audio
latents stayed unchanged. The local rank-64 adapter passed all 302 projection
target checks against the FL2VA base. Edit-mode schedules and request validation
are covered by host tests; edit weights were not available for a visual test.

Timeline-pin tests cover frame validation, image fitting, rotary placement,
sequence removal, and sigma-based release for both modes and custom counts.
CUDA denoising tests and Vulkan device tests verify target continuity and
unchanged source audio; Vulkan also covers cancellation and repeated runs.
The swap INT8 checkpoint completed two-pin (frames 0 and 60), 90-frame
160x96 execution tests on CUDA at six steps and Vulkan at twelve steps, using
the default 0.9 release threshold. Both produced video and audio successfully
and retained the pinned audio latents. These small renders validate execution,
not visual quality or pixel-exact agreement with upstream.
