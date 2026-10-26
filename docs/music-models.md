# YuE2 and MiniMax Music3

Native CUDA and Vulkan music generation is available in the CLI and C API 1.38. YuE2 supports lyric and style conditioning with optional ABC planning. MiniMax Music3 supports caption and lyric conditioning with autoregressive acoustic codes, flow synthesis and overlapping audio windows. Both return stereo audio without a Python runtime or an online service.

## Checkpoints

The supplied single-file Comfy INT8 ConvRot checkpoints are supported directly:

```text
weights/yue2_3b_int8_convrot.safetensors
weights/minimax_music3_dit_int8_convrot.safetensors
weights/minimax_music3_dav.safetensors
weights/text_encoder/minimax_music3_text_encoder_pruned_int8_convrot.safetensors
```

YuE2 contains its tokenizer, autoregressive and flow transformers, and audio decoder in one bundle. MiniMax requires all three files. The pruned MiniMax language model was downloaded to Downloads, verified against SHA256 `010b7416d2336a08c711bc22ee65849c9623069ddb7d89bec011a75699e52014`, and moved to the text encoder folder.

The CLI discovers these files under `weights/`, `weights/transformer/`, `weights/text_encoder/` and `weights/vae/`. Explicit paths use `--transformer`, `--text-encoder` and `--audio-vae`. The family can also be inferred from an explicit transformer. Raw Hugging Face sharded directories and unfused Diffusers layouts are not supported checkpoint inputs. The music implementation does not download weights at runtime.

## Generate audio

From the repository root on Windows:

```powershell
build/Release/slopfab.exe generate --model yue2 --prompt "Warm piano ballad with a gentle female vocal" --lyrics "[verse] Morning light across the sea" --music-duration 30 --seed 42 --out yue2.wav
build/Release/slopfab.exe generate --model yue2 --prompt "Gentle instrumental piano and strings" --music-planning off --music-duration 8 --seed 42 --out piano.wav
build/Release/slopfab.exe generate --model minimax_music3 --prompt "Warm piano ballad with a gentle female vocal" --lyrics "[verse] Morning light across the sea" --music-duration 30 --seed 42 --out music3.wav
```

For Vulkan, add `--inference-backend vulkan --vulkan-arithmetic portable` to either model's command:

```powershell
build/Release/slopfab.exe generate --model yue2 --inference-backend vulkan --vulkan-arithmetic portable --prompt "Gentle instrumental piano and strings" --music-planning off --music-duration 8 --seed 42 --out yue-vulkan.wav
build/Release/slopfab.exe generate --model minimax_music3 --inference-backend vulkan --vulkan-arithmetic portable --prompt "Gentle instrumental piano with warm strings" --music-duration 12 --seed 42 --out music3-vulkan.wav
```

Vulkan executes every neural stage, including autoregressive generation, flow synthesis and waveform decoding. It does not call CUDA. Builds with `-DSLOPFAB_ENABLE_CUDA=OFF` support both music models. Vulkan music requires portable arithmetic; exact arithmetic is rejected because cross-backend bit parity is not guaranteed. Music selects fixed attention kernels internally, so leave `--attention` unset.

`--prompt` is the style/caption, and `--lyrics` supplies lyrics. Duration defaults to 30 seconds, accepts `(0,300]`, and limits generated acoustic frames: the model may emit EOS earlier. It is rounded up to acoustic frames at 25 Hz. YuE2 returns native 48 kHz audio; the supplied MiniMax DAV returns native 44.1 kHz audio. Codec framing and window crops can make the final WAV duration differ slightly from the requested limit. MiniMax's model card describes a 32 kHz service output; slopfab preserves this checkpoint's 44.1 kHz decoder output without resampling.

`--steps` controls flow integration, independently of the autoregressive frame budget. Defaults are 32 midpoint steps for YuE2 and 30 Euler steps per MiniMax window. MiniMax uses windows of up to 200 acoustic frames, advances by 100 frames, carries the preceding latent context and crops decoded overlaps. Long requests therefore involve multiple flow windows.

## Music controls

`--music-settings` accepts a JSON object. Individual `--lyrics`, `--music-duration` and `--music-planning` options are also available. For multiline lyrics or externally supplied ABC, JSON avoids shell newline escaping.

| Setting | Default and meaning |
| --- | --- |
| `lyrics` | Empty string; lyric text with optional section tags |
| `duration` | 30 seconds; positive, at most 300 |
| `planning` | YuE2 `full`, `melody` or `off`; full includes chords in ABC |
| `abc` | YuE2 external ABC; requires full or melody planning and bypasses ABC generation |
| `abc_max_tokens` | YuE2 planning budget, 4096; range 32–4096 |
| `temperature` | YuE2 1; zero selects greedy sampling |
| `top_p` | YuE2 0.95; positive, at most 1 |
| `top_k` | Zero selects YuE2 100 or MiniMax 50; MiniMax maximum 1024 |
| `repetition_penalty` | YuE2 1.2; at least 1 |
| `semantic_guidance` | Zero selects YuE2 1 for ABC planning, 1.01 with planning off, or MiniMax 1.5; explicit values must be at least 1 |
| `flow_guidance` | MiniMax only; zero selects 1.7; 1 disables the conditional/unconditional difference |

MiniMax uses top-k sampling and rejects nondefault YuE2-only controls. Unknown JSON keys and invalid ranges are rejected. Seeded generation is reproducible on the same native backend; seeds are not equivalent to PyTorch RNG seeds. User text is tokenized as ordinary text so literal special-token spellings cannot inject protocol controls. YuE2 includes NFC normalization and the checkpoint's whole-token merge behavior.

## DLL integration

Use `dist/windows/slopfab.dll` with the matching `capi.h`. The existing C ABI remains compatible.

1. Set `SLOPFAB_MODEL_TRANSFORMER` to the YuE2 bundle or MiniMax flow checkpoint with `slopfab_request_set_model_path`. MiniMax additionally needs `SLOPFAB_MODEL_TEXT_ENCODER` and `SLOPFAB_MODEL_AUDIO_VAE`.
2. Set family `yue2` or `minimax_music3`, or leave it empty for checkpoint detection. Unlike the CLI, the C API requires explicit model paths.
3. Set the caption with `slopfab_request_set_prompt` and music controls with `slopfab_request_set_music_settings(request, json)`. The JSON setter replaces all music settings; a rejected call leaves the previous settings intact.
4. Select CUDA or select `SLOPFAB_INFERENCE_VULKAN` with `slopfab_request_set_inference_backend`. For Vulkan, also call `slopfab_request_set_vulkan_arithmetic(request, SLOPFAB_VULKAN_ARITHMETIC_PORTABLE)`. Leave attention unset or explicitly select `flash2`; music resolves this default even in Vulkan-only builds. Music chooses its own kernels internally, including FP32 softmax/PV for the MiniMax flow transformer.
5. Start and wait using the existing generation API. Read the existing output structure: video is empty, audio is interleaved stereo float data, and `audio_sample_rate` reports the native rate. Copy audio before destroying the generation handle.

Progress and cancellation are checked between transformer blocks and pipeline stages. Model memory is released between MiniMax AR, flow and decoding stages. When expanded weights are cached, their compressed GPU copies are released to avoid retaining both representations. Real generation was tested on an RTX 5090 with 32 GB VRAM; smaller GPUs have not been characterized.

YuE2 reports `TRANSFORMER_LOAD` before loading its bundle. During `CONDITIONING`,
`step` is the zero-based index of the last completed ABC or semantic token, with
`-1` before the first token. `total_steps` is the combined token budget and shrinks
if generation ends early at EOS. During `DENOISING`, the same fields count completed
flow iterations across all chunks, without resetting between chunks. Each midpoint
iteration includes two model predictions. Use the callback's `total_steps`, rather
than the plan's model-evaluation estimate, for the progress denominator.

For either YuE2 phase, display `100.0 * (step + 1) / total_steps` when
`total_steps > 0`. This is progress within the current phase. Block heartbeats can
repeat the same values while work is in flight; they preserve responsive
cancellation and do not represent additional completed steps. Existing callback
structures and function signatures are unchanged.

## CUDA performance

CUDA music reuses temporary activation and attention buffers within each autoregressive stage and flow window/chunk. Buffers are scoped to a generation stage, use the runtime's default stream, and are released before the next stage; checkpoint weights keep their existing lifetime. MiniMax also collects each frame's eight hidden vectors on the GPU before a single download. Single-token attention batches query heads across the KV heads, reducing MiniMax's per-layer attention submissions from 96 to 9 and YuE2's from 48 to 5. Prefill and FP32 flow attention retain their existing kernels.

Batched GEMMs retain BF16 inputs/probabilities and FP32 accumulation but can reorder sums. Generated audio can therefore differ from older builds with the same seed. Repeatability within a build remains tested. Synthetic attention checks compare both the previous per-head path and an independent CPU reference, including partially filled KV caches.

Measured on an RTX 5090 under Windows on 2026-10-26, using the supplied INT8 ConvRot checkpoints, caption `Warm instrumental piano`, seed 42, and two serial C API generations per build. Values are mean seconds, including model preparation in the reported stages. MiniMax uses a 12-second ceiling and 30 steps per overlapping window; YuE2 uses an 8-second ceiling, 32 steps and planning off.

| Model / stage | Before | After | Speedup |
| --- | ---: | ---: | ---: |
| MiniMax conditioning | 39.34 | 18.80 | 2.09x |
| MiniMax flow | 49.66 | 29.27 | 1.70x |
| MiniMax total | 90.23 | 49.31 | 1.83x |
| YuE2 conditioning | 9.36 | 4.43 | 2.11x |
| YuE2 flow | 6.73 | 2.80 | 2.40x |
| YuE2 total | 17.27 | 8.40 | 2.06x |

Both builds produced 529408 stereo frames for MiniMax and 383936 for YuE2. Repeated runs of the optimized build produced byte-identical audio. Buffer pooling alone also preserved MiniMax's previous audio exactly; batched attention changes rounding as described above. Benchmark JSON, raw audio and logs are retained locally under `build/music-perf/`.

Validation passed 69 CUDA runtime checks, the unit/music CLI/C API/embedded-tokenizer suites, an 8-second YuE2 run with full planning capped at 128 ABC tokens and all 32 flow steps, and cancellation within both conditioning and flow followed by successful recovery. MiniMax's shared hidden-vector collection also passed repeated Vulkan audio and active-cancellation checks.

To measure each stage with local checkpoints, the smoke harness accepts longer durations, flow steps and optional YuE2 planning. It reports per-run timing JSON while checking finite audio, same-seed repeatability and cancellation:

```powershell
python tools/music_generation_smoke.py --library build/Release/slopfab.dll --family minimax_music3 --duration 12 --steps 30 --timings-json build/music-perf/minimax.json
python tools/music_generation_smoke.py --library build/Release/slopfab.dll --family yue2 --duration 8 --steps 32 --timings-json build/music-perf/yue.json
python tools/music_generation_smoke.py --library build/Release/slopfab.dll --cancel-stages
```

The harness defaults to YuE2 planning off; add `--planning full --abc-max-tokens 128` for a bounded planning test. `--cancel-stages` additionally cancels after several transformer blocks in conditioning and flow, then verifies that a fresh generation succeeds. Compare timings with no competing GPU jobs and account for the first run's initialization costs.

## Limits and diagnostics

Music requires CUDA or a Vulkan device with 16-bit storage, shader int64, timeline semaphores, 256-thread workgroups and the required FP32 arithmetic features. Cooperative BF16 matrix multiplication is used when supported; other devices use the portable scalar GEMM path. Hardware validation so far covers an RTX 5090, not AMD or Intel GPUs. Refmods, LoRA, reference audio, video/image controls, H3 sampling schedules/caches, precomputed prompt embeddings, upscaling and continuation/bridge controls are unsupported. MiniMax text is limited to 5000 tokens and the combined AR context to 10240 tokens. YuE2 uses its 24576-token context with flow chunking.

`--dump-latents` saves FP32 values of the BF16 flow result: YuE2 `audio_latents` has shape `[acoustic_frames,64]` and `semantic_tokens` records the generated token IDs; MiniMax saves `audio_chunk_0`, `audio_chunk_1`, etc., each `[flow_latent_frames,128]`. These are diagnostic tensors, not resumable H3 latent archives. MiniMax latent frame counts are `floor(acoustic_frames * 441 / 128)` per window.

## Validation

Portable music and C API tests cover settings, rejected controls, audio geometry, step counts and overlapping windows. Development tools exercise local checkpoints:

```powershell
python tools/music_generation_smoke.py --library dist/windows/slopfab.dll
python tools/music_generation_smoke.py --library dist/windows/slopfab.dll --backend vulkan --repeats 5
```

The Windows release passed all nine selected unit, C API, tokenizer, shader-control and CLI regression suites, including music CLI defaults. Both models passed DLL audio delivery, same-seed byte-for-byte repeatability and cancellation checks. Real CLI runs include YuE2 full ABC planning with 32 flow steps and an 8-second frame budget, and MiniMax with a 12-second frame budget, two overlapping windows and 30 flow steps per window. Example WAVs and logs are under `build/music-support/`.

`tools/yue_reference.py`, `tools/yue_tokenizer_reference.py` and `tools/minimax_music_reference.py` provide independent PyTorch equation, tokenizer and upstream codec comparisons using `slopfab_musicprobe`. They require development dependencies and upstream decoder source paths; production inference does not.

CUDA relative RMS errors against independent reference implementations on fixed fixtures:

| Stage | YuE2 | MiniMax Music3 |
| --- | --- | --- |
| Global autoregressive transformer | 0.414% | 1.171% |
| Local codebook transformer | Not applicable | 1.082% |
| Acoustic conditioning projection | Not applicable | Exact for equivalent GEMM convolution |
| Flow transformer, single prediction | 1.294% | See full-window validation below |
| Waveform decoder | 0.000333% | 0.000202% |

Convolution comparisons use equivalent GEMM equations; cuDNN may accumulate products in a different order. These checks validate implementation arithmetic and are not perceptual music-quality benchmarks.

### MiniMax positional precision correction

The initial `music3-piano.wav` contained garbled audio. The original BF16 rotary phase calculation merged adjacent positions beyond 256, corrupting full music windows. The earlier short synthetic fixtures did not cross that precision boundary. Rotary phases and Fourier timestep features now evaluate in FP32 before narrowing their sine/cosine outputs. Transformer matrix operations remain BF16.

The corrected 12-second sample is `build/music-support/music3-piano-fixed.wav`. The original filename also receives the corrected audio; `music3-piano-before-fix.wav` preserves the faulty sample for comparison. The replacement uses the same caption, seed and 30 steps per window.

Validation now includes a complete 200-frame window (689 audio latent rows plus the timestep token), all 30 sampling steps, real piano conditioning, and identical starting noise. The corrected production sampler output differs by 2.635% relative RMS from the upstream FP16 flow implementation, with cosine similarity 0.999663. Upstream BF16 with prematurely rounded phases differed from FP16 by 65.5%; preserving phase precision reduced this to 2.8%. A teacher-forced 301-frame trajectory also matches the upstream Qwen3 global model within 1.52% and the upstream local decoder within 0.61% relative RMS.

`tools/minimax_upstream_ar.py` checks the recorded autoregressive trajectory. `tools/minimax_upstream_flow.py --dtype fp16 --check-native` checks the complete sampling window against the upstream module; use `--help` for required local checkpoint, source and fixture paths. `slopfab_musicprobe mm-render DIT HIDDEN OUTPUT.wav VAE` replays captured conditioning through the production sampler and decoder, including overlapping windows. The portable positional test covers the 256/257 and 512/513 boundaries. The CLI test also verifies that music selects its required default attention mode.

### Vulkan validation

The CUDA-disabled CLI completed YuE2 with 200 semantic frames and all 32 midpoint steps, producing `build/music-support/yue-vulkan-piano.wav` (383936 stereo frames at 48 kHz). It also completed fresh MiniMax autoregressive generation for 300 acoustic frames and two overlapping 30-step flow windows, producing `build/music-support/music3-vulkan-piano.wav` (529408 stereo frames at 44.1 kHz). These are separate outputs from the existing CUDA examples.

The full MiniMax sampling fixture uses the same captured piano conditioning and starting noise as CUDA. Across all 30 steps and 689 latent rows, Vulkan differs from CUDA by **1.318% relative RMS**, with cosine similarity **0.999913**. Relative to the upstream FP16 reference, the error is **2.717%**, cosine **0.999641**. This fixture crosses the positional precision boundaries that caused the earlier distorted sample. On the short fixed YuE2 fixture, flow prediction differs from CUDA by 3.593% relative RMS and the audio decoder matches exactly. The MiniMax decoder differs from the upstream fixture by 0.000187% relative RMS. Seeded trajectories can diverge across backends when near-tied token probabilities change.

`slopfab_musicprobe_vulkan` accepts the same modes and fixtures as `slopfab_musicprobe`, including `mm-sample` for full-window comparisons and `mm-render` for production replay through overlap handling and waveform decoding. GPU regressions cover YuE's rounded inverse RMS, grouped-query attention with 256-wide heads and partially filled KV caches, and fused bias before BF16 rounding on full and tail GEMM tiles. The C API smoke harness checks both backends for geometry, finite samples, seeded repeatability and cancellation.

CUDA and Vulkan share token sampling, prompt construction, flow integration, windowing and codec orchestration. Vulkan caches expanded BF16 matrices without retaining compressed GPU duplicates and tiles large vocabulary weights below storage-buffer limits. It currently waits for each music GPU operator before submitting the next. Repeated seeded tests exposed intermittent corruption in batched music execution; batching remains disabled until that path is reliable. This conservative submission path passed five consecutive generations for each model with byte-identical audio, plus cancellation checks. It adds CPU submission overhead. The shader sources and embedded SPIR-V binaries are checked by the build manifest. Cooperative pipelines explicitly enable their Vulkan memory-model and full-subgroup requirements. Shader-control tests check capability declarations and reject duplicate execution modes; GPU regressions also pass with synchronization and shader race validation enabled.

## Upstream references

- [YuE2 model and usage](https://huggingface.co/m-a-p/YuE2-3B)
- [MiniMax Music3 model and usage](https://huggingface.co/MiniMaxAI/MiniMax-Music3)
- [Comfy MiniMax Music3 checkpoints](https://huggingface.co/Comfy-Org/MiniMax-Music-3)
- [Comfy MiniMax model implementation](https://github.com/Comfy-Org/ComfyUI/tree/master/comfy/ldm/minimax_music)
