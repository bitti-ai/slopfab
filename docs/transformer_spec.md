# MiniMax H3 — Transformer + FL2VA Text-to-Video Denoising Pipeline
## Implementation Specification for a C++/CUDA port

Target: the 33 B `MiniMaxH3DiTModel` / `MiniMaxH3Transformer3DModel` and the
`t2va` (text-only) denoising loop, as shipped in
`weights/transformer/fl2va_pruned_fp8_scaled.safetensors`.

Everything here is grounded in:

- `ref/diffusers/transformer_minimax_h3.py` — the diffusers port of the model
  (cited as `transformer.py:NNN`)
- `ref/diffusers/scheduling_minimax_h3.py` (`scheduler.py:NNN`)
- `ref/diffusers/convert_minimax_h3_to_diffusers.py` (`convert.py:NNN`) — the
  authority on original-name → diffusers-name and on every tensor transform
- `ref/diffusers/modular/{packing,before_encoder,encoders,before_denoise,denoise,decoders,modular_blocks_minimax_h3,modular_pipeline}.py`
  (cited as `packing.py:NNN` etc.)
- `ref/transformer/config.json`, `ref/FL2VA/transformer/config.json` (the
  *original* sglang-native config), `ref/text_encoder/config.json`,
  `ref/FL2VA/model_index.json`, `ref/README.md`
- the checkpoint header itself, read directly

> **Read §3 first if you are only here for the AdaLN.** The released
> `transformer_minimax_h3.py` implements **full** AdaLN from
> `time_embed_dim = 2688`. Our checkpoint ships a **factorised rank-8** form
> that is *not described anywhere in the reference sources*. Both are specified;
> the rank-8 grid semantics are **UNRESOLVED** (§10.1).

---

## 0. Resolved configuration

`ref/transformer/config.json` and `ref/FL2VA/transformer/config.json` agree
field for field (the second uses the original names):

| diffusers name | original name | value |
|---|---|---|
| `num_attention_heads` | `num_attention_heads` | 56 |
| `attention_head_dim` | `attention_head_dim` | 128 |
| `hidden_size` | `hidden_size` | **5376** |
| — (derived) | — | `inner_dim = 56*128 =` **7168** ≠ 5376 |
| `num_layers` | `num_layers` | 50 |
| `num_refiner_layers` | `token_refiner_num_layers` | 2 |
| `ffn_dim` | `ffn_hidden_size` | 14336 |
| `in_channels` | `latents_dim` | 24 |
| `audio_in_channels` | `audio_latents_dim` | 32 |
| `patch_size` | `patch_size` | `[1, 2, 2]` |
| `text_dim` | `text_dim` | 5120 |
| `freq_dim` | `timestep_input_dim` | 256 |
| `time_embed_hidden_dim` | `time_embed_hidden_size` | 5376 |
| `time_embed_dim` | `time_embed_dim` | 2688 |
| `rope_freq_dim` | `rope_inv_freq_len` | 16 |
| `rope_theta` | *(absent; 10000.0)* | 10000.0 |
| `norm_eps` / `qk_norm_eps` / `final_norm_eps` | same | 1e-5 |
| — | `adaln_out_features` | **96768** = 18 × 5376 |
| — | `final_adaln_out_features` | **10752** = 2 × 5376 |

Derived constants:

```
video_patch_dim   = in_channels * prod(patch_size) = 24 * 1*2*2 = 96
inner_dim         = 7168
ffn fused width   = 2 * ffn_dim = 28672
adaln width       = 6 * 3 * hidden_size = 96768
MODALITY_NUM      = 3          (transformer.py:36)
tags: VIDEO=0, TEXT=1, AUDIO=2, PAD=-1   (packing.py:49-51, transformer.py:560)
```

Scheduler shifts (`ref/FL2VA/model_index.json` → `_minimax_h3.sigma_shift_scales`,
consumed by `convert.py:630-653`): **video 12.0, audio 3.0**.

### 0.1 The single path this spec covers

Specified: **single GPU, `t2va` (text only), batch = 1, no CFG, no padding rows.**

Explicitly **excluded**:

- `ref2va` — `packing_ref2va.py`, `MiniMaxH3Ref2VA*` blocks, `transformer_ref`.
- `fl2va` keyframe conditioning. The layout code is shared and §2 gives the
  general formula, but the keyframe VAE encode
  (`encoders.py:264-302`), the `noise_aug = 0.999` anchor
  (`packing.py:84`) and the conditioning-row timestep
  (`before_denoise.py:417`) are described only as far as they change the
  `t2va` index arithmetic (they do not: `num_condition_video_rows = 0`).
- Classifier-free guidance / negative prompts. **Confirmed absent**:
  `ref/README.md:180` — *"The released checkpoints are CFG-distilled Omni
  Transformer model weights."* — and `denoise.py:124-127`, `encoders.py:89-91`,
  `modular_pipeline.py:32-33` all state there is no guider, no
  `negative_prompt`, no `guidance_scale`, and **exactly one forward pass per
  step**. There is no `guidance` input to `forward` at all
  (`transformer.py:530-544`).
- Padding rows and the attention mask (`transformer.py:622-625`). The modular
  blocks never emit padding; the reference sglang implementation pads to a
  multiple of 64 and keeps the tail as a separate attention document. A port
  that does not pad needs **no attention mask at all**.
- Batch > 1. `_check_prompt` (`encoders.py:48-53`) rejects it: one request = one
  packed sequence.
- Training, gradient checkpointing, LoRA (`attention_kwargs` / `apply_lora_scale`).
- The two VAEs (see `docs/vae_decoder_spec.md`) and the Qwen3-VL conditioner
  itself (§1.2 specifies only the interface).

---

## 1. End-to-end text-to-video dataflow

### 1.1 Symbols

| symbol | meaning |
|---|---|
| `L` | number of text tokens = `prompt_embeds.shape[1]` |
| `H, W` | pixel canvas, multiples of 32 |
| `Hl, Wl` | latent size = `H/16, W/16` (`modular_pipeline.py:54-57`, spatial compression 16) |
| `F` | video latent frames |
| `R` | rows per latent frame = `(Hl/2) * (Wl/2)` (`packing.py:399`) |
| `A` | audio latents per channel |
| `V = F*R` | target video rows |
| `Sa = 2*A` | audio rows (stereo, channel-major) |
| `S = L + Sa + V` | packed sequence length (`t2va`: no condition rows) |
| `N` | `num_inference_steps` as passed; the loop runs `N-1` model evaluations |

Worked example, the reference `t2va` request
(`ref/scripts/readme/reproducible-768p-t2va-request.sh`: 16:9, short edge 768,
10 s):

```
resolve_canvas_size(16, 9)  -> H=768,  W=1344        (packing.py:131-166)
align_num_frames(240)       -> 243     (243 = 17*14 + 5)
video_latent_num_frames     -> F = 14*5 + 2 = 72
Hl, Wl                      -> 48, 84
R                           -> 24 * 42 = 1008
V                           -> 72576
A = round(243/24*40)        -> 405        Sa = 810
S                           -> 73386 + L
```

The default `num_frames = 124` gives `F = 37`, `V = 37296`, `A = 207`,
`Sa = 414`, `S = 37710 + L`.

### 1.2 Prompt → conditioning embedding

**There is no chat template and there are no special tokens.**
`MiniMaxH3TextEncoderStep.encode_prompt` (`encoders.py:112-195`):

1. `t2va` has `images = None`, so the token list is exactly
   `tokenizer(prompt, add_special_tokens=False)["input_ids"]`
   (`encoders.py:167`). No BOS, no EOS, no `<|im_start|>`, no system prompt.
   `text_token_tags` is `[1] * L` (`encoders.py:169`).
   *(For `fl2va` each keyframe prepends `"<Picture {i+1}>: "` +
   `<|vision_start|>` + `<|image_pad|>`×`num_image_tokens` + `<|vision_end|>`,
   and the vision-block rows are tagged **0 = video**, `encoders.py:159-166`.)*
   The prompt *content* is expected to be MiniMax's "Context-IR" structure
   (`integrated_multimodal_description:` / `overall_soundscape:` /
   `non_diegetic_music:`, see the reference request script and
   `ref/README.md:65`), but that is an authoring convention, **not** anything
   the code applies.
2. `mm_token_type_ids` = all zeros for pure text
   (`encoders.py:174-176`; Qwen3-VL uses it for its per-modality 3-D mrope).
3. `text_encoder.model(...)` is called with `output_hidden_states=True` — the
   **submodule**, not the top-level model: the LM head is never run
   (`encoders.py:177-193`).
4. `prompt_embeds = outputs.hidden_states[50]` (`encoders.py:194`,
   `MINIMAX_H3_TEXT_ENCODER_LAYER = 50` at `packing.py:75`).

**Which hidden state, exactly.** `hidden_states` is the standard HF tuple with
`hidden_states[0]` = the embedding output (`packing.py:73-74` says so
explicitly). Index 50 is therefore:

- **0-based into the tuple**, i.e. the output of the **50th decoder layer
  counting from 1**, i.e. **0-based decoder layer index 49**, of 64 total
  (`ref/text_encoder/config.json` → `text_config.num_hidden_layers = 64`).
- **Pre-layernorm / unnormalised.** `packing.py:73` calls it "the
  *unnormalized* hidden state", and `encoders.py:146-149` raises if the encoder
  has ≤ 50 layers with the message *"The last hidden state of a stack truncated
  to exactly 50 layers is **post-norm** and is not the conditioning MiniMax-H3
  expects."* It is the raw residual stream after layer 49 (0-based), **before**
  the model's final `norm`.

Output shape `(1, L, 5120)`, cast to the transformer dtype.

### 1.3 Latent initialisation

`MiniMaxH3PrepareLatentsStep.prepare_latents` (`before_denoise.py:272-327`).
Draw order off the single request generator is part of reproducibility
(`before_denoise.py:284-292`):

1. *(fl2va only)* conditioning noise, one `randn` per keyframe, drawn in
   `keyframe_condition_noise` (`packing.py:501-538`) — **first**.
2. Video noise: `randn_tensor((1, 24, F, Hl, Wl), fp32)` on the execution
   device (`before_denoise.py:310-315`), then patchified to `(V, 96)` rows.
3. Audio noise: `randn_tensor((2*A, 32), fp32)` — drawn **directly in row
   layout**, `(Sa, 32)` (`before_denoise.py:319-324`). Note this is *not* the
   same as drawing `(2, 32, A)` and permuting; a port must reproduce the row-
   major `(Sa, 32)` draw to match a seed.

Both stay **float32** for the whole loop. Passing `latents` explicitly skips
draw 2; passing `audio_latents` (shape `(2, 32, A)`) skips draw 3 and is
`permute(0,2,1).reshape(-1,32)`-ed (`before_denoise.py:326`).

### 1.4 Patchify

`patchify_video_latents` (`packing.py:246-275`), patch `(1,2,2)`:

```
(1, 24, F, Hl, Wl)
  -> reshape (1, 24, F/1, 1, Hl/2, 2, Wl/2, 2)
  -> permute (0, 2, 4, 6, 1, 3, 5, 7)          # (b, f, hh, ww, c, pt, ph, pw)
  -> reshape (F*R, 96)
```

Explicit index arithmetic (batch 1):

```
row      = (f * (Hl/2) + hh) * (Wl/2) + ww        # 0 <= row < V
feature  = c * 4 + dh * 2 + dw                    # 0 <= feature < 96
value    = latents[0, c, f, 2*hh + dh, 2*ww + dw]
```

i.e. **frame-major, then latent-row-major, then column**; within a row,
**channel-major** over the 2×2 patch.

`unpatchify_video_tokens` (`packing.py:278-312`) is the exact inverse.

### 1.5 The forward pass

`_predict_velocity` (`denoise.py:99-115`) → `MiniMaxH3Transformer3DModel.forward`
(`transformer.py:530-644`):

```
video_embeds = proj_in(latents)                      (1, V, 96)    -> (1, V, 5376)
audio_embeds = audio_proj_in(audio_latents)          (1, Sa, 32)   -> (1, Sa, 5376)
text_embeds  = context_embedder(prompt_embeds)       (1, L, 5120)  -> (1, L, 5376)
text_embeds  = token_refiner(text_embeds)                          -> (1, L, 5376)

hidden = zeros(1, S, 5376)
hidden[:, text_indices]  = text_embeds
hidden[:, video_indices] = video_embeds
hidden[:, audio_indices] = audio_embeds              (transformer.py:602-605)

rotary_emb = rope(position_ids)                      (S, 96) cos, (S, 96) sin
temb       = time_embedder(time_proj(timestep))      (T, 2688)   [full model]
adaln_idx  = timestep_indices * 3 + clamp(token_tags, min=0)   (transformer.py:616)

for block in 50 blocks:  hidden = block(hidden, temb, adaln_idx, rotary_emb, None)

hidden       = norm_out(hidden, temb, timestep_indices)          (1, S, 5376)
video_output = proj_out(hidden)[:, video_indices]                (1, V, 96)
audio_output = audio_proj_out(hidden)[:, audio_indices]          (1, Sa, 32)
```

Both output heads run over **every** row and are selected afterwards
(`transformer.py:638-640`). A port should instead gather the rows first and run
each head only on its own rows — mathematically identical, because `norm_out`
is a per-row operation, and it saves `S × 5376 × 128` flops per step.

The three index tensors are disjoint and their union is `[0, S)` for a padless
sequence, so the scatter/gather is a permutation.

### 1.6 The loop

`MiniMaxH3DenoiseLoopWrapper.__call__` (`denoise.py:298-306`) +
`MiniMaxH3LoopSchedulerStep` (`denoise.py:258-275`):

```
scheduler.set_timesteps(N)          # shift 12.0, video
audio_scheduler.set_timesteps(N)    # shift  3.0, audio
row_timestep_plan = [ build_row_timesteps(layout, t_v, t_a, max(t_v, 0.999), 1.0)
                      for t_v, t_a in zip(timesteps, audio_timesteps) ]   (before_denoise.py:405-422)

for i, t in enumerate(timesteps):                       # len == N-1
    v_pred, a_pred = transformer(..., *row_timestep_plan[i], ...)
    latents      [C_v:] = scheduler      .step(v_pred[0, C_v:].float(), t,                  latents      [C_v:])
    audio_latents[C_a:] = audio_scheduler.step(a_pred[0, C_a:].float(), audio_timesteps[i], audio_latents[C_a:])
```

with `C_v = num_condition_video_rows` and `C_a = num_condition_audio_rows`
(both **0** for `t2va`). The two schedulers are stepped **independently** with
their own timestep and their own sigma grid inside one loop iteration; there is
one transformer call per iteration serving both.

Conditioning rows are re-imposed *by construction* — the loop simply never
writes them (`denoise.py:186-188`).

### 1.7 Out

`MiniMaxH3VideoDecodeStep` (`decoders.py:90-120`):

```
latents = unpatchify_video_tokens(latents[C_v:], F, Hl, Wl, 24, (1,2,2))   # (1,24,F,Hl,Wl)
latents = latents * latents_std + latents_mean     # per-channel, from vae config
-> AutoencoderKLMiniMaxH3.decode   (see docs/vae_decoder_spec.md)
```

Audio (`decoders.py:178-195`) is `unpack_audio_tokens` (`packing.py:315-328`):
`(Sa, 32) -> reshape(2, A, 32) -> permute(0,2,1) -> (2, 32, A)`, denormalised
the same way, then the mono audio VAE decodes the two stereo channels as two
batch items.

---

## 2. Token packing

`build_packed_sequence` (`packing.py:369-466`). Row order:

```
[ text (L) | keyframe conditions (C) | target audio (Sa = 2A) | target video (V = F*R) ]
```

For `t2va`, `C = 0`. Offsets (`packing.py:406-408`):

```
condition_start = L
audio_start     = L + C
video_start     = L + C + Sa
S               = L + C + Sa + V
```

**There are no separator tokens, no BOS/EOS, no learned modality embeddings and
no padding.** Modality is carried entirely by (a) which input projection wrote
the row, (b) the per-row `token_tags`, and (c) the rotary coordinate.

### 2.1 Index tensors (`packing.py:448-455`)

```
text_indices  = [0, L)
audio_indices = [audio_start, video_start)
video_indices = [condition_start, audio_start)  ++  [video_start, S)
```

Note `video_indices` is the concatenation of the conditioning rows **and** the
target rows, conditioning first — so `latents` (the row buffer) is
`[conditions | targets]` and `latents[C_v:]` is the generated part.

`token_tags` (`packing.py:452-455`) are assigned in this order — **the order
matters**:

```
token_tags[text_indices]  = text_token_tags       # 1, except fl2va vision rows -> 0
token_tags[audio_indices] = 2
token_tags[video_indices] = 0
```

So a text row belonging to a keyframe's vision block is tagged `0` (video) even
though it lives in the text region and is fed by `context_embedder`. For `t2va`
all `L` text rows are tagged `1`.

### 2.2 Attention

**One full-attention stream. No mask, no causality, no cross-attention.**
`MiniMaxH3AttnProcessor` (`transformer.py:157-207`) issues a single
`dispatch_attention_fn(q, k, v, attn_mask=None, is_causal=False)`. The class
docstring (`transformer.py:158-159`) states outright: *"Full self-attention over
one packed sequence. There is no cross-attention anywhere in MiniMax-H3."*
The mask is built only when padding rows exist (`transformer.py:622-625`), and
then only to split live rows from padding — irrelevant for a padless port.

Consequence for the port: one dense `S × S` attention per layer, `S ≈ 73 k` for
a 10 s 768p request. This is the dominant cost; use a FlashAttention-style
kernel with `head_dim = 128`, 56 heads, no mask.

### 2.3 Rotary coordinates (`packing.py:412-445`)

Built in **float64**. The shared clock is **40 rotary units per second**: video
advances `5/3` units per pixel frame at 24 fps, audio advances 1 unit per latent
at 40 latents/s (`packing.py:28-31, 89-94`).

**Text rows** `i ∈ [0, L)`:

```
pos[i] = (i, 0, 0)
```

**Spatial grids** (`_spatial_position_grid`, `packing.py:331-341`), with
`sqrt_area = sqrt(Hl * Wl)`:

```
ratio_h = Hl / sqrt_area          left_h = (1 - ratio_h) / 2
h_grid[k] = 32 * ( left_h + k * ratio_h / (Hl/2) )      k in [0, Hl/2)
```

and the same with `Wl`. This is `np.linspace(left, left+ratio, n, endpoint=False)`
— **numpy's** formula `start + arange(n)*(stop-start)/n`, which the comment at
`packing.py:338-340` warns is *not* `torch.linspace`. Reproduce it exactly; a
square canvas spans `[0, 32)`.

`frame_grid` (`packing.py:418`) is `meshgrid(h_grid, w_grid, indexing="ij")`
flattened row-major, i.e. `frame_grid[hh*(Wl/2) + ww] = (h_grid[hh], w_grid[ww])`
— the same `r = hh*(Wl/2) + ww` ordering as the patchifier (§1.4). Good.

**Audio rows** `audio_start + c*A + a`, `c ∈ {0,1}` (channel-major),
`a ∈ [0, A)` (`packing.py:433-440`):

```
pos.t = L + a
pos.h = 0
pos.w = w_grid[0]   if c == 0  else  w_grid[-1]
```

Careful: `packing.py:433` uses `float(num_text_tokens)`, i.e. `L` — **not**
`audio_start`. Keyframe condition rows do not advance the audio clock.
Audio carries **no height coordinate** and is pinned to the two extremes of the
width grid, which is how the two stereo channels are distinguished.

**Video rows** `video_start + f*R + r` (`packing.py:442-445`):

```
pos.t     = L + T(f)
pos.(h,w) = frame_grid[r]
```

with the non-uniform temporal grid (`_temporal_position_grid`, `packing.py:344-353`):

```
span[j] = (5/3) * FRAMES_PER_LATENT[j % 5],   FRAMES_PER_LATENT = (1, 4, 4, 4, 4)
T(0)    = 0
T(f)    = sum_{j<f} span[j]                   # cumulative sum
```

so `T = 0, 5/3, 5/3+20/3, ... ` and each group of 5 latent frames advances
`(5/3)*17 = 28.3333…` units = 17 pixel frames. Compute `T` as a `cumsum` in
float64.

*(fl2va only, `packing.py:420-429`: a `"first"`-anchored keyframe sits at
`t = L`; a `"last"`-anchored one at `L + _temporal_position_span(F) - 5/3`,
where `_temporal_position_span` must be summed with numpy's **pairwise**
summation, `packing.py:356-366` — the two summation orders differ in the last
ulp from 16 latent frames on.)*

### 2.4 Unpacking

`video_output` and `audio_output` come back already gathered, in
`video_indices` / `audio_indices` order (`transformer.py:639-640`), so the
denoise loop indexes them as plain `(V, 96)` and `(Sa, 32)` row buffers.
Dropping the conditioning prefix and calling §1.4's inverse is the whole
unpack.

---

## 3. `adaln_t_table` and the rank-8 AdaLN

### 3.1 What the released `transformer_minimax_h3.py` does — full AdaLN

**Yes: the released reference implements full AdaLN from
`time_embed_dim = 2688` and knows nothing about `adaln_t_table`.** Both forms
are described here because our checkpoint ships the factorised one.

Timestep → conditioning vector (`transformer.py:483-486, 610-611`):

```
temb0 = Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0)(timestep)
temb  = time_embedder(temb0)
      = linear_2( SiLU( linear_1(temb0) ) )      # 256 -> 5376 -> 2688
```

`timestep` is the `(T,)` vector of **distinct** timesteps in the sequence, in
`[0, 1]`, **unscaled** (no ×1000) — `transformer.py:416, 554-556`.
`Timesteps` is diffusers' standard sinusoid:

```
half = 128
freq[j] = exp( -ln(max_period) * j / 128 ),  j in [0,128)
arg     = t * freq
emb     = concat( cos(arg), sin(arg) )      # flip_sin_to_cos=True puts cos first
```

`max_period = 10000` is the diffusers library default and is **not visible in
any file in `ref/`** — see §10.4.

Per block (`MiniMaxH3AdaLayerNormModulation`, `transformer.py:100-128`):

```
m = adaln_proj.linear( SiLU(temb) )          # (T, 2688) -> (T, 96768)
m = m.view(T * 3, 6 * 5376)
shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = m.chunk(6, -1)
```

Final layer (`MiniMaxH3AdaLayerNormOut`, `transformer.py:131-154`):

```
shift, scale = norm_out.linear( SiLU(temb) ).chunk(2, -1)     # (T, 10752) -> 2 x (T, 5376)
```

The `SiLU` runs at `temb`'s precision (float32 in the mixed-precision
checkpoint) and only the *result* is cast to the projection dtype
(`transformer.py:122-126, 148-149`).

### 3.2 The 18 modulation vectors — resolved

`96768 = 6 * 3 * 5376` (`convert.py:222`) = **6 parameters × 3 modalities ×
5376 channels**. The flat output layout of `adaln_proj.linear` is
**modality-outer, parameter-inner**, because `view(-1, 6*hidden)` splits the
96768 axis into 3 groups of 32256 and `chunk(6)` then splits each group
(`transformer.py:127-128`), and the convert script maps the weight through
**unchanged, with no permutation and no transpose** (`convert.py:221-226`,
`convert.py:18` "There are no transposes anywhere").

Definitive table — row `p` of `adaln_proj.linear.weight` (`p ∈ [0, 96768)`):

| flat rows | modality | parameter | applies to |
|---|---|---|---|
| `0 – 5375` | 0 = video | `shift_msa` | pre-attention norm shift |
| `5376 – 10751` | 0 = video | `scale_msa` | pre-attention norm scale |
| `10752 – 16127` | 0 = video | `gate_msa` | attention output gate |
| `16128 – 21503` | 0 = video | `shift_mlp` | pre-FFN norm shift |
| `21504 – 26879` | 0 = video | `scale_mlp` | pre-FFN norm scale |
| `26880 – 32255` | 0 = video | `gate_mlp` | FFN output gate |
| `32256 – 64511` | 1 = **text** | same 6, same order | |
| `64512 – 96767` | 2 = **audio** | same 6, same order | |

Decomposition: `p = modality*32256 + param*5376 + channel`, with
`param ∈ {0:shift_msa, 1:scale_msa, 2:gate_msa, 3:shift_mlp, 4:scale_mlp, 5:gate_mlp}`
and `modality ∈ {0:video, 1:text, 2:audio}` matching
`MINIMAX_H3_VIDEO_TAG/TEXT_TAG/AUDIO_TAG` (`packing.py:49-51`).

**This ordering is now empirically confirmed, not just read off the convert
script.** Evaluating `W_8 @ adaln_t_table[0] + b` for block 0 and taking the
mean of each 5376-wide parameter slice gives:

| modality | shift_msa | scale_msa | gate_msa | shift_mlp | scale_mlp | gate_mlp |
|---|---|---|---|---|---|---|
| video | −0.0024 | **−0.8161** | −0.0019 | −0.0011 | **−0.8062** | +0.0008 |
| text | +0.0024 | **−0.4275** | −0.0016 | +0.0007 | **+0.2217** | −0.0032 |
| audio | +0.0011 | **−0.6696** | +0.0012 | +0.0003 | **−0.9126** | +0.0007 |

Slots 1 and 4 carry large non-zero means; slots 0, 2, 3 and 5 all sit within
±0.004 of zero. That is exactly the signature of a `1 + scale` parameterisation
— the scale parameters are centred away from zero, the shifts and gates are
centred on it — and it lands on slots 1 and 4 and nowhere else. Any other
assignment of the six parameters would put the two non-zero-mean slices in the
wrong place. The resulting `1 + scale_msa` spans `[−0.15, 1.87]` with 3.5 % of
channels non-positive, which is a sane modulation range.

The same evaluation confirms the final layer's `[shift; scale]` order: rows
`0:5376` have mean −0.0002 with standard deviation 0.010 (a shift), and rows
`5376:10752` give `1 + scale` a mean of 0.198 (a scale).

Row selection (`transformer.py:616`, docstring `transformer.py:105-108`):

```
adaln_indices[s] = timestep_indices[s] * 3 + max(token_tags[s], 0)
```

so the effective modulation table has `T*3` rows laid out
`[t0_video, t0_text, t0_audio, t1_video, …]`. The `clamp(min=0)` only exists to
keep padding rows (`tag = -1`) from indexing backwards.

The **final layer has no modality dependence**: `norm_out` selects with
`timestep_indices` directly, not `adaln_indices` (`transformer.py:147, 152-154`),
and its 10752 outputs are `[shift, scale]` in that order (`transformer.py:137-139,
149`).

### 3.3 Application formula (`transformer.py:353-371`)

```
# --- attention half ---
residual = x
n = RMSNorm(x, w=norm1.weight, eps=1e-5)
n = n * (1 + scale_msa[adaln_indices]) + shift_msa[adaln_indices]
a = Attention(n, rotary_emb)
x = residual + gate_msa[adaln_indices] * a

# --- feed-forward half ---
residual = x
n = RMSNorm(x, w=norm2.weight, eps=1e-5)
n = n * (1 + scale_mlp[adaln_indices]) + shift_mlp[adaln_indices]
f = SwiGLU_FFN(n)
x = residual + gate_mlp[adaln_indices] * f
```

Note `1 + scale` (not bare `scale`) and that the gate multiplies the **branch
output only** — the residual is ungated. Final layer
(`transformer.py:151-154`):

```
x = RMSNorm(x, w=final_layer.norm.weight, eps=1e-5)
x = x * (1 + scale[timestep_indices]) + shift[timestep_indices]
```

### 3.4 What our checkpoint actually ships — the rank-8 factorisation

Read directly from the safetensors header of
`weights/transformer/fl2va_pruned_fp8_scaled.safetensors`:

```
adaln_t_table                            F32  [1025, 8]
blocks.{0..49}.adaln_proj.linear.weight  F16  [96768, 8]      <-- not [96768, 2688]
blocks.{0..49}.adaln_proj.linear.bias    F16  [96768]         <-- unchanged
final_layer.adaln_proj.linear.weight     F16  [10752, 8]      <-- not [10752, 2688]
final_layer.adaln_proj.linear.bias       F16  [10752]         <-- unchanged
time_embedder.*                          ABSENT
```

For comparison, the original checkpoint's key list
(`ref/FL2VA/transformer/model.safetensors.index.json`) *does* contain
`time_embedder.proj_in.{weight,bias}` and `time_embedder.proj_out.{weight,bias}`
and its `adaln_proj.linear.weight` is `[96768, 2688]`.

So the pruned checkpoint has replaced the entire
`t → sinusoid → MLP → 2688 → SiLU` front end with a **table lookup producing an
8-vector**, and every AdaLN projection has been contracted onto that 8-dim
basis:

```
full:   m(t) = W_full @ SiLU(temb(t)) + b        W_full in R^{96768 x 2688}
rank-8: m(t) = W_8    @ c(t)          + b        W_8    in R^{96768 x 8}
        c(t) = adaln_t_table[ row(t) ]  in R^8
```

The biases are **identical in shape and role**, which is the strongest evidence
that only the input side was factorised: `W_8 = W_full @ B` for some shared
basis `B ∈ R^{2688 x 8}` with `SiLU(temb(t)) ≈ B @ c(t)`. The same `c(t)` feeds
all 50 blocks *and* `final_layer` — one table, 51 consumers.

This is exactly the saving `ref/README.md:141` advertises: *"approximately 13B
parameters residing in AdaLN-related branches. Because the AdaLN modulation
outputs can be precomputed and cached, these parameters do not need to be
loaded for inference-only deployment."* Check: `50 × 96768 × 2688 = 13.00 × 10⁹`.

**Critical implementation consequences:**

1. **Do not apply SiLU (or any activation) to `adaln_t_table` rows.** The `SiLU`
   in `transformer.py:126` sits *before* the projection, so it is already baked
   into the table.
2. `timestep`, `time_proj` and `time_embedder` are not needed at all. The
   `forward` signature's `timestep: (T,)` tensor is consumed only to produce
   `c(t)` per distinct timestep.
3. The per-step AdaLN cost collapses to 51 GEMVs of `[out, 8] × [8]` per
   distinct timestep — for `t2va` there are **at most 2 distinct timesteps**
   (video `t_v`, audio `t_a`; text shares `t_v`). Precompute
   `(T, 3, 6, 5376)` per block once per denoising step: `50 × 2 × 96768 × 4 B
   = 38.7 MB` in fp32. Trivial; do it in fp32 even though the weights are F16.
4. `W_8`/`b` are **F16**, not BF16 — the only F16 tensors in the file besides
   nothing else. Convert to fp32 on load.

### 3.5 Timestep → table row — **UNRESOLVED**

The grid semantics of `adaln_t_table [1025, 8]` cannot be determined from any
source in `ref/`: the reference implements the full form and the pruned
checkpoint ships no metadata describing the mapping (its `__metadata__` is
absent; the only per-tensor metadata is ComfyUI's `comfy_quant` JSON on the fp8
weights, which says nothing about the table). The tensor payload itself lives at
byte offset 20 870 775 256 of a 20 958 205 608-byte file and was **not yet
downloaded** at the time of writing, so it could not be inspected either.

What can be asserted:

- `1025 = 2¹⁰ + 1`, so it is almost certainly a **dyadic grid of 1024 intervals
  over `t ∈ [0, 1]`**, with row 0 ↔ `t = 0` and row 1024 ↔ `t = 1`. H3's `t` is
  `1 - sigma` with `t = 1` meaning clean (`scheduler.py:22-24`), and `t` is a
  continuous real (the shifted sigma grid produces arbitrary values), so the
  table must be *addressed by value*, not by step number.
- The two candidate lookups are

  ```
  u = t * 1024
  nearest:       c = table[ round(u) ]
  interpolated:  j = floor(u);  f = u - j;  c = (1-f)*table[j] + f*table[j+1]
  ```

  They agree exactly on grid points. Nearest quantises `t` to ~9.8e-4, which is
  large next to the conditioning constant `noise_aug = 0.999` and next to the
  spacing of a 50-step shifted schedule near `t = 1`; **linear interpolation is
  the safer default** because it is exact wherever nearest is and strictly
  closer everywhere else, given that `SiLU(temb(t))` is smooth in `t`.

**Verification recipe once the download completes** (do this before trusting
either):

1. Read `adaln_t_table`; confirm it is smooth in the row index and that
   consecutive-row differences are `O(1/1024)` of the total range (a smooth
   curve ⇒ interpolation is meaningful; a step-like table ⇒ it is a lookup of
   discrete training timesteps and the mapping is different).
2. Rebuild `SiLU(time_embedder(Timesteps(t)))` for `t = j/1024`, `j ∈ [0,1024]`
   from the **original** `MiniMaxAI/MiniMax-H3` `transformer/` shards, take an
   8-component PCA/SVD of the resulting `(1025, 2688)` matrix, and check that
   the coefficient matrix matches `adaln_t_table` up to an 8×8 change of basis
   `M`. If it does, recover `M` and confirm
   `W_8 ≈ W_full @ B` for the corresponding `B`. That pins the grid, the
   direction (`t` vs `sigma`) and the interpolation question at once.
3. Cheap fallback if the original weights are unavailable: run the port at
   `t = 0.5` with nearest vs interpolated lookup and compare against a
   `t = 512/1024` run — they must be bit-identical under both schemes, so
   instead compare `t = 0.4995` (which lands mid-interval) between the two
   schemes and against a known-good ComfyUI run of the same checkpoint.

Until then, treat the mapping as a **configurable knob** in the port and
default to linear interpolation on `u = t * 1024`.

---

## 4. Block structure

`MiniMaxH3TransformerBlock` (`transformer.py:317-371`).

**Pre-norm, everywhere.** Both sub-layers are `x + gate * f(modulate(norm(x)))`.
There is no post-norm and no residual scaling.

### 4.1 Norms

| norm | shape | eps | source | checkpoint key |
|---|---|---|---|---|
| `norm1` | RMSNorm(5376), affine | 1e-5 | `transformer.py:334` | `blocks.i.norm1.weight` (BF16) |
| `norm2` | RMSNorm(5376), affine | 1e-5 | `transformer.py:341` | `blocks.i.norm2.weight` (BF16) |
| `attn.norm_q` | RMSNorm(**128**), affine | 1e-5 | `transformer.py:231` | `blocks.i.attn.q_norm.weight` (BF16) |
| `attn.norm_k` | RMSNorm(**128**), affine | 1e-5 | `transformer.py:232` | `blocks.i.attn.k_norm.weight` (BF16) |
| `norm_out.norm` | RMSNorm(5376), affine | 1e-5 | `transformer.py:144` | `final_layer.norm.weight` (BF16) |
| `token_refiner.final_norm` | RMSNorm(5376), affine | 1e-5 | `transformer.py:305` | `token_refiner.final_norm.weight` (BF16) |

All are **RMSNorm**, not LayerNorm: `x * rsqrt(mean(x²) + eps) * w`, no mean
subtraction, no bias. Every eps is 1e-5 (`ref/transformer/config.json`).

### 4.2 Attention — the 7168 vs 5376 mismatch

`MiniMaxH3Attention.__init__` (`transformer.py:214-237`):

```
to_q, to_k, to_v : Linear(5376 -> 7168, bias=False)      # inner_dim = 56*128
to_out[0]        : Linear(7168 -> 5376, bias=False)
to_out[1]        : Dropout(0.0)                          # no-op at inference
```

The residual stream is 5376 wide; attention **widens to 7168 on the way in and
projects back on the way out**. The projection back happens in `to_out[0]`
(`transformer.py:205`), i.e. `blocks.i.attn.out_proj.weight` of shape
`[5376, 7168]`. There is no bias anywhere in attention (`use_bias = False`,
`transformer.py:227`).

Our checkpoint keeps the fused form: `blocks.i.attn.qkv_proj.weight` is
`[21504, 5376]` = `3 × 7168` rows.

### 4.3 Ordering inside the attention (`transformer.py:172-207`)

```
1. q, k, v = qkv_proj(x).split(7168)                  # or three separate GEMMs
2. reshape each to (B, S, 56, 128)                    # unflatten(-1, (heads, -1))
3. q = RMSNorm(q, q_norm.weight, 1e-5)                # per head, over 128 channels
   k = RMSNorm(k, k_norm.weight, 1e-5)
4. q = rope(q, cos, sin);  k = rope(k, cos, sin)      # AFTER qk-norm
5. attn = softmax(q k^T / sqrt(128)) v                # no mask, not causal
6. flatten heads -> (B, S, 7168)
7. out_proj -> (B, S, 5376)
```

**QK-norm is applied before RoPE**, not after (`transformer.py:183-188`). This
is the opposite of some other DiTs; get it right.

`v` is **not** normalised. Attention scale is the default `1/sqrt(head_dim)`
(nothing overrides it).

### 4.4 SwiGLU FFN — which half is the gate

`FeedForward(5376, inner_dim=14336, activation_fn="swiglu", bias=False)`
(`transformer.py:342`), i.e. `net.0.proj : Linear(5376 -> 28672)` and
`net.2 : Linear(14336 -> 5376)`, no biases.

`convert.py:264-270` is explicit about the layout difference:

> *"The reference computes `fc2(silu(gate) * value)` from a fused `[gate; value]`;
> diffusers' `SwiGLU` computes `value * silu(gate)` from a fused `[value; gate]`,
> so the two halves swap places."*

Our checkpoint uses the **original** key name `blocks.i.mlp.fc1.weight`, so it
is in the **original** layout:

```
fc1.weight  [28672, 5376]
    rows     0 .. 14335   =  GATE   (goes through SiLU)
    rows 14336 .. 28671   =  VALUE  (linear)

y = fc2( SiLU(x @ gate^T) * (x @ value^T) )
fc2.weight  [5376, 14336]
```

**The first half is the gate.** If you ever load a diffusers-converted
`ff.net.0.proj.weight` instead, the halves are swapped.

### 4.5 Residual / gate arrangement

Restated for the kernel author — see §3.3 for the exact expression. Per block,
per row `s`:

```
x[s] += gate_msa[a(s)] * AttnOut[s]
x[s] += gate_mlp[a(s)] * FFOut[s]
```

with `a(s) = timestep_indices[s]*3 + token_tags[s]`. There is no learned
LayerScale, no `1/sqrt(depth)` and no stochastic depth.

---

## 5. MM-RoPE

`MiniMaxH3RotaryPosEmbed` (`transformer.py:73-97`) + `_apply_rotary_emb`
(`transformer.py:56-70`).

### 5.1 Frequencies

```
inv_freq[j] = 1 / rope_theta ** ( 2j / (2 * rope_freq_dim) )
            = 1 / 10000 ** ( j / 16 )        j in [0, 16)
```

(`transformer.py:85-87`: `arange(0, 2*16, 2) / (2*16)` = `[0, 2, …, 30]/32`.)
16 entries, float32. The checkpoint ships `rope.inv_freq` `F32 [16]`; the
diffusers port **drops it and recomputes**, noting the recomputed value is
bitwise equal in both released variants (`convert.py:104-107`). A port may do
either; recomputing in fp64 then rounding to fp32 is safest.

**One `inv_freq` is shared by all three axes** — there is no per-axis frequency
split of the kind Qwen/mrope uses. Each axis contributes all 16 frequencies.

### 5.2 Angle construction (`transformer.py:90-97`)

```
position_ids : (S, 3) float64  ->  cast to float32                (transformer.py:92)
freqs = position_ids[:, :, None] * inv_freq[None, None, :]        # (S, 3, 16)
ft, fh, fw = freqs.unbind(1)                                      # each (S, 16)
freqs = cat(ft, fh, fw, dim=-1)                                   # (S, 48)
freqs = cat(freqs, freqs, dim=-1)                                 # (S, 96)
cos, sin = freqs.cos(), freqs.sin()                               # (S, 96) fp32
```

So the 48-wide half-period is `[T(16) | H(16) | W(16)]` and it is **duplicated**
so that the `rotate_half` convention works.

### 5.3 Which head dims rotate (`transformer.py:56-70`)

```
rotary_dim = cos.shape[-1] = 96
x_rot  = x[..., :96]        <- rotated
x_pass = x[..., 96:]        <- passed through UNCHANGED (32 of 128 channels)
x1, x2 = x_rot.chunk(2, -1)             # x1 = ch 0..47, x2 = ch 48..95
x_rot  = x_rot * cos + cat(-x2, x1) * sin
out    = cat(x_rot, x_pass, -1)
```

Written out, for `j ∈ [0, 48)`:

```
out[j]      = x[j]    * cos[j] - x[j+48] * sin[j]
out[j+48]   = x[j+48] * cos[j] + x[j]    * sin[j]
out[96..127] = x[96..127]
```

with

```
cos[j] = cos( pos.t * inv_freq[j]      )   for j in [ 0, 16)
cos[j] = cos( pos.h * inv_freq[j-16]   )   for j in [16, 32)
cos[j] = cos( pos.w * inv_freq[j-32]   )   for j in [32, 48)
```

Summary:

- **Pairing convention: GPT-NeoX half-split**, but the split is over the **96
  rotary channels only**, pairing channel `j` with channel `j + 48` — *not*
  `j` with `j + 64`.
- **96 of 128 head dims are rotated; channels 96–127 pass through untouched.**
- Axis split is **T:16, H:16, W:16** in that order, occupying rotary channel
  pairs `(0-15, 48-63)`, `(16-31, 64-79)`, `(32-47, 80-95)`.
- Applied identically to `q` and `k`, per head, with the same `(S, 96)` tables
  broadcast over the head axis (`transformer.py:65-66`).

### 5.4 Where the coordinates come from

Per §2.3. Recap of the three cases:

| row kind | `t` | `h` | `w` |
|---|---|---|---|
| text | row index `i ∈ [0, L)` | 0 | 0 |
| audio (channel `c`, latent `a`) | `L + a` | 0 | `w_grid[0]` (c=0) / `w_grid[-1]` (c=1) |
| video (frame `f`, patch `hh,ww`) | `L + T(f)` | `h_grid[hh]` | `w_grid[ww]` |

Text rows advance only the T axis and sit at `h = w = 0`. There is **no offset
and no reserved band** separating text from media on the spatial axes — the
separation is purely on the T axis, where text occupies `[0, L)` and media
starts at `L`. Note the spatial grids are *centred*, so `0` generally falls
inside their range: for a square canvas they span `[0, 32)`, and for
`Hl, Wl = 48, 84` they run `h ∈ [3.9, 28.1]`, `w ∈ [-5.2, 37.2]`.

---

## 6. Token refiner

`MiniMaxH3TokenRefiner` (`transformer.py:279-314`) and
`MiniMaxH3TokenRefinerBlock` (`transformer.py:248-276`).

**Where it sits:** immediately after `context_embedder` and *before* the text
rows are scattered into the packed sequence (`transformer.py:599-603`):

```
text_embeds = context_embedder(prompt_embeds)      # (1, L, 5120) -> (1, L, 5376)
text_embeds = token_refiner(text_embeds)           # (1, L, 5376) -> (1, L, 5376)
hidden[:, text_indices] = text_embeds
```

It therefore operates on the **text stream alone**, `L` rows, never seeing
video or audio rows. It runs **once per request**, not once per denoising step —
a port should hoist it out of the loop entirely (the reference re-runs it every
step because `forward` is stateless; it is `O(L²)` with `L` in the low
thousands, so caching it is a free win).

**Structure** — 2 blocks, plain pre-norm, **no AdaLN, no RoPE, no timestep
dependence, no mask**:

```
x = x + attn( RMSNorm(x, norm1.weight, 1e-5) )         # 56 heads x 128, inner 7168
x = x + ff  ( RMSNorm(x, norm2.weight, 1e-5) )         # SwiGLU 14336
...
x = RMSNorm(x, final_norm.weight, 1e-5)                # final_norm_eps = 1e-5
```

The attention module is the *same class* as the main blocks
(`transformer.py:264-269`) — same 7168 inner dim, same `q_norm`/`k_norm`
RMSNorm(128, 1e-5) — but `MiniMaxH3TokenRefinerBlock.forward`
(`transformer.py:273-276`) calls `self.attn(self.norm1(x))` with **no
`rotary_emb` argument**, so `_apply_rotary_emb` is skipped
(`transformer.py:186`). The refiner is position-agnostic; positional
information reaches the text rows only through the packed sequence's RoPE in
the 50 main blocks.

Checkpoint keys (all BF16, no fp8, no scales):
`token_refiner.blocks.{0,1}.{norm1,norm2}.weight`,
`.attn.{qkv_proj,out_proj,q_norm,k_norm}.weight`,
`.mlp.{fc1,fc2}.weight`, `token_refiner.final_norm.weight`.
Same shapes and same conventions as a main block minus `adaln_proj`.

---

## 7. Scheduler

`MiniMaxH3Scheduler` (`scheduling_minimax_h3.py`). Two instances per request:
`shift = 12.0` (video), `shift = 3.0` (audio).

### 7.1 Sigma grid (`scheduler.py:158-162`)

```
base   = linspace(1.0, 0.0, N)                    # N points, float32, ON CPU
sigmas = shift * base / (1 + (shift - 1) * base)  # exponential shift
sigmas = unique_consecutive(sigmas)               # collapse float32 collisions
```

`linspace(1, 0, N)` gives `base[k] = 1 - k/(N-1)` — **the terminal 0 is part of
the requested step count**, and the shift maps `0 ↦ 0` and `1 ↦ 1` exactly.
Build this on the **host in float32**, not on the device
(`scheduler.py:145-146`).

`unique_consecutive` removes *adjacent* duplicates only. The shift compresses
the grid near `sigma = 1`, so for large `N` two neighbouring float32 sigmas can
collide there; the collapse then silently shortens the schedule. For
`N ≤ ~100` no collision occurs at either shift, but the port must implement it
(and see §9.5).

### 7.2 Timesteps (`scheduler.py:169-171`)

```
timesteps = 1 - sigmas[:-1]          # length len(sigmas) - 1
num_inference_steps = len(timesteps) # NOTE: overwritten to N-1
```

`t = 1 - sigma`, **`t = 1` means clean** — the opposite direction from
diffusers' flow-match schedulers, and on a `[0,1]` scale rather than
`×1000` (`scheduler.py:22-24`). The terminal sigma gets **no model
evaluation**: `N` grid points drive `N-1` forward passes.

`_step_index` and `_begin_index` are reset to `None`
(`scheduler.py:172-173`); the first `step` resolves `_step_index` by locating
`timestep` in `timesteps` (`scheduler.py:262-263`), and it increments by one per
call. A port should just carry `i` explicitly.

### 7.3 Euler update (`scheduler.py:262-284`)

```
sigma_from_t = 1 - t                         # in the SAMPLE's dtype (fp32 here)
denoised     = x_t + sigma_from_t * v        # NOTE the PLUS

sigma        = sigmas[i]                     # from the grid, fp32
sigma_next   = sigmas[i+1]
ratio        = sigma_next / sigma
x_next       = ratio * x_t + (1 - ratio) * denoised
```

Three things to get exactly right:

1. **The velocity sign is reversed.** `denoised = x_t + sigma*v`, not
   `x_t - sigma*v` (`scheduler.py:20-21, 274`). H3's transformer predicts a
   *data-ward* velocity.
2. **The two sigmas come from different sources on purpose.** `sigma_from_t` is
   recomputed from the timestep the transformer was conditioned on
   (`1 - t`); `ratio` uses the stored grid. For `sigma < 0.5` the float32 round
   trip `1 - (1 - sigma)` is not exact, and the reference deliberately keeps
   them apart (`scheduler.py:265-268`). Do not "simplify" this.
3. **Evaluate in float32** (`scheduler.py:277-281`). Our latents are already
   fp32 (§1.3) and `denoise.py:264, 270` casts the model output with `.float()`
   before stepping, so the whole update is fp32 end to end.

At the last step `sigma_next = 0` ⇒ `ratio = 0` ⇒ `x_next = denoised`.

`eta = 0`: **no noise is ever re-injected**, despite the reference class being
named "euler ancestral" (`scheduler.py:33-34`).

### 7.4 `scale_noise` (`scheduler.py:196-225`) — fl2va only

```
x_t = t * x_0 + (1 - t) * noise
```

Used for keyframe anchors at `t = 0.999`; `timestep` is taken at face value and
**not** looked up in `timesteps`. Unused for `t2va`.

### 7.5 Row timesteps (`packing.py:469-498`, `before_denoise.py:410-422`)

Per step `i`:

```
row_t[:]                = timesteps[i]                      # default: video t
row_t[cond video rows]  = max(timesteps[i], 0.999)
row_t[gen audio rows]   = audio_timesteps[i]
row_t[cond audio rows]  = 1.0
timestep, timestep_indices = torch.unique(row_t, sorted=True, return_inverse=True)
```

**Text rows keep the video timestep** (they are never overridden). For `t2va`
this yields `timestep` = the ascending-sorted unique of `{t_v, t_a}` — normally
**2 entries**, collapsing to 1 iff `t_v == t_a`. `timestep_indices[s] ∈ {0, 1}`
maps each row into it.

Note `torch.unique(sorted=True)` sorts **ascending**, so which of `t_v`/`t_a`
is index 0 depends on their values and changes over the schedule. A port must
reproduce this, since it feeds `adaln_indices`.

---

## 8. Tensor-name → operation map (fp8 checkpoint)

Header facts, read directly from
`weights/transformer/fl2va_pruned_fp8_scaled.safetensors`: **1082 tensors**,
payload 20 958 205 608 B, no `__metadata__`. Original (non-diffusers) key
spelling throughout.

### 8.1 Layout and transpose conventions

**There are no transposes anywhere** (`convert.py:18`). Every `*.weight` of a
linear layer is stored PyTorch-style **`[out_features, in_features]`**, so the
op is `y = x @ W^T + b`. A cuBLAS-style port wanting `y = W_col_major * x`
can consume the rows directly as columns without a copy.

**QKV layout — verified empirically, and it is *not* what `convert.py` warns
about.** `convert.py:110-132` says the *raw Hub shards* store fused QKV
per-head interleaved (`[head0: q k v, head1: q k v, …]`) and that the reference
de-interleaves at load. Our pruned file has already been de-interleaved: the
mean `|w|` of `qkv_proj.weight` under the contiguous partition separates
cleanly while the interleaved partition is flat, for every block checked:

```
blocks.0.attn.qkv_proj:  contiguous [q,k,v] = 8.450, 7.866, 5.691
                         interleaved       = 7.247, 7.396, 7.363
blocks.1.attn.qkv_proj:  contiguous [q,k,v] = 6.054, 5.992, 4.638
                         interleaved       = 5.586, 5.598, 5.500
```

⇒ **`qkv_proj.weight[0:7168] = W_q`, `[7168:14336] = W_k`,
`[14336:21504] = W_v`.** Split on contiguous thirds; do **not** apply
`reorder_interleaved_qkv`. (Re-verify on one block after any re-download —
§10.3.)

### 8.2 fp8 dequantisation

Three tensors per quantised linear:

| suffix | dtype | shape | meaning |
|---|---|---|---|
| `.weight` | `F8_E4M3` | `[out, in]` | quantised weight |
| `.weight_scale` | `F32` | scalar `[]` | per-tensor weight scale |
| `.input_scale` | `F32` | scalar `[]` | per-tensor **activation** scale (absent on `mlp.fc2`) |
| `.comfy_quant` | `U8` | 27 or 63 B | ASCII JSON tag |

`comfy_quant` payloads, read verbatim:

```
attn.qkv_proj / attn.out_proj / mlp.fc1 :  {"format": "float8_e4m3fn"}
mlp.fc2                                 :  {"format": "float8_e4m3fn", "full_precision_matrix_mult": true}
```

**Verified across all 50 blocks** (not just block 0): `input_scale` is present on
`attn.qkv_proj`, `attn.out_proj` and `mlp.fc1` in every one of the 50 blocks and
**absent on `mlp.fc2` in every one of the 50**. The `comfy_quant` payloads split
exactly 141 × `{"format": "float8_e4m3fn"}` (the three scaled linears) and
50 × `{"format": "float8_e4m3fn", "full_precision_matrix_mult": true}`
(`mlp.fc2`). There is no block anywhere in the file that deviates.

Observed ranges of `448 × scale` over the blocks measured:

| tensor | `448 × weight_scale` | `448 × input_scale` |
|---|---|---|
| `attn.qkv_proj` | 2.33 – 11.88 | 11.95 – 178.50 |
| `attn.out_proj` | 1.48 – 6.16 | 24.00 – 858.00 |
| `mlp.fc1` | 3.03 – 20.00 | 4.59 – 139.50 |
| `mlp.fc2` | 2.41 – 19.50 | — |

Note the calibrated activation maximum reaching **858** on `out_proj`. That is
comfortably inside fp16 and bf16 range, but it rules out any scheme that
assumes activations are O(1).

Measured scale semantics (block 0):

| tensor | fp8 max\|w\| | `weight_scale` | `448 × weight_scale` | `input_scale` | `448 × input_scale` |
|---|---|---|---|---|---|
| `attn.qkv_proj` | 448.0 | 8.1264e-3 | 3.6406 | 3.53655e-2 | 15.84375 |
| `attn.out_proj` | 448.0 | 4.1330e-3 | 1.8516 | 5.35714e-2 | 24.00000 |
| `mlp.fc1` | 448.0 | 1.86942e-2 | 8.3750 | 1.349749e-2 | 6.046875 |
| `mlp.fc2` | 448.0 | 1.07422e-2 | 4.8125 | — | — |

Every quantised tensor's fp8 maximum is **exactly 448.0** (the `e4m3fn` max) and
contains **no NaNs**, so

```
weight_scale = amax(|W_real|) / 448        =>  W_real = W_fp8 * weight_scale
```

is exact by construction. `448 × input_scale` lands on clean fp16-representable
activation maxima (24.0, 15.84375, 6.046875), so the same convention holds on
the activation side:

```
input_scale = amax_calibrated(|x|) / 448   =>  x_fp8 = clamp(x / input_scale, -448, 448)
```

**Recommended kernel for the three tensors that carry `input_scale`:**

```
x_q  = quantise_e4m3( x / input_scale )
acc  = fp8_gemm(x_q, W_fp8)                    # fp32 accumulate
y    = acc * (weight_scale * input_scale)
```

**`mlp.fc2` must not use an fp8 GEMM.** It carries `full_precision_matrix_mult:
true` and, consistently, ships **no `input_scale`** — its input is the SwiGLU
product, which has no static calibration. Dequantise
`W = W_fp8 * weight_scale` to bf16/fp16/fp32 and run an ordinary GEMM.

A port that does not want fp8 tensor cores at all can dequantise every weight
and run bf16 GEMMs throughout; the `input_scale` values are then unused. That
is numerically *better*, not worse — the fp8 path is a speed/VRAM trade.

### 8.3 Top-level tensors (32)

| key | dtype | shape | operation |
|---|---|---|---|
| `video_patch_proj.weight` | F32 | `[5376, 96]` | `proj_in` — patchified video rows → residual stream |
| `video_patch_proj.bias` | F32 | `[5376]` | |
| `audio_patch_proj.weight` | F32 | `[5376, 32]` | `audio_proj_in` — audio latent rows → residual stream |
| `audio_patch_proj.bias` | F32 | `[5376]` | |
| `condition_proj.weight` | BF16 | `[5376, 5120]` | `context_embedder` — Qwen3-VL hidden state → residual stream |
| `condition_proj.bias` | BF16 | `[5376]` | |
| `adaln_t_table` | F32 | `[1025, 8]` | §3.4 — the rank-8 timestep code, one row per grid point |
| `rope.inv_freq` | F32 | `[16]` | §5.1; recomputable, bitwise-equal |
| `token_refiner.blocks.{0,1}.*` | BF16 | see §6 | 2 refiner blocks, same shapes as a main block minus `adaln_proj` |
| `token_refiner.final_norm.weight` | BF16 | `[5376]` | RMSNorm eps 1e-5 |
| `final_layer.norm.weight` | BF16 | `[5376]` | `norm_out.norm`, RMSNorm eps 1e-5 |
| `final_layer.adaln_proj.linear.weight` | **F16** | `[10752, 8]` | rank-8 final modulation; rows `0:5376` = **shift**, `5376:10752` = **scale** |
| `final_layer.adaln_proj.linear.bias` | **F16** | `[10752]` | same split |
| `final_layer.video_out.weight` | F32 | `[96, 5376]` | `proj_out` — video velocity head |
| `final_layer.video_out.bias` | F32 | `[96]` | |
| `final_layer.audio_out.weight` | F32 | `[32, 5376]` | `audio_proj_out` — audio velocity head |
| `final_layer.audio_out.bias` | F32 | `[32]` | |

The dtype split matches `MINIMAX_H3_FP32_SOURCE_PREFIXES` (`convert.py:96-102`):
the two patch projections and the two output heads are F32, everything else was
BF16 in the original. `time_embedder.*` — present in the original index — is
**absent** here (§3.4).

### 8.4 Per-block tensors, `i ∈ [0, 50)` (21 keys × 50 = 1050)

| key | dtype | shape | operation |
|---|---|---|---|
| `blocks.i.norm1.weight` | BF16 | `[5376]` | pre-attention RMSNorm, eps 1e-5 |
| `blocks.i.norm2.weight` | BF16 | `[5376]` | pre-FFN RMSNorm, eps 1e-5 |
| `blocks.i.attn.qkv_proj.weight` | F8 | `[21504, 5376]` | contiguous `[Wq; Wk; Wv]`, each `[7168, 5376]`, no bias |
| `blocks.i.attn.qkv_proj.weight_scale` | F32 | `[]` | ×448 = amax |
| `blocks.i.attn.qkv_proj.input_scale` | F32 | `[]` | activation scale |
| `blocks.i.attn.qkv_proj.comfy_quant` | U8 | `[27]` | `{"format": "float8_e4m3fn"}` |
| `blocks.i.attn.q_norm.weight` | BF16 | `[128]` | per-head RMSNorm on q, eps 1e-5, **before RoPE** |
| `blocks.i.attn.k_norm.weight` | BF16 | `[128]` | per-head RMSNorm on k, eps 1e-5, **before RoPE** |
| `blocks.i.attn.out_proj.weight` | F8 | `[5376, 7168]` | 7168 → 5376 projection back, no bias |
| `blocks.i.attn.out_proj.{weight,input}_scale` | F32 | `[]` | |
| `blocks.i.attn.out_proj.comfy_quant` | U8 | `[27]` | |
| `blocks.i.mlp.fc1.weight` | F8 | `[28672, 5376]` | fused `[gate(14336); value(14336)]` — **gate first** |
| `blocks.i.mlp.fc1.{weight,input}_scale` | F32 | `[]` | |
| `blocks.i.mlp.fc1.comfy_quant` | U8 | `[27]` | |
| `blocks.i.mlp.fc2.weight` | F8 | `[5376, 14336]` | 14336 → 5376, no bias |
| `blocks.i.mlp.fc2.weight_scale` | F32 | `[]` | **no `input_scale`** |
| `blocks.i.mlp.fc2.comfy_quant` | U8 | `[63]` | `…"full_precision_matrix_mult": true` |
| `blocks.i.adaln_proj.linear.weight` | **F16** | `[96768, 8]` | rank-8 modulation, output layout per §3.2 |
| `blocks.i.adaln_proj.linear.bias` | **F16** | `[96768]` | same layout |

Coverage: `32 + 1050 = 1082` ✓ — every tensor in the file is accounted for.

### 8.5 Name reconciliation (original ↔ diffusers)

From `convert.py:169-227`, for reading the diffusers port alongside:

```
video_patch_proj.*            -> proj_in.*
audio_patch_proj.*            -> audio_proj_in.*
condition_proj.*              -> context_embedder.*
time_embedder.proj_in.*       -> time_embedder.linear_1.*      (absent in our file)
time_embedder.proj_out.*      -> time_embedder.linear_2.*      (absent in our file)
blocks.i.*                    -> transformer_blocks.i.*
token_refiner.blocks.i.*      -> token_refiner.refiner_blocks.i.*
  .attn.qkv_proj.weight       -> .attn.to_{q,k,v}.weight       (split contiguous thirds)
  .attn.q_norm/.k_norm        -> .attn.norm_q/.norm_k
  .attn.out_proj              -> .attn.to_out.0
  .mlp.fc1.weight             -> .ff.net.0.proj.weight         (HALVES SWAPPED to [value; gate])
  .mlp.fc2                    -> .ff.net.2
final_layer.norm.*            -> norm_out.norm.*
final_layer.adaln_proj.linear.* -> norm_out.linear.*
final_layer.video_out.*       -> proj_out.*
final_layer.audio_out.*       -> audio_proj_out.*
rope.inv_freq                 -> dropped (recomputed)
```

---

## 9. Numerical-fidelity notes

### 9.1 What is computed in fp32 despite lower-precision storage

- **The scheduler step.** `scheduler.py:277-281` forces float32 for half inputs;
  `denoise.py:264, 270` casts the model output to fp32 before stepping. The
  latents live in fp32 for the whole run (`before_denoise.py:314, 322`). Keep
  them fp32.
- **The rotary angle computation.** `position_ids` is built in **float64**
  (`packing.py:412`) and cast to float32 *inside* `rope.forward`
  (`transformer.py:92`); `inv_freq` is fp32; the product, `cos` and `sin` are
  fp32. Do not build the grid in fp32 from the start — `_temporal_position_grid`
  and `_spatial_position_grid` accumulate in fp64 and the reference explicitly
  reproduces numpy's summation order (`packing.py:338-340, 356-366`).
- **The AdaLN activation.** In the full model, `SiLU(temb)` runs at `temb`'s own
  precision (fp32, since `time_embedder` is an fp32 module) and only the result
  is cast down (`transformer.py:122-126, 148`). The comment there is worth
  reading: *"Every block reads the same `temb`, so a rounding applied before the
  activation biases every block's modulation parameters identically at every
  sampling step, which accumulates coherently over the denoising trajectory."*
  In the rank-8 form the equivalent statement is: **evaluate
  `W_8 @ c(t) + b` in fp32** even though `W_8`/`b` are F16. It is 51 tiny GEMVs;
  there is no reason not to.
- **The two patch projections and the two output heads** are F32 tensors in the
  checkpoint (`convert.py:96-102`, `transformer.py:442-449`). The reference
  aligns the activation with the parameter dtype at each of them
  (`transformer.py:597-599, 638`). Run `proj_in`, `audio_proj_in`, `proj_out`,
  `audio_proj_out` in fp32.
- **RMSNorm mean-of-squares.** Accumulate `mean(x²)` in fp32 regardless of the
  activation dtype. See §10.5 for what the reference actually guarantees.

### 9.2 The RoPE precision edge

`_apply_rotary_emb` **casts `cos`/`sin` down to the activation dtype**
(bf16, if you run the block stack in bf16) before the rotation
(`transformer.py:65-66`). If your kernel keeps them in fp32 you will *not* match
the reference bit for bit — you will be *more* accurate, but different. Decide
deliberately.

Separately, note the magnitude of the T coordinate: it reaches
`L + (5/3)·num_frames`, i.e. a few thousand for a long prompt. fp32 ulp at 4096
is ~2.4e-4; multiplied by `inv_freq[0] = 1` that is the angle error floor of the
reference itself. Matching it requires performing the *same* fp64→fp32 cast at
the *same* point (before multiplying by `inv_freq`), not computing in fp64
throughout.

### 9.3 Epsilon placement

Every RMSNorm is `x * rsqrt(mean(x²) + eps) * w` with `eps` **inside** the
square root, added to the mean of squares — not to the norm, not outside. All
six eps values are 1e-5 (§4.1). `qk_norm` normalises over the **128-wide head
dimension**, not over 7168.

### 9.4 Operator-order hazards

1. **`1 + scale`, then `+ shift`.** `n * (1 + scale) + shift`, in that order
   (`transformer.py:357-359, 365-367, 152-154`). Fusing to `n*scale + (n + shift)`
   changes the rounding.
2. **QK-norm before RoPE** (§4.3). Reversing them is a real, silent quality bug.
3. **Gate multiplies the branch, not the sum.** `x + g*f(x)`, never `g*(x+f(x))`.
4. **SwiGLU half order**: gate = first half of `fc1` in the *original* naming
   (§4.4).
5. **`denoised = x + sigma*v`** with a plus (§7.3), and `sigma_from_t = 1 - t`
   recomputed rather than read from the grid.
6. **`torch.unique(sorted=True)`** in `build_row_timesteps` sorts ascending, so
   the identity of `timestep_indices == 0` flips over the schedule (§7.5).
7. **Audio noise is drawn as `(2A, 32)`**, not `(2, 32, A)` (§1.3) — a port that
   draws the latter and permutes will not reproduce a seed.
8. **`patchify` is frame → row → column, channel-major within the patch** (§1.4);
   `frame_grid` uses the same `hh*(Wl/2)+ww` order (§2.3). A mismatch between the
   two silently scrambles the spatial RoPE.

### 9.5 Schedule-length hazard

`before_denoise.py:410-421` builds the row-timestep plan with
`zip(timesteps, audio_timesteps)`. If `unique_consecutive` (§7.1) collapses the
two shifted grids to *different* lengths, `zip` silently truncates to the
shorter one while `denoise.py:302` iterates over the **video** timesteps —
producing an `IndexError` or a truncated run depending on which is shorter.
It does not happen at practical `N`, but a port should assert
`len(timesteps) == len(audio_timesteps)` after `set_timesteps`.

### 9.6 Things that are deliberately *not* high precision

- The block stack runs at the loaded dtype (bf16 in diffusers, fp8-GEMM here).
  The reference's `_skip_layerwise_casting_patterns = ["norm"]`
  (`transformer.py:437`) keeps norm weights out of any layerwise cast.
- `to_out[1]` is `Dropout(0.0)` — a no-op; omit it.
- The `attention_mask` is `None` on the `t2va` path; do not synthesise an
  all-zero float mask (`transformer.py:190-193` notes it would hard-fail flash /
  sage backends).

---

## 10. Open questions / UNRESOLVED

### 10.1 PARTLY RESOLVED — the `adaln_t_table` index mapping

The file is now complete and the table has been read and analysed. Two of the
three sub-questions are settled by measurement; the third is not, and the
reason it cannot be is stated precisely below.

**SETTLED — the grid is uniform over `[0, 1]` in 1024 intervals.** The table is
smooth in the row index with no plateaus, steps or irregular spacing. A lookup
table of discrete *training* timesteps would be piecewise-flat or unevenly
spaced; it is neither.

**SETTLED — the lookup is linearly interpolated, not nearest-neighbour.**
Measured on the real tensor:

| column | range | max row-to-row step, as fraction of range | max 2nd difference / range |
|---|---|---|---|
| c0 | 0.870090 | 0.0029 | 2.3e-5 |
| c1 | 0.314754 | 0.0093 | 6.8e-5 |
| c2 | 0.126696 | 0.0229 | 3.8e-4 |
| c3 | 0.102978 | 0.0140 | 2.4e-4 |
| c4 | 0.010183 | 0.0361 | 9.7e-4 |
| c5 | 0.002224 | 0.0419 | 1.4e-3 |
| c6 | 0.001175 | 0.0690 | 4.1e-3 |
| c7 | 0.000653 | 0.0559 | 2.7e-3 |

The decisive test: drop every other row, linearly interpolate the omitted ones
back, and measure the error. Worst case over all eight columns is **2.4e-5
absolute, ≤ 0.2 % of the column range**. At the full grid the error is roughly
four times smaller again, because the error of linear interpolation is
second-order in the step. Nearest-neighbour, by contrast, incurs up to half a
grid step — **1.5 % to 3.5 % of range on the small components, one to two
orders of magnitude worse**. Interpolate.

**Incidentally confirmed — the rank-8 factorisation is an SVD/PCA truncation
(§10.2).** The per-column ranges decay geometrically (0.870, 0.315, 0.127,
0.103, 0.0102, 0.00222, 0.00118, 0.000653), which is a singular-value profile.
The components are ordered by decreasing importance, exactly as a truncated SVD
produces. This is now evidence rather than inference.

**STILL UNRESOLVED — the direction: does row 0 mean `t = 0` or `t = 1`?**
Because the grid is uniform, "over `t`" and "over `sigma = 1 - t`" differ *only*
by reversing the row order, so this is a single bit. It cannot be recovered
from the local files: doing so requires rebuilding `SiLU(time_embedder(t))`
from the **original** `MiniMaxAI/MiniMax-H3` `transformer/` shards, and those
weights are not present — the pruned checkpoint deleted `time_embedder.*`
entirely, which is the whole point of the pruning.

Evidence favouring **row 0 ↔ `t = 0`**, which is the implemented default:

- `‖row‖` is 0.5387 at row 0 and 0.4252 at row 1024, with a minimum of 0.0666
  near the middle. Row 0 is the more extreme endpoint.
- `‖row_{j+1} − row_j‖` is largest at row 0 (0.00489) and decays to 0.00254
  before rising slightly to 0.00272 at the far end.
- Both are consistent with `Timesteps(t=0)` being the distinguished point of
  the sinusoid manifold: with `flip_sin_to_cos=True`, `s(0)` is exactly
  `(1,…,1, 0,…,0)` — all 128 cosines at their maximum and all 128 sines exactly
  zero — while `s(1)` is a generic point.

This is suggestive, not conclusive, because the sinusoid passes through an
unknown two-layer MLP and a SiLU before reaching the table. **Keep the switch.**
If the direction is wrong the symptom is specific and easy to see end to end:
the sample degrades rather than improves along the denoising trajectory. That
is §3.5's step 3, and flipping the default is a one-line change.

---

**Original statement of the question, retained for context:**

**What is unknown:** how a continuous timestep `t ∈ [0,1]` selects (or blends)
a row of `adaln_t_table [1025, 8]`; whether the grid is over `t` or over
`sigma = 1 - t`; whether the lookup is nearest-neighbour or linearly
interpolated.

**What was checked:** all of `ref/diffusers/**` (the reference implements the
full `time_embed_dim = 2688` path and has no notion of a table);
`ref/FL2VA/transformer/config.json` and `ref/transformer/config.json` (no table
field); `ref/FL2VA/model_index.json` and `ref/model_index.json`; the
safetensors header of the pruned checkpoint (no `__metadata__`; the only
per-tensor metadata is ComfyUI's `comfy_quant` format tag on the fp8 weights);
`ref/README.md` (mentions that AdaLN outputs "can be precomputed and cached" at
line 141 but describes no format). The tensor payload itself is at byte offset
20 870 775 256 and the local file was 1.92 GB of 20.96 GB at the time of
writing, so it could not be inspected.

**Recommended action:** §3.5 gives a verification recipe. Until it is run,
implement the lookup behind a switch and default to linear interpolation on
`u = t * 1024`.

### 10.2 UNRESOLVED — whether the rank-8 factorisation is exact or approximate

The algebraic constraint `W_8 @ c(t) + b ≡ W_full @ SiLU(temb(t)) + b` cannot
hold exactly for a smooth non-linear curve in `R^2688`, so the pruned checkpoint
is presumably an 8-component least-squares/PCA approximation. **This is
inference, not a sourced claim** — no file in `ref/` documents the pruning, and
the tool that produced the checkpoint is not present. The error it introduces
into every modulation vector is unknown and cannot be bounded without the
original `time_embedder` weights. Anyone comparing this port against the
official MiniMax output should expect a small systematic difference from this
alone.

### 10.3 RESOLVED — QKV interleave, re-verified on late blocks

§8.1's conclusion (contiguous `[q; k; v]`, *not* per-head interleaved) originally
rested on blocks 0 and 1. Re-run on blocks 30 and 35 (the latest blocks whose
fp8 QKV payload had arrived), sampling every 8th row's mean `|w|`:

```
block  0: contiguous [q,k,v] = 9.019, 8.282,  5.674   interleaved = 7.635,  7.670,  7.670
block 30: contiguous [q,k,v] = 4.030, 4.102,  5.077   interleaved = 4.381,  4.410,  4.418
block 35: contiguous [q,k,v] = 6.535, 6.493,  8.286   interleaved = 7.040,  7.125,  7.150
block 45: contiguous [q,k,v] = 8.867, 8.539, 12.659   interleaved = 9.813, 10.151, 10.101
block 49: contiguous [q,k,v] = 4.482, 4.532,  7.652   interleaved = 5.450,  5.789,  5.426
```

Blocks 45 and 49 were re-run on the completed file, which is what §10.3
originally asked for.

The contiguous partition separates at every depth (note the ordering flips —
`v` is *smallest* at block 0 and *largest* by block 30, which is a real depth
trend, not an artefact) while the interleaved partition is flat to within 1 %
everywhere. **Confirmed: contiguous `[Wq; Wk; Wv]`, do not de-interleave.**

The same pass re-confirmed §8.2 on blocks 0/30/35: fp8 `amax` is exactly
`448.0`, NaN count is exactly 0, and `448 × weight_scale` lands on clean
fp16-representable values (3.640625, 11.875001, 7.750000).

### 10.4 UNRESOLVED (irrelevant to our checkpoint) — sinusoid `max_period`

The full-AdaLN timestep sinusoid uses diffusers'
`get_timestep_embedding(..., max_period=?)` default. `Timesteps` is constructed
at `transformer.py:483` without a `max_period` argument, and the diffusers
library source is not in `ref/`. The library default is 10000, which §3.1
assumes. **Our checkpoint does not use this path at all** (no `time_embedder`),
so it matters only if you later reconstruct `SiLU(temb(t))` for the §3.5
verification — in which case pin the exact diffusers version.

### 10.5 UNRESOLVED (low risk) — RMSNorm internal accumulation dtype

The reference uses `torch.nn.RMSNorm`, whose internal accumulation dtype for
half-precision inputs is a PyTorch implementation detail that varies by version
and backend and is not pinned anywhere in `ref/`. §9.1's recommendation (fp32
mean-of-squares) is the numerically safe choice but may differ from a given
PyTorch build in the last bits.

### 10.6 UNRESOLVED — token-refiner attention mask

`MiniMaxH3TokenRefinerBlock.forward` (`transformer.py:273-276`) passes no
`attention_mask`, and `t2va` never pads the text stream, so full bidirectional
attention over all `L` rows is unambiguous **for our path**. Whether the
original sglang implementation masked anything in the refiner (e.g. across the
vision blocks of a multi-keyframe `fl2va` request) cannot be determined from the
diffusers port. Not a blocker for `t2va`.

### 10.7 Not investigated (out of scope, flagged for completeness)

- The Qwen3-VL conditioner itself — its 3-D mrope
  (`mrope_section [24,20,20]`, `mrope_interleaved: true`, `rope_theta 5e6`),
  its GQA (64 q heads / 8 kv heads), and the exact tokenizer. §1.2 specifies
  only the interface and the layer index; the encoder needs its own spec.
- Both VAEs. The video decoder is covered by `docs/vae_decoder_spec.md`; the
  audio VAE (`AutoencoderKLMiniMaxH3Audio`, DAC + BigVGAN) is not.
- `fl2va` keyframe encoding: the fp16 rounding of the sampled posterior
  (`encoders.py:299-300`), the fixed encode seed 42 (`packing.py:87`), and the
  conditioning-noise draw order (`packing.py:512-515`) are all reproducibility-
  critical and are documented in those files, but were not verified here.
