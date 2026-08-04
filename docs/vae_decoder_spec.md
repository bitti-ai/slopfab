# MiniMax H3 Video VAE — Decoder Implementation Specification

Target: a C++/CUDA port of the **decoder** half of `AutoencoderKLLegacy`
(`_class_name` in `ref/FL2VA/video_vae/source/config.json`), as shipped in
`weights/vae/minimax_h3_video_vae_fp16.safetensors` (562 tensors, all F16).

Everything here is grounded in:

- `D:\Projects\vidfab\ref\FL2VA\video_vae\*.py` (the inference-only reference bundle)
- `D:\Projects\vidfab\ref\FL2VA\video_vae\source\config.json` — **the constructor config that actually built the checkpoint**
- `D:\Projects\vidfab\ref\FL2VA\video_vae\config.json` — the wrapper/runtime config
- `D:\Projects\vidfab\ref\vae\config.json` — a *re-expressed* config for the diffusers-native `AutoencoderKLMiniMaxH3` port (different module names, same model)
- the checkpoint itself, enumerated with `vidfab.exe inspect`

> **Read this first.** The `ref/vae/config.json` names quoted in the task brief
> (`block_out_channels`, `spatial_downsample_factors`, `layers_per_block`,
> `norm_num_groups`, …) describe the **encoder only**. There is **no CNN
> upsampler stack in this decoder.** See §3.

---

## 0. Resolved configuration

`AutoencoderKLLegacy.__init__` (`klvae.py:1097-1197`) is instantiated from
`source/config.json`:

```
use_3d_conv       = true      causal_encoder = true    causal_decoder = false
z_channels        = 24        embed_dim      = 24      out_ch = 3
ch                = 128       ch_mult        = [1,2,2,4,4,8]
space_down        = [2,2,2,2,1,1]   time_down = [1,2,2,1,1,1]
padding_mode      = "reflect" padding_mode_t = null    use_t_isolated_gn = true
pixel_norm_type   = "imagenet"  scaling_factor = 1.0   shift_factor = 0.0
use_vit_decoder   = true
vit_decoder_kwargs = { num_layers:36, heads:32, dim_head:64,
                       norm_type:"rms_norm", norm_affine:true,
                       qk_norm_type:"rms_norm", qk_norm_affine:false,
                       ffn_activation_fn:"silu", ffn_use_gated:true,
                       rope_theta:100.0, rope_dim_ratio:0.75 }
```

Derived in `klvae.py:1147-1148`:

```python
self.vae_ratio   = int(np.cumprod(space_down)[-1])   # 2*2*2*2*1*1 = 16
self.vae_ratio_t = int(np.cumprod(time_down)[-1])    # 1*2*2*1*1*1 = 4
```

`ViT3DDecoder` is then built with (`klvae.py:1181-1189`):

```python
vit_kwargs = {"patch_size": self.vae_ratio,      # 16
              "in_channels": z_channels,         # 24
              "out_channels": out_ch,            # 3
              **vit_decoder_kwargs}
vit_kwargs.setdefault("patch_size_t", self.vae_ratio_t)   # 4
vit_kwargs.setdefault("t_causal", causal_decoder)         # False
```

Values **not** in `vit_decoder_kwargs` therefore take `ViT3DDecoder`
defaults (`vae_vit.py:220-243`): `num_register_tokens=4`, `eps=1e-5`,
`bias=True`, `mask_config={}`. `FeedForward` defaults
(`base_module.py:55-65`): `mult=4`, `glu_balanced=False`.
`TransformerBlock` default `use_scale=True` (`base_module.py:213`).

Resulting constants:

| symbol | value | source |
|---|---|---|
| `dim` (model width) | **2048** | `heads*dim_head = 32*64` (`vae_vit.py:246`) |
| `rope_apply_dim` | **48** | `int(64 * 0.75)` (`vae_vit.py:247`) |
| FFN inner dim | **8192** | `round(2048*4*1)` (`base_module.py:68`) |
| `patch_dim` (proj_out) | **3072** | `3*4*16*16` (`vae_vit.py:281`) |
| spatial upsample | **16×** | `patch_size` |
| temporal upsample | **4×** | `patch_size_t` |

Runtime knobs from `ref/FL2VA/video_vae/config.json`, passed through
`MiniMaxH3VideoVAE.from_pretrained` (`minimax_h3_video_vae.py:90-101`):

```
clip_length=17  token_drop=3  decoder_tiling=1  parallel_tiling=1
tile_size=256   tile_overlap_min=64   decoder_parallel=0  chunk_dim=-1
```

`setup_forward` (`klvae.py:99-119`) then computes, for this model:

```python
frame_drop         = token_drop * vae_ratio_t             # 3*4  = 12  (encode-only)
frame_pre_padding  = (-clip_length) % vae_ratio_t         # (-17)%4 = 3
tokens_chunk_size  = ceil(clip_length / vae_ratio_t)      # ceil(17/4) = 5
token_overlap      = (-token_drop) % tokens_chunk_size    # (-3)%5 = 2
frame_overlap      = max(token_overlap*vae_ratio_t - frame_pre_padding, 0)   # 2*4-3 = 5
isolated_first_frame = isolated_last_frame = isolated_key_frame = False
decoder_tile_size        = tile_size        = 256   # klvae.py:116
decoder_tile_overlap_min = tile_overlap_min = 64    # klvae.py:117
```

**Memorise these five numbers — the whole temporal chunking scheme is
`tokens_chunk_size=5`, `token_overlap=2`, `frame_pre_padding=3`,
`frame_overlap=5`, `chunk_dec = 5*4 = 20`.**

### Code paths excluded

The single-GPU, non-distributed, eager inference path is specified. Excluded:

- **Spatial/sequence parallel** (`parallel.py`). `decoder_parallel=0` and
  `_ensure_vae_parallel_state()` (`minimax_h3_video_vae.py:40-59`) seeds
  `sp_size=1, sp_enabled=False`, so `Attention.spatial_parallel=False`
  (`attention.py:84`), `ViT3DDecoder.spatial_parallel=False`
  (`vae_vit.py:276`), and every `all_to_all_4D` / `get_subseq` /
  `gather_subseq` / `exchange_borders` call is skipped. `_pad_for_sp`
  returns `(hidden_states, img_ids, pack_info, 0)` unchanged
  (`vae_vit.py:178-179`).
- **FlashAttention / FlexAttention.** `flash.py` in this bundle is already a
  PyTorch-SDPA shim (`flash.py:2`, `flash.py:55-72`). Since
  `t_causal=False`, `pack_info` is empty (`vae_vit.py:326-334`) and
  `flash_attn` takes the unmasked branch `_sdpa_attention(q,k,v,causal=False)`
  (`flash.py:178`). **The decoder uses plain full bidirectional attention with
  no mask.**
- **Training / masking.** `mask_config={}` ⇒ `mask_prob=0` ⇒
  `mask_enabled=False` (`vae_vit.py:128-130`), so
  `apply_mask_preprocess`/`apply_mask_postprocess` are identities
  (`vae_vit.py:159-164`, `vae_vit.py:202-207`). Gradient checkpointing raises
  (`attention.py:29-34`).
- **`torch.compile` fast paths** (`base_module.py:107-134`, `func.py:109-141`)
  — all default to OFF and are numerically identical to eager.
- **Encoder** (`vae_cnn.py`, `EncoderFCN3D`) — documented only in §3.3 and §4.5
  because its 116 tensors must be accounted for.

---

## 1. Top-level dataflow

### 1.1 Symbols

```
B                      batch
T_lat, H_lat, W_lat    latent extents
T_px  = T_lat * 4      pixel frames (before chunk trimming)
H_px  = H_lat * 16
W_px  = W_lat * 16
```

### 1.2 Full chain

```
(0)  latents from the diffusion transformer      [B, 24, T_lat, H_lat, W_lat]
        |
(1)  DE-NORMALISE:  z = z_norm * latents_std + latents_mean      (per channel, dim=1)
        |                                                 [B, 24, T_lat, H_lat, W_lat]
(2)  decode_base -> decode_temporal            klvae.py:790-809, 678-788
        |   temporal chunking: window 7 tokens, stride 5 tokens
        |   per chunk:  clip_z                 [B, 24, 7, H_lat, W_lat]
        |
(3)      _adaptive_decode -> tiled_decode      klvae.py:437-441, 365-429
        |   spatial tiling in PIXEL space: 256x256 tiles, >=64 px overlap
        |   per tile:  z_tile                  [B, 24, 7, 16, 16]   (256/16)
        |
(4)        decode(z_tile)                      klvae.py:1214-1229
        |     post_quant_conv: Conv3d(24->24, k=1)   [B, 24, 7, h, w]
        |     ViT3DDecoder(...)                      [B,  3, 28, 16h, 16w]
        |
(5)      tile blend + trim  ->                 [B, 3, 28, H_px, W_px]
        |
(6)    chunk split/blend/stitch ->             [B, 3, T_out, H_px, W_px]
        |
(7)  trim_output(recon, frame_num)             klvae.py:452-459   (tail-aligned)
        |
(8)  DE-NORMALISE PIXELS: rgb = x * imagenet_std + imagenet_mean, clamp(0,1)
     VAEProcessor.revert_tensor                vae_processor.py:212-221
        |
(9)  uint8 frames                              [B, 3, T_out, H_px, W_px]
```

### 1.3 `latents_mean` / `latents_std` — multiply-then-add

**Direction: `z = z_norm * latents_std + latents_mean`.** Multiply by std
first, then add mean.

Grounding and caveat:

- `latents_mean` and `latents_std` exist as real F16 tensors of shape `[24]`
  in the checkpoint, and are duplicated in the safetensors `__metadata__`
  under the key `minimax_h3_video_vae`.
- They appear in `ref/vae/config.json` and `ref/FL2VA/video_vae/config.json`
  but **nowhere in any `.py` file in `ref/`** (verified by grep over all of
  `D:\Projects\vidfab\ref`: the only hits are the four `config.json` files).
  The application site is in the *pipeline*
  (`MiniMaxH3ModularPipeline` / `AutoencoderKLMiniMaxH3` in diffusers
  0.36.0.dev0), which is **not part of the provided inputs**.
- The VAE itself performs no scaling: `scaling_factor = 1.0` and
  `shift_factor = 0.0` in `source/config.json`, and neither is referenced in
  `klvae.py` outside `__init__`'s signature (`klvae.py:1124-1125`). So there
  is no additional `/scaling_factor` term to worry about.
- Because the transformer is trained on `z_norm = (z - mean)/std`, the decode
  side is forced to be the exact inverse `z = z_norm*std + mean`. The magnitudes
  agree with this reading: `latents_std` ∈ [0.45, 3.28], `latents_mean` ∈
  [-1.37, 1.07] — these are raw per-channel statistics of encoder output, not
  reciprocals.

**Layout:** both are length-24 vectors indexed by the latent channel
(dim 1 of `[B,24,T,H,W]`). Broadcast as `[1,24,1,1,1]`.

**Precision:** the checkpoint stores them as F16, which loses precision versus
the fp32 values in `config.json`. **Prefer the fp32 literals from
`ref/vae/config.json` (or the safetensors `__metadata__` JSON, which carries
full fp32 text) over the F16 tensors**, and do the de-normalisation in fp32
before casting `z` to the compute dtype. See §5.1.

> Note: `ref/FL2VA/video_vae/config.json` contains **corrupted**
> `latents_std` values — e.g. `1.68317747116088865` and `0.96531379222869875`
> have an extra digit versus `ref/vae/config.json` /
> the safetensors metadata (`1.6831774711608887`, `0.9653137922286987`).
> Use `ref/vae/config.json` or the checkpoint metadata, not the FL2VA copy.

### 1.4 `clip_length: 17` and `token_drop: 3` — temporal chunked decoding

`clip_length` and `token_drop` never appear directly in the decode loop; they
act **only** through the five derived constants in §0. Their decode-time
meaning:

- `clip_length=17` frames per encoder chunk, with `vae_ratio_t=4` ⇒
  `tokens_chunk_size = ceil(17/4) = 5` latent frames per chunk, and
  `frame_pre_padding = (-17) % 4 = 3` — i.e. a 5-token chunk decodes to 20
  frames but only 17 are real; the **leading 3** decoded frames are
  pre-padding artefacts of the causal encoder and are dropped.
- `token_drop=3` ⇒ `token_overlap = (-3) % 5 = 2` extra latent tokens of
  right-context fed to every chunk, and `frame_overlap = 2*4 - 3 = 5` output
  frames of overlap between consecutive chunks. It also sets
  `split_count = int(token_drop > 0) + 1 = 2` (`klvae.py:583`, `klvae.py:758`).

#### 1.4.1 Algorithm (`decode_temporal`, `klvae.py:678-788`; streaming variant `klvae.py:571-676`)

Inference default takes the **streaming** path
(`_resolve_temporal_stream_cat()` defaults to `"1"`, `klvae.py:50-52`); it is
numerically identical to the list-concat path — it just preallocates the
output and copies in place.

With `isolated_first_frame = isolated_last_frame = False`, `z_head = z_tail =
None` and `isolated_token_num = 0`:

```python
pseudo_total_tokens = T_lat + token_drop                       # T_lat + 3
remainder = pseudo_total_tokens % 5
pad_tokens = (5 - remainder) if remainder else 0               # inference only
# pad by repeating the LAST latent frame:
z = cat([z, z[:,:,-1:].repeat(1,1,pad_tokens,1,1)], dim=2)     # klvae.py:711-713
pseudo_total_tokens += pad_tokens
num_chunks = pseudo_total_tokens // 5 - 1                      # -1 because token_drop>0
```

For each chunk `i in [0, num_chunks)`:

```python
t_start = i * 5
t_end   = t_start + 5 + 2          # = t_start + 7   (chunk + token_overlap)
clip_z  = z[:, :, t_start:t_end]                       # 7 latent frames
clip_dec = _adaptive_decode(clip_z)                    # -> 7*4 = 28 pixel frames
```

Then the 28 frames are split with `chunk_dec = tokens_chunk_size *
vae_ratio_t = 20`, `split_count = 2` (`klvae.py:639-655`):

| `j` | slice of `clip_dec` | after `[frame_pre_padding:]` = `[3:]` | role |
|---|---|---|---|
| 0 | `[0:20]` (20 frames) | frames **3..19** → **17 frames** | primary output |
| 1 | `[20:28]` (8 frames) | frames **23..27** → **5 frames** | `dec_overlap`, carried to next chunk |

Stitching:

- For `j == 0`, if a `dec_overlap` from chunk `i-1` exists, the current
  17-frame block is **linearly cross-faded** with it over `frame_overlap = 5`
  frames: `blend(dec_overlap, clip_dec_chunk, 5, dim=-3)` (`klvae.py:647-649`).
- `blend` (`klvae.py:220-250`):
  ```python
  positions = arange(5, dtype=b.dtype)          # [0,1,2,3,4]
  weight_a  = 1 - positions/5                   # [1.0, 0.8, 0.6, 0.4, 0.2]
  weight_b  = positions/5                       # [0.0, 0.2, 0.4, 0.6, 0.8]
  out[0:5]  = a[-5:]*weight_a + b[0:5]*weight_b
  out[5:]   = b[5:]
  ```
  Note the ramp is **asymmetric**: the first blended frame is 100 % from the
  previous chunk, the last is 80 % `b` / 20 % `a` — it never reaches 100 % `b`.
- After the last chunk, the trailing `dec_overlap` (5 frames) is appended
  verbatim (`klvae.py:657-660`, `klvae.py:775-776`).

Total logical frames = `num_chunks * 17 + 5`.

Finally, frames generated purely from the repeated padding token are dropped
from the tail. In the list path this is an explicit slice
(`klvae.py:784-786`); in the streaming path the writer simply stops at
`output_frames` (`klvae.py:600-609`). The count is
`_decode_temporal_pad_frames` (`klvae.py:514-529`); with `intra_tail =
clip_length % vae_ratio_t = 17 % 4 = 1 ≠ 0`:

```python
z_len_before_pad = z.shape[2] - pad_tokens     # original T_lat
pad_frames = sum(1 if (z_len_before_pad + k) % 5 == 0 else 4
                 for k in range(pad_tokens))
```

#### 1.4.2 The natural (unpadded) case

Encoding `L = 17K + 5` frames yields `T_lat = 5K + 2` tokens
(`VAEProcessor.get_latent_length`, `vae_processor.py:173-186`:
`(L - frame_overlap)//clip_length * tokens_chunk_size + token_overlap`).
Then `pseudo_total = 5K + 5`, `remainder = 0`, `pad_tokens = 0`,
`num_chunks = K`, and the output is exactly `17K + 5 = L` frames.
The last chunk consumes tokens `[5(K-1), 5K+2)` — exactly 7, with nothing
left over. **A correct port should assert `(T_lat + 3) % 5 == 0` in the
common path and only fall back to the padding branch otherwise.**

Absolute frame mapping (no padding): chunk `i` writes output frames
`[17i, 17i+17)`, and its `dec_overlap` covers `[17i+17, 17i+22)`, which is
exactly the first 5 frames of chunk `i+1`'s primary block — the same
absolute frames decoded twice from overlapping token windows, hence the
cross-fade.

#### 1.4.3 `trim_output`

`decode_base` (`klvae.py:790-809`) calls `trim_output(recon, frame_num)`.
Because `causal_encoder=True` (`klvae.py:454`), trimming is **tail-aligned**:
`dec = dec[:, :, -target_frames:]`. If the caller passes no `frame_num`, it is
a no-op.

### 1.5 Spatial tiling (`tiled_decode`, `klvae.py:365-429`)

`decoder_tiling=1`, so `_adaptive_decode` is `tiled_decode`
(`klvae.py:437-441`). Tiling is computed in **pixel** space and then
divided by `vae_ratio=16` to slice the latent.

`split_tiles(input_len, is_decoder=True)` (`klvae.py:192-218`) with
`tile_size=256`, `tile_overlap_min=64`:

- If `256 >= input_len`: single tile, no overlap. **For output up to
  256×256 pixels (16×16 latent) tiling is a no-op.**
- Otherwise `N = ceil(input_len/256)`, increased until
  `256*N - 64*(N-1) - input_len >= 0`; the surplus is redistributed by
  bumping overlaps in units of `vae_ratio = 16`, round-robin
  (`klvae.py:209-211`). Since `input_len`, 256 and 64 are all multiples of
  16, tiles cover the input exactly.

Per tile: `z_tile = z[..., i_pos//16 : (i_pos+256)//16, j_pos//16 :
(j_pos+256)//16]` (`klvae.py:384-392`), decoded independently, then merged
(`klvae.py:414-428`) in this exact order for tile `(i,j)`:

1. if `i > 0`: `blend(rows[i-1][j], tile, y_overlap[i-1], dim=-2)` — vertical
2. if `j > 0`: `blend(row[j-1],   tile, x_overlap[j-1], dim=-1)` — horizontal
3. if not last row: drop the last `y_overlap[i]` rows
4. if not last col: drop the last `x_overlap[j]` cols

`rows[i-1][j]` and `row[j-1]` are the **unmodified** decoded tiles (the loop
never writes back into `rows`), so blending always reads raw neighbours.
Rows are concatenated along `-1` then along `-2`.

**Nesting order: temporal chunking is the OUTER loop, spatial tiling the
INNER loop** — `decode_temporal` → `_adaptive_decode` → `tiled_decode` →
`decode`.

### 1.6 Pixel de-normalisation

The ViT emits ImageNet-normalised values. `VAEProcessor.revert_tensor`
(`vae_processor.py:212-221`) applies `transform_rev`, built by
`get_denormalize_transform("imagenet")` (`normalize.py:35-39`):

```
inv_mean = -m/s ;  inv_std = 1/s
Normalize(inv_mean, inv_std)(x) = (x - inv_mean)/inv_std = x*s + m
```

with `m = (0.485, 0.456, 0.406)`, `s = (0.229, 0.224, 0.225)`
(`normalize.py:9-10`), followed by `.clamp(0, 1)`.

So: **`rgb01[c] = x[c]*s[c] + m[c]`, then clamp to [0,1], then ×255 for
uint8.** Per RGB channel over dim 1.

---

## 2. The 36-layer ViT decoder

Module: `ViT3DDecoder` (`vae_vit.py:216-366`), blocks `TransformerBlock`
(`base_module.py:200-282`), attention `Attention` (`attention.py:37-163`).

### 2.1 Token construction

Input to `ViT3DDecoder.forward` is `x = post_quant_conv(z)`,
`[B, 24, T, H, W]` (T,H,W are the *clip/tile* latent extents, e.g.
`7 × 16 × 16`).

1. **Patchify (trivial).** `_pack_tensors_3d(x, 1, 1)` (`vae_vit.py:301`,
   impl `vae_vit.py:55-74`) with `patch_size = patch_size_t = 1`: it is a pure
   layout change `[B,24,T,H,W] → [B, T*H*W, 24]`, channel-last, tokens in
   **(t, h, w) row-major** order: `n = (t*H + h)*W + w`.
   *There is no convolutional patch embedding — the decoder consumes one
   token per latent voxel and expands 16×16×4 at the output.*
2. **Embed.** `x_embedder`: `nn.Linear(24 → 2048)` (`vae_vit.py:251`), applied
   via `_linear_with_module_dtype` inside `torch.autocast("cuda",
   enabled=False)` (`vae_vit.py:304-305`).
   `N = T*H*W`.  → `[B, N, 2048]`
3. **Register tokens.** `self.register_tokens` `[1, 4, 2048]` (a learned
   `nn.Parameter`, `vae_vit.py:153`) is `expand(B,-1,-1)` and **appended after**
   the patch tokens (`vae_vit.py:311-313`). Positions `N .. N+3`.
4. **CLS token.** `has_cls_token=False`, so there is **no learned CLS
   parameter**. Instead a literal zero token is appended
   (`vae_vit.py:315-316`):
   ```python
   cls_token = torch.zeros_like(hidden_states[:, 0:1, :])
   ```
   Position `N+4`. It contributes K/V to attention (and receives updates via
   the residual stream) but starts at zero. **Do not omit it — it changes the
   softmax denominator for every token.**
5. Sequence length is `N + num_suffix` where `num_suffix = 1 +
   num_register_tokens = 5` (`vae_vit.py:299`).
6. **Position ids.** `create_token_ids((T,H,W), device, x.dtype)`
   (`func.py:12-47`) → `[1, N, 3]`, expanded to `B`. Suffix ids are
   **zeros** `[B, 5, 3]` (`vae_vit.py:321-322`), so the 5 suffix tokens get
   angle 0 ⇒ `cos = 1, sin = 0` ⇒ identity rotation.
7. **Discarding.** After the blocks, `norm_out` and `proj_out` are applied to
   **all `N+5`** tokens, and only then is the tail sliced off:
   `output = output[:, :num_patches, :]` (`vae_vit.py:359`).
   Both ops are token-wise, so **slicing before `norm_out` is numerically
   identical and is the recommended optimisation** (saves 5 × 3072-wide
   GEMM rows).

### 2.2 Block structure — **pre-norm, with LayerScale**

`TransformerBlock.forward` (`base_module.py:262-282`):

```python
h = norm1(x)                       # RMSNorm, computed in fp32 (see §5.2)
a = attn(h, rope, pack_info)
x = x + a * scale1                 # LayerScale, per-channel [2048]
h = norm2(x)
f = ff(h)
x = x + f * scale2
```

- **Pre-norm** (norm applied to the residual-stream input, output added back).
- **`scale1` / `scale2`** are learned per-channel gates (`nn.Parameter`,
  zero-initialised at construction, `base_module.py:245`, `base_module.py:260`;
  present in the checkpoint as `[2048]` F16). They multiply the *sub-layer
  output* elementwise **before** the residual add.
- There is **no final residual scaling** and **no dropout**.

### 2.3 Norms

| site | type | shape in ckpt | affine | eps | evidence |
|---|---|---|---|---|---|
| `norm1`, `norm2` | **`nn.RMSNorm`** | `weight [2048]`, no bias | yes | **1e-5** | `norm_type="rms_norm"` in `source/config.json`; `base_module.py:222-233`; checkpoint has `norm1.weight` but **no** `norm1.bias` |
| `attn.norm_q`, `attn.norm_k` | **`nn.RMSNorm`** over `dim_head=64` | *(no tensors)* | **no** (`qk_norm_affine=false`) | **1e-5** | `attention.py:68-74` |
| `norm_out` | **`nn.LayerNorm`** | `weight [2048]` + `bias [2048]` | yes | **1e-5** | `vae_vit.py:280`; checkpoint has both weight and bias |

RMSNorm semantics (PyTorch): `y = x / sqrt(mean(x²) + eps) * weight` —
**no mean subtraction, eps inside the sqrt, added to the mean-square.**

LayerNorm semantics: `y = (x - mean)/sqrt(var + eps) * weight + bias`, with
biased variance, over the last dim (2048).

The presence/absence of `.bias` keys in the checkpoint is a hard confirmation
of RMSNorm vs LayerNorm at each site.

### 2.4 Attention

`Attention.forward` (`attention.py:128-163`):

```python
qkv = self.to_qkv(hidden_states)                       # [B, S, 6144]
qkv = qkv.view(B, S, -1, 3*dim_head)                   # [B, S, 32, 192]
q, k, v = torch.chunk(qkv, 3, dim=-1)                  # each [B, S, 32, 64]
q = norm_q(q.float()).to(q.dtype)                      # RMSNorm over last dim
k = norm_k(k.float()).to(k.dtype)
q = apply_rotary_pos_emb(q, rope)
k = apply_rotary_pos_emb(k, rope)
out = flash_attn(q, k, v)                              # SDPA, no mask
out = out.reshape(B, S, 2048)
out = self.to_out(out)
```

- **QKV is FUSED** into one `nn.Linear(2048 → 6144)`, with **bias**
  (`bias=True`, `attention.py:80`; `to_qkv.bias [6144]` in the checkpoint).
- **CRITICAL layout.** The `view(B, S, -1, 192)` before the chunk means Q/K/V
  are **interleaved per head**, not blocked. For output row index
  `r ∈ [0, 6144)` of `to_qkv.weight`:
  ```
  r = h*192 + s ,  h ∈ [0,32) , s ∈ [0,192)
  s ∈ [  0,  64) -> Q[head h][dim s]
  s ∈ [ 64, 128) -> K[head h][dim s-64]
  s ∈ [128, 192) -> V[head h][dim s-128]
  ```
  This is corroborated by the diffusers-native re-export
  (`ref/vae/diffusion_pytorch_model.safetensors.index.json`), which stores
  separate `to_q/to_k/to_v` per block — the conversion must perform exactly
  this de-interleave.
- **`to_out`** is `nn.Linear(2048 → 2048)` **with bias**
  (`out_bias = bias = True`, `attention.py:56, 82`).
- **QK-norm IS present**: RMSNorm over `dim_head = 64`, **non-affine**
  (no parameters), eps 1e-5, applied **before** RoPE
  (`attention.py:146-153`). Order matters: **norm, then rotate.**
- **No mask.** `t_causal=False` ⇒ `pack_info == {}`
  (`vae_vit.py:326-334`) ⇒ unmasked SDPA (`flash.py:155, 178`).
- **Softmax scale** is PyTorch SDPA's default `1/sqrt(64) = 0.125`.
- SDPA operates on `[B, H, S, D]` (`flash.py:58-60`); the reference then
  applies `.nan_to_num(0.0)` (`flash.py:72`) — a no-op in the unmasked path
  but harmless to replicate.
- Head count 32, head dim 64, `heads * dim_head == embed_dim` so `to_out` is
  square.

### 2.5 RoPE — 3D axial, 48 of 64 head dims rotated

`RotaryEmbeddingND(dim=48, rotary_base=100.0, n_dim=3, use_angle=True)`
(`vae_vit.py:249`; impl `base_module.py:157-196`).

**Frequency table.** `base_module.py:173-175`:

```python
inv_freq = 1 / rotary_base ** torch.arange(0, 1, 2*n_dim/dim, dtype=torch.float32)
#        = 1 / 100.0 ** arange(0, 1, 6/48 = 0.125)     -> 8 entries
```

| f | exponent | `inv_freq[f] = 100^(-f/8)` |
|---|---|---|
| 0 | 0.000 | 1.0 |
| 1 | 0.125 | 0.56234132519 |
| 2 | 0.250 | 0.31622776601 |
| 3 | 0.375 | 0.17782794100 |
| 4 | 0.500 | 0.1 |
| 5 | 0.625 | 0.05623413251 |
| 6 | 0.750 | 0.03162277660 |
| 7 | 0.875 | 0.01778279410 |

**Coordinates.** `create_token_ids(..., id_type="length_normalized")`
(`func.py:34-37`), independently per axis of size `D`:

```python
coords = arange(0.5, D) / D            # 0.5/D, 1.5/D, ...
coords = 2.0*coords - 1.0              # in (-1, 1), symmetric about 0
```

Meshgrid over `(T, H, W)` with `indexing="ij"`, stacked last, flattened →
`[N, 3]` in the same (t,h,w) row-major order as the token packing.

**Note this is resolution-normalised, not absolute-index, RoPE**: the
coordinates depend on the *clip/tile* extents `T`, `H`, `W`. Because temporal
chunks are always 7 tokens and spatial tiles are 16×16 latent (when tiling is
active), the coordinate grids are identical across chunks/tiles — but for an
untiled decode of a larger frame the grid stretches. **Any port must compute
the ids from the actual `(T,H,W)` of the tensor entering the ViT, not from a
global frame index.**

**Angle assembly.** `base_module.py:183-196`:

```python
angles = (2*pi) * img_ids[:, :, :, None] * inv_freq[None,None,None,:]   # [B,N,3,8]
angles = angles.flatten(2, 3)          # [B,N,24]  order: t*8, then h*8, then w*8
angles = angles.tile(2)                # [B,N,48]  last dim duplicated
angles = angles.unsqueeze(2)           # [B,N,1,48]
cos, sin = cos(angles), sin(angles)    # [B,N,1,48]
```

`use_angle=True` ⇒ `angle_scale = 2π` (`base_module.py:168-169`), so the
normalised coordinate spans one full turn at the base frequency.

**Axis split of the 24 unique angles:**

```
j =  0.. 7  ->  T axis, freq f = j
j =  8..15  ->  H axis, freq f = j-8
j = 16..23  ->  W axis, freq f = j-16
```
and `angles[j+24] == angles[j]` by construction of `tile(2)`.

**Application.** `_apply_rotary_pos_emb_impl` (`func.py:82-103`) on
`t` of shape `[B, N, 32, 64]`:

```python
rot_dim = 48 ;  t_dim = 64  ->  rot_dim < t_dim
t_rot, t_pass = t[..., :48], t[..., 48:]
t_rot = t_rot*cos + rotate_half(t_rot)*sin
t = cat([t_rot, t_pass], -1)
```

with `rotate_half(x) = cat([-x2, x1], -1)` where `x1, x2 = chunk(x, 2, -1)`
(`func.py:77-79`), i.e. **GPT-NeoX half-split convention**: dimension `d` is
paired with `d + 24`, for `d ∈ [0, 24)`.

**Summary: head dims 0..47 are rotated as 24 (d, d+24) pairs — 8 pairs for T,
8 for H, 8 for W. Head dims 48..63 pass through unrotated.**

### 2.6 FFN — gated SiLU (SwiGLU), mult 4

`FeedForward` (`base_module.py:55-105`), `activation_fn="silu"`,
`use_gated=True`, `glu_balanced=False`, `mult=4`, `bias=True`:

```python
inner_dim = round(2048 * 4 * 1) = 8192
w1: Linear(2048 -> 16384)      # 2*inner_dim because gated
w2: Linear(8192 -> 2048)

h = w1(x)                              # [B,S,16384]
gate, val = h.chunk(2, dim=-1)         # gate = FIRST 8192, val = SECOND 8192
h = SiLU(gate) * val
out = w2(h)
```

**Order is `silu(first half) * (second half)` — `gate` is the first chunk**
(`base_module.py:99-100`). Getting this backwards is a silent correctness bug.

`SiLU(x) = x * sigmoid(x)`; PyTorch's exact (non-approximate) form.

`glu_balanced=False` means **no 2/3 shrink** — inner dim is the full `dim*mult`
(`base_module.py:67-68`), matching the checkpoint's `16384 × 2048` `w1`.

### 2.7 Output head

```python
hidden_states = norm_out(hidden_states)            # LayerNorm(2048), eps 1e-5
output = proj_out(hidden_states)                   # Linear(2048 -> 3072), bias
output = output[:, :N, :]                          # drop 4 register + 1 cls
output = _unpack_tensors_3d(output, 16, 4, T*4, H*16, W*16)
```

`_unpack_tensors_3d` (`vae_vit.py:77-93`):

```python
view   (B, T, H, W, 3, 4, 16, 16)                  # (t, h, w, c, pt, ph, pw)
permute(0, 4, 1, 5, 2, 6, 3, 7) -> (B, 3, T, 4, H, 16, W, 16)
reshape(B, 3, T*4, H*16, W*16)
```

**Channel layout inside the 3072-vector** (strides into the flat index):

```
index = c*1024 + pt*256 + ph*16 + pw
c  ∈ [0,3)   RGB, slowest
pt ∈ [0,4)   sub-frame within the 4× temporal expansion
ph ∈ [0,16)  sub-row
pw ∈ [0,16)  sub-column, fastest
```

This is a **depth-to-space / pixel-shuffle** with a channel-major (`c` outermost)
ordering — **not** the PyTorch `nn.PixelShuffle` `(c, ph, pw)` ordering, and
not the more common `(pt, ph, pw, c)` ordering. Get this wrong and the output
is a plausible-looking but scrambled image.

---

## 3. "The CNN upsampler stack" — **it does not exist in this decoder**

**There is no CNN upsampler, no resblock stack, no transposed conv, no
nearest-neighbour upsample, and no GroupNorm anywhere on the decode path.**

Evidence:

1. `AutoencoderKLLegacy.__init__` hard-refuses any other decoder
   (`klvae.py:1133-1136`):
   ```python
   if not use_3d_conv or not use_vit_decoder:
       raise NotImplementedError(
           "this release only supports use_3d_conv=True with use_vit_decoder=True")
   ```
   and `source/config.json` sets `use_vit_decoder: true`.
2. `self.decoder = ViT3DDecoder(**vit_kwargs)` (`klvae.py:1189`) is the only
   decoder assignment; `decode()` calls `self.decoder(z2)` on the
   `use_vit_decoder` branch (`klvae.py:1222-1223`).
3. There is **no decoder CNN class in the bundle at all** — `vae_cnn.py`
   defines only `Downsample3D`, `ResnetBlock3D`, `EncoderFCN3D`. No
   `DecoderFCN3D`, no `Upsample3D`.
4. The checkpoint contains **zero** `decoder.up.*`, `decoder.conv_in.*`,
   `decoder.conv_out.*`, `decoder.mid.*` or `decoder.*.norm*.bias` tensors.
   Every one of the 440 `decoder.*` tensors is a ViT tensor (§4).
5. `space_up = [1,2,2,2,2,1]`, `time_up = null` and
   `num_res_blocks_decoder = null` in `source/config.json` are **legacy
   constructor kwargs that are accepted and then never read** — grep
   `klvae.py` for `space_up`/`time_up`: they appear only in the `__init__`
   signature (`klvae.py:1113-1114`).

**All spatial (16×) and temporal (4×) upsampling happens in a single step:
the `proj_out` linear + the depth-to-space unpack of §2.7.** There is no
staged upsampling, therefore no question of sequencing spatial vs temporal
factors, no GroupNorm placement, no activation and no padding on the decode
path.

### 3.1 Consequences for the port

- No `reflect` padding is needed anywhere in the decoder.
- No causal temporal convolution is needed anywhere in the decoder.
- `norm_eps: 1e-6`, `norm_num_groups: 32`, `spatial_padding_mode: "reflect"`,
  `block_out_channels`, `layers_per_block`, `spatial_downsample_factors`,
  `temporal_downsample_factors` from `ref/vae/config.json` are **encoder-only
  parameters**.
- The only 3D convolution on the decode path is `post_quant_conv`, a
  `nn.Conv3d(24, 24, kernel_size=1)` with **no padding** — mathematically a
  per-voxel 24×24 matmul plus bias.

### 3.2 Causality on the decode path

`causal_decoder: false` in `source/config.json` ⇒ `t_causal=False` ⇒ the ViT
attention is **fully bidirectional in time** within a chunk
(`vae_vit.py:327-334` is skipped). Temporal causality is enforced *only* by
the encoder and by the chunking scheme (each chunk sees 5 own tokens + 2
lookahead tokens).

### 3.3 Encoder conv/norm behaviour (documented for completeness only)

Asked about explicitly; **none of this runs at decode time.**

- **`BaseConv3d` temporal padding is CAUSAL** (`conv.py:40-60`). With
  `causal=True` (encoder default, `klvae.py:1169` passes
  `causal=causal_encoder=True`) and `padding_mode_t=None` ⇒
  `pad_mode_t = "constant"` (zeros) (`conv.py:37`):
  ```python
  pad_size = (0,0, 0,0, self.padding[0]*2 if causal else padding[0],
                        0              if causal else padding[0])
  ```
  For `kernel=3, padding=1` this prepends **2 zero frames** and appends
  **none** — a strictly causal temporal conv. If `D == 1`, it prepends
  `kernel_size[0]-1 = 2` zero frames instead (`conv.py:53-58`).
  Had `causal=False` it would be symmetric `replicate` padding.
- **Spatial padding is `reflect`** and is applied **before** the temporal
  padding, via one `F.pad(x, (W,W,H,H,0,0), mode="reflect")`
  (`conv.py:66-70`). The conv itself then runs with `padding=0`
  (`conv.py:80-87`).
- **Stride-2 downsample** (`vae_cnn.py:65-80`) pads asymmetrically
  **right/bottom only** with `F.pad(x, (0,1,0,1,0,0), mode="reflect")`
  *outside* the conv, and the conv carries `padding=(1,0,0)` (temporal only).
- **GroupNorm**: 32 groups, `eps=1e-6`, affine
  (`norm.py:342-357`), and because `use_t_isolated_gn=true`,
  `TemporalIsolatedSpatialParallelGroupNorm` (`norm.py:245-252`) reshapes
  `[B,C,T,H,W] → [B*T, C, 1, H, W]` first, so **statistics are per-frame,
  never pooled across time**.
- **Placement** in `ResnetBlock3D` (`vae_cnn.py:154-174`) is full
  pre-activation: `x + conv2(silu(norm2(conv1(silu(norm1(x))))))`, with a
  1×1×1 `nin_shortcut` on the residual when channels change.
- **Activation** is `F.silu` everywhere (`vae_cnn.py:27-31`).

---

## 4. Tensor-name → operation map (all 562 tensors)

Totals: **440** `decoder.*` + **116** `encoder.*` + **4**
(`quant_conv`/`post_quant_conv`) + **2** (`latents_mean`/`latents_std`)
= **562**.

Conventions used below:

- **`nn.Linear` weights are stored `[out_features, in_features]`.**
  `y = x @ W.T + b`. For a row-major C++ GEMM this means the checkpoint
  buffer is already `K`-contiguous per output row — i.e. it is the
  `B^T` operand; use `cublasGemmEx` with `op(B) = T` on a row-major
  `[M=tokens, K=in]` activation, or transpose once at load time.
- **`nn.Conv3d` weights are stored `[out_channels, in_channels/groups, kD, kH, kW]`.**
  Both convs here are `k=1`, so the trailing 3 dims are singleton and the
  tensor is just an `[out, in]` matrix.
- All 562 tensors are **F16** in this file.

### 4.1 Decoder — top level (8 tensors)

| tensor | shape | consumed by |
|---|---|---|
| `decoder.x_embedder.weight` | `2048 × 24` | `nn.Linear(24→2048)` at `vae_vit.py:251`, called `vae_vit.py:305`. `[out=2048, in=24]`. Applied to the channel-last packed latent `[B, N, 24]`. |
| `decoder.x_embedder.bias` | `2048` | same linear |
| `decoder.register_tokens` | `1 × 4 × 2048` | learned parameter (`vae_vit.py:153`); `expand(B,-1,-1)` and concatenated **after** the `N` patch tokens (`vae_vit.py:311-313`). Row `r ∈ [0,4)` becomes sequence position `N+r`. Gets **zero** position ids ⇒ identity RoPE. |
| `decoder.norm_out.weight` | `2048` | `nn.LayerNorm(2048, eps=1e-5)` gain (`vae_vit.py:280`, applied `vae_vit.py:352`) |
| `decoder.norm_out.bias` | `2048` | same LayerNorm's shift |
| `decoder.proj_out.weight` | `3072 × 2048` | `nn.Linear(2048→3072)` (`vae_vit.py:282`, called `vae_vit.py:357`). Output index decodes as `c*1024 + pt*256 + ph*16 + pw` (§2.7). |
| `decoder.proj_out.bias` | `3072` | same linear |
| `decoder.mask_token` | `1 × 1 × 2048` | **UNUSED at decode.** Registered as a persistent buffer by `init_mask_config` (`vae_vit.py:125`). `mask_config={}` ⇒ `mask_prob=0` ⇒ `mask_enabled=False` (`vae_vit.py:128-130`), and the buffer→Parameter swap at `vae_vit.py:146-148` never fires. `apply_mask_preprocess`/`postprocess` are identities in eval (`vae_vit.py:159-164`, `202-207`). **Do not load it.** |

### 4.2 Decoder — per transformer block, `i ∈ [0, 36)` (12 × 36 = 432 tensors)

| tensor | shape | consumed by |
|---|---|---|
| `decoder.transformer_blocks.{i}.norm1.weight` | `2048` | `nn.RMSNorm(2048, eps=1e-5)` gain, pre-attention (`base_module.py:229-233`, applied `base_module.py:268`). **No bias key exists ⇒ RMSNorm, not LayerNorm.** |
| `decoder.transformer_blocks.{i}.attn.to_qkv.weight` | `6144 × 2048` | fused QKV `nn.Linear` (`attention.py:80`). Row `r = h*192 + s`: `s∈[0,64)`→Q head `h`, `s∈[64,128)`→K head `h`, `s∈[128,192)`→V head `h`. **Per-head interleaved, NOT [allQ | allK | allV].** |
| `decoder.transformer_blocks.{i}.attn.to_qkv.bias` | `6144` | same, same index rule |
| `decoder.transformer_blocks.{i}.attn.to_out.weight` | `2048 × 2048` | attention output projection (`attention.py:82`, applied `attention.py:161`) |
| `decoder.transformer_blocks.{i}.attn.to_out.bias` | `2048` | same |
| `decoder.transformer_blocks.{i}.scale1` | `2048` | LayerScale on the attention output: `x = x + attn_out * scale1` (`base_module.py:271`). Elementwise over the 2048 channel dim. |
| `decoder.transformer_blocks.{i}.norm2.weight` | `2048` | `nn.RMSNorm(2048, eps=1e-5)` gain, pre-FFN (`base_module.py:247-251`, applied `base_module.py:275`) |
| `decoder.transformer_blocks.{i}.ff.w1.weight` | `16384 × 2048` | gated FFN input (`base_module.py:73`). Rows `[0,8192)` = **gate** (SiLU applied), rows `[8192,16384)` = **value** (`base_module.py:99-100`). |
| `decoder.transformer_blocks.{i}.ff.w1.bias` | `16384` | same, same split |
| `decoder.transformer_blocks.{i}.ff.w2.weight` | `2048 × 8192` | FFN output (`base_module.py:86`, applied `base_module.py:104`) |
| `decoder.transformer_blocks.{i}.ff.w2.bias` | `2048` | same |
| `decoder.transformer_blocks.{i}.scale2` | `2048` | LayerScale on the FFN output: `x = x + ff_out * scale2` (`base_module.py:278`) |

**No `attn.norm_q.weight` / `attn.norm_k.weight` exist** — `qk_norm_affine=false`
makes those RMSNorms parameter-free (`attention.py:69-74`). This is expected,
not a missing tensor.

### 4.3 Latent normalisation (2 tensors — used)

| tensor | shape | consumed by |
|---|---|---|
| `latents_mean` | `24` | added **after** the std multiply, at the decode entry point (§1.3). Broadcast `[1,24,1,1,1]`. Not referenced by any `.py` in `ref/`. |
| `latents_std` | `24` | multiplied first. Same broadcast. |

### 4.4 Quant convs (4 tensors — 2 used, 2 unused)

| tensor | shape | consumed by |
|---|---|---|
| `post_quant_conv.weight` | `24 × 24 × 1 × 1 × 1` | **USED.** `nn.Conv3d(24, 24, 1)` (`klvae.py:1176`), applied first in `decode()` (`klvae.py:1219`). k=1 ⇒ a per-voxel `[24×24]` matmul: `out[o] = Σ_i W[o,i]*in[i] + b[o]`. Plain `nn.Conv3d`, **not** `SpatialParallelConv3d` — no padding, no causal behaviour. |
| `post_quant_conv.bias` | `24` | same |
| `quant_conv.weight` | `48 × 48 × 1 × 1 × 1` | **UNUSED at decode.** Encoder-side moment projection (`klvae.py:1175`, applied `klvae.py:1207`). |
| `quant_conv.bias` | `48` | **UNUSED at decode.** |

### 4.5 Encoder (116 tensors — **all UNUSED at decode**)

Listed by pattern so every tensor is accounted for. Module: `EncoderFCN3D`
(`vae_cnn.py:177-300`). `block_mid = [128,256,256,512,512,1024]`,
`block_in = [128,128,256,256,512,512]`.

| pattern | count | shape(s) | role (encode only) |
|---|---|---|---|
| `encoder.conv_in.{weight,bias}` | 2 | `128×3×3×3×3`, `128` | causal reflect Conv3d 3→128, k=3 p=1 (`vae_cnn.py:221-223`) |
| `encoder.down.{L}.block.{b}.norm1.{weight,bias}` | 24 | `[block_in or block_mid]` | per-frame GroupNorm(32, eps 1e-6) (`vae_cnn.py:104`) |
| `encoder.down.{L}.block.{b}.conv1.{weight,bias}` | 24 | `Cout×Cin×3×3×3` | `vae_cnn.py:124-132` |
| `encoder.down.{L}.block.{b}.norm2.{weight,bias}` | 24 | `[block_mid[L]]` | `vae_cnn.py:105` |
| `encoder.down.{L}.block.{b}.conv2.{weight,bias}` | 24 | `Cout×Cout×3×3×3` | `vae_cnn.py:134-142` |
| `encoder.down.{L}.block.0.nin_shortcut.{weight,bias}` | 6 | `256×128×1³`, `512×256×1³`, `1024×512×1³` | residual channel match, only at `L ∈ {1,3,5}` where `block_in ≠ block_mid` (`vae_cnn.py:144-152`) |
| `encoder.down.{L}.downsample.conv.{weight,bias}` | 8 | `C×C×3×3×3` for `L ∈ {0,1,2,3}` | strided conv, `stride=(time_down[L], space_down[L], space_down[L])` = `(1,2,2), (2,2,2), (2,2,2), (1,2,2)` (`vae_cnn.py:52-61`). **Levels 4 and 5 have no `downsample` key** because `space_down*time_down == 1` and `block_out == block_mid` (`vae_cnn.py:241-256`) — this is why the count is 8, not 12. |
| `encoder.norm_out.{weight,bias}` | 2 | `1024` | `vae_cnn.py:261-263` |
| `encoder.conv_out.{weight,bias}` | 2 | `48×1024×3×3×3`, `48` | 1024 → `2*z_channels = 48` (mean ‖ logvar) (`vae_cnn.py:272-278`) |
| **total** | **116** | | |

Per-level tensor counts, for cross-checking a loader:
L0 = 18, L1 = 20, L2 = 18, L3 = 20, L4 = 16, L5 = 18; plus `conv_in` 2,
`norm_out` 2, `conv_out` 2 ⇒ 116.

### 4.6 Coverage summary

| group | count | decode-time status |
|---|---|---|
| `decoder.*` ViT (blocks + head + embedder + register tokens) | 439 | **used** |
| `decoder.mask_token` | 1 | unused |
| `post_quant_conv.*` | 2 | **used** |
| `latents_mean`, `latents_std` | 2 | **used** |
| `quant_conv.*` | 2 | unused (encoder) |
| `encoder.*` | 116 | unused (encoder) |
| **total** | **562** | **443 used / 119 unused** |

A decoder-only C++ build needs **443 tensors ≈ 4.51 GiB** in F16
(36 blocks × 128 MiB of big weights = 4.5 GiB, plus ~12 MiB for
`proj_out`/`x_embedder`/`norm_out`/`register_tokens`/`post_quant_conv` and
~2.4 MiB of biases, norms and LayerScales). The 116 encoder tensors +
`quant_conv` account for the remaining ~0.33 GiB of the 4.85 GiB file.

---

## 5. Numerical-fidelity notes

### 5.1 Weight/activation dtype — read this before benchmarking

The reference's default load path is
`from_config(...)` (fp32 parameters) followed by
`model.load_state_dict(state_dict)` (`minimax_h3_video_vae.py:107-111`),
which **copies fp16 values into fp32 parameters**. Unless the caller
subsequently calls `.half()`, **the reference executes the whole decoder in
fp32 with fp16-quantised weights.** A fp16 C++ port will therefore *not* match
the reference bit-for-bit even in principle; it can only match a
`.half()`-ed reference. Decide which target you are matching before writing
tolerance tests. → also see §6.

`latents_mean`/`latents_std` are stored F16 here but exist at full fp32
precision in `ref/vae/config.json` and in the safetensors `__metadata__`;
use the fp32 values (see §1.3).

### 5.2 Norms are computed in fp32 — but not all of them

`_vit_norm_input` (`base_module.py:45-48`, duplicated at `attention.py:22-26`)
checks `MINIMAX_H3_VAE_DECODER_VIT_FP32_NORM` which **defaults to `"1"`**:

```python
def _vit_norm_input(module, hidden_states):
    if _env_flag("MINIMAX_H3_VAE_DECODER_VIT_FP32_NORM", "1"):
        return hidden_states.float()
    return hidden_states.to(module.weight.dtype)
```

so by default:

- `norm1`, `norm2`: input upcast to fp32, RMSNorm in fp32, result cast **back
  to the residual dtype** before the sub-layer (`base_module.py:268`, `275`).
- `norm_q`, `norm_k`: same, cast back to `query.dtype` before RoPE
  (`attention.py:147-149`).
- **`norm_out` is NOT wrapped** — `vae_vit.py:352` calls
  `self.norm_out(hidden_states)` directly, so it runs in the activation dtype.
  This asymmetry is easy to miss.

Port guidance: accumulate the RMS sum-of-squares in fp32 (and the LayerNorm
mean/variance in fp32) regardless; that matches the default path and is
strictly better for `norm_out`.

### 5.3 RoPE precision — the sharp edge

`create_token_ids(latent_size, x.device, **x.dtype**)` (`vae_vit.py:320`):
the position coordinates are materialised in the **activation dtype**. In a
fp16 model:

1. `arange(0.5, D, dtype=half)`, `/D`, `2*c - 1` — all in fp16, so the
   normalised coordinates are quantised to ~2⁻¹¹ relative.
2. `self.angle_scale * img_ids[...]` (`base_module.py:185-186`): `angle_scale`
   is a **Python float**, so PyTorch scalar promotion keeps this multiply in
   **fp16** — the `2π·coord` product is rounded to half precision *before*
   meeting `inv_freq`.
3. `... * inv_freq` promotes to fp32 (`inv_freq` is registered
   `dtype=torch.float32`, `base_module.py:173-176`) — but note it is a
   *floating-point buffer*, so `model.half()` **would** convert it to fp16 and
   change step 3 too. Non-persistent (`persistent=False`), so it is not in the
   checkpoint and must be regenerated.
4. `cos`/`sin` are evaluated in fp32 then **cast back to `img_ids.dtype`**
   (`base_module.py:196`) — i.e. fp16 cos/sin tables.
5. `apply_rotary_pos_emb` casts `cos`/`sin` to `t.dtype` (`func.py:90-91`)
   and does the rotation in fp16.

To match a fp16 reference exactly you must reproduce the fp16 rounding at
steps 1, 2 and 4. To match a fp32 reference, compute everything in fp32.
**Computing the coordinates in fp32 and the tables in fp32 is the higher-
quality choice and is what a fp32 reference does; do not blindly copy the
fp16 chain unless you are chasing bit-exactness against a `.half()`-ed run.**

Also: `rotate_half` is the **half-split** convention (`d` pairs with `d+24`),
not the adjacent-pair (`2d`, `2d+1`) convention used by many other RoPE
implementations. And only the **first 48** of 64 head dims are rotated.

### 5.4 Blend weights are computed in the activation dtype

`blend` (`klvae.py:225-227`) builds
`positions = arange(blend_extent, dtype=b.dtype)` and divides by
`blend_extent`. For `blend_extent = 5` in fp16, `1/5, 2/5, 3/5, 4/5` are all
inexact. This applies to **both** the temporal cross-fade
(`frame_overlap = 5`) and the spatial tile blends (overlaps of 64+ px).
For a fp32 port, compute the ramp in fp32 — it is more accurate, but will
differ from a fp16 reference in the last bits of every overlap region.

The ramp is asymmetric (weight_b tops out at `(n-1)/n`, never 1.0) — see
§1.4.1. Reproduce that exactly; it is not a rounding artefact.

### 5.5 Attention

- SDPA with `attn_mask=None, dropout_p=0, is_causal=False`
  (`flash.py:64-71`). Scale `1/sqrt(64) = 0.125`, applied to Q·Kᵀ before the
  softmax.
- PyTorch's fused kernels accumulate the softmax statistics in fp32 even for
  fp16 inputs; a naive fp16-accumulating CUDA kernel **will** diverge
  noticeably over `N+5` tokens (for a 7×16×16 tile that is 1797 tokens; for an
  untiled 7×48×80 latent it is 26 885 tokens).
- `.nan_to_num(0.0)` on the SDPA output (`flash.py:72`).
- Q/K are normalised (RMSNorm, non-affine, eps 1e-5) **before** RoPE. Applying
  RoPE first changes the result because RMSNorm is not rotation-equivariant
  across the norm's full 64-dim window (only 48 dims are rotated).

### 5.6 Operator-order hazards

- **LayerScale folding.** `x + attn_out * scale1` is mathematically equal to
  folding `scale1` into `to_out.weight` *and* `to_out.bias`. It is a valid
  optimisation but changes fp16 rounding; do not fold if chasing bit-exactness.
- **Residual accumulation.** 36 sequential `x = x + sublayer*scale` adds. In
  fp16 this is the dominant error path. Keeping the residual stream in fp32
  (bf16 at minimum) while doing GEMMs in fp16 is the standard mitigation and
  moves *toward* the fp32 reference, not away from it.
- **FFN chunk order**: gate is the **first** 8192 rows of `w1`.
- **QKV interleave**: per-head `[q|k|v]` blocks of 64, not global `[Q|K|V]`
  blocks of 2048.
- **Depth-to-space order**: `c` outermost, then `pt`, `ph`, `pw`.
- **`proj_out` / `x_embedder`** are wrapped in
  `torch.autocast("cuda", enabled=False)` (`vae_vit.py:304`, `356`) and
  `_linear_with_module_dtype` (`vae_vit.py:19-25`) casts the input to the
  weight dtype and the output back. Under an explicit-dtype C++ port this is a
  no-op, but it documents that these two projections are intended to run at
  the weight's precision, not at an autocast-reduced precision.
- **The zero CLS token** at position `N+4` is not optional (§2.1.4).
- **Suffix position ids are zeros**, giving `cos=1, sin=0`. If you skip RoPE
  for the suffix tokens entirely, that is equivalent — but only because
  `rotate_half(x)*0 == 0`; verify your kernel doesn't produce NaNs there.
- **Tiling/chunking nesting order** and the vertical-then-horizontal blend
  order (§1.5) affect the result in overlap corners.

### 5.7 Things that are *not* fp32 in the reference

- The residual stream, all GEMMs, RoPE application, LayerScale, the blends and
  `norm_out` all run in the model dtype.
- `DiagonalGaussianDistribution` upcasts to fp32 (`vae_module.py:7-10`) — but
  that is **encoder-only** and irrelevant here.
- `MINIMAX_H3_VAE_DECODER_TEMPORAL_CAT_DTYPE` (`klvae.py:29-47`) defaults to
  `None` = "keep", so decoded chunks are concatenated in whatever dtype the
  decoder produced. No implicit upcast at stitch time.

---

## 6. Open questions / UNRESOLVED

1. **UNRESOLVED — the exact pipeline call site and formula for
   `latents_mean`/`latents_std`.** Checked: every `.py` under
   `D:\Projects\vidfab\ref` (grep for `latents_mean|latents_std` returns hits
   only in the four `config.json` files); the safetensors `__metadata__`;
   `model_index.json`; `README.md`; `docs/`. The consumer is
   `AutoencoderKLMiniMaxH3` / `MiniMaxH3ModularPipeline` in
   diffusers 0.36.0.dev0, which is not installed and not vendored here.
   `z = z_norm * std + mean` is forced by the definition of the normalisation
   and by `scaling_factor=1.0, shift_factor=0.0`, but the literal source line
   was not read. **Verify against a reference decode before shipping.**

2. **UNRESOLVED — the dtype the reference actually runs in.** The bundled
   loader produces fp32 parameters holding fp16-rounded values
   (§5.1); the checkpoint here is a repackaged all-F16 file with two extra
   tensors (`latents_mean`, `latents_std`) that `load_state_dict(...,
   strict=True)` at `minimax_h3_video_vae.py:111` would **reject** — so this
   file was produced by a different, unseen packaging step. The original
   `source/model.safetensors` referenced by
   `ref/FL2VA/video_vae/config.json` is **not present** (only
   `source/config.json` is). Its dtype is unknown.

3. **UNRESOLVED — whether the shipping pipeline enables decoder tiling.**
   `ref/FL2VA/video_vae/config.json` says `vae_decoder_tiling: 1` with
   `vae_tile_size: 256`, which would decode 768p in 4×3 = 12 tiles per
   temporal chunk with ≥64 px blended overlaps. But the diffusers-native
   `ref/vae/config.json` has **no tiling keys at all**, so the diffusers
   `AutoencoderKLMiniMaxH3` port may use different (or no) tiling. This
   changes the output pixels in overlap regions and changes the RoPE
   coordinate grid (which is normalised per-tile, §2.5). **This is the single
   largest open risk to matching reference output.**

4. **UNRESOLVED — whether `decode_base` is called with an explicit
   `frame_num`.** `trim_output` is tail-aligned (`klvae.py:454-456`) and
   drops leading frames. Without knowing the caller's `frame_num` the exact
   output length for a given latent is under-determined beyond
   `num_chunks*17 + 5 - pad_frames`.

5. **UNRESOLVED — behaviour for `T_lat` values that yield `num_chunks == 0`.**
   E.g. `T_lat ∈ {1, 2}` gives `num_chunks = 0`, `dec is None`, and the
   streaming path raises `RuntimeError` (`klvae.py:666-667`); the list path
   would fail at `torch.cat([])`. Single-image decode presumably goes through
   `decode_base(..., process_image=True)` (`klvae.py:791-795`), which bypasses
   `decode_temporal` entirely — but no image-decode entry point exists in this
   bundle (there is `encode_images`, no `decode_images`), so the intended
   single-frame decode path is unverified.

6. **UNRESOLVED — `_decode_temporal_pad_frames` for non-canonical `T_lat`.**
   The formula (`klvae.py:514-529`) is implemented and reproduced verbatim in
   §1.4.1, but it was only hand-verified for the canonical
   `T_lat = 5K+2, pad_tokens = 0` case. The padded branch has an unusual
   `(z_len_before_pad + k) % tokens_chunk_size == 0 → intra_tail` rule that
   should be differential-tested against the reference rather than trusted.

7. **Not determined — `parallel.py` `get_parallel_state()` default when
   `vae_parallel_tiling` is falsy.** `_ensure_vae_parallel_state()` is only
   invoked when `bool(config["vae_parallel_tiling"])`
   (`minimax_h3_video_vae.py:88-89`); here it is `1`, so the single-process
   state is seeded and every parallel branch is inert. Irrelevant for a
   single-GPU port, noted only so the exclusion in §0 is auditable.

8. **Note, not a question** — `ref/FL2VA/video_vae/config.json` carries
   subtly corrupted `latents_std` digits versus `ref/vae/config.json` and the
   checkpoint metadata (§1.3). Use `ref/vae/config.json`.
