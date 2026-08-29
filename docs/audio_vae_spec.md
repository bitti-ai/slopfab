# MiniMax H3 Audio VAE — Decoder Implementation Specification

Target: a C++/CUDA port of the **decode half** of `DacAudioVAE`
(`ref/FL2VA/audio_vae/dac_audio_vae.py:120`), as shipped in
`weights/vae/minimax_h3_audio_vae_fp32.safetensors` (917 tensors, all F32,
577.12 MiB of payload).

Everything here is grounded in:

- `D:\Projects\vidfab\ref\FL2VA\audio_vae\*.py` — the inference-only reference
  bundle, and **the authority** for every formula below
- `D:\Projects\vidfab\ref\FL2VA\audio_vae\config.yaml` and `metadata.json` —
  the constructor kwargs the checkpoint was built with
- `D:\Projects\vidfab\ref\FL2VA\audio_vae\config.json` — the wrapper config
  (`latents_mean` / `latents_std`)
- `D:\Projects\vidfab\ref\audio_vae\config.json` — a *re-expressed* config for
  the diffusers-native `AutoencoderKLMiniMaxH3Audio` port (same model)
- `D:\Projects\vidfab\ref\diffusers\modular\decoders.py:123-198` and
  `packing.py:315-328` — how `decode` is called from the pipeline
- the checkpoint itself, enumerated with `vidfab.exe inspect --list`

> **Read §2 and §6 first.** Two things in this model produce output that is
> bounded, audible and completely wrong if taken the obvious way: the Snake
> activation's log-space `alpha`/`beta` (§6), and the phase/crop arithmetic of
> the anti-aliased activation wrapper (§7). Neither raises an error.

---

## 0. Resolved configuration

`MiniMaxH3AudioVAE.from_pretrained` (`minimax_h3_audio_vae.py:43-77`) builds
`DacAudioVAE` from `metadata.json` plus two fields of `config.yaml`:

```python
model = DacAudioVAE(
    encoder_rates=[2, 4, 4, 5, 5],      decoder_rates=[5, 5, 2, 2, 2, 2, 2],
    attn_proj=True,                     decoder_type="bigvgan",
    decoder_dim=1024,                   vae_latent_channels=32,
    sample_rate=32000,
)                                       # minimax_h3_audio_vae.py:67-75
```

**`latent_dim` is deliberately not passed**, even though `metadata.json` carries
`"latent_dim": 2048`. It therefore falls to the default at
`dac_audio_vae.py:143-146`:

```python
if latent_dim is None:
    latent_dim = encoder_dim * (2 ** len(encoder_rates))   # 64 * 2**5 = 2048
```

which happens to agree. Do not "simplify" by reading `latent_dim` from the
metadata and skipping the check — they agree here and there is no guarantee
they agree in a future checkpoint; the constructor default is what built these
weights.

`attn_proj_dim` (`dac_audio_vae.py:151-155`): `2048 % 32 == 0`, so
`attn_proj_dim = vae_latent_channels = 32`.

`sample_rate == 32000` selects the second hardcoded BigVGAN table
(`dac_audio_vae.py:175-186`):

```
resblock                = "1"                     -> AMPBlock1
num_mels                = latent_dim  = 2048
upsample_rates          = [5, 5, 2, 2, 2, 2, 2]
upsample_kernel_sizes   = [9, 9, 4, 4, 4, 4, 4]
upsample_initial_channel= decoder_dim = 1024
resblock_kernel_sizes   = [3, 7, 11]
resblock_dilation_sizes = [[1,3,5], [1,3,5], [1,3,5]]
use_tanh_at_final       = False                   -> clamp, not tanh   (§9)
use_bias_at_final       = False                   -> conv_post has no bias
activation              = "snakebeta"
snake_logscale          = True                    -> alpha/beta are logs (§6)
```

Derived constants:

| symbol | value | source |
|---|---|---|
| total upsample | **800** | `5*5*2*2*2*2*2` |
| latents per second | **40** | `32000 / 800` |
| latent channels (`z`) | **32** | `vae_latent_channels` |
| `dec_in_proj` output width | **2048** | `latent_dim` |
| BigVGAN entry width | **1024** | `decoder_dim` |
| BigVGAN exit width | **8** | `1024 >> 7` |
| anti-alias up/down ratio | **2 / 2** | `Activation1d` defaults (`dac_alias_free_act.py:12-13`) |
| anti-alias kernel size | **12 / 12** | `Activation1d` defaults (`dac_alias_free_act.py:14-15`) |

`decoder_kernel_sizes` cross-checks: `ref/audio_vae/config.json:24-32` lists
`[9,9,4,4,4,4,4]`, and `convert_minimax_h3_to_diffusers.py:574` derives them as
`2*rate - (rate % 2)`. Both agree with the reference's hardcoded table. **The
naive `kernel = 2*rate` is wrong for the odd rates**: rate 5 pairs with kernel
9, not 10. The checkpoint settles it — `decoder.ups.0.0.weight` is
`[1024, 512, 9]`.

### Code paths excluded

- **The whole encoder.** `DacAudioVAE` in this bundle has no `encode()` at all
  (`dac_audio_vae.py:120-225` defines only `__init__`, `preprocess`, `decode`).
  `encoder.*`, `pre_block.*`, `mean_proj.*` and `logs_proj.*` — 136 tensors,
  281.2 MiB — are dead weight for this port. See §11.
- **`preprocess`** (`dac_audio_vae.py:201-209`) pads *audio* before encoding;
  nothing in the decode path calls it.
- **Weight normalisation.** See §3.

---

## 1. Entry point and tensor conventions

`decoders.py:183-195`:

```python
audio_latents = unpack_audio_tokens(rows[num_condition_audio_rows:], num_audio_latents)
audio_latents = audio_latents * audio_latents_std + audio_latents_mean   # [1,32,1] broadcast
audio = components.audio_vae.decode(audio_latents, return_dict=False)[0]
block_state.audio = audio.float().permute(1, 0, 2)                       # (1, 2, N)
```

`unpack_audio_tokens` (`packing.py:315-328`) returns
`(2, latent_channels, num_audio_latents)` — "One batch item per stereo channel,
which is what the mono audio VAE consumes."

So `AudioDecoder::decode` receives **`[2, 32, A]` fp32, already de-normalised**,
laid out channel-major with contiguous time (`z[b][c][a]` at
`((b*32)+c)*A + a`), and every internal activation keeps that `[B, C, T]`
layout. The de-normalisation is the caller's job; `latents_mean()` /
`latents_std()` expose the checkpoint's `[32]` fp32 tensors so the caller does
not have to parse the metadata JSON. (They are byte-identical to
`ref/FL2VA/audio_vae/config.json:13-80` — verified.)

`decode` returns `[B, 1, A*800]`; this port interleaves it to
`samples[t*2 + c]`.

---

## 2. The decode graph

`DacAudioVAE.decode` (`dac_audio_vae.py:211-225`) is two lines:

```python
z = self.dec_in_proj(z)      # nn.Conv1d(32, 2048, 1)     dac_audio_vae.py:160
return self.decoder(z)       # BigVGAN                    dac_bigvgan.py:180
```

`BigVGAN.forward` (`dac_bigvgan.py:180-206`):

```python
x = self.conv_pre(x)                                   # 2048 -> 1024, k7 p3
for i in range(7):
    x = self.ups[i][0](x)                              # ConvTranspose1d, §5
    xs = None
    for j in range(3):                                 # resblock_kernel_sizes
        r = self.resblocks[i*3 + j](x)                 # AMPBlock1, §8
        xs = r if xs is None else xs + r
    x = xs / 3
x = self.activation_post(x)                            # Activation1d(SnakeBeta(8))
x = self.conv_post(x)                                  # 8 -> 1, k7 p3, no bias
x = torch.clamp(x, -1.0, 1.0)                          # use_tanh_at_final=False
```

Note `xs = xs / self.num_kernels` at `dac_bigvgan.py:195` — the three resblocks
are **averaged, not summed**. Dropping the `/3` gives output 3x too loud, which
then clips at §9 and sounds distorted rather than erroring.

### Shapes for `A` input latents

| step | out channels | out length | tensors |
|---|---|---|---|
| input `z` | 32 | `A` | — |
| `dec_in_proj` | 2048 | `A` | `dec_in_proj.{weight,bias}` |
| `decoder.conv_pre` | 1024 | `A` | `decoder.conv_pre.{weight,bias}` |
| `ups[0]` (r=5, k=9) | 512 | `5A` | `decoder.ups.0.0.{weight,bias}` |
| resblocks 0,1,2 (k=3,7,11) | 512 | `5A` | 12 conv + 12 snake + 12 filter |
| `ups[1]` (r=5, k=9) | 256 | `25A` | `decoder.ups.1.0.*` |
| resblocks 3,4,5 | 256 | `25A` | ” |
| `ups[2]` (r=2, k=4) | 128 | `50A` | `decoder.ups.2.0.*` |
| resblocks 6,7,8 | 128 | `50A` | ” |
| `ups[3]` (r=2, k=4) | 64 | `100A` | `decoder.ups.3.0.*` |
| resblocks 9,10,11 | 64 | `100A` | ” |
| `ups[4]` (r=2, k=4) | 32 | `200A` | `decoder.ups.4.0.*` |
| resblocks 12,13,14 | 32 | `200A` | ” |
| `ups[5]` (r=2, k=4) | 16 | `400A` | `decoder.ups.5.0.*` |
| resblocks 15,16,17 | 16 | `400A` | ” |
| `ups[6]` (r=2, k=4) | 8 | `800A` | `decoder.ups.6.0.*` |
| resblocks 18,19,20 | 8 | `800A` | ” |
| `activation_post` | 8 | `800A` | `decoder.activation_post.*` |
| `conv_post` | 1 | `800A` | `decoder.conv_post.weight` |

**Every stage preserves or exactly multiplies the length** — see §4, §5, §7.
There is no trimming anywhere, so

```
num_samples = A * 800        exactly
```

For `A = 405` that is 324,000 samples = 10.125 s at 32 kHz.

---

## 3. Weight normalisation — resolved by inspection

Every conv in the reference is built through `weight_norm`
(`dac_audio_vae.py:21-22`, `dac_bigvgan.py:51`, `:68`, `:125`, `:139`, `:170`),
which is `torch.nn.utils.parametrizations.weight_norm`. Such a module's
`state_dict` keys are `…parametrizations.weight.original0` /`original1`, with a
compat pre-hook accepting the legacy `…weight_g` / `…weight_v` spelling.
`convert_minimax_h3_to_diffusers.py:586-588` explicitly claims the checkpoint
uses the `weight_g`/`weight_v` spelling.

**The file we ship does not.** All 917 keys were enumerated: there is no `_g`,
no `_v`, no `parametrizations.` anywhere. Every conv carries a single plain
`weight` of the full `[C_out, C_in, K]` shape.

> **Contradiction with the reference.** As shipped,
> `weights/vae/minimax_h3_audio_vae_fp32.safetensors` could not be loaded by
> `minimax_h3_audio_vae.py:76` (`load_state_dict(..., strict=True)`) — every key
> would be unexpected and every parametrisation key missing. The file is a
> repack with weight norm already folded (`w = g * v / ||v||`).

**Decision: use `weight` verbatim. Do not compute a norm.** Confirmed
empirically: running the reference graph on these tensors yields a waveform with
RMS 0.105 and peak 0.40 over a random latent — un-normalised `v` tensors driven
through 21 residual blocks would not land in that range by accident.

---

## 4. Plain 1-D convolutions

All of them are `stride=1` and "same"-padded; only dilation varies.

`get_padding` (`dac_utils.py:11-12`):

```python
def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)     # == dilation*(k-1)//2
```

For odd `k` this is exactly SAME padding: `L_out = L + 2p - d(k-1) - 1 + 1 = L`.

| conv | k | dilation | padding | source |
|---|---|---|---|---|
| `dec_in_proj` | 1 | 1 | 0 | `dac_audio_vae.py:160` |
| `conv_pre` | 7 | 1 | 3 | `dac_bigvgan.py:125` |
| `resblocks.*.convs1.j` | 3/7/11 | 1,3,5 | `get_padding(k,d)` | `dac_bigvgan.py:49-63` |
| `resblocks.*.convs2.j` | 3/7/11 | **1** | `get_padding(k,1)` | `dac_bigvgan.py:66-81` |
| `conv_post` | 7 | 1 | 3, **no bias** | `dac_bigvgan.py:170` |

Resolved padding per (kernel, dilation):

```
k=3 :  d=1 -> 1    d=3 -> 3    d=5 -> 5
k=7 :  d=1 -> 3    d=3 -> 9    d=5 -> 15
k=11:  d=1 -> 5    d=3 -> 15   d=5 -> 25
```

**`convs2` always uses dilation 1**, whatever `convs1` used. Copying the
dilation across both halves of the pair is the easy mistake; it changes the
receptive field and the padding, and still produces the same output length.

Arithmetic (PyTorch `conv1d` is cross-correlation, **not** a flipped
convolution — the kernel is *not* reversed):

```
y[b][co][n] = bias[co] + sum_ci sum_k  x[b][ci][n + k*dilation - padding] * w[co][ci][k]
```

with out-of-range `x` reading zero (`F.pad` default). Padding for these convs is
**zero padding**, not replicate — replicate padding only appears inside the
anti-alias resampler (§7).

---

## 5. Transposed convolutions (the upsamplers)

`dac_bigvgan.py:134-150`:

```python
ConvTranspose1d(upsample_initial_channel // 2**i,
                upsample_initial_channel // 2**(i+1),
                k, u, padding=(k - u) // 2)
```

`output_padding` is left at its default **0**, `dilation` at **1**. PyTorch:

```
L_out = (L_in - 1)*stride - 2*padding + dilation*(K-1) + 1 + output_padding
```

| i | in | out | k | u | padding `(k-u)//2` | `L_out` |
|---|---|---|---|---|---|---|
| 0 | 1024 | 512 | 9 | 5 | 2 | `(L-1)*5 - 4 + 9 = 5L` |
| 1 | 512 | 256 | 9 | 5 | 2 | `5L` |
| 2 | 256 | 128 | 4 | 2 | 1 | `(L-1)*2 - 2 + 4 = 2L` |
| 3 | 128 | 64 | 4 | 2 | 1 | `2L` |
| 4 | 64 | 32 | 4 | 2 | 1 | `2L` |
| 5 | 32 | 16 | 4 | 2 | 1 | `2L` |
| 6 | 16 | 8 | 4 | 2 | 1 | `2L` |

Every stage is **exactly** `u x`. That is what makes `num_samples = 800A` exact,
and it is the cheapest end-to-end assertion available.

Weight layout is `[C_in, C_out, K]` — **input channels first**, the opposite of
`Conv1d`. The checkpoint confirms it: `decoder.ups.0.0.weight` is
`[1024, 512, 9]` with `decoder.ups.0.0.bias` of `[512]`. Indexing it as
`[C_out, C_in, K]` would be a silent transpose for the square case; here it
would at least fail a shape check.

Scatter form (the definition):

```
y[b][co][i*u + k - p] += x[b][ci][i] * w[ci][co][k]
```

Gather form (what the kernel implements), for output `n`:

```
m = n + p ;  phase = m mod u ;  j0 = m div u
y[b][co][n] = bias[co] + sum_{t>=0, phase+t*u < K} sum_ci x[b][ci][j0 - t] * w[ci][co][phase + t*u]
              (terms with j0 - t outside [0, L) drop out)
```

With `(K,u) = (9,5)` or `(4,2)` only **two** taps per output survive; the
transposed convs are ~3% of the decoder's arithmetic despite the channel counts.

---

## 6. SnakeBeta — the log-space trap

Definition (`dac_activations.py:9-15` and `:47-60`):

```python
def snakebeta(x, alpha, beta):
    return x + (beta + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)

def forward(self, x):
    alpha = self.alpha[None, :, None]
    beta  = self.beta[None, :, None]
    if self.alpha_logscale:            # h.snake_logscale == True
        alpha = torch.exp(alpha)
        beta  = torch.exp(beta)
    return snakebeta(x, alpha, beta)
```

Resolved, all three of the questions in the brief:

1. **It divides by `beta`, not `alpha`.** `alpha` is inside the sine (frequency);
   `beta` is the reciprocal scale (magnitude). The docstring at
   `dac_activations.py:51` says so too: `SnakeBeta := x + 1/b * sin^2 (xa)`.
2. **`alpha` and `beta` are stored in log space.** `snake_logscale=True` at
   `dac_audio_vae.py:186` reaches `SnakeBeta(channels, alpha_logscale=...)` at
   `dac_bigvgan.py:88` and `:164`.
3. **The `1e-9` guard is added to the *exponentiated* beta**, after `exp`, not
   to the stored log. `exp(b) + 1e-9`, never `exp(b + 1e-9)`.

So, per channel `c`:

```
y = x + sin(exp(log_alpha[c]) * x)^2 / (exp(log_beta[c]) + 1e-9)
```

**Checkpoint evidence.** Across the 18,296 decoder `alpha` values: min -1.756,
max 1.491, mean 0.038, and **33.9% are negative**. Betas: min -1.228, max 1.726.
Parameters centred on zero with a third negative are logs; `exp` maps them to
`alpha in [0.173, 4.443]`, `beta in [0.293, 5.620]`. (Contrast the *encoder's*
`Snake1d`, `dac_audio_vae.py:34-40`, which has no `beta` and is **not**
log-scaled: its 31 alphas are all positive, 0.005 to 2.872. Do not share one
code path between the two.)

**What the wrong variants do.** Measured end to end on the real checkpoint with
a random `[2,32,2]` latent:

| variant | output RMS | peak | samples clipped at ±1 |
|---|---|---|---|
| **as specified** (`exp`, divide by beta) | 0.084 | 0.329 | 0.00% |
| divide by `alpha` instead of `beta` | 0.122 | 0.455 | 0.00% |
| **no `exp`** (treat as linear scale) | 0.974 | 1.000 | **91.9%** |
| identity (no activation at all) | 0.136 | 0.458 | 0.00% |

Omitting `exp` is loud and obvious. Dividing by the wrong parameter is not: it
produces a plausible, unclipped, wrong-timbre waveform. There is no runtime
check that catches it — only a reference comparison.

`alpha`/`beta` ship as `[C]`, broadcast over batch and time.

---

## 7. The anti-aliased activation wrapper

`Activation1d` (`dac_alias_free_act.py:8-30`) with all defaults
(`up_ratio=2, down_ratio=2, up_kernel_size=12, down_kernel_size=12`):

```python
x = self.upsample(x)      # UpSample1d(2, 12)
x = self.act(x)           # SnakeBeta, §6
x = self.downsample(x)    # DownSample1d(2, 12)
```

### 7.1 The filters

Both filters are `kaiser_sinc_filter1d(cutoff=0.5/2, half_width=0.6/2,
kernel_size=12)` (`dac_alias_free_resample.py:19` and `:39-43`), i.e. identical
arguments, so **the up filter and the down filter are the same 12 taps**. They
ship as buffers (`…upsample.filter`, `…downsample.lowpass.filter`, both
`[1,1,12]`) so nothing has to be recomputed — but the arithmetic at
`dac_alias_free_filter.py:28-58` was reproduced to confirm the shipped values
are what the reference would build:

```
delta_f = 4*0.3 = 1.2 ;  A = 2.285*(6-1)*pi*1.2 + 7.95 = 51.021235 ;  A > 50
beta    = 0.1102*(A - 8.7) = 4.663800128
window  = kaiser(12, beta, periodic=False)
time    = arange(-6, 6) + 0.5                      # even kernel -> half-sample offset
f       = 2*cutoff * window * sinc(2*cutoff*time)  # sinc(x) = sin(pi x)/(pi x)
f      /= f.sum()
```

Max deviation from the shipped tensor: **2.6e-8**. The taps are

```
+0.00202897 +0.00938946 -0.02554346 -0.05765738 +0.12857261 +0.44320980
+0.44320980 +0.12857261 -0.05765738 -0.02554346 +0.00938946 +0.00202897
```

They are **symmetric** and sum to 1, so the correlate-vs-convolve question does
not bite here; the crop offsets in §7.2 do. All 254 filter tensors in the
checkpoint are bitwise identical to each other. This port still loads each one
separately rather than sharing a single buffer — the saving is 12 KB and the
assumption is not worth carrying.

Two consequences of `sum(f) == 1` worth testing:
- Downsample of a constant is that constant.
- Because `f` is symmetric, its even-index and odd-index taps each sum to 0.5,
  so upsample-by-2 of a constant is that constant (see the `ratio *` factor
  below). **The whole wrapper is therefore an identity on a constant signal
  apart from the Snake activation** — a cheap, sharp end-to-end check.

### 7.2 `UpSample1d(ratio=2, kernel_size=12)`

`dac_alias_free_resample.py:14-30`. Constants first:

```
stride    = ratio                                     = 2
pad       = kernel_size // ratio - 1                  = 5
pad_left  = pad*stride + (kernel_size - stride)//2    = 10 + 5 = 15
pad_right = pad*stride + (kernel_size - stride + 1)//2= 10 + 5 = 15
```

Forward:

```python
x = F.pad(x, (pad, pad), mode="replicate")            # +5 each side  -> L+10
x = ratio * F.conv_transpose1d(x, filter.expand(C,-1,-1), stride=ratio, groups=C)
x = x[..., pad_left : -pad_right]                     # drop 15 each side
```

Length: `(L+10-1)*2 + 12 = 2L+30`, minus 30, = **`2L`**.

**Three separate things must all be right**, and each is silent when wrong:
the padding mode is **replicate** (edge), not zero; the `ratio *` scale factor
(without it the output is half amplitude, which sounds like a quiet-but-fine
decode); and the asymmetric-looking 15/15 crop, which is what puts the
interpolated samples on the correct half-sample phase.

Closed form used by this port (derived from the above; `f` has 12 taps,
`x[.]` clamps its index into `[0, L)` which is exactly what replicate padding
does):

```
for n in [0, 2L):
    m    = n + 15
    par  = m & 1                      # tap parity: only 6 of 12 taps contribute
    J    = m >> 1
    up[n] = 2 * sum_{i=0..5} x[clamp(J - i - 5, 0, L-1)] * f[par + 2*i]
```

Spot check `n=0`: `m=15`, `par=1`, `J=7`, source indices `clamp(2,1,0,-1,-2,-3)`
= `2,1,0,0,0,0` against taps `f[1],f[3],f[5],f[7],f[9],f[11]` — which is exactly
what the padded transposed convolution produces.

### 7.3 `DownSample1d(ratio=2, kernel_size=12)`

`dac_alias_free_resample.py:33-48` wraps `LowPassFilter1d(cutoff=0.25,
half_width=0.3, stride=2, kernel_size=12)`. From
`dac_alias_free_filter.py:79-97`:

```
even      = True
pad_left  = kernel_size//2 - int(even) = 5           # note the -1
pad_right = kernel_size//2             = 6           # asymmetric!
padding_mode = "replicate"                           # class default, not overridden
```

```python
x   = F.pad(x, (5, 6), mode="replicate")             # -> L+11
out = F.conv1d(x, filter.expand(C,-1,-1), stride=2, groups=C)
```

Length: `floor((L+11-12)/2) + 1 = floor((L-1)/2) + 1`; for the `L = 2T` this
wrapper always sees, that is **`T`**. Closed form:

```
for n in [0, T):
    down[n] = sum_{k=0..11} x[clamp(2*n + k - 5, 0, 2T-1)] * f[k]
```

**The 5/6 asymmetry is the phase**, and it is the counterpart of the 15/15 crop
in §7.2. A symmetric 6/6 or 5/5 pad still runs, still returns the right length,
and shifts the signal by half a sample per activation — 127 activations deep,
that is a large group delay plus aliasing.

### 7.4 Net effect

`Activation1d` **preserves length exactly**: `T -> 2T -> 2T -> T`. That is
worth asserting in code, because it is the invariant that keeps every
`get_padding` "same" convolution downstream honest.

---

## 8. `AMPBlock1` (the dilated residual blocks)

`dac_bigvgan.py:24-106`. Per block: 3 dilations x (one dilated conv + one
dilation-1 conv), each preceded by its own anti-aliased SnakeBeta — 6 convs and
6 activations.

```python
acts1, acts2 = self.activations[::2], self.activations[1::2]
for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
    xt = a1(x); xt = c1(xt); xt = a2(xt); xt = c2(xt)
    x  = xt + x
```

The `[::2]` / `[1::2]` slicing means **iteration `j` uses `activations[2j]` and
`activations[2j+1]`**, i.e. the checkpoint's `activations.0…5` pair up as
`(0,1) -> j=0 (dil 1)`, `(2,3) -> j=1 (dil 3)`, `(4,5) -> j=2 (dil 5)`. Reading
them as "first three before convs1, last three before convs2" is wrong and
produces a plausible waveform.

Note the activation comes **before** each convolution (pre-activation residual),
and the residual add is **inside** the dilation loop, not around it — the three
dilations are sequential, not parallel branches.

21 blocks total, `resblocks[i*3 + j]` for upsample stage `i` and kernel
`resblock_kernel_sizes[j]` (`dac_bigvgan.py:153-157`):

| stage `i` | channels | resblock indices | kernels |
|---|---|---|---|
| 0 | 512 | 0, 1, 2 | 3, 7, 11 |
| 1 | 256 | 3, 4, 5 | 3, 7, 11 |
| 2 | 128 | 6, 7, 8 | 3, 7, 11 |
| 3 | 64 | 9, 10, 11 | 3, 7, 11 |
| 4 | 32 | 12, 13, 14 | 3, 7, 11 |
| 5 | 16 | 15, 16, 17 | 3, 7, 11 |
| 6 | 8 | 18, 19, 20 | 3, 7, 11 |

---

## 9. `activation_post`, `conv_post`, and the output bound

`dac_bigvgan.py:160-178` and `:197-204`:

```python
activation_post = SnakeBeta(ch, alpha_logscale=h.snake_logscale)   # ch = 8
self.activation_post = Activation1d(activation=activation_post)    # anti-aliased too
self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3, bias=self.use_bias_at_final))
...
x = self.activation_post(x)
x = self.conv_post(x)
if self.use_tanh_at_final:  x = torch.tanh(x)
else:                       x = torch.clamp(x, min=-1.0, max=1.0)
```

`use_bias_at_final = False` and `use_tanh_at_final = False`
(`dac_audio_vae.py:184-185`), so:

- **`conv_post` has no bias.** The checkpoint agrees: `decoder.conv_post.weight`
  `[1,8,7]` exists, `decoder.conv_post.bias` does not.
- **The output is clamped to `[-1, 1]`, not `tanh`-ed.** `tanh` would compress
  everything; `clamp` is transparent below full scale. On real content the
  clamp is inactive (measured peak 0.40 on random latents), so substituting
  `tanh` would be an *almost* inaudible error — all the more reason to get it
  right from the config rather than from listening.

`activation_post` is the full anti-aliased wrapper, not a bare `SnakeBeta`; it
carries its own `upsample.filter` and `downsample.lowpass.filter`.

---

## 10. Numerical envelope

Measured on the real checkpoint with a deterministic random `[2,32,3]` latent
(the NumPy transcription of §2 in float64):

```
dec_in_proj  [2,2048,3]     min -0.424  max +0.540
conv_pre     [2,1024,3]     min -0.993  max +1.019
ups0/stage0  [2, 512,15]    min -1.430  max +1.382
ups1/stage1  [2, 256,75]    min -1.811  max +1.622
stage2       [2, 128,150]   min -2.121  max +1.896
stage3       [2,  64,300]   min -3.492  max +3.320     <- widest
stage4       [2,  32,600]   min -1.532  max +2.016
stage5       [2,  16,1200]  min -1.076  max +1.111
stage6       [2,   8,2400]  min -0.661  max +0.789
conv_post    [2,   1,2400]  min -0.403  max +0.342
```

Nothing here needs more than fp32. There is no normalisation layer anywhere in
the decoder, so a magnitude bug does not get rescaled away — it compounds.

---

## 11. Tensor coverage — all 917 accounted for

| # | pattern | count | bytes | role |
|---|---|---|---|---|
| 1 | `latents_mean`, `latents_std` | 2 | 256 B | §1, caller's de-normalisation |
| 2 | `dec_in_proj.{weight,bias}` | 2 | 0.26 MiB | §2, `Conv1d(32,2048,1)` |
| 3 | `decoder.conv_pre.{weight,bias}` | 2 | 56.00 MiB | §4, `Conv1d(2048,1024,7)` p3 |
| 4 | `decoder.ups.{0..6}.0.{weight,bias}` | 14 | 23.17 MiB | §5, 7 `ConvTranspose1d` |
| 5 | `decoder.resblocks.{0..20}.convs{1,2}.{0..2}.{weight,bias}` | 252 | 168.06 MiB | §8, 126 convs |
| 6 | `decoder.resblocks.{0..20}.activations.{0..5}.act.{alpha,beta}` | 252 | 0.14 MiB | §6, log-space |
| 7 | `decoder.resblocks.{0..20}.activations.{0..5}.upsample.filter` | 126 | 6 KB | §7.2 |
| 8 | `decoder.resblocks.{0..20}.activations.{0..5}.downsample.lowpass.filter` | 126 | 6 KB | §7.3 |
| 9 | `decoder.activation_post.{act.alpha,act.beta,upsample.filter,downsample.lowpass.filter}` | 4 | 160 B | §9 |
| 10 | `decoder.conv_post.weight` | 1 | 224 B | §9, no bias |
| | **decode path (rows 2-10)** | **779** | **247.64 MiB** | uploaded to the device |
| 11 | `encoder.block.{0,7}.{weight,bias}` | 4 | | encoder in/out convs |
| 12 | `encoder.block.6.alpha` | 1 | | encoder `Snake1d`, linear scale |
| 13 | `encoder.block.{1..5}.block.{0..2}.block.{0..3}.{weight,bias,alpha}` | 90 | | 15 `ResidualUnit`s |
| 14 | `encoder.block.{1..5}.block.{3,4}.{weight,bias,alpha}` | 15 | | per-`EncoderBlock` Snake + stride conv |
| 15 | `pre_block.*` (`AttnProjection`) | 22 | 48.33 MiB | §12 |
| 16 | `mean_proj.{weight,bias}`, `logs_proj.{weight,bias}` | 4 | | posterior heads |
| | **encoder path (rows 11-16)** | **136** | **281.13 MiB** | **not loaded** |
| | **total** | **917** | **577.12 MiB** | |

`2 + 779 + 136 = 917`. Verified against `vidfab.exe inspect --list --limit 1000`
by classifying every name; the classifier has no "other" bucket.

Row-5 arithmetic: 21 blocks x 2 conv lists x 3 dilations x {weight,bias} = 252.
Rows 6-8: 21 x 6 activations x {alpha,beta} = 252, and x {up,down} filter = 252.

---

## 12. Where the attention projection sits — **it is not in the decode path**

`attn_proj: true` (`metadata.json:4`) makes `dac_audio_vae.py:195-196` build

```python
self.pre_block = AttnProjection(latent_dim=2048, self.attn_proj_dim=32, num_heads=8)
```

`AttnProjection` (`dac_attn_proj.py:69-88`) is a causal-attention block that
*narrows* 2048 down to 32. Its checkpoint shapes confirm the direction:
`pre_block.proj.weight [32,2048]`, `pre_block.norm1.weight [2048]`,
`pre_block.norm2.weight [32]`, `pre_block.attn.qkv.weight [6144,2048]` (the
`in_dim > out_dim` branch at `dac_attn_proj.py:33-37`, `head_dim = 2048/8 =
256`), `pre_block.attn.proj.weight [32,32]`, `mlp` hidden 64 = `32 * mlp_ratio`.

It maps the **encoder's** 2048-wide output down to the 32-wide latent that
`mean_proj` / `logs_proj` (`dac_audio_vae.py:157-158`, both `Conv1d(32,32,1)`)
turn into a posterior. `DacAudioVAE.decode` never references it.

**Building the attention block into the decoder is the single most expensive
wrong turn available here** — 8 heads, causal masking, GeGLU MLP, ~48 MiB of
weights, none of it executed.

---

## 13. Implementation notes for this port

### 13.1 Kernels (`include/vidfab/cuda/audio_vae_kernels.cuh`)

Channel counts run 2048 down to 1 while lengths run to 324,000, so nothing here
is GEMM-shaped; all five kernels are direct.

| launcher | shape | notes |
|---|---|---|
| `launch_conv1d` | `[B,Cin,L] -> [B,Cout,Lout]` | stride 1, arbitrary pad/dilation. Shared-memory tiled: 512 time x 8 out-channels per block, 4 input channels per pass. |
| `launch_conv_transpose1d` | `[B,Cin,L] -> [B,Cout,Lout]` | gather form of §5. |
| `launch_aa_upsample_snake` | `[B,C,L] -> [B,C,2L]` | §7.2 fused with §6; one block per (time tile, channel, batch) so `expf` on `log_alpha`/`log_beta` is hoisted out of the element loop. |
| `launch_aa_downsample` | `[B,C,2T] -> [B,C,T]` | §7.3. |
| `launch_snake_beta` | in place | §6 alone; exists so the activation can be tested without the resampler. |
| `launch_add_inplace`, `launch_scale_inplace`, `launch_clamp_inplace` | | residual, `/3`, §9. |

`launch_conv1d` is deliberately **stride-1 only** and rejects anything else:
no convolution in the decode path strides, and supporting it would double the
shared-memory tile arithmetic for dead code. The encoder (which does stride) is
not ported.

### 13.2 Memory

One `cudaMalloc` for all 779 decode-path tensors (259,672,032 B = 247.64 MiB),
staged through a single host vector, with a name -> offset table. Activations
are six equal buffers sized from `max(C*T) * B` — for `A=405` that is
`2 * 256 * 10125 = 5,184,000` floats = 19.8 MiB each, the product `C*T` being
constant at 2,592,000 from stage 1 onward — plus one double-width buffer for the
2x anti-alias intermediate. Eight buffer-widths in total, ~158 MiB, independent
of which stage is running. No streaming or chunking is needed at these sizes and
none is implemented.

### 13.3 Measured

RTX 5090 (sm_120), CUDA 13.0, `A = 405` (10.125 s of stereo at 32 kHz,
324,000 samples per channel):

```
decode wall time      127.3 ms     (both stereo channels, exact math, one call)
weights               247.64 MiB
peak device memory    408.00 MiB   (weights + activation pool, cudaMemGetInfo delta)
```

The exact baseline canonicalizes subnormal/NaN operands and every convolution
FMA boundary, and uses shared fixed-polynomial exp/sin implementations for
SnakeBeta instead of vendor transcendental instructions.

Accuracy against the float64 NumPy transcription of §2 driven by the same
checkpoint, `A = 3`: **worst absolute deviation 4.14e-7** over the sampled
output, three orders of magnitude inside the project's 1e-3 tolerance. At that
margin the two implementations are running the same arithmetic, not merely
agreeing to tolerance.

### 13.4 Exact CUDA/Vulkan primitive substrate

`audio_primitives.h` owns backend-neutral Conv1D/ConvTranspose1D shape
arithmetic and typed checkpoint loading. The loader accepts the shipped folded
`name.weight` layout and legacy `weight_g`/`weight_v`; legacy folding uses a
double-precision ascending square sum, one host `sqrt`, and a pinned float
scale. The shipped decoder uses 779 fp32 tensors (247.64 MiB), and the
primitive loader does not retain a second host copy after upload.

Vulkan records Conv1D, transposed Conv1D, residual add, `/3`, clamp,
interleave, SnakeBeta, and the asymmetric anti-alias up/down operations through
one persistent five-binding pipeline. Each invocation owns one output and
convolution reductions visit channels then taps in ascending order. Buffers
stay device-resident and multiple primitives can share one `TensorBatch`; no
CUDA fallback, host round-trip, per-operator allocation, or per-operator
submission is present. The runtime rejects invalid shape/dtype/alias contracts
before modifying batch access state. Subnormal operands/results become signed
zero and NaNs become `0x7fc00000` at the same CUDA/Vulkan boundaries. Padding
still executes `fma(0, weight, accumulator)`, matching CUDA even for exceptional
weight bits. Exact exp maps `-Inf` to zero and `+Inf` to `+Inf`; exact sine
returns canonical NaN for non-finite input or finite magnitudes at least
`2^31`, before any float-to-integer conversion. SnakeBeta propagates that
canonical NaN and treats an infinite beta reciprocal as signed zero. The
bitwise exceptional matrix covers both signed subnormals, distinct qNaN
payloads, both infinities, and `+/-1e20` in input, log-alpha, and log-beta.

The embedded shader was generated with Microsoft DirectXShaderCompiler
v1.9.2607 (`dxcompiler.dll` 1.9.0.5402), downloaded from the official release
asset `dxc_2026_07_29.zip` (SHA-256
`A1DFB116BA3EEAE6A1582291B53A8E7BF65AD760676BD3194685C8F7367CD241`):

```
dxc -spirv -fspv-target-env=vulkan1.2 -T cs_6_6 -E main -O3 -Gis \
    -Fo tensor_audio.comp.spv tensor_audio.hlsl
```

Pinned source/SPIR-V SHA-256 values are
`C5D8F8D334D64AE5CAD62E8F725BAD1BB818D25718CE3D1B4D723004F738DBEC`
and `51C0834880118790B70ECCBD5FEF7267C8DFD8BB98256EE2F67D3250A1E90AB5`.

The opt-in real test (`VIDFAB_AUDIO_VAE_REAL=1`) binds replay to
`minimax_h3_audio_vae_fp32.safetensors`, 605,254,808 bytes, SHA-256
`8E505D95DD1561D47ABD43D4238FD40D9BB1AE9E147ED0A4CBA778D76AE4DB48`.
It exercises the production `[2048,32,1]` input projection, the
`[1024,512,9]` first transposed convolution, a `[512,512,11]` dilation-5
residual convolution, and shipped activation-post Snake/anti-alias tensors.
CUDA/Vulkan results are byte exact with aggregate FNV64
`2F78225C9577A73B`; the selected raw weights contain zero fp32 subnormals. On
the measured RTX 5090 the first real transpose and k11/d5 convolution
record/submit/waits are 0.56 and 0.93 ms. Live tensors plus bounded
upload/readback staging use 65.5 MiB from a 69.0 MiB pool; repeated replay keeps
the pool and descriptor count stable.

The production-shape replay uses stereo batch 2 at latent length A=405: the
input projection is `32->2048` at length 405, the first transpose is
`1024->512`, length `405->2025`, and the dilation-5 residual convolution is
`512->512` at length 2025. The transpose and residual are chained in one
device-resident Vulkan batch. Their combined exact digest is
`2A3EEBA122112082`. Measured CUDA/Vulkan wall times (including synchronization)
are 0.13/0.38 ms for the input projection and 4.12/19.42 ms for the chained
transpose plus residual. Production live tensors add 25.4 MiB; total Vulkan
pool use is 90.9 MiB of 91.1 MiB and a repeated replay holds descriptor
allocations at two with no pool growth.

### 13.5 Complete Vulkan decoder and generation routing

`vulkan::AudioDecoder` now owns the entire decode path described in this
document. It loads all 779 decode tensors, keeps them resident, and records the
two channels, seven upsample stages, 21 residual blocks, 126 activations, final
convolution, clamp, and channel interleave as one bounded 497-operator Vulkan
transaction. Six flat activation arenas and one double-width anti-alias arena
are reused across every stage; there is no allocation, upload, readback,
submission, or descriptor creation between individual graph operators.

The real-checkpoint graph test uses the checkpoint identity pinned above. At
`A=3`, final float samples are bit exact against the CUDA decoder and the PCM16
WAV files are byte exact. At the production `A=405` shape on an RTX 5090, the
complete CUDA/Vulkan decodes measured 131.4/655.6 ms and produced the same
FNV64 `0B9084D3F1C6355A`. Vulkan holds 247.6 MiB of weights, accounts a 405.9
MiB logical peak, reaches 517.9 MiB allocator-used high-water, and uses 517.8
MiB after decode (524.7 MiB reserved on the clean production path). The
persistent upload/readback staging pair is 112.0 MiB; the streaming host loader
peaks at 56.0 MiB, so it never retains a second 247.6 MiB host weight image. The 476
descriptor allocations and reserved pool size remain unchanged across a
second `A=405` decode and an `A=3` decode after it. `unload()` releases all
decoder-owned weights and arenas (direct-accounted bytes return to zero); the
context retains only bounded runtime metadata plus its allocator pages for a
later load.

Neural backend selection is explicit and separate from the output colour
converter. `RunOptions::inference_backend` and the C API default to CUDA.
Vulkan requires exact attention and either synthetic rows or an explicit F32
captured prompt embedding. Native conditioning, Ref2VA, AB2 and caches remain
fail-closed, with no remapping or CUDA fallback. Selecting exact attention also selects
`ViTTransformerMode::kExact` for the CUDA video
VAE, so parity runs compare the same deterministic graph. An opt-in real
`run_generate` test writes one backend-neutral init-latent safetensors archive
(FNV64 `5529904CB8C9DC9E`) and runs the minimum 32x32, 22-frame vertical slice
through CUDA exact and Vulkan exact. Pinned FNV64 values are
`E2CA5273E36E9ED7` for all 67,584 PixelBuffer values, `5933499108CE9C79`
for all 59,200 interleaved PCM values, `D55DBD1D534B8787` for Y4M, and
`6B066C7CF430117D` for PCM16 WAV. The float buffers match bit for bit and both
containers are byte identical.

The diagnostic replay copies dec-in projection, pre-convolution, all seven
post-stage averages, final activation, final convolution, clamp and interleave
inside the same command buffer and compares every float to CUDA. It adds 13
copies for a 510-operator diagnostic transaction; production stays one
497-operator transaction. A deliberately truncated reload fails after the two
input convolutions without replacing or leaking the old graph. The lifecycle
test is load -> A3 -> failed reload -> A405 -> A3 -> unload -> reload -> A3.

The Vulkan library and decoder contract test also build and run with CUDA
disabled. It loads the real checkpoint, verifies the 497-operator contract,
pins A3 FNV64 `528F17A83D5EF7EE`, repeats without pool/descriptor growth and
unloads, while linking no CUDA target. This guards decoder-library purity. The
top-level runner still belongs to `vidfab_cuda`, so the CUDA-off CLI does not
yet expose synthetic Vulkan generation.

---

## 14. UNRESOLVED

Two items cannot be settled from `ref/`. Neither changes a sample value.

**U1 — Which stereo channel is batch item 0.**
`packing.py:315-328` and `decoders.py:194` establish that batch item `b` becomes
output channel `b`, but nothing in `ref/` states whether channel 0 is left or
right; it is fixed further upstream, in whatever produced the packed rows.
*Decision: batch 0 -> left, batch 1 -> right*, i.e. `samples[t*2 + b]`. This
matches the WAV/`permute(1,0,2)` ordering and is the only interpretation under
which a mono-in-both-channels decode is unaffected. Swapping it is a
one-character change in `AudioDecoder::decode`.

**U2 — The exact position of `pre_block` in the encoder.**
There is no `encode()` in this bundle (§0), so `AttnProjection`'s placement is
inferred from tensor shapes (§12) rather than read from source. *Decision:
irrelevant — the encoder is not ported.* Recorded so that a future encoder port
starts from "this was never verified" rather than from this document.

Additionally, one item is resolved *against* the reference rather than from it:

**C1 — weight normalisation (§3).** `convert_minimax_h3_to_diffusers.py:586-588`
says the checkpoint ships `weight_g`/`weight_v`; the file we have ships folded
`weight` and would fail `load_state_dict(strict=True)`. The file wins; weights
are used verbatim.
