# Qwen3-VL-32B as MiniMax H3's Text Conditioner
## Implementation Specification for a C++/CUDA port

Target: the conditioner behind `prompt_embeds` in `docs/transformer_spec.md` §1.2,
as shipped in `weights/text_encoder/qwen3vl_32b_int8_convrot.safetensors`.

H3 does not use a text encoder in the usual sense. It runs the **decoder** stack
of a 32 B VLM for exactly 50 layers and takes the **raw, unnormalised residual
stream**: no final norm, no LM head, no chat template, no special tokens, no
vision tower. This document specifies that path and nothing else.

Everything here is grounded in:

- `ref/diffusers/modular/encoders.py` — `MiniMaxH3TextEncoderStep.encode_prompt`,
  lines 112-195 (cited as `encoders.py:NNN`)
- `ref/diffusers/modular/packing.py:73-75` (`packing.py:NNN`)
- `ref/FL2VA/text_encoder/config.json` → `text_config` (cited as `config.json`)
- `ref/FL2VA/text_encoder/model.safetensors.index.json` — the **original**
  64-layer key list (cited as `index.json`)
- `ref/comfy_quants/convrot.py` — the offline rotation, read line by line
- `docs/convrot_notes.md` — ConvRot, **solved**; `docs/transformer_spec.md`
  §1.2 and §10.7 — the consumer side and the hand-off
- the checkpoint header and payload, read directly
- **`transformers` v4.57.1**, `models/qwen3_vl/modeling_qwen3_vl.py`,
  `models/qwen3_vl/modular_qwen3_vl.py`, `modeling_rope_utils.py`,
  `utils/generic.py`. **These are not present in `ref/` and `transformers` is
  not installed locally** (`python -c "import transformers"` →
  `ModuleNotFoundError`). They were fetched from the pinned upstream tag
  matching `config.json`'s `"transformers_version": "4.57.0.dev0"` and are cited
  as `hf/<file>`. Every claim sourced that way is marked; see §10.1 for the
  residual risk and the one-command way to close it.

> **Read §2 first.** The headline result is that Qwen3-VL's 3-D interleaved
> mrope **degenerates exactly to ordinary 1-D GPT-NeoX RoPE** for a pure-text
> sequence. That is proved, not assumed, and it removes the entire mrope
> machinery from the port. §9 lists the things that produce plausible-but-wrong
> output if taken the obvious way.

---

## 0. Resolved configuration

From `config.json` → `text_config`. Every value below is literal in the file;
nothing is inferred.

| field | value | note |
|---|---|---|
| `hidden_size` | **5120** | the residual stream |
| `num_attention_heads` | **64** | query heads |
| `num_key_value_heads` | **8** | grouped-query, 8 groups of 8 |
| `head_dim` | **128** | explicit — **not** `5120/64 = 80` |
| `intermediate_size` | **25600** | SwiGLU inner width, `5 × hidden_size` |
| `hidden_act` | `"silu"` | |
| `rms_norm_eps` | **1e-6** | **not** the transformer's 1e-5 |
| `rope_theta` | **5000000** | i.e. `5e6` |
| `rope_scaling.rope_type` | `"default"` | ⇒ `attention_factor = 1.0` |
| `rope_scaling.mrope_interleaved` | **true** | §2.2 |
| `rope_scaling.mrope_section` | **[24, 20, 20]** | T, H, W; sums to 64 = `head_dim/2` |
| `attention_bias` | **false** | **no biases on any projection** |
| `attention_dropout` | 0.0 | inference no-op |
| `vocab_size` | **151936** | |
| `tie_word_embeddings` | false | irrelevant — no LM head is loaded |
| `max_position_embeddings` | 262144 | never binding; `L` is a few thousand |
| `dtype` | `"bfloat16"` | the residual stream's reference dtype |
| `num_hidden_layers` | 64 *(in config)* | **but the checkpoint ships 50** — §1.4 |

Derived constants, resolved to numbers:

```
q width       = 64 * 128 = 8192           q_proj  [8192,  5120]
kv width      =  8 * 128 = 1024           k_proj  [1024,  5120],  v_proj [1024, 5120]
o input       = 8192                      o_proj  [5120,  8192]
kv groups     = 64 / 8   = 8              query head h reads kv head h / 8
attn scale    = head_dim^-0.5 = 1/sqrt(128) = 0.08838834764831845
rotary pairs  = head_dim/2 = 64           j pairs with j + 64
mrope sum     = 24 + 20 + 20 = 64  ✓
convrot group = 256;  5120/256 = 20,  8192/256 = 32,  25600/256 = 100
```

`ROPE_INIT_FUNCTIONS["default"]` is `_compute_default_rope_parameters`
(`hf/modeling_rope_utils.py`), which returns `attention_factor = 1.0`
("Unused in this type of RoPE"), so the `* self.attention_scaling` in the rotary
forward is a multiply by one. Omit it.

### 0.1 The single path this spec covers

Specified: **`t2va` (text only), batch 1, one unpadded prompt, single GPU, no
KV cache, no generation.**

Explicitly **excluded**:

- **The `visual.*` tower.** 351 tensors, 1.19 GB, present in the file and
  **never loaded**. It is reached only by the `fl2va` keyframe path
  (`encoders.py:153-166` — the `if images:` branch) and by `ref2va`
  (`encoders.py:399-424`); this port is `t2va`, so `pixel_values is None`
  (`encoders.py:151, 189`) and the tower is dead weight. §8.4 enumerates it so
  the coverage count closes; §7.3 says how to skip it on load.
- **DeepStack.** Qwen3-VL injects merged visual features into the hidden states
  of the first 3 decoder layers (`vision_config.deepstack_visual_indexes =
  [8, 16, 24]`, `hf/modeling_qwen3_vl.py` `Qwen3VLTextModel.forward`, the
  `if deepstack_visual_embeds is not None` branch). With no images,
  `deepstack_visual_embeds is None` and the branch never runs. There is nothing
  to implement.
- **The LM head and the final norm.** Neither is in the checkpoint. `encoders.py`
  calls `text_encoder.model(...)` — the submodule — precisely so the head never
  runs (`encoders.py:177-181`).
- **KV cache / incremental decode.** `use_cache=False` (`encoders.py:191`). One
  forward over `L` tokens, done.
- **Batching.** `_check_prompt` (`encoders.py:48-53`) rejects anything but a
  single string.
- **CFG / negative prompt.** The checkpoint is guidance-distilled
  (`encoders.py:88-91`). One encode per request.
- **Chat template.** `ref/FL2VA/text_encoder/chat_template.json` exists and is
  **never applied** on this path — see §1.2.

---

## 1. End-to-end dataflow

### 1.1 Symbols

| symbol | meaning |
|---|---|
| `L` | number of prompt tokens; `prompt_embeds.shape[1]` downstream |
| `D` | 5120, hidden size |
| `Hq`, `Hk` | 64 query heads, 8 kv heads |
| `dh` | 128, head dim |
| `I` | 25600, MLP intermediate |
| `N` | 50, decoder layers present in the checkpoint |

`L` is bounded in practice by MiniMax's Context-IR prompt convention (a few
hundred to a few thousand tokens; `docs/transformer_spec.md` §1.2). Nothing in
the port should assume a hard cap, but sizing buffers for `L ≤ 8192` is ample.

### 1.2 Tokenisation — the interface

**Solved elsewhere.** `include/slopfab/text/tokenizer.h` implements the Qwen2
byte-level BPE and has been verified against the HuggingFace reference over 98
cases including fuzz. This spec fixes only how it is *called*.

```
token_ids = Tokenizer::encode(prompt)          // slopfab/text/tokenizer.h:35
```

with the following, all load-bearing:

1. **`add_special_tokens=False`** (`encoders.py:167`). **No BOS (151643), no EOS
   (151645), no `<|im_start|>`, no system prompt, no chat template.** The token
   list is exactly the prompt's own tokens. `Tokenizer::encode` already adds
   nothing, so this is satisfied by not doing anything extra — but do not "fix"
   it later by adding a BOS.
2. **Vocabulary source** — `ref/FL2VA/text_encoder/tokenizer.json` (or the
   byte-identical `ref/FL2VA/tokenizer/tokenizer.json`). Both carry the same
   151 936-entry vocab; `model.embed_tokens.weight` is `[151936, 5120]`, so ids
   index it directly with no offset.
3. **`mm_token_type_ids` is all zeros** for pure text
   (`encoders.py:172-176`: the processor derives it from the vision pad ids, and
   there are none). Its only consumer is `get_rope_index`'s modality grouping
   (§2.1), which with a single all-zero run reduces to `arange(L)`. **The port
   does not need to compute or carry `mm_token_type_ids` at all.**
4. **`attention_mask = ones_like(input_ids)`** (`encoders.py:187`). All ones =
   nothing is padded. This is a *padding* mask, not a request for bidirectional
   attention — see §3, which is the single most dangerous misreading in this
   file.
5. `text_token_tags` (`encoders.py:169`) is `[1] * L` for `t2va` and is consumed
   by the *transformer*, not by the conditioner. It is not this module's output;
   see `docs/transformer_spec.md` §2.1.

Empty prompts are not handled by the reference and should be rejected at the
API boundary rather than producing `L = 0`.

### 1.3 The forward pass in one page

```
x = embed_tokens[token_ids]                          (L, 5120)   bf16

cos, sin = rope(positions = 0..L-1)                  (L, 128) each, fp32   # §2

for i in 0 .. 49:                                    # every layer in the file
    # --- attention half ---
    r = x
    n = RMSNorm(x, W=input_layernorm[i], eps=1e-6)   (L, 5120)
    q = n @ q_proj[i]^T                              (L, 8192)  -> (L, 64, 128)
    k = n @ k_proj[i]^T                              (L, 1024)  -> (L,  8, 128)
    v = n @ v_proj[i]^T                              (L, 1024)  -> (L,  8, 128)
    q = RMSNorm(q, W=q_norm[i], eps=1e-6)            # per head, over the 128 axis
    k = RMSNorm(k, W=k_norm[i], eps=1e-6)            # per head, over the 128 axis
    q = rope_apply(q, cos, sin)                      # AFTER qk-norm
    k = rope_apply(k, cos, sin)
    a = causal_attention(q, k, v)                    # GQA 64/8, scale 1/sqrt(128)
    a = flatten(a)                                   (L, 8192)
    x = r + a @ o_proj[i]^T                          (L, 5120)

    # --- MLP half ---
    r = x
    n = RMSNorm(x, W=post_attention_layernorm[i], eps=1e-6)
    g = n @ gate_proj[i]^T                           (L, 25600)
    u = n @ up_proj[i]^T                             (L, 25600)
    x = r + (SiLU(g) * u) @ down_proj[i]^T           (L, 5120)

# NO final norm. NO lm_head.
prompt_embeds = x                                    (L, 5120)  -> fp32 to host
```

Every `@ W^T` above is a ConvRot int8 GEMM: **the activation must be rotated
online before it enters** (§5). No `+ bias` appears anywhere — `attention_bias:
false` and the MLP is unconditionally bias-free
(`hf/modeling_qwen3_vl.py` `Qwen3VLTextMLP.__init__`, all three `nn.Linear(...,
bias=False)`). The checkpoint agrees: there is no `*.bias` key under
`model.layers.*` (§8.3).

Verbatim reference for the two halves
(`hf/modeling_qwen3_vl.py`, `Qwen3VLTextDecoderLayer.forward`):

```python
residual = hidden_states
hidden_states = self.input_layernorm(hidden_states)
hidden_states, _ = self.self_attn(...)
hidden_states = residual + hidden_states

residual = hidden_states
hidden_states = self.post_attention_layernorm(hidden_states)
hidden_states = self.mlp(hidden_states)
hidden_states = residual + hidden_states
```

**Pre-norm, both halves. The residual is never normalised, never gated, never
scaled.** There is no LayerScale, no `1/sqrt(depth)`, no post-norm.

### 1.4 Which hidden state, exactly — RESOLVED, and it is not obvious

**Answer: the residual stream leaving decoder layer index 49 (0-based) — that
is, the output of the last layer present in the checkpoint — with no
normalisation applied afterwards.**

The chain of reasoning, in order, because an off-by-one here silently conditions
the transformer on the wrong layer:

**(a) What H3 asks for.** `prompt_embeds = outputs.hidden_states[50]`
(`encoders.py:194`), with `MINIMAX_H3_TEXT_ENCODER_LAYER = 50`
(`packing.py:75`). `packing.py:73-74` states the convention outright:

> *"MiniMax-H3 conditions on the **unnormalized** hidden state its Qwen3-VL
> conditioner produces after the 50th of its 64 decoder layers, i.e.
> `hidden_states[50]` (`hidden_states[0]` being the embedding output)."*

**(b) What HF's `hidden_states` tuple actually contains.** In transformers
≥ 4.55 the tuple is not built by an explicit loop; `Qwen3VLTextModel.forward`
carries a `@check_model_inputs` decorator and returns only
`BaseModelOutputWithPast(last_hidden_state=...)`. The collection happens in
`hf/utils/generic.py`, which wraps each decoder layer's `forward`:

```python
if key == "hidden_states" and len(collected_outputs[key]) == 0:
    collected_outputs[key] += (args[0],)          # layer 0's INPUT, once
...
output = orig_forward(*args, **kwargs)
collected_outputs[key] += (output,)               # each layer's OUTPUT
...
elif hasattr(outputs, "last_hidden_state"):
    collected_outputs[key] = collected_outputs[key][:-1]   # DROP the last
    collected_outputs[key] += (outputs.last_hidden_state,) # append the NORMED one
```

So for an `N`-layer stack the tuple has `N + 1` entries:

| index | contents |
|---|---|
| `0` | embedding output (input to layer 0) |
| `i`, `1 ≤ i ≤ N-1` | **raw, unnormalised** output of decoder layer `i-1` (0-based) |
| `N` | `model.norm(output of layer N-1)` — **post-norm** |

**(c) Apply it to the full 64-layer model.** `hidden_states[50]` with `N = 64`
falls in the middle band, so it is the **raw output of 0-based decoder layer
49** = the 50th layer counting from 1. Not the input to layer 50 as a separate
thing — those are the same tensor — and emphatically not post-norm.

**(d) The guard confirms the tuple semantics independently.**
`encoders.py:142-149` raises when `num_hidden_layers <= 50`:

> *"The last hidden state of a stack truncated to exactly 50 layers is
> **post-norm** and is not the conditioning MiniMax-H3 expects."*

That statement is only true if entry `N` is the normed one — which is exactly
what (b) shows. Two independent sources agree.

**(e) Now map it onto our checkpoint.** The shipped file is *already truncated*.
Its safetensors `__metadata__`, read directly, is:

```json
{"minimax_h3_te": "{\"num_hidden_layers\": 50, \"output\": \"unnormalized_hidden_after_layer_50\"}"}
```

and it contains `model.layers.0` … `model.layers.49` and **no `model.norm`, no
`lm_head`**. For comparison, `index.json` (the original) has 1058 tensors
including `model.language_model.layers.0…63`, `model.language_model.norm.weight`
and `lm_head.weight`. The repack kept layers 0-49, dropped layers 50-63, dropped
the norm and dropped the head — precisely the tensors that (c) says are not
needed.

**Therefore:**

> The wanted tensor is the **output of the last layer present** (layer 49),
> taken **before** any normalisation. It is *not* the input to layer 49.

**(f) The trap this creates.** If you loaded this 50-layer file into HF with
`num_hidden_layers = 50` and asked for `hidden_states[50]`, you would get
entry `N` — the *post-norm* value — which is the exact failure
`encoders.py:142-149` exists to prevent. Our port must not reproduce that
indexing. It cannot, in fact, because `model.norm` is absent; but a port that
"helpfully" substitutes an identity-weight RMSNorm to "complete the model" would
reintroduce the bug in a form that still produces well-scaled output. **Do not
add a final norm.**

Sanity check available to the implementer. An earlier draft of this section said
the per-row *RMS* reaches "hundreds"; that was wrong, and measurement on the real
checkpoint corrects it. It is the per-row **L2 norm** that reaches the hundreds.
The RMS over 5120 channels is 2.9–3.8 for ordinary tokens, and 214 for token 0 —
the usual attention-sink massive activation. So an absolute RMS threshold is a
bad check: it would either fire on every ordinary row or miss a crept-in norm
entirely.

Check the **spread across rows** instead. A final RMSNorm flattens every row to
`RMS(w)`, so the ratio between the largest and smallest per-row RMS collapses
toward 1. We measure a 75× spread, which no post-norm output can produce. That
is the discriminating test, and it is what `tests/test_encoder.cu` asserts.

### 1.5 Output

```
PromptEmbedding { num_tokens = L, hidden_size = 5120, data = [L * 5120] fp32 }
```

matching `include/slopfab/text/encoder.h`. fp32 on the **host**: `L` is at most a
few thousand, so at `L = 4096` this is 84 MB — trivially copied, and it lets the
encoder `unload()` all 24.4 GB before the transformer loads (§7).

The downstream consumer casts it to the transformer dtype and feeds
`condition_proj` / `context_embedder` `[5376, 5120]`
(`docs/transformer_spec.md` §1.5, §8.3). Emitting fp32 rather than bf16 costs
nothing and keeps one rounding out of the chain.

---

## 2. Positional encoding — mrope degenerates **exactly** to 1-D RoPE

This is the headline result and it is established by construction, not assumed.

### 2.1 `position_ids` for a text-only sequence — all three axes are the token index

Two independent code paths both produce this, which is why the conclusion is
robust:

**Path A — `get_rope_index`, the multimodal path.**
`Qwen3VLModel.forward` computes `position_ids` from `mm_token_type_ids` when
none is supplied. `get_rope_index` groups the sequence into maximal runs of one
modality with `itertools.groupby` and, for a text run (`modality_type == 0`):

```python
if modality_type == 0:  # text
    text_len = end_idx - start_idx
    llm_pos_ids_list.append(
        torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + current_pos
    )
    current_pos += text_len
```

(`hf/modeling_qwen3_vl.py`, `Qwen3VLModel.get_rope_index`.) The `.expand(3, -1)`
is the whole answer: **one `arange` broadcast across all three axes**. For a pure
text prompt `mm_token_type_ids` is all zeros (`encoders.py:172-176`), so the
whole sequence is a single run starting at `current_pos = 0`, and

```
position_ids[axis, s] = s          for axis ∈ {t, h, w}, s ∈ [0, L)
```

**Path B — the text model's own default.** If `position_ids` is `None` on
arrival at `Qwen3VLTextModel.forward`:

```python
position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
```

— the same `arange`, expanded to 4 rows (the leading row is the *text* position
used for the causal mask; rows 1-3 are t/h/w and go to the rotary embedding:
`text_position_ids = position_ids[0]; position_ids = position_ids[1:]`).

Both paths land on the same thing. `past_seen_tokens = 0` because
`use_cache=False` (`encoders.py:191`).

> **For the port: there are no 3-D positions. There is one integer per token,
> equal to its index.**

### 2.2 What `mrope_interleaved: true` changes — resolved to an explicit table

The rotary forward (`hf/modular_qwen3_vl.py`,
`Qwen3VLTextRotaryEmbedding.forward`):

```python
inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1).to(x.device)
position_ids_expanded = position_ids[:, :, None, :].float()      # (3, bs, 1, positions)
with maybe_autocast(device_type=device_type, enabled=False):     # Force float32
    freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
    freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos() * self.attention_scaling
    sin = emb.sin() * self.attention_scaling
```

with

```python
def apply_interleaved_mrope(self, freqs, mrope_section):
    """Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
    interleaved [THWTHWTHW...TT], preserving frequency continuity."""
    freqs_t = freqs[0]                                   # start from T everywhere
    for dim, offset in enumerate((1, 2), start=1):       # H, W
        length = mrope_section[dim] * 3
        idx = slice(offset, length, 3)
        freqs_t[..., idx] = freqs[dim, ..., idx]
    return freqs_t
```

Working the slices out for `mrope_section = [24, 20, 20]` over `j ∈ [0, 64)`:

- start: every `j` is **T**
- `dim=1` (H), `offset=1`, `length = 20*3 = 60` → `j ∈ {1, 4, 7, …, 58}`, 20 indices → **H**
- `dim=2` (W), `offset=2`, `length = 20*3 = 60` → `j ∈ {2, 5, 8, …, 59}`, 20 indices → **W**

giving the axis map

```
j:      0 1 2 3 4 5 ... 57 58 59 | 60 61 62 63
axis:   T H W T H W ...  T  H  W |  T  T  T  T
counts: T = 20 + 4 = 24,  H = 20,  W = 20      (= mrope_section, exactly)
```

which is literally the docstring's `[THWTHW…TT]`. Contrast **blocked** mrope
(Qwen2-VL), where the same `[24,20,20]` would mean `j ∈ [0,24)` → T,
`[24,44)` → H, `[44,64)` → W. The two differ by a permutation of the 64
frequency slots — and for text-only input **that permutation is applied to three
identical tensors, so it is a no-op**. See §2.3.

The port does not need to know any of this. It is recorded so that (a) the
degeneracy argument can be checked, and (b) if someone later adds `fl2va`
keyframes, the table above is the thing they need.

### 2.3 The degeneracy claim — it holds **bitwise**, not merely mathematically

`freqs` before the interleave is

```
freqs[axis, b, s, j] = position_ids[axis, b, s] * inv_freq[j]
```

For text-only, §2.1 gives `position_ids[0,b,s] == position_ids[1,b,s] ==
position_ids[2,b,s] == s`. The three axis-slices are therefore computed from
*identical* inputs:

- `inv_freq_expanded` is a **broadcast `expand`** of one `inv_freq` buffer over
  the axis dimension — literally the same memory, not three copies, so there is
  no possibility of the three axes differing.
- `position_ids_expanded` is `position_ids.float()`; the integer→fp32 cast of
  the same integer is the same fp32 value.
- The product is a batched matmul over a batch of 3 with element-identical
  operands and identical shapes. A deterministic kernel returns element-identical
  results across the batch.

Hence `freqs[0] == freqs[1] == freqs[2]` **bit for bit**, and
`apply_interleaved_mrope`, which only ever copies elements from `freqs[1]` and
`freqs[2]` into positions of `freqs[0]`, writes values that are already there.

> **Conclusion: for a pure-text sequence the interleaved 3-D mrope is bitwise
> identical to `freqs[0]`, i.e. to standard 1-D RoPE with `theta = 5e6` over
> `head_dim = 128`. Not "numerically close" — identical.**

Two corollaries worth stating, because they are what make this safe to rely on:

1. **The result is independent of `mrope_section` and of `mrope_interleaved`.**
   Any assignment of frequency slots to axes gives the same answer when all axes
   carry the same position. A port that ignores both fields is not making an
   approximation.
2. **The degeneracy is a property of the input, not of the model.** It fails the
   moment a vision block appears, because `get_rope_index` then advances the
   three axes differently over the image rows. If `fl2va` is ever added, §2.2's
   table becomes load-bearing and this section must be revisited. Flag it in the
   code, not just here.

### 2.4 The pairing convention — GPT-NeoX half-split, confirmed

**This is the trap.** Getting it wrong is silent: interleaved-adjacent-pair
RoPE applied to a half-split checkpoint produces a well-scaled tensor that
degrades generation subtly rather than crashing.

Confirmed, not assumed. `hf/modeling_qwen3_vl.py` defines

```python
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
```

and applies it as

```python
q_embed = (q * cos) + (rotate_half(q) * sin)
k_embed = (k * cos) + (rotate_half(k) * sin)
```

This was confirmed against three things, all consistent:

1. The `rotate_half` source above, from the Qwen3-VL model file itself (not a
   shared Llama import that might be overridden).
2. `emb = torch.cat((freqs, freqs), dim=-1)` in the rotary forward (§2.2), which
   duplicates the 64 angles to 128. That duplication is *only* correct for a
   half-split rotation: it makes `cos[j] == cos[j+64]`, which is exactly what
   pairing `j` with `j+64` needs. An interleaved-adjacent convention would
   instead require `repeat_interleave`, and the file does not use it.
3. `docs/transformer_spec.md` §5.3, where the H3 DiT — a different model —
   independently uses `rotate_half` with a `cat(freqs, freqs)` duplication. Same
   family, same convention.

Written out, for a head vector `x[0..127]` and `j ∈ [0, 64)`:

```
out[j]      = x[j]    * cos[j] - x[j+64] * sin[j]
out[j+64]   = x[j+64] * cos[j] + x[j]    * sin[j]

cos[j] = cos(s * inv_freq[j]),   sin[j] = sin(s * inv_freq[j]),   s = token index
```

**All 128 head dims rotate.** There is no pass-through tail here — unlike the H3
transformer, which rotates only 96 of its 128 (`docs/transformer_spec.md` §5.3).
Do not carry that habit across.

### 2.5 Resolved frequencies

`_compute_default_rope_parameters` (`hf/modeling_rope_utils.py`):

```python
base = config.rope_theta                                        # 5000000
partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)   # 1.0
head_dim = getattr(config, "head_dim", None) or hidden_size // num_attention_heads   # 128
dim = int(head_dim * partial_rotary_factor)                     # 128
attention_factor = 1.0
inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(dtype=torch.float) / dim))
```

so, resolved:

```
inv_freq[j] = 5e6 ** (-j / 64),      j ∈ [0, 64)
```

Note `head_dim` comes from the config (128), **not** from `hidden_size /
num_attention_heads` (which would be 80). Computed values:

| j | `inv_freq[j]` | period `2π/f`, in tokens |
|---|---|---|
| 0 | 1.000000000 | 6.3 |
| 1 | 7.858300209e-01 | 8.0 |
| 2 | 6.175287366e-01 | 10.2 |
| 4 | 3.813417554e-01 | 16.5 |
| 8 | 1.454215497e-01 | 43.2 |
| 16 | 2.114742622e-02 | 297 |
| 21 | 6.337244529e-03 | 992 |
| 24 | 3.075291170e-03 | 2 043 |
| 32 | 4.472136206e-04 | 14 050 |
| 48 | 9.457416127e-06 | 664 366 |
| 63 | 2.545079667e-07 | 24 687 578 |

With `theta = 5e6` and `L` in the low thousands, only channels `j ≲ 22` complete
a full turn; the rest barely rotate. That is expected for a 262 k-context model
and is a useful smoke test: if your `cos` table is not ≈ 1 for large `j`, the
exponent sign or the `/64` is wrong.

### 2.6 Implementation

```
for s in 0 .. L-1:
    for j in 0 .. 63:
        angle    = (float) s * inv_freq[j]        // fp32
        cos[s][j] = cosf(angle);  cos[s][j+64] = cos[s][j]
        sin[s][j] = sinf(angle);  sin[s][j+64] = sin[s][j]
```

`(L, 128)` fp32 for each of `cos` and `sin`; at `L = 4096` that is 2 MB each.
Build once per request and share across all 50 layers and both `q` and `k`
(`position_embeddings` is computed once in `Qwen3VLTextModel.forward` and passed
down). Applied identically to `q` (64 heads) and `k` (8 heads), broadcast over
the head axis.

Precision: the reference computes the angles inside an explicit
`maybe_autocast(enabled=False)` block — **forced fp32** — then casts `cos`/`sin`
down to the activation dtype on return (`return cos.to(dtype=x.dtype), ...`).
See §6.3 for whether to follow that last cast.

---

## 3. Attention masking — **CAUSAL**. Unambiguously.

The temptation is to reason "H3 uses this as an encoder, so it must be
bidirectional". **That is wrong, and it changes every output value.**

The evidence, all pointing the same way:

1. **`Qwen3VLTextAttention.__init__` sets `self.is_causal = True`**
   (`hf/modeling_qwen3_vl.py`). Unconditional; there is no config flag that
   turns it off.
2. **`Qwen3VLTextModel.forward` calls `create_causal_mask(...)` explicitly**,
   with `config`, `inputs_embeds`, the incoming `attention_mask`,
   `past_key_values` and `position_ids`. The mask handed to every decoder layer
   is that return value.
3. **`attention_mask=torch.ones_like(input_ids)` (`encoders.py:187`) is a
   padding mask, not a bidirectionality switch.** In HF's convention a 2-D
   `attention_mask` marks which *key* positions are real (1) versus padding (0).
   All-ones means "nothing is padded". `create_causal_mask` then combines it
   with the causal triangle; an all-ones padding mask leaves the triangle
   untouched. There is no code path in which a 2-D all-ones mask disables
   causality.
4. **`use_cache=False`** (`encoders.py:191`) with no `past_key_values` means
   `past_seen_tokens = 0`, so the mask is the plain lower triangle over
   `[0, L) × [0, L)` with no cache offset.
5. **Nothing in `ref/` overrides it.** `encoders.py` calls
   `text_encoder.model(...)` — the stock `Qwen3VLModel` — with no
   `attn_implementation` change, no custom mask, no monkey-patch. H3 is
   *consuming* a causal LM's mid-stack activations; it is not converting the LM
   into an encoder.

> **Implement: strict causal. Token `s` attends to keys `0 … s` inclusive.**

Consequences for the kernel:

- No mask tensor is needed. There is no padding (batch 1, one prompt), so the
  mask is *exactly* the causal triangle and should be expressed as a kernel flag
  (`is_causal = true`), not as a materialised `[L, L]` buffer.
- Roughly half the score matrix is skipped; a causal FlashAttention-style kernel
  should exploit that (skip whole key-tiles above the diagonal).
- **Row 0 attends only to itself.** Its output is `v[0]` for every head, exactly.
  That is a free unit test: for any prompt, `prompt_embeds[0]` must be
  reproducible from the first token alone, and must not change when you append
  more tokens. If appending a word changes row 0, your mask is bidirectional.
  This test costs one extra forward and catches the error immediately.
- Conversely, `prompt_embeds[L-1]` sees the whole prompt. The conditioning is
  therefore *positionally asymmetric*: early rows carry little context. That is
  the intended behaviour, not a bug to be "fixed".

Attention scale is `self.scaling = head_dim ** -0.5 = 1/sqrt(128)` — the plain
default, nothing overrides it. `attention_dropout = 0.0`.

---

## 4. Decoder layer structure

### 4.1 RMSNorm — all four of them

`hf/modeling_qwen3_vl.py`, `Qwen3VLTextRMSNorm.forward`:

```python
input_dtype = hidden_states.dtype
hidden_states = hidden_states.to(torch.float32)
variance = hidden_states.pow(2).mean(-1, keepdim=True)
hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
return self.weight * hidden_states.to(input_dtype)
```

i.e. `y = w * (x * rsqrt(mean(x²) + eps))`. Note the exact order, which is
prescriptive:

- **The upcast to fp32 is explicit and mandatory**, not an implementation detail
  — unlike `docs/transformer_spec.md` §10.5, where the reference used
  `torch.nn.RMSNorm` and the accumulation dtype was a PyTorch internal. Here the
  model file spells it out. Do the mean-of-squares and the `rsqrt` in fp32.
- **`eps` is inside the square root**, added to the *mean of squares*.
- **The downcast happens before the weight multiply**:
  `self.weight * hidden_states.to(input_dtype)`. The normalised value is rounded
  to bf16 *first*, then scaled by `w`. A port that keeps everything in fp32 and
  multiplies by `w` at fp32 is more accurate but not bit-identical. §6.2.
- No mean subtraction, no bias. It is RMSNorm, not LayerNorm.

The four instances, all with `eps = config.rms_norm_eps = ` **1e-6**
(`Qwen3VLTextDecoderLayer.__init__` and `Qwen3VLTextAttention.__init__` both
pass `eps=config.rms_norm_eps`):

| instance | normalised axis | weight shape | checkpoint key |
|---|---|---|---|
| `input_layernorm` | 5120 | `[5120]` | `model.layers.i.input_layernorm.weight` |
| `post_attention_layernorm` | 5120 | `[5120]` | `model.layers.i.post_attention_layernorm.weight` |
| `self_attn.q_norm` | **128** (per head) | `[128]` | `model.layers.i.self_attn.q_norm.weight` |
| `self_attn.k_norm` | **128** (per head) | `[128]` | `model.layers.i.self_attn.k_norm.weight` |

> **`eps = 1e-6`, not 1e-5.** The H3 transformer uses 1e-5 everywhere
> (`docs/transformer_spec.md` §4.1). Sharing an RMSNorm kernel between the two
> modules with a hard-coded eps is a real hazard; pass it as a parameter.

`q_norm`/`k_norm` are **shared across heads**: one `[128]` vector applied to
every one of the 64 (resp. 8) heads, normalising over the 128-wide head axis
only. Do not normalise over 8192 or 1024.

Measured on the checkpoint, worth knowing before you debug a NaN:

```
layer 0  input_layernorm.weight :  min -0.0007  max  0.0272  mean 0.0129
layer 0  q_norm.weight          :  min  0.0014  max  3.6406  mean 1.7850
layer 0  k_norm.weight          :  min  0.0471  max 20.7500  mean 1.9857
layer 49 post_attention_ln      :  min -0.0000  max  1.3281  mean 0.6800
```

`k_norm` reaching **20.75** is real, not corruption. After qk-norm the key
vectors have per-channel scales spanning ~440×; the subsequent `q·k` sums 128
such products. Accumulate attention scores in fp32 (§6.1).

### 4.2 Attention — GQA, and where qk-norm sits

`hf/modeling_qwen3_vl.py`, `Qwen3VLTextAttention.forward`, verbatim on the
ordering question:

```python
query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
key_states   = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

cos, sin = position_embeddings
query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
```

**Order, and it is prescriptive:**

```
1. project        n -> q (L,8192), k (L,1024), v (L,1024)
2. reshape        q -> (L, 64, 128),  k, v -> (L, 8, 128)
3. RMSNorm        q_norm on q, k_norm on k          <-- per head, over 128
4. RoPE           on q and k                        <-- AFTER qk-norm
5. GQA expand     each kv head serves 8 query heads
6. attention      softmax(q·kᵀ / sqrt(128) + causal) · v
7. flatten        (L, 64, 128) -> (L, 8192)
8. o_proj         -> (L, 5120)
```

> **QK-norm is applied BEFORE RoPE.** Reversing steps 3 and 4 is a **silent
> quality bug**: both orders produce finite, well-scaled activations and a
> plausible video. The reason it matters is that RMSNorm rescales each head
> vector by a data-dependent scalar and then by the per-channel weight `w`; RoPE
> mixes channel `j` with channel `j+64`, whose `w` entries differ (by up to
> 440× on `k_norm`, §4.1). `norm ∘ rope ≠ rope ∘ norm`. Get it right, and note
> that the H3 DiT happens to use the same order (`docs/transformer_spec.md`
> §4.3) — so at least the two modules agree.

**`v` is not normalised.** Only `q` and `k`.

**GQA mapping.** `num_key_value_groups = 64 / 8 = 8`. From `repeat_kv`:

```python
hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
```

The `expand` inserts the repeat axis *after* the kv-head axis and the `reshape`
flattens `(kv_head, rep)` in that order, so the mapping is **contiguous
blocks, not interleaved**:

```
query head h  reads  kv head  h / 8         (integer division)
kv head    g  serves query heads  [8g, 8g + 8)
```

i.e. query heads 0-7 → kv head 0, heads 8-15 → kv head 1, …, heads 56-63 → kv
head 7. Getting this backwards (`h % 8`) is another silent, plausible-output
bug; there is no shape check that catches it. A port should not materialise the
expanded `k`/`v` at all — index the 8 kv heads directly from the 64 query heads.

Memory note: `k` and `v` are `(L, 8, 128)` = `L × 1024` each, an eighth of `q`.
At `L = 4096` that is 8 MB each in bf16. Keeping both resident for the whole
layer is free.

### 4.3 SwiGLU MLP — which projection goes through SiLU

`hf/modeling_qwen3_vl.py`, `Qwen3VLTextMLP`:

```python
self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)   # [25600, 5120]
self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=False)   # [25600, 5120]
self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)   # [5120, 25600]
self.act_fn    = ACT2FN[config.hidden_act]                               # silu

def forward(self, x):
    return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

Resolved:

```
y = down_proj( SiLU(gate_proj(n)) * up_proj(n) )

SiLU(z) = z * sigmoid(z) = z / (1 + exp(-z))
```

- **`gate_proj` goes through SiLU. `up_proj` does not.** The names are honest
  here — unlike the H3 transformer, where the fused `fc1` has the gate in the
  first half and a diffusers conversion swaps them (`docs/transformer_spec.md`
  §4.4). Because Qwen3-VL ships `gate_proj` and `up_proj` as **separate
  tensors**, there is no fused-halves ambiguity to get wrong. Confirmed against
  the checkpoint: both keys exist independently, both `[25600, 5120]` (§8.3).
- `gate_proj` and `up_proj` **share the same input** `n = post_attention_layernorm(x)`.
  Rotate it once for ConvRot and feed both GEMMs (§5.3).
- The product is elementwise over 25600 channels, then `down_proj` contracts.
- No bias on any of the three.

Two `(L, 25600)` intermediates are live simultaneously. At `L = 4096` in bf16
that is 210 MB each, 419 MB for the pair — the dominant activation cost. §7.

### 4.4 Residual arrangement — restated

Per §1.3 and the verbatim `Qwen3VLTextDecoderLayer.forward` in §1.3:

```
x = x + Attention(RMSNorm_in(x))
x = x + MLP(RMSNorm_post(x))
```

The residual carries the *unnormalised* stream throughout. There is no gate, no
scale, no dropout, no post-norm. The stream grows in magnitude with depth — this
is the standard pre-norm behaviour and is why §1.4's output has a large RMS.

---

## 5. Quantisation: int8 + ConvRot, and nvfp4 + AWQ

### 5.1 What is quantised, and the dequantisation formula

Seven linears per layer are int8; everything else is BF16 or F32. Per-tensor
metadata, read verbatim from the file: **all 350 quantised tensors carry the
identical payload**

```json
{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}
```

(72 bytes of ASCII in a `U8 [72]` tensor named `<linear>.comfy_quant`;
`7 × 50 = 350`, confirmed by reading every one of them).

**`weight_scale` is per-output-channel, `[out, 1]` F32, despite the format tag
reading "tensorwise".** Observed shapes `[8192,1]`, `[1024,1]`, `[5120,1]`,
`[25600,1]` — one scale per row. This matches `docs/convrot_notes.md`, which
settled it against the bf16 ground truth.

Verified directly on this file: for every quantised tensor sampled (layers 0 and
49, all seven linears), **each int8 row's max absolute value is exactly 127**,
and no value exceeds ±127. So the quantiser is symmetric per-row with
`scale = amax(|W_rot_row|) / 127`, and

```
W_rot[o][i] = (float) int8_weight[o][i] * weight_scale[o]        // exact by construction
```

Measured scale ranges (layer 0 `q_proj`: 6.37e-5 … 1.58e-3; layer 0 `gate_proj`:
3.47e-5 … 1.36e-3; layer 49 `down_proj`: 4.62e-4 … 1.89e-3) — all comfortably
inside fp32 and bf16.

There is **no `input_scale`** anywhere in this file (contrast the transformer's
fp8 checkpoint, `docs/transformer_spec.md` §8.2). Activations are not statically
calibrated. Two consequences: (a) an int8×int8 tensor-core path would have to
dynamically row-quantise the activation itself, and (b) the simplest correct
implementation is to dequantise `W_rot` to bf16 and run a bf16 GEMM — which is
what §7 recommends, since this module runs once per request and is not on the
critical path.

### 5.2 The rotation — solved; reference it, do not re-derive

`docs/convrot_notes.md` is the authority and the answer is **verified against
bf16 ground truth at 0.84 % relative error (pure int8 noise), with two wrong
candidates ruled out at 138 % and 150 %**. Restating only what the kernel needs:

```
h4 = [[ 1,  1,  1, -1],       # regular Hadamard: symmetric, row sums 2
      [ 1,  1, -1,  1],       # NOT Sylvester/Walsh, NOT 2I - J
      [ 1, -1,  1,  1],
      [-1,  1,  1,  1]]

H = kron(h4, h4, h4, h4) / sqrt(256)      # 256x256; symmetric, orthogonal, involutory (H@H = I)
```

`ref/comfy_quants/convrot.py` confirms the construction line by line —
`build_hadamard` builds exactly this `h4`, Kronecker-powers it up to `size`, and
returns `h / (size ** 0.5)`; its docstring says "normalized by `1/sqrt(size)`
(so it is symmetric and orthogonal: `H @ H == I`)". The offline half is

```python
grouped = weight.reshape(out_f, n_groups, group_size)
rotated = torch.matmul(grouped, hadamard.T.to(dtype=weight.dtype))   # rotate_weight()
```

**The online activation rotation is mandatory.** With `W_rot = W Hᵀ` and `H`
orthogonal,

```
x Wᵀ = (x Hᵀ)(W Hᵀ)ᵀ = x_rot W_rotᵀ
```

Skipping it computes `x H Wᵀ`, which is a well-scaled tensor of noise — no
crash, no NaN, plausible-looking output. `ref/comfy_quants/convrot.py`'s
`rotate_activation` is the reference form (`x_rot = grouped @ H`), and because
`H` is symmetric, weight and activation use the *same* matrix.

Implement it as a 4-stage radix-4 butterfly over each 256-wide block, one
`1/16` at the end (`docs/convrot_notes.md`):

```
for stride in (1, 4, 16, 64):
    for each group of 4 elements at offsets {i, i+stride, i+2*stride, i+3*stride}:
        a,b,c,d = x[i], x[i+stride], x[i+2*stride], x[i+3*stride]
        x[i]          =  a + b + c - d
        x[i+stride]   =  a + b - c + d
        x[i+2*stride] =  a - b + c + d
        x[i+3*stride] = -a + b + c + d
x /= 16
```

Every `in_features` here is 5120, 8192 or 25600 — all divisible by 256 — so
**every quantised layer in this checkpoint is rotated**. There is no
skip-when-not-divisible case to handle (unlike the transformer, where
`docs/convrot_notes.md` warns to check per tensor).

Do the butterfly in **fp32**. The `±1/16` entries are exactly representable and
the sums are small, but the input may be a bf16 activation with 8 bits of
mantissa and the transform sums 256 of them.

### 5.3 Where the rotation applies — four rotations per layer, not seven

The rotation acts on the *input* (contraction) dimension, so it is a property of
the activation feeding a GEMM, not of the GEMM. Projections sharing an input
share one rotation:

| rotated activation | width | blocks | feeds |
|---|---|---|---|
| `RMSNorm_in(x)` | 5120 | 20 | `q_proj`, `k_proj`, `v_proj` |
| attention output (flattened heads) | 8192 | 32 | `o_proj` |
| `RMSNorm_post(x)` | 5120 | 20 | `gate_proj`, `up_proj` |
| `SiLU(g) * u` | 25600 | 100 | `down_proj` |

Four rotations per layer, 200 for the stack. Cost per token per layer:
`(20 + 32 + 20 + 100) × 256 × 4 stages × 4 ops ≈ 705 k` flops — under 0.1 % of
the layer's ~975 MFLOP/token and entirely bandwidth-bound. Negligible.

> **`H` does not commute with the RMSNorm weight.** `H · diag(w) ≠ diag(w) · H`.
> The rotation must be applied to the **complete** RMSNorm output, after the
> per-channel `w` multiply — not to the raw `x`, and not folded into `w`. (It
> *does* commute with the per-row scalar `rsqrt(...)`, but that buys nothing.)

Likewise the rotation must **not** be applied to `x` on the residual path: the
residual bypasses the GEMM entirely and stays in the original basis. Only the
four activations in the table above are ever rotated, and each is rotated into a
scratch buffer that the GEMM consumes and then discards.

### 5.4 A free correctness cross-check — use it once

`H` is involutory (`H @ H = I`), so `W_rot @ H = W Hᵀ H = W`. That gives two
mathematically equivalent implementations:

- **(a)** dequantise `W_rot`, rotate the **activation**, GEMM. *(recommended)*
- **(b)** dequantise `W_rot`, rotate the **weight** back to `W` (same butterfly,
  applied along the input axis), GEMM against the **unrotated** activation.

(a) is far cheaper — it rotates `L × in` values instead of `out × in`, and
`L ≈ 10³` against `out ∈ [1024, 25600]`. But (b) is trivial to write, and the
two must agree to ~1e-3 relative (they are not bitwise identical: de-rotating
the dequantised weight redistributes the int8 quantisation noise, though
orthogonality preserves its norm). **Run one layer both ways during bring-up.**
Given that a wrong ConvRot is silent, this is the single highest-value test in
the module. A third check: `butterfly(butterfly(v)) == v` to fp32 round-off, for
random `v`, which catches stride/sign errors in the butterfly itself.


### 5.5 The other build: nvfp4 + AWQ

The same model ships a second time as
`qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` (14.61 GiB, 2054 tensors). It is
a different quantiser's output, not a repacking, and it differs from section 5.1
in every respect that matters. Both builds load; which one a file is comes from
its own `comfy_quant` descriptors, never from a flag.

Per quantised linear:

| tensor | dtype | shape |
|---|---|---|
| `.weight` | U8 | `[out, in/2]` — two E2M1 nibbles per byte |
| `.weight_scale` | F8_E4M3 | `[out, in/16]` — one scale per 16 contracted elements |
| `.weight_scale_2` | F32 | scalar — one global scale per tensor |
| `.pre_quant_scale` | BF16 | `[in]` — **present on `o_proj` and `down_proj` only** |
| `.comfy_quant` | U8 | `[55]` |

34 tensors per layer against the int8 build's 25: 4 norms + 7 x (weight,
weight_scale, weight_scale_2, comfy_quant) + 2 pre_quant_scale. 50 layers gives
1700, plus 3 for the embedding, plus 351 `visual.*` that are present and never
loaded, which is the 2054.

The `comfy_quant` payload is exactly
`{"format": "nvfp4", "full_precision_matrix_mult": true}` on **all 350**
quantised linears. So every layer is checkpoint-declared full precision: it
dequantises and runs bf16, and must never take a native fp4 GEMM. That comes
from the file and is asserted at validation; nothing may infer it from which
scales happen to be present. This build is also **not** ConvRot-rotated —
`convrot` is absent — so nothing in section 5.2 or 5.3 applies to it.

The embedding table does not follow the linears. Here it is
**I8 `[151936, 5120]` with an F32 per-row `weight_scale` `[151936, 1]`** and a
`comfy_quant` of `{"format": "int8_tensorwise"}`; in the int8+ConvRot build the
same tensor is BF16. The row scale multiplies.

#### 5.5.1 Four things that must be established empirically

Every one of these is silent: the wrong answer is finite, correctly shaped and
plausibly scaled, and passes every structural check in section 1.4. All four
were pinned by comparing this build elementwise against the int8 build of the
same model, after de-rotating the latter with the ConvRot Hadamard.
`tools/nvfp4_layout_probe.py` prints the grid; the control that the two files
are the same model at all is `model.embed_tokens`, which agrees at relative
L2 0.0094 / correlation +0.99996.

**1. The HIGH nibble holds the even-indexed element.** Not the low one. Scored
on layer-0 `q_proj` against the de-rotated int8 weight:

| nibble order | relative L2 | correlation |
|---|---|---|
| high = even | **0.0981** | **+0.995178** |
| low = even | 1.4765 | +0.000844 |

Same verdict on `o_proj`, `gate_proj` and `down_proj`. The wrong order is not
merely worse — it is uncorrelated.

**2. `weight_scale` is stored swizzled, not row-major.** Its declared shape is
`[out, in/16]`, but the bytes are in a 128x4 tile layout. For output row `m` and
block `k`, with `K = in_features/16`:

```
tile   = (m / 128) * (K / 4) + (k / 4)
offset = tile * 512 + (m % 32) * 16 + ((m % 128) / 32) * 4 + (k % 4)
```

The tile is 128 rows x 4 blocks = 512 bytes, laid out row-of-tiles major. Inside
it the slowest index is `m % 32` at stride 16, then which quarter of the 128
rows at stride 4, then `k % 4` at stride 1 — so four consecutive blocks of one
row are four consecutive bytes, which is the only part visible by eye. In-tile
maximum is `31*16 + 3*4 + 3 = 511`, a bijection onto the tile.

Read row-major instead, with everything else correct, layer-0 `o_proj` scores
relative L2 **0.7707** at correlation **+0.793**. That is well-scaled noise, and
no shape check, dtype check or finiteness check would ever catch it.

Requires `out_features % 128 == 0` and `(in_features/16) % 4 == 0`. Both hold
for all seven linears with nothing left over — q 8192x320, k/v 1024x320,
o 5120x512, gate/up 25600x320, down 5120x1600 — and the stored byte count equals
`out * in/16` exactly, which is independent evidence that there is no padding.
A padded layout is plausible but no shipped file exercises one, so the loader
**throws** rather than guessing which convention it would follow.

**3. The activation is multiplied by `pre_quant_scale`.** The AWQ convention is
`y = (x*s) @ (W/s)^T`: the stored weight has already been divided, so it is the
activation that is scaled. Layer-0 `o_proj`, correct nibble order and swizzle:

| fold | relative L2 | correlation |
|---|---|---|
| `W * s` (multiply the activation) | **0.0999** | **+0.995014** |
| `W / s` | 0.7798 | +0.639554 |
| no fold | 0.5023 | +0.946950 |

`down_proj` gives 0.1073 / 1.0401 / 0.3608 for the same three.

**4. The other five linears had it folded into the preceding norm.** `q_proj`,
`k_proj` and `v_proj` into `input_layernorm`; `gate_proj` and `up_proj` into
`post_attention_layernorm`. Check per tensor — never infer it from the layer's
name, and never synthesise a value for it: a null pointer means "already
accounted for", not "unknown".

The evidence, and it is a clean control. Between the two builds at layer 0:

| tensor | identical? | median relative difference |
|---|---|---|
| `input_layernorm.weight` | no | 0.5092 |
| `post_attention_layernorm.weight` | no | 0.8250 |
| `self_attn.q_norm.weight` | **yes, bitwise** | 0.0000 |
| `self_attn.k_norm.weight` | **yes, bitwise** | 0.0000 |

`q_norm` and `k_norm` sit *after* the projections and have nothing to absorb, so
their being bitwise identical while the other two differ is exactly the
signature of the fold. And the ratio `nvfp4_norm / int8_norm` *is* the folded
scale: using it as the fold recovers `q_proj` at relative L2 0.0981 and
`gate_proj` at 0.1054, the same floor as the two linears that store their scale
explicitly.

#### 5.5.2 The cross-check to run once

Sections 5.4's advice applies here in a different form. Run the same prompt
through both builds and compare `hidden_states[50]`.

They are two *different quantisations* of one model, so the bar is a few
percent, **not** the project's 1e-3 / 1e-2 per-tensor tolerance — that tolerance
is for two paths computing the same thing, and confusing the two bars will make
you chase a gap that is not there. Measured over a 190-token prompt: **0.0187**
of the int8 output's norm overall, worst row 0.171.

That bar has real power despite being loose. A wrong AWQ fold direction, a
row-major scale read or a swapped nibble order all miss by order one, while
leaving output that passes every structural check in section 1.4 — including the
token-0 massive activation, the row-RMS spread, and finiteness. Bracket it on
both sides: agreement much better than a percent would mean something is
comparing an output with itself.

---

## 6. Numerical precision

The reference residual stream is **bf16** (`config.json` → `text_config.dtype`).
The following must be higher precision regardless.

### 6.1 Must be fp32

- **RMSNorm mean-of-squares and `rsqrt`.** Not optional and not an
  implementation detail — `Qwen3VLTextRMSNorm.forward` casts to fp32 explicitly
  (§4.1). Summing 5120 squared bf16 values in bf16 loses the norm outright.
- **RoPE angles, `cos`, `sin`.** The reference forces fp32 with an explicit
  `maybe_autocast(..., enabled=False)` block (§2.2). Also compute
  `s * inv_freq[j]` in fp32 from an fp32 `(float)s` — at `L = 4096`, `s` is
  exactly representable and the product is the reference's own error floor.
- **Attention softmax.** Max-subtracted, accumulated in fp32, `expf` in fp32.
  With `k_norm` weights up to 20.75 (§4.1) the pre-softmax logits have a wide
  dynamic range; bf16 accumulation over `L` keys will lose small terms.
- **The `q·kᵀ` and `probs·v` accumulators.** fp32 accumulate over bf16 inputs —
  i.e. standard tensor-core `bf16 × bf16 → fp32`.
- **The ConvRot butterfly** (§5.2).
- **The residual add**, ideally. The stream grows over 50 layers and the
  branch output is often much smaller than the accumulated residual; adding in
  fp32 and rounding once costs one buffer and removes 50 roundings. Optional —
  the reference adds in bf16 — but if you keep an fp32 master residual, say so,
  because it will show up as a small systematic difference against a PyTorch
  run.
- **The output.** `[L, 5120]` fp32 to the host (§1.5).

### 6.2 May be bf16

- The residual stream and all activations between ops.
- GEMM inputs. Dequantised weights are bf16 (`int8 * scale` computed in fp32,
  rounded to bf16 — the scale is fp32, so do the multiply in fp32 and round once).
- `q`, `k`, `v` after projection; the attention output.
- The SwiGLU product. `SiLU` itself should evaluate in fp32 (`expf`) and round
  the result, which costs nothing on a memory-bound elementwise kernel.

### 6.3 Two places where matching the reference bit-for-bit and being *more
accurate* diverge

1. **RMSNorm's downcast-before-weight.** The reference computes
   `self.weight * hidden_states.to(input_dtype)` — normalised value rounded to
   bf16 *first*, then multiplied by `w`. Keeping fp32 through the `w` multiply
   is strictly more accurate and will not match a PyTorch run in the last bits.
2. **`cos`/`sin` downcast.** The rotary forward returns
   `cos.to(dtype=x.dtype)`, i.e. bf16 if you run in bf16, and the rotation is
   then done at bf16. Keeping the tables in fp32 through the rotation is more
   accurate and again does not match. `docs/transformer_spec.md` §9.2 makes the
   same observation about the DiT.

Pick one policy and record it. The recommendation is **fp32 tables and fp32
weight-multiply** — more accurate, negligible cost — with the understanding that
a bitwise comparison against PyTorch is then off the table and validation must be
by relative error (expect ≲ 1e-2 relative on the layer-49 output, dominated by
int8 weight quantisation, which is far larger than any of this).

### 6.4 `inv_freq` construction — a last-ulp note

The reference computes `base ** (arange/dim)` in **fp32** (`base` is a Python
float, promoted to the fp32 tensor's dtype). Computing in fp64 and rounding to
fp32 differs by up to **1.19e-7 relative** (measured over all 64 entries). At
`L = 4096` and `inv_freq[0] = 1`, that is an angle error below 5e-4 rad on the
worst channel — well under the bf16 activation noise. Either is fine; fp64
is marginally more accurate. Not worth a switch.

---

## 7. Memory strategy

Byte budget, measured from the header (payload sums exactly to
`27 141 342 152 − 181 112` header bytes ✓):

```
model.layers.*             24 394 802 800 B  = 24.39 GB     <- load
model.embed_tokens.weight   1 555 824 640 B  =  1.56 GB     <- host only
visual.*                    1 190 533 600 B  =  1.19 GB     <- stream only for images
                           --------------
                           27 141 161 040 B  = 27.14 GB
```

Per layer (× 50 = 24 394 802 800 ✓):

```
int8 weights   487 587 840 B     (q 41.94 M, k 5.24 M, v 5.24 M, o 41.94 M,
                                  gate 131.07 M, up 131.07 M, down 131.07 M)
weight_scale       286 720 B     (71 680 F32)
norms               20 992 B     (5120 + 5120 + 128 + 128, BF16)
comfy_quant            504 B     (7 x 72)
               -------------
               487 896 056 B  =  0.488 GB
```

**Dequantising up front does not fit.** `24 379 392 000` int8 weight bytes → bf16
is **48 758 784 000 B = 48.76 GB**, against a 32 GB card. Not an option, at any
`L`.

### 7.1 Recommended: int8 resident, dequantise into a reusable scratch buffer

Keep all 50 layers' int8 weights, scales and norms on the device — 24.39 GB —
and immediately before each GEMM dequantise that one weight into a **single
reusable scratch buffer**, run the GEMM, and reuse the buffer for the next.

```
largest weight        25600 x 5120 = 131 072 000 elements
scratch (bf16)        262 144 000 B  = 262 MB        <- one buffer, reused 7x per layer
```

One buffer suffices: the seven GEMMs in a layer are strictly sequential
(`q,k,v` share an input but not a weight; `gate` and `up` likewise). It is the
*activations* that must be double-buffered, not the weights.

Peak device footprint at `L = 4096`, bf16 activations:

```
int8 weights + scales + norms                        24.39 GB
weight scratch (bf16)                                 0.26 GB
residual x                                    L*5120  0.04 GB
two (L, 25600) MLP intermediates              2 x     0.42 GB
q (L,8192) + k,v (L,1024 each)                        0.08 GB
attention workspace (flash-style, tiled)          ~   0.10 GB
rope cos/sin (L,128) fp32 x2                          0.004 GB
rotation scratch (largest L x 25600 fp32)             0.42 GB
                                                     --------
peak                                             ~   25.7 GB
```

leaving ~6 GB of headroom on a 32 GB card. At a more typical `L = 1024` the
activation total drops to ~0.23 GB and the peak is ~24.9 GB.

Notes:

- **Keep `model.embed_tokens.weight` on the host.** Only `L` of its 151 936 rows
  are ever read. Gather them host-side into an `[L, 5120]` buffer and upload
  that: 40 MB instead of 1.56 GB. `include/slopfab/text/encoder.h` already
  specifies this.
- **Fuse dequantisation with the rotation of the activation, not with the
  weight.** The activation rotation is `L × in`; the weight dequant is
  `out × in`. Do not attempt to fuse dequant into the GEMM's inner loop unless
  you are writing a custom int8 kernel — a plain `int8 → bf16` scale-and-widen
  kernel followed by cuBLAS bf16 GEMM is fast enough here.
- The whole stack reads 24.4 GB of weights once. At 800 GB/s that is 30 ms of
  pure bandwidth; the arithmetic is ~48.8 GFLOP/token × `L`, so at `L = 1024`
  about 50 TFLOP plus ~1.7 TFLOP of attention. Sub-second either way.
- **`unload()` before the transformer loads.** 24.39 GB here plus 19.3 GB there
  does not fit; `include/slopfab/text/encoder.h` already documents that the
  pipeline enforces the ordering, and §1.5's host-side fp32 output is what makes
  it possible.

### 7.2 Fallback: layer streaming

If 25.7 GB is too close to the edge — other processes, a smaller card, a display
attached — stream the layers instead. Text encoding runs **once per request**,
not once per denoising step, so even a slow stream costs seconds and it removes
all VRAM risk.

Memory-map the safetensors file and, for each layer `i`, upload its 487.9 MB of
weights into one of two ping-pong device buffers while layer `i-1` computes:

```
device residency:  2 x 487.9 MB layer buffers      = 0.98 GB
                 + 262 MB dequant scratch          = 0.26 GB
                 + activations (as above, L=4096)  = 1.06 GB
                                                    --------
peak                                             ~   2.3 GB
```

— an **11× reduction**, from 25.7 GB to 2.3 GB.

Transfer cost for the full 24.39 GB:

| path | effective bandwidth | time |
|---|---|---|
| PCIe 4.0 x16, pinned staging, page cache warm | ~20 GB/s | **~1.2 s** |
| PCIe 3.0 x16, pinned | ~11 GB/s | ~2.2 s |
| cold from NVMe (~3.5 GB/s), no page cache | ~3.5 GB/s | ~7 s |
| cold from SATA SSD (~500 MB/s) | ~0.5 GB/s | ~49 s |

With compute at well under a second, the stream is entirely transfer-bound and
overlapping it with compute buys little — but do it anyway, it is two streams
and one event. The warm-page-cache case is the realistic one on a machine that
has just written or read the file, and 1-2 s per request is acceptable for a
video generation that takes minutes.

**Do not stream from an unpinned `mmap` directly.** `cudaMemcpyAsync` from
pageable memory serialises and falls back to a staging copy. Allocate two pinned
host buffers of 488 MB, `memcpy` from the mapping into one while the other
uploads.

Choose between the two at runtime from the complete peak, not weights alone.
The native loader adds the exact `max_prompt_tokens` layer workspace and
persistent activation/RoPE buffers to the packed weights, then preserves the
larger of 2 GiB or 20% of total VRAM for WDDM/driver commitment. This matters
because WDDM may accept a large `cudaMalloc` reservation and report OOM only
when an asynchronous upload first commits its pages. Auto mode streams when
that bound is not free; explicit resident mode fails synchronously before
mapping registration, allocation, or upload. Both modes produce identical
results; only the schedule differs. Any later load failure is transactionally
drained and releases partial buffers, mapping registration, events, streams,
and cuBLAS before the error is rethrown.

### 7.3 Skipping `visual.*` on load

Filter by key prefix at header-parse time, before any allocation:

```
load    keys matching  ^model\.layers\.(\d+)\.
gather  key            ^model\.embed_tokens\.weight$        (host only)
SKIP    keys matching  ^visual\.
```

Skipping is 351 tensors and 1.19 GB. Do not "load it just in case" — it is dead
weight on the `t2va` path and there is no code that would ever read it (§0.1).
Assert that exactly 1250 layer tensors and 1 embedding tensor were consumed; if
the count differs, the checkpoint is not the one this spec describes.

---

## 8. Tensor-name → operation map (all 1602 tensors)

Header facts, read directly from
`weights/text_encoder/qwen3vl_32b_int8_convrot.safetensors`: **1602 tensors**,
header 181 104 B, payload 27 141 161 040 B, file 27 141 342 152 B.
`__metadata__` is present and reads

```json
{"minimax_h3_te": "{\"num_hidden_layers\": 50, \"output\": \"unnormalized_hidden_after_layer_50\"}"}
```

### 8.1 Layout and naming conventions

**No transposes.** Every `*.weight` of a linear is stored PyTorch-style
`[out_features, in_features]`, so the op is `y = x Wᵀ`. A cuBLAS port wanting
column-major `W` can consume the rows as columns with no copy.

**The repack flattened Qwen3-VL's module nesting.** Comparing against
`index.json` (the original 64-layer release):

| original key | this checkpoint |
|---|---|
| `model.language_model.embed_tokens.weight` | `model.embed_tokens.weight` |
| `model.language_model.layers.i.*` (i < 50) | `model.layers.i.*` |
| `model.language_model.layers.i.*` (50 ≤ i < 64) | **dropped** |
| `model.language_model.norm.weight` | **dropped** |
| `lm_head.weight` | **dropped** |
| `model.visual.*` | `visual.*` |

Original: `1 + 1 + 64×11 + 1 + 351 = 1058` tensors, `total_size` 66 714 780 128
(bf16). Ours: `1 + 50×25 + 351 = 1602`, 27 141 161 040 (int8 + bf16). The three
dropped groups are exactly the ones §1.4 shows are unnecessary.

**No biases.** `attention_bias: false` and the MLP is bias-free; there is no
`*.bias` under `model.layers.*` in either the original or this file. (The 351
`visual.*` tensors *do* have biases — that tower is a ViT with LayerNorm — which
is one more reason not to route them through the same loader.)

### 8.2 Top-level (1 tensor)

| key | dtype | shape | operation |
|---|---|---|---|
| `model.embed_tokens.weight` | BF16 | `[151936, 5120]` | token embedding lookup. **Host-resident**; gather `L` rows (§7.1). No scaling, no positional add — the embedding output *is* `hidden_states[0]`. |

Measured over the first 4096 rows: values in `[−0.785, 0.961]`, per-row RMS mean
0.0204, max 0.0291. Small — the stream is scaled up by the norms downstream.

### 8.3 Per decoder layer, `i ∈ [0, 50)` — 25 keys × 50 = **1250**

| key | dtype | shape | operation |
|---|---|---|---|
| `model.layers.i.input_layernorm.weight` | BF16 | `[5120]` | pre-attention RMSNorm, **eps 1e-6** |
| `model.layers.i.self_attn.q_proj.weight` | I8 | `[8192, 5120]` | `q = n Wᵀ`, 64 heads × 128. ConvRot, no bias |
| `model.layers.i.self_attn.q_proj.weight_scale` | F32 | `[8192, 1]` | per-output-channel, `= amax/127` |
| `model.layers.i.self_attn.q_proj.comfy_quant` | U8 | `[72]` | `{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}` |
| `model.layers.i.self_attn.k_proj.weight` | I8 | `[1024, 5120]` | `k`, 8 kv heads × 128. ConvRot, no bias |
| `model.layers.i.self_attn.k_proj.weight_scale` | F32 | `[1024, 1]` | |
| `model.layers.i.self_attn.k_proj.comfy_quant` | U8 | `[72]` | |
| `model.layers.i.self_attn.v_proj.weight` | I8 | `[1024, 5120]` | `v`, 8 kv heads × 128. ConvRot, no bias |
| `model.layers.i.self_attn.v_proj.weight_scale` | F32 | `[1024, 1]` | |
| `model.layers.i.self_attn.v_proj.comfy_quant` | U8 | `[72]` | |
| `model.layers.i.self_attn.q_norm.weight` | BF16 | `[128]` | RMSNorm over the head axis, eps 1e-6, **before RoPE**, shared across all 64 heads |
| `model.layers.i.self_attn.k_norm.weight` | BF16 | `[128]` | same, on `k`, shared across all 8 kv heads |
| `model.layers.i.self_attn.o_proj.weight` | I8 | `[5120, 8192]` | 8192 → 5120 after head flatten. ConvRot (32 blocks), no bias |
| `model.layers.i.self_attn.o_proj.weight_scale` | F32 | `[5120, 1]` | |
| `model.layers.i.self_attn.o_proj.comfy_quant` | U8 | `[72]` | |
| `model.layers.i.post_attention_layernorm.weight` | BF16 | `[5120]` | pre-MLP RMSNorm, eps 1e-6 |
| `model.layers.i.mlp.gate_proj.weight` | I8 | `[25600, 5120]` | **goes through SiLU**. ConvRot, no bias |
| `model.layers.i.mlp.gate_proj.weight_scale` | F32 | `[25600, 1]` | |
| `model.layers.i.mlp.gate_proj.comfy_quant` | U8 | `[72]` | |
| `model.layers.i.mlp.up_proj.weight` | I8 | `[25600, 5120]` | **linear, no activation**. Same input as `gate_proj` |
| `model.layers.i.mlp.up_proj.weight_scale` | F32 | `[25600, 1]` | |
| `model.layers.i.mlp.up_proj.comfy_quant` | U8 | `[72]` | |
| `model.layers.i.mlp.down_proj.weight` | I8 | `[5120, 25600]` | contracts the SwiGLU product. ConvRot (100 blocks), no bias |
| `model.layers.i.mlp.down_proj.weight_scale` | F32 | `[5120, 1]` | |
| `model.layers.i.mlp.down_proj.comfy_quant` | U8 | `[72]` | |

All 350 `comfy_quant` payloads are byte-identical (verified by reading every
one). All 350 `weight_scale` tensors are `[out, 1]`. All 350 int8 weights are
ConvRot-rotated, because every `in_features ∈ {5120, 8192, 25600}` is divisible
by 256.

### 8.4 `visual.*` — 351 tensors

The text-only CUDA path does not upload these tensors. The exact Vulkan archive
boundary validates all 351 tensors transactionally, and its multimodal overload
streams them on demand without keeping expanded copies.

| key pattern | count | dtype |
|---|---|---|
| `visual.patch_embed.proj.{weight,bias}` | 2 | BF16 |
| `visual.pos_embed.weight` | 1 | BF16 |
| `visual.blocks.i.{norm1,norm2}.{weight,bias}` (i ∈ [0,27)) | 4 × 27 = 108 | BF16 |
| `visual.blocks.i.attn.{qkv,proj}.{weight,bias}` | 4 × 27 = 108 | BF16 |
| `visual.blocks.i.mlp.{linear_fc1,linear_fc2}.{weight,bias}` | 4 × 27 = 108 | BF16 |
| `visual.merger.{norm,linear_fc1,linear_fc2}.{weight,bias}` | 6 | BF16 |
| `visual.deepstack_merger_list.i.{norm,linear_fc1,linear_fc2}.{weight,bias}` (i ∈ [0,3)) | 6 × 3 = 18 | BF16 |

`27 × 12 = 324`, `+ 18 + 6 + 2 + 1 = 351`. Note these are **not quantised** —
all BF16, all with biases, and the tower uses LayerNorm (weight *and* bias), not
RMSNorm. Structurally unlike the language stack; a shared loader would be wrong
as well as wasteful. The 3 `deepstack_merger_list` entries correspond to
`vision_config.deepstack_visual_indexes = [8, 16, 24]`.

### 8.5 Coverage

```
model.embed_tokens.weight                              1
model.layers.{0..49}.*            25 x 50    =      1 250
visual.*                                              351
                                              -----------
                                                    1 602   ✓  matches the header
```

Bytes:

```
1 555 824 640  +  24 394 802 800  +  1 190 533 600  =  27 141 161 040
27 141 161 040 + 181 104 (header) + 8 (length prefix) = 27 141 342 152  ✓  = file size
```

Text-only CUDA loads **1251 tensors, 25.95 GB on disk, 24.39 GB uploaded**
(the embedding stays on the host). Vulkan validates the full archive and
streams only one compressed text layer or BF16 visual component at a time.

---

## 9. Operator-order hazards — the silent-garbage list

Every item here produces finite, well-scaled, plausible output when done wrong.
None of them crash. They are ordered by how much damage they do.

1. **ConvRot: not rotating the activation.** Computes `x H Wᵀ` instead of
   `x Wᵀ`. Pure noise, correctly scaled. §5.2. Cross-check with §5.4 before
   trusting anything.
2. **ConvRot: the wrong Hadamard.** Sylvester/Walsh gives 150 % relative error,
   `kron⁴(2I − J)` gives 139 %, both looking like well-behaved tensors. Use the
   regular `h4` in §5.2 verbatim. This is settled in `docs/convrot_notes.md`;
   do not "optimise" it into an FWHT.
3. **Bidirectional instead of causal attention.** §3. Changes every row except
   the last. Test: appending a token must not change `prompt_embeds[0]`.
4. **The wrong hidden state.** Taking the *input* to layer 49 instead of its
   output (off by one layer), or applying a final norm that does not exist.
   §1.4. Test: output rows should have large RMS, not ≈ 1.
5. **QK-norm after RoPE instead of before.** §4.2. Silent quality degradation.
6. **GQA mapping `h % 8` instead of `h / 8`.** §4.2. Every query head reads the
   wrong kv head; shapes all check out.
7. **`eps = 1e-5` instead of `1e-6`.** §4.1. The transformer uses 1e-5; sharing
   a kernel with a hard-coded eps silently changes every norm here.
8. **`head_dim = 80`** from `5120/64` instead of the config's explicit **128**.
   This one *does* fail loudly (shapes mismatch `[8192, 5120]`), but it will
   send you looking in the wrong place. `hidden_size / num_heads ≠ head_dim` for
   this model.
9. **Interleaved-adjacent RoPE pairs `(2j, 2j+1)` instead of half-split
   `(j, j+64)`.** §2.4. Classic, silent, and the reason §2.4 confirms rather
   than assumes.
10. **Rotating only 96 of 128 head dims**, by analogy with the H3 DiT
    (`docs/transformer_spec.md` §5.3). Here **all 128 rotate**. §2.4.
11. **Adding a BOS/EOS or applying the chat template.** `add_special_tokens=False`
    (`encoders.py:167`). One extra leading token shifts every RoPE position and,
    because attention is causal, changes every row.
12. **Applying SiLU to `up_proj` instead of `gate_proj`.** §4.3. The separate
    tensor names make this hard to get wrong here, but the H3 DiT's fused `fc1`
    conditions people to expect a swap.
13. **Rotating the residual, or the raw `x` instead of the normed `x`.** §5.3.
    `H` does not commute with `diag(w)`.
14. **RMSNorm accumulating in bf16.** §6.1. Explicit fp32 upcast in the
    reference, not a PyTorch internal.
15. **Loading `visual.*`.** Not incorrect, just 1.19 GB of waste — and it will
    tempt someone into wiring up DeepStack, which for text-only must not run.

---

## 10. Open questions / UNRESOLVED

### 10.1 UNRESOLVED (low risk) — `transformers` is not local; the model source was fetched

**What is unknown:** nothing specific — but §§2, 3, 4 and 6 rest on
`modeling_qwen3_vl.py`, `modular_qwen3_vl.py`, `modeling_rope_utils.py` and
`utils/generic.py`, none of which are in `ref/`.

**What was checked:** `python -c "import transformers"` →
`ModuleNotFoundError: No module named 'transformers'` (Python 3.10.11). A
filesystem search for `modeling_qwen3_vl*.py` and for any `qwen3_vl` directory
under `C:\Users\NN` and the project drives found nothing. `ref/` contains only
the diffusers pipeline, the MiniMax configs and `comfy_quants/convrot.py`; the
Qwen3-VL model code is not vendored anywhere. The sources were therefore fetched
from upstream at tag **v4.57.1**, chosen to match
`config.json`'s `"transformers_version": "4.57.0.dev0"`. Some fetches resolved
against `main`, where the file has since drifted (`check_model_inputs` is now a
deprecated alias); every claim in this spec was taken from the **v4.57.1**
fetch or cross-checked against it.

**Risk:** a version skew between 4.57.0.dev0 and 4.57.1 in the rotary,
attention or MLP code. Low — these are stable across the 4.57 line, and every
structural claim is independently corroborated by the checkpoint's own tensor
names and shapes (§8.3), which match the described modules exactly.

**Recommended verification (one command, do it once):**

```
pip install "transformers==4.57.1" torch --no-deps    # in a throwaway venv
```

then instantiate `Qwen3VLTextConfig(**text_config)` and dump
`Qwen3VLTextRotaryEmbedding(cfg).forward(x, arange(L).expand(3,1,L))[0]` for
`L = 16`, and compare against the port's `cos` table. That single comparison
closes §2 entirely (angles, interleave, duplication and the `attention_scaling`
question at once). If a GPU with 32 GB is available, also run the reference
against the port on a 20-token prompt and compare layer-49 outputs; expect
≲ 1e-2 relative from int8 alone.

**Default to implement meanwhile:** everything as specified. Nothing here is a
coin-flip.

### 10.2 RESOLVED — the mrope degeneracy

Recorded as resolved because the working hypothesis was correct and it is worth
stating that it was *established*, not assumed.

- **Position ids:** all three axes equal the token index, from two independent
  code paths (`get_rope_index`'s `.expand(3, -1)` and `Qwen3VLTextModel`'s
  `.expand(4, ...)` default). §2.1.
- **`mrope_interleaved`:** resolved to an explicit `[THWTHW…TT]` slot table with
  counts 24/20/20 matching `mrope_section` exactly. §2.2.
- **Degeneracy:** holds **bitwise**, because the three axis slices of `freqs`
  are computed from a broadcast `expand` of one `inv_freq` and three identical
  position rows, so the interleave copies values onto themselves. §2.3.
- **Pairing:** GPT-NeoX half-split `(j, j+64)`, confirmed against the model
  file's own `rotate_half`, against the `cat((freqs, freqs))` duplication that
  only makes sense for a half-split, and against the H3 DiT's independent use of
  the same convention. §2.4.
- **Consequence:** implement plain 1-D RoPE, `theta = 5e6`, `head_dim = 128`,
  all 128 dims rotated. `mrope_section` and `mrope_interleaved` may be ignored
  **for text-only input**, and the code should say so at the point of ignoring
  them.

### 10.3 UNRESOLVED (low risk, easily bounded) — the effect of int8 + ConvRot on H3's conditioning

**What is unknown:** how much the int8-quantised conditioner's layer-49 output
differs from the released bf16 conditioner's, and therefore how much the final
video differs.

**What was checked:** `docs/convrot_notes.md` measured **0.84 % relative error**
on a single dequantised weight row against bf16 ground truth. That is per-weight
error, not end-to-end error over 50 layers, and per-layer errors in a pre-norm
residual stack can accumulate or cancel; no measurement of the composite exists.
The bf16 reference checkpoint
(`Comfy-Org/MiniMax-H3 : text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors`,
47.97 GB) is **not** present locally — only the int8 file is.

**Risk:** low and unavoidable. This is the checkpoint we have; there is no
higher-precision option that fits. Quantisation error of this kind shows up as a
mild loss of prompt fidelity, not as corruption.

**Recommended action:** none required. If the bf16 file is ever downloaded, run
both through the same 20-token prompt and record the relative error of the
layer-49 output as a regression baseline. Do **not** spend VRAM on a
higher-precision path — dequantising to bf16 needs 48.8 GB (§7).

### 10.4 UNRESOLVED (no risk on this path) — `attention_scaling`

**What is unknown:** whether `attention_scaling` is exactly 1.0 at runtime.

**What was checked:** `config.json` sets `rope_scaling.rope_type = "default"`;
`hf/modeling_rope_utils.py`'s `_compute_default_rope_parameters` returns
`attention_factor = 1.0` with the comment "Unused in this type of RoPE". The
rotary forward multiplies `cos` and `sin` by it. The only way it could differ is
a `dynamic_rope_update` decorator rewriting `inv_freq` mid-run, which applies
only to `rope_type ∈ {"dynamic", "longrope", "llama3", ...}` — not `"default"`.

**Risk:** none. **Default to implement:** omit the multiply.

### 10.5 RESOLVED for conditioning — `fl2va` / `ref2va`

The backend-neutral image processor supplies merge-group-major 16x16 patches,
learned position indices and exact THW coordinates. The visual tower is patch
projection plus learned position, 27 exact blocks, DeepStack extraction after
visual blocks 8/16/24, and main/deep mergers. The decoder replaces image-pad
rows, builds the non-degenerate interleaved THW mRoPE table, then injects those
three features after decoder layers 0/1/2 respectively. All activation traffic
between these stages is device-to-device in the Vulkan implementation.

The authorities use the production I8+ConvRot archive SHA-256
`BC2CED0FBEA64757FA9ACDDCCFC0B3F4819D1DCF1DA6C124D690D368BE283923`
and NVFP4+AWQ SHA-256
`33E69E3EDAB846D52949BAFDB00378BD3F5A93F78124FC83D5EF109DC4A1FCBB`
with a deterministic 256x256 RGB image passed through the real patchifier.
CUDA and Vulkan match every one of 27 I8 visual BF16 boundaries and all 50
decoder boundaries for both archives; final FP32 embedding FNV64 values are
respectively `A875C128AA7A0E9D` and `EBC9E36A30C843CD`. A CUDA-disabled replay
pins the I8 result. At the trace-free production maximum S16384/L4100, the
actual-SHA I8 and NV runs pin visual main/DeepStack FNV64 values
`63A7DD4533D82D51`, `DA837B598AE29C53`, `0EDF2B1389F62CE2`, and
`D8F970E5016F22B7`; all 351 `visual.*` metadata and payloads are also compared
byte-for-byte between archives. Final I8/NV hashes are `EDE491026C069661` and
`4435938303E77287`. Both measure a 2676.3 MiB logical and non-staging allocator
peak; cold/warm Vulkan times are 60.79/54.96 s and 54.52/51.65 s. Repeat
pool/reservation/descriptor counts are exact and unload returns pooled-used
memory to the staging-only baseline.
Top-level Ref2VA is still rejected because keyframe
video-VAE *encoding* is a separate unported component, not because conditioner
vision semantics remain unknown.

### 10.6 UNRESOLVED (very low risk) — exact prompt length limits

**What is unknown:** whether MiniMax's pipeline truncates long prompts anywhere.

**What was checked:** `encoders.py:112-195` performs no truncation and passes no
`max_length` to the tokenizer; `_check_prompt` (`encoders.py:48-53`) checks only
the type. `config.json` gives `max_position_embeddings = 262144`, far above
anything reachable. `docs/transformer_spec.md` §1.1 treats `L` as unbounded and
adds it to the packed sequence length `S`.

**Risk:** an unbounded `L` grows the packed sequence the DiT attends over
(`S = L + Sa + V`) and the encoder's own `O(L²)` attention. A 100 k-token prompt
would be a memory problem in the *transformer*, not here.

**Recommended default:** do not truncate. Reject or warn above, say, 8192
tokens at the API boundary, and size the encoder's buffers for that. Note that
truncation would be *observable* to the user as a silently shortened prompt, so
a hard error is better than a silent cut.
