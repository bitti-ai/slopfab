# Ref2VA Qwen vision contract

This records the image-conditioning path in the local MiniMax-H3 upstream
snapshot (`ref/diffusers/modular/encoders.py`) and the matching Qwen3-VL
configuration. It is an implementation contract, not a proposal.

## Checkpoint sufficiency

Ref2VA has no separate image encoder. Its `text_encoder` is
`Qwen3VLForConditionalGeneration`, and the same checkpoint used for text owns
the visual tower. The local `ref/text_encoder/model.safetensors.index.json`
contains all **351** `model.visual.*` tensors: patch embed, 27 ViT blocks, the
main merger, and three DeepStack mergers. Therefore the repository's Qwen
checkpoint is sufficient input data. A repacked checkpoint may flatten the
prefix to `visual.*`; both forms mean the same tower.

The tower is not optional for reference images. Merely gathering the token
embedding for id 151655 would ignore the pixels.

## Pixel and token flow

1. `processor.image_processor(images=..., return_tensors="pt")` returns
   `pixel_values` and one `[grid_t, grid_h, grid_w]` row per image.
2. H3's processor uses patch size 16, temporal patch size 2, spatial merge 2,
   RGB mean `[.5,.5,.5]`, standard deviation `[.5,.5,.5]`, minimum 65,536
   pixels, and maximum 16,777,216 pixels. The smart-resize factor is
   `16 * 2 = 32`. A still image has `grid_t=1`; the processor duplicates it
   across the temporal patch internally.
3. The unmerged visual patch count is `grid_t*grid_h*grid_w`. The language
   token count is that value divided by `merge_size**2`, i.e. four.
4. For image `i` (one-based), MiniMax tokenizes `"<Picture i>: "` with
   `add_special_tokens=False`, then appends `<|vision_start|>`, exactly the
   merged token count of `<|image_pad|>`, and `<|vision_end|>`. The verbatim
   user prompt follows all references. There is no chat template, BOS, EOS,
   system message, `im_start`, or `im_end`.
5. `create_mm_token_type_ids` marks the image-pad run as image modality. Qwen
   derives its three-axis mRoPE positions from these ids and `image_grid_thw`.
6. The visual tower produces one 5120-wide merged embedding per image-pad
   token. Those embeddings replace the ordinary token embeddings at pad
   positions. Its three DeepStack mergers additionally produce features that
   are injected into text decoder layers 8, 16, and 24.
7. MiniMax asks Qwen for all hidden states and consumes
   `hidden_states[50]`: the raw, unnormalized residual after decoder layer 49.
   The LM head and final text norm are unused.

All rows from a vision block, including its start/end sentinels, receive H3's
video modality tag for transformer AdaLN. The `<Picture i>: ` label and final
prompt receive the text tag. This H3 tag stream is distinct from Qwen's own
multimodal token-type ids.

## Current port boundary

`qwen_vision.h` now implements the exact smart-resize/grid/token-count and
presentation-block contracts on the host. End-to-end reference conditioning
still requires the CUDA ViT forward, merged embedding replacement, 3-D mRoPE,
and DeepStack injection. Until all four are connected, the runtime must keep
failing closed instead of claiming that presence of 351 tensors alone makes
the current executable image-capable.
