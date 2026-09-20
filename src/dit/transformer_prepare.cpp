#include "transformer_impl.h"

namespace slopfab::dit {

void Transformer::prepare_text(const float* prompt_embeds, int num_tokens) {
  cuda::StageMemorySpan memory("transformer.text");
  Impl& s = *impl_;
  s.require_loaded("prepare_text");
  if (num_tokens < 0) throw std::runtime_error("transformer: negative token count");

  s.num_text = num_tokens;
  if (num_tokens == 0) {
    s.text_cache.reset();
    return;
  }
  s.begin_transformer_text_capture(prompt_embeds, num_tokens);

  const int hidden = s.cfg.hidden_size;
  const int inner = s.cfg.inner_dim();
  const size_t rows = static_cast<size_t>(num_tokens);

  // The refiner is O(L^2) with L in the low thousands and, unlike everything
  // else here, has no timestep dependence — so it runs once per request rather
  // than once per step, which the reference only fails to do because its
  // forward is stateless (spec 6).
  SequenceLayout text_only;
  text_only.num_text = num_tokens;
  const Carve text_carve = plan_carve(s.cfg, text_only, s.row_chunk,
                                      s.attention_mode,
                                      stack_uses_convrot(s.refiner),
                                      s.compact_queries(s.attention_mode == AttentionMode::kExact
                                          ? AttentionMode::kExact : AttentionMode::kFlash2));

  Workspace& ws = s.ws;
  // The refiner needs one extra bf16 [L, text_dim] buffer that the main path
  // does not, for the fp32 prompt embedding narrowed to the block dtype.
  ws.reserve(std::max(ws.capacity(),
                      text_carve.total + align_up(rows * s.cfg.text_dim * sizeof(__nv_bfloat16))));
  ws.clear();

  DeviceBuffer<float> d_prompt(rows * s.cfg.text_dim);
  d_prompt.copy_from_host(prompt_embeds, d_prompt.size(), s.stream.get());

  __nv_bfloat16* xin = ws.alloc_n<__nv_bfloat16>(rows * s.cfg.text_dim);
  cuda::launch_narrow_to_bf16(d_prompt.get(), xin, d_prompt.size(), s.stream.get());

  s.text_cache.allocate(rows * hidden);
  __nv_bfloat16* x = s.text_cache.get();
  if (s.attention_mode == AttentionMode::kExact) {
    if (s.condition_proj.format != QuantFormat::kBF16 ||
        s.condition_proj.bias_format != QuantFormat::kF32 ||
        s.condition_proj.pre_quant_scale != nullptr || s.condition_proj.convrot) {
      throw std::runtime_error(
          "transformer: exact condition projection requires plain BF16 weight and fp32 bias");
    }
    const uint32_t tiled = static_cast<uint32_t>(num_tokens) / 64 * 64;
    if (tiled != 0) {
      cuda::launch_deterministic_bf16_gemm_nt(
          xin, static_cast<const __nv_bfloat16*>(s.condition_proj.data),
          s.condition_proj.bias, x, tiled, static_cast<uint32_t>(hidden),
          static_cast<uint32_t>(s.cfg.text_dim), DenseGemmBias::kFloat32,
          0, 0, s.stream.get());
    }
    if (tiled != static_cast<uint32_t>(num_tokens)) {
      cuda::launch_deterministic_scalar_gemm_nt(
          xin, s.condition_proj.data, s.condition_proj.bias, x,
          static_cast<uint32_t>(num_tokens) - tiled,
          static_cast<uint32_t>(hidden), static_cast<uint32_t>(s.cfg.text_dim),
          DenseGemmMode::kBFloat16, DenseGemmBias::kFloat32,
          tiled, tiled, s.stream.get());
    }
  } else {
    s.linear.forward(s.condition_proj, xin, num_tokens, x, ws);
  }
  s.emit_stage("condition_proj", x, num_tokens, hidden);

  __nv_bfloat16* q = ws.alloc_n<__nv_bfloat16>(text_carve.query);
  __nv_bfloat16* k = ws.alloc_n<__nv_bfloat16>(rows * inner);
  __nv_bfloat16* v = ws.alloc_n<__nv_bfloat16>(rows * inner);
  __nv_bfloat16* attn_out = ws.alloc_n<__nv_bfloat16>(text_carve.query);
  __nv_bfloat16* normed = ws.alloc_n<__nv_bfloat16>(text_carve.normed);
  __nv_bfloat16* fused = ws.alloc_n<__nv_bfloat16>(text_carve.fused);
  __nv_bfloat16* act = ws.alloc_n<__nv_bfloat16>(text_carve.act);
  __nv_bfloat16* branch = ws.alloc_n<__nv_bfloat16>(text_carve.normed);

  const Carve saved = s.carve;
  s.carve = text_carve;
  for (const BlockWeights& b : s.refiner) {
    // No AdaLN, no RoPE, no mask: `mod_base` and `cos` are null.
    // Preserve every legacy mode's historical Flash2 refiner. Exact is the
    // only selection whose contract deliberately covers both stacks.
    const AttentionMode refiner_mode = s.attention_mode == AttentionMode::kExact
        ? AttentionMode::kExact : AttentionMode::kFlash2;
    s.run_block(b, nullptr, num_tokens, x, nullptr, nullptr, nullptr, q, k, v, attn_out, normed,
                fused, act, branch, refiner_mode);
  }
  s.carve = saved;

  for (int start = 0; start < num_tokens; start += text_carve.chunk) {
    const int n = std::min(text_carve.chunk, num_tokens - start);
    __nv_bfloat16* part = x + static_cast<size_t>(start) * hidden;
    cuda::launch_rmsnorm(part, s.refiner_final_norm, normed, n, hidden, s.cfg.norm_eps,
                         s.stream.get());
    SLOPFAB_CUDA_CHECK(cudaMemcpyAsync(part, normed, static_cast<size_t>(n) * hidden * sizeof(__nv_bfloat16),
                                      cudaMemcpyDeviceToDevice, s.stream.get()));
  }
  SLOPFAB_CUDA_CHECK(cudaStreamSynchronize(s.stream.get()));
  s.emit_stage("final_norm", x, num_tokens, hidden);
  s.transformer_capture.text_active = false;
  ws.clear();
  s.attention_configuration_locked = true;
}

void Transformer::prepare_sequence(const SequenceLayout& layout, const PackedIndices& indices,
                                   const std::vector<double>& position_ids) {
  Impl& s = *impl_;
  s.require_loaded("prepare_sequence");

  const int seq = layout.total_rows();
  if (seq <= 0) throw std::runtime_error("transformer: empty packed sequence");
  if (layout.num_text != s.num_text) {
    throw std::runtime_error("transformer: layout has " + std::to_string(layout.num_text) +
                             " text rows but prepare_text cached " + std::to_string(s.num_text));
  }
  if (position_ids.size() != static_cast<size_t>(seq) * 3) {
    throw std::runtime_error("transformer: position_ids has " +
                             std::to_string(position_ids.size()) + " entries, expected 3 * " +
                             std::to_string(seq));
  }
  if (static_cast<int>(indices.tags.size()) != seq) {
    throw std::runtime_error("transformer: token tags do not cover the sequence");
  }

  s.has_sequence = false;
  s.layout = layout;
  s.indices = indices;
  s.carve = plan_carve(s.cfg, layout, s.row_chunk, s.attention_mode,
                       stack_uses_convrot(s.blocks), s.compact_queries(s.attention_mode), s.is_vsa());
  if (s.architecture == TransformerArchitecture::kRef2VAFullAdaLN) {
    const size_t old_scratch = s.carve.scratch;
    s.carve.scratch = std::max(s.carve.scratch, cuda::linear_workspace_bytes(
        s.blocks.front().full_adaln, 2, ComputeType::kF32));
    s.carve.scratch = std::max(s.carve.scratch, cuda::linear_workspace_bytes(
        s.final_full_adaln, 2, ComputeType::kF32));
    s.carve.total += align_up(s.carve.scratch) - align_up(old_scratch);
  }

  // Drop a larger text/refiner reservation before allocating sequence buffers.
  // Previous preparation/forward has completed before this stage is entered.
  SLOPFAB_CUDA_CHECK(cudaStreamSynchronize(s.stream.get()));
  if (cuda::StepProfiler::instance().enabled()) {
    constexpr double gib = 1024.0 * 1024 * 1024;
    std::printf("memory      sequence %d rows (text %d, condition video %d, condition audio %d, video %d, audio %d); "
                "row chunk %d, compact queries %s; workspace %.3f -> %.3f GiB, "
                "K/V %.3f GiB, Q/output %.3f GiB\n",
                layout.total_rows(), layout.num_text, layout.num_condition_video,
                layout.num_condition_audio, layout.num_video_rows, layout.num_audio_rows, s.carve.chunk,
                s.carve.chunked_attention ? "yes" : "no", s.ws.capacity() / gib,
                s.carve.total / gib, 4.0 * s.carve.qkv / gib, 4.0 * s.carve.query / gib);
    std::fflush(stdout);
  }
  s.ws.resize(s.carve.total);
  if (s.is_vsa()) {
    if (s.attn_band > 0) throw std::runtime_error("VSA-H3 cannot use frame-banded attention");
    const auto tiles = build_vsa_tiles(layout);
    s.vsa_rows.allocate(tiles.rows.size());
    s.vsa_sizes.allocate(tiles.sizes.size());
    s.vsa_row_tiles.allocate(tiles.row_tiles.size());
    s.vsa_rows.copy_from_host(tiles.rows.data(), tiles.rows.size(), s.stream.get());
    s.vsa_sizes.copy_from_host(tiles.sizes.data(), tiles.sizes.size(), s.stream.get());
    s.vsa_row_tiles.copy_from_host(tiles.row_tiles.data(), tiles.row_tiles.size(), s.stream.get());
    s.vsa_compressed.allocate(tiles.sizes.size() * s.cfg.inner_dim());
    s.vsa_config = {static_cast<int>(tiles.sizes.size()), tiles.prefix_tiles,
                    s.cfg.num_attention_heads, s.cfg.attention_head_dim,
                    s.vsa_rows.get(), s.vsa_sizes.get(), s.vsa_row_tiles.get()};
    // The map vectors are local pageable storage; finish their transfers.
    SLOPFAB_CUDA_CHECK(cudaStreamSynchronize(s.stream.get()));
  }

  // Frame-banded attention: one small table for the whole request, ~300 entries
  // at the default geometry, rebuilt here because it depends on the layout and
  // nothing else. Off leaves the buffer empty, which is what the attention call
  // site turns into a null `band_ranges` and so into the unbanded kernel.
  //
  // Built from the kernel's own tiling rather than from constants repeated here:
  // a query tile or key alignment that drifted from the kernel's would produce a
  // band subtly misaligned with the loop, which is a wrong model rather than an
  // error.
  s.d_band.reset();
  s.host_band.clear();
  if (s.attn_band > 0) {
    const int query_tile = s.attention_mode == AttentionMode::kExact
        ? static_cast<int>(cuda::deterministic_h3_query_tile())
        : cuda::attention_fused_query_tile();
    const int key_align = s.attention_mode == AttentionMode::kExact
        ? static_cast<int>(cuda::deterministic_h3_key_align())
        : cuda::attention_fused_key_align();
    const dit::BandedKeyRanges band = dit::build_banded_key_ranges(
        layout, s.attn_band, query_tile, key_align);
    s.host_band = band.ranges;
    s.d_band.allocate(band.ranges.size());
    s.d_band.copy_from_host(band.ranges.data(), band.ranges.size(), s.stream.get());
  }

  const dit::H3RopeTables rope = dit::build_h3_rope_tables(
      position_ids, s.cfg.rope_theta, static_cast<uint32_t>(s.cfg.rope_freq_dim));
  s.host_rope_cos = rope.cosine;
  s.host_rope_sin = rope.sine;
  s.rope_cos.allocate(rope.cosine.size());
  s.rope_sin.allocate(rope.sine.size());
  s.rope_cos.copy_from_host(rope.cosine.data(), rope.cosine.size(), s.stream.get());
  s.rope_sin.copy_from_host(rope.sine.data(), rope.sine.size(), s.stream.get());
  SLOPFAB_CUDA_CHECK(cudaStreamSynchronize(s.stream.get()));

  auto upload_idx = [&](const std::vector<int32_t>& src, DeviceBuffer<int32_t>& dst) {
    dst.allocate(std::max<size_t>(src.size(), 1));
    if (!src.empty()) dst.copy_from_host(src.data(), src.size(), s.stream.get());
  };
  upload_idx(indices.text, s.d_text_idx);
  upload_idx(indices.audio, s.d_audio_idx);
  upload_idx(indices.video, s.d_video_idx);

  s.hidden.allocate(static_cast<size_t>(seq) * s.cfg.hidden_size);
  // Cleared before the allocation, not after it. `DeviceBuffer::allocate` calls
  // `reset()` first, so an allocation that throws leaves the buffer null — and
  // clearing the flag afterwards would leave it *true* over a null pointer,
  // from a previous successful run. A caller that caught the out-of-memory and
  // carried on would then take the reuse path into an async illegal access
  // instead of a clean throw, and out-of-memory is the likeliest failure this
  // feature has at the production geometry.
  s.bc_have_delta = false;
  // Sized here with `hidden` because it is the same shape and shares its
  // lifetime. A re-`prepare_sequence` with a different geometry reallocates it,
  // which invalidates any captured delta — the stream it was captured against
  // no longer has the same rows.
  if (s.block_cache.enabled() && s.bc_span.valid()) {
    s.bc_delta.allocate(static_cast<size_t>(seq) * s.cfg.hidden_size);
  }
  s.d_adaln.allocate(static_cast<size_t>(seq));
  s.d_ts_video.allocate(std::max<size_t>(indices.video.size(), 1));
  s.d_ts_audio.allocate(std::max<size_t>(indices.audio.size(), 1));

  s.d_video_rows.allocate(std::max<size_t>(indices.video.size(), 1) * s.cfg.video_patch_dim());
  s.d_video_head.allocate(s.d_video_rows.size());
  s.d_audio_rows.allocate(std::max<size_t>(indices.audio.size(), 1) * s.cfg.audio_in_channels);
  s.d_audio_head.allocate(s.d_audio_rows.size());

  s.ws.reserve(s.carve.total);
  s.has_sequence = true;
  SLOPFAB_CUDA_CHECK(cudaStreamSynchronize(s.stream.get()));
  s.attention_configuration_locked = true;
}

// ---------------------------------------------------------------------------


}  // namespace slopfab::dit
