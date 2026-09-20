#include "transformer_impl.h"

namespace slopfab::dit {

void Transformer::Impl::build_modulation(const std::vector<float>& timesteps) {
    const int T = static_cast<int>(timesteps.size());
    if (T <= 0) throw std::runtime_error("transformer: no distinct timesteps");

    const int code_dim = is_pruned_table_architecture(architecture)
                             ? AdaLNTable::kRank
                             : cfg.timestep_embed_dim;
    if (is_pruned_table_architecture(architecture)) {
      host_code.assign(static_cast<size_t>(T) * code_dim, 0.0f);
      for (int i = 0; i < T; ++i) {
        const std::array<float, AdaLNTable::kRank> c = table.lookup(timesteps[i], lookup);
        for (int k = 0; k < code_dim; ++k) {
          host_code[static_cast<size_t>(i) * code_dim + k] = c[k];
        }
      }
    } else {
      host_code = timestep_embedding.forward(timesteps);
      // Each AdaLN module applies SiLU to the shared time MLP result before
      // casting/projecting. Applying it once is identical because every one
      // of the 51 consumers reads the same tensor and FP8 forward_f32 widens
      // the stored weight rather than narrowing the activation.
      for (float& value : host_code) value = value / (1.0f + std::exp(-value));
    }
    if (d_code.size() < host_code.size()) d_code.allocate(host_code.size());
    d_code.copy_from_host(host_code.data(), host_code.size(), stream.get());

    mod_timesteps = T;
    const size_t per_block = block_mod_stride();
    const size_t need = per_block * blocks.size();
    if (mod.size() < need) mod.allocate(need);
    const size_t final_need = static_cast<size_t>(kFinalParams) * T * cfg.hidden_size;
    if (final_mod.size() < final_need) final_mod.allocate(final_need);

    for (size_t b = 0; b < blocks.size(); ++b) {
      if (is_pruned_table_architecture(architecture)) {
        cuda::launch_adaln_expand(blocks[b].adaln_w, blocks[b].adaln_b, d_code.get(),
                                  mod.get() + b * per_block, T, kNumModalities, kNumParams,
                                  cfg.hidden_size, code_dim, stream.get());
      } else {
        linear.forward_f32(blocks[b].full_adaln, d_code.get(), T,
                           mod.get() + b * per_block, ws);
      }
    }
    if (is_pruned_table_architecture(architecture)) {
      cuda::launch_adaln_expand(final_adaln_w, final_adaln_b, d_code.get(), final_mod.get(), T,
                                /*num_modality=*/1, kFinalParams, cfg.hidden_size, code_dim,
                                stream.get());
    } else {
      linear.forward_f32(final_full_adaln, d_code.get(), T, final_mod.get(), ws);
    }
  }

void Transformer::Impl::run_block(const BlockWeights& b, const float* mod_base, int rows, __nv_bfloat16* x,
                 const int32_t* adaln_idx, const float* cos, const float* sin, __nv_bfloat16* q,
                 __nv_bfloat16* k, __nv_bfloat16* v, __nv_bfloat16* attn_out,
                 __nv_bfloat16* normed, __nv_bfloat16* fused, __nv_bfloat16* act,
                 __nv_bfloat16* branch, AttentionMode block_attention_mode, int layer) {
    const int hidden = cfg.hidden_size;
    const int inner = cfg.inner_dim();
    const int chunk = carve.chunk;
    const float eps = cfg.norm_eps;
    const size_t stride = mod_row_stride();
    // Only armed between `forward`'s begin_step and end_step, so the token
    // refiner — which shares this function — contributes nothing.
    cuda::StepProfiler& prof = cuda::StepProfiler::instance();

    // Spec 3.2's parameter order. Six tables, each [T*3, hidden], indexed by
    // adaln_idx[row] = timestep_index*3 + tag.
    const float* shift_msa = mod_base != nullptr ? mod_base + 0 * stride : nullptr;
    const float* scale_msa = mod_base != nullptr ? mod_base + 1 * stride : nullptr;
    const float* gate_msa = mod_base != nullptr ? mod_base + 2 * stride : nullptr;
    const float* shift_mlp = mod_base != nullptr ? mod_base + 3 * stride : nullptr;
    const float* scale_mlp = mod_base != nullptr ? mod_base + 4 * stride : nullptr;
    const float* gate_mlp = mod_base != nullptr ? mod_base + 5 * stride : nullptr;
    if (layer >= 0)
      begin_block_capture(x, adaln_idx, cos, sin, rows, layer,
                          block_attention_mode);
    if (layer >= 0)
      begin_graph_capture(x, adaln_idx, cos, sin, rows, layer,
                          block_attention_mode);
    // Adapter keys identify logical projections even when two blocks reuse
    // the same device address. Their staged factors share the block's slot.
    auto adapter_key = [&](const QuantWeight& weight) -> const void* {
      if (layer < 0 || !streamer || !streamer->contains(layer)) return weight.data;
      const BlockWeights& original = blocks[layer];
      const QuantWeight* current[] = {&b.wq, &b.wk, &b.wv, &b.out_proj, &b.fc1, &b.fc2};
      const QuantWeight* logical[] = {&original.wq, &original.wk, &original.wv,
                                      &original.out_proj, &original.fc1, &original.fc2};
      for (int i = 0; i < 6; ++i) if (&weight == current[i]) return logical[i]->data;
      return weight.data;
    };
    auto project = [&](const QuantWeight& weight,
                       const __nv_bfloat16* dense,
                       const __nv_bfloat16* input, int count,
                       __nv_bfloat16* output) {
      if (block_attention_mode != AttentionMode::kExact) {
        linear.forward_prepared(weight, dense, input, count, output, ws);
        lora.apply(adapter_key(weight), input, count, output, blas, stream.get(), false);
        return;
      }
      if (weight.bias != nullptr)
        throw std::runtime_error(
            "transformer: exact H3 projection requires bias-free weights");
      // The deterministic GEMM consumes the same logical activation as the
      // regular LinearRunner. ConvRot INT8 checkpoints therefore still need
      // their online activation rotation; exactness changes the GEMM, not the
      // checkpoint's coordinate system. Rewind the temporary buffers after
      // each projection just as LinearRunner::forward_prepared does. Stream
      // order keeps the preceding GEMM ahead of the next projection's reuse.
      Workspace::Scope projection_scope(ws);
      const __nv_bfloat16* projected_input = input;
      if (weight.pre_quant_scale != nullptr) {
        __nv_bfloat16* scaled = ws.alloc_n<__nv_bfloat16>(
            static_cast<size_t>(count) * weight.in_features);
        cuda::launch_pre_quant_scale(projected_input, weight.pre_quant_scale,
                                     scaled, count, weight.in_features,
                                     stream.get());
        projected_input = scaled;
      }
      if (weight.convrot && weight.convrot_group > 0 &&
          weight.in_features % weight.convrot_group == 0) {
        __nv_bfloat16* rotated = ws.alloc_n<__nv_bfloat16>(
            static_cast<size_t>(count) * weight.in_features);
        cuda::launch_convrot(projected_input, rotated, count,
                             weight.in_features, weight.convrot_group,
                             stream.get());
        projected_input = rotated;
      }
      const uint32_t tiled = static_cast<uint32_t>(count) / 64 * 64;
      if (tiled != 0) {
        cuda::launch_deterministic_bf16_gemm_nt(
            projected_input, dense, nullptr, output, tiled,
            static_cast<uint32_t>(weight.out_features),
            static_cast<uint32_t>(weight.in_features), DenseGemmBias::kNone,
            0, 0, stream.get());
      }
      if (tiled != static_cast<uint32_t>(count)) {
        cuda::launch_deterministic_scalar_gemm_nt(
            projected_input, dense, nullptr, output,
            static_cast<uint32_t>(count) - tiled,
            static_cast<uint32_t>(weight.out_features),
            static_cast<uint32_t>(weight.in_features),
            DenseGemmMode::kBFloat16, DenseGemmBias::kNone,
            tiled, tiled, stream.get());
      }
      lora.apply(adapter_key(weight), input, count, output, blas, stream.get(), true);
    };

    if (carve.chunked_attention) {
      // All keys/values must be formed from the original residual stream.
      // Once they are ready each query chunk can update only its own rows of
      // x: later queries read disjoint rows, so no full Q or output is needed.
      auto normalize = [&](int start, int n) {
        const size_t off = static_cast<size_t>(start) * hidden;
        if (mod_base != nullptr)
          cuda::launch_rmsnorm_modulate(x + off, b.norm1, scale_msa, shift_msa,
              adaln_idx + start, normed, n, hidden, eps, stream.get());
        else
          cuda::launch_rmsnorm(x + off, b.norm1, normed, n, hidden, eps, stream.get());
        prof.tick("attn.norm1", stream.get());
      };
      {
        Workspace::Scope kv_scope(ws);
        const __nv_bfloat16* dk = linear.prepare(b.wk, ws);
        const __nv_bfloat16* dv = linear.prepare(b.wv, ws);
        prof.tick("attn.dequant", stream.get());
        for (int start = 0; start < rows; start += chunk) {
          const int n = std::min(chunk, rows - start);
          normalize(start, n);
          const size_t off = static_cast<size_t>(start) * inner;
          project(b.wk, dk, normed, n, k + off);
          project(b.wv, dv, normed, n, v + off);
          prof.tick("attn.qkv_proj", stream.get());
        }
      }
      cuda::launch_head_rmsnorm(k, b.k_norm, rows, cfg.num_attention_heads,
                                cfg.attention_head_dim, eps, stream.get());
      if (cos != nullptr)
        cuda::launch_rope_h3(k, cos, sin, rows, cfg.num_attention_heads,
                             cfg.attention_head_dim, stream.get());
      prof.tick("attn.qknorm_rope", stream.get());

      AttentionConfig acfg;
      acfg.seq_len = rows;
      acfg.num_heads = cfg.num_attention_heads;
      acfg.head_dim = cfg.attention_head_dim;
      acfg.band_ranges = layer >= 0 && d_band.size() > 0 ? d_band.get() : nullptr;
      {
        Workspace::Scope query_scope(ws);
        const __nv_bfloat16* dq = linear.prepare(b.wq, ws);
        const __nv_bfloat16* dout = linear.prepare(b.out_proj, ws);
        prof.tick("attn.dequant", stream.get());
        for (int start = 0; start < rows; start += chunk) {
          const int n = std::min(chunk, rows - start);
          const size_t off = static_cast<size_t>(start) * hidden;
          normalize(start, n);
          project(b.wq, dq, normed, n, q);
          prof.tick("attn.qkv_proj", stream.get());
          cuda::launch_head_rmsnorm(q, b.q_norm, n, cfg.num_attention_heads,
                                    cfg.attention_head_dim, eps, stream.get());
          if (cos != nullptr) {
            // H3 rotary tables store the 96 rotated channels per row.
            const size_t rope_off = static_cast<size_t>(start) * 96;
            cuda::launch_rope_h3(q, cos + rope_off, sin + rope_off, n,
                                 cfg.num_attention_heads, cfg.attention_head_dim, stream.get());
          }
          prof.tick("attn.qknorm_rope", stream.get());
          cuda::attention_forward_query_chunk(stream.get(), q, k, v, attn_out,
                                               acfg, start, n);
          diagnose("attention", attn_out, static_cast<size_t>(n) * inner, layer);
          prof.tick("attn.flash2", stream.get());
          project(b.out_proj, dout, attn_out, n, branch);
          diagnose("out_proj", branch, static_cast<size_t>(n) * hidden, layer);
          prof.tick("attn.out_proj", stream.get());
          if (mod_base != nullptr)
            cuda::launch_add_gated(x + off, branch, gate_msa, adaln_idx + start,
                                   n, hidden, stream.get());
          else
            cuda::launch_add_rows_bf16(x + off, branch, static_cast<size_t>(n) * hidden,
                                       stream.get());
          prof.tick("attn.residual", stream.get());
          diagnose("attention_residual", x + off, static_cast<size_t>(n) * hidden, layer);
        }
      }
      if (layer < 0) ++attention_routes.generic_refiner;
      else ++attention_routes.generic_main;
    } else {
      // Dequantised once per block, not once per row-chunk. The dense copy of a
      // weight does not depend on which rows are being projected, so re-deriving
      // it inside the loop was the largest single piece of redundant memory
      // traffic in a step: 771 MB per block per chunk, at ten chunks and fifty
      // blocks. The scope holds them until the loop ends; `compute_carve` sizes
      // the arena for the three thirds at once, which is exactly the fused
      // qkv_proj it already reserved for.
      {
        Workspace::Scope qkv_scope(ws);
        const __nv_bfloat16* dq = linear.prepare(b.wq, ws);
        const __nv_bfloat16* dk = linear.prepare(b.wk, ws);
        const __nv_bfloat16* dv = linear.prepare(b.wv, ws);
        // Its own phase. Billed to `attn.norm1` it would look like a norm that
        // got slower when the dequantisation moved out of the loop, which is the
        // opposite of what happened.
        prof.tick("attn.dequant", stream.get());
        for (int start = 0; start < rows; start += chunk) {
          const int n = std::min(chunk, rows - start);
          const size_t off = static_cast<size_t>(start) * hidden;
          if (mod_base != nullptr) {
            cuda::launch_rmsnorm_modulate(x + off, b.norm1, scale_msa, shift_msa, adaln_idx + start,
                                          normed, n, hidden, eps, stream.get());
          } else {
            cuda::launch_rmsnorm(x + off, b.norm1, normed, n, hidden, eps, stream.get());
          }
          prof.tick("attn.norm1", stream.get());
          // Three GEMMs against contiguous thirds of `qkv_proj` rather than one
          // fused GEMM plus a split: identical arithmetic, and it writes straight
          // into the full-sequence q/k/v without a [chunk, 21504] staging buffer.
          const size_t qoff = static_cast<size_t>(start) * inner;
          project(b.wq, dq, normed, n, q + qoff);
          project(b.wk, dk, normed, n, k + qoff);
          project(b.wv, dv, normed, n, v + qoff);
          prof.tick("attn.qkv_proj", stream.get());
        }
      }

      // QK-norm over the 128-wide head dimension, then RoPE — in that order
      // (spec 4.3). `v` is not normalised and never rotated.
      cuda::launch_head_rmsnorm(q, b.q_norm, rows, cfg.num_attention_heads, cfg.attention_head_dim,
                                eps, stream.get());
      cuda::launch_head_rmsnorm(k, b.k_norm, rows, cfg.num_attention_heads, cfg.attention_head_dim,
                                eps, stream.get());
      if (cos != nullptr) {
        cuda::launch_rope_h3(q, cos, sin, rows, cfg.num_attention_heads, cfg.attention_head_dim,
                             stream.get());
        cuda::launch_rope_h3(k, cos, sin, rows, cfg.num_attention_heads, cfg.attention_head_dim,
                             stream.get());
      }
      if (layer >= 0) capture_block_qkv(q, k, v);
      prof.tick("attn.qknorm_rope", stream.get());

      AttentionConfig acfg;
      acfg.seq_len = rows;
      acfg.num_heads = cfg.num_attention_heads;
      acfg.head_dim = cfg.attention_head_dim;
      // The label follows the backend that actually ran. It used to say
      // "attn.fused" unconditionally, so the profile could not distinguish the
      // fused path from a fallback to the blocked one — only the magnitudes
      // could, which is not a check, it is a reader noticing.
      AttentionBackend backend = AttentionBackend::kFused;
      if (block_attention_mode == AttentionMode::kNone) backend = AttentionBackend::kBlocked;
      if (block_attention_mode == AttentionMode::kSage2) backend = AttentionBackend::kSage2;
      // Released H3 policy: dense for the first ten denoiser evaluations and
      // for blocks 0 and 1 on every later evaluation. Refiner layer=-1 is dense.
      if (is_sol_attention(block_attention_mode))
        backend = AttentionBackend::kSol;
      // Empty unless this request asked for a band, so the default path hands the
      // kernel a null pointer and gets the unbanded instantiation.
      // The text refiner is always full attention. On a reused model `d_band`
      // may still hold the preceding main sequence's table, so key this on the
      // block kind as well as buffer presence rather than relying on first-run
      // allocation order.
      acfg.band_ranges = layer >= 0 && d_band.size() > 0 ? d_band.get() : nullptr;
      if (backend == AttentionBackend::kSol && rows == layout.total_rows()) {
        acfg.exact_prefix = layout.video_start();
        acfg.sol_beta = sol_schedule.beta;
        acfg.sol_error_k=sol_schedule.error_k;
        acfg.sol_error_v=sol_schedule.error_v;
        acfg.sol_pipeline = block_attention_mode == AttentionMode::kSolExperimental ||
                            sol_pipeline_diag;
      }
      if (layer >= 0 && !sol_capture_path.empty()) capture_sol_inputs(q, k, v, rows, layer);
      if (layer >= 0 && is_vsa()) {
        cuda::vsa_attention_forward(stream.get(), q, k, v, attn_out,
                                    vsa_compressed.get(), vsa_config, ws);
        ++attention_routes.generic_main;
      } else if (block_attention_mode == AttentionMode::kExact) {
        cuda::launch_deterministic_h3_attention(
            stream.get(), q, k, v, attn_out, acfg.band_ranges,
            static_cast<uint32_t>(rows), static_cast<uint32_t>(cfg.num_attention_heads),
            static_cast<uint32_t>(cfg.attention_head_dim), acfg.effective_scale());
        if (layer < 0) ++attention_routes.exact_refiner_full;
        else if (acfg.band_ranges == nullptr) ++attention_routes.exact_main_full;
        else ++attention_routes.exact_main_banded;
      } else {
        const auto attention_plan = cuda::AttentionPlan::compile(acfg, cfg.num_attention_heads, backend);
        attention_plan.forward(blas, stream.get(), q, k, v, attn_out, ws);
        if (layer < 0) ++attention_routes.generic_refiner;
        else ++attention_routes.generic_main;
      }
      if (layer >= 0) capture_block_attention(attn_out);
      diagnose("attention",attn_out,size_t(rows)*inner,layer);
      const char* label = layer >= 0 && is_vsa() ? "attn.vsa-h3" : block_attention_mode == AttentionMode::kExact ? "attn.exact" :
                          backend == AttentionBackend::kFused ? "attn.flash2" :
                          backend == AttentionBackend::kSage2 ? "attn.sage2" :
                          backend == AttentionBackend::kSol ?
                            (block_attention_mode==AttentionMode::kSolExperimental?
                              "attn.sol.experimental":"attn.sol") : "attn.none";
      prof.tick(label, stream.get());

      if (layer >= 0 && is_vsa()) {
        Workspace::Scope gate_scope(ws);
        const __nv_bfloat16* dense_gate = linear.prepare(b.compress_gate, ws);
        for (int start = 0; start < rows; start += chunk) {
          const int n = std::min(chunk, rows - start);
          cuda::launch_rmsnorm_modulate(x + size_t(start) * hidden, b.norm1,
              scale_msa, shift_msa, adaln_idx + start, normed, n, hidden, eps, stream.get());
          // The fused FFN buffer is idle here and large enough for the gate.
          project(b.compress_gate, dense_gate, normed, n, fused);
          cuda::vsa_add_compression(stream.get(), attn_out + size_t(start) * inner,
              fused, vsa_compressed.get(), start, n, vsa_config);
        }
        prof.tick("attn.vsa_gate", stream.get());
      }

      {
        Workspace::Scope out_scope(ws);
        const __nv_bfloat16* dout = linear.prepare(b.out_proj, ws);
        prof.tick("attn.dequant", stream.get());
        for (int start = 0; start < rows; start += chunk) {
          const int n = std::min(chunk, rows - start);
          const size_t off = static_cast<size_t>(start) * hidden;
          project(b.out_proj, dout,
                  attn_out + static_cast<size_t>(start) * inner, n, branch);
          diagnose("out_proj",branch,size_t(n)*hidden,layer);
          prof.tick("attn.out_proj", stream.get());
          if (mod_base != nullptr) {
            cuda::launch_add_gated(x + off, branch, gate_msa, adaln_idx + start, n, hidden,
                                   stream.get());
          } else {
            cuda::launch_add_rows_bf16(x + off, branch, static_cast<size_t>(n) * hidden,
                                       stream.get());
          }
          prof.tick("attn.residual", stream.get());
          diagnose("attention_residual",x+off,size_t(n)*hidden,layer);
        }
      }
    }
    if (layer >= 0) capture_block_attention_residual(x);
    emit_stage("attn", x, rows, hidden);

    {
      // Both FFN weights stay live across the loop, which is why the arena has
      // to hold fc1 and fc2 at once rather than the larger of the two.
      Workspace::Scope mlp_scope(ws);
      const __nv_bfloat16* d1 = linear.prepare(b.fc1, ws);
      const __nv_bfloat16* d2 = linear.prepare(b.fc2, ws);
      prof.tick("mlp.dequant", stream.get());
      for (int start = 0; start < rows; start += chunk) {
        const int n = std::min(chunk, rows - start);
        const size_t off = static_cast<size_t>(start) * hidden;
        if (mod_base != nullptr) {
          cuda::launch_rmsnorm_modulate(x + off, b.norm2, scale_mlp, shift_mlp, adaln_idx + start,
                                        normed, n, hidden, eps, stream.get());
        } else {
          cuda::launch_rmsnorm(x + off, b.norm2, normed, n, hidden, eps, stream.get());
        }
        prof.tick("mlp.norm2", stream.get());
        diagnose("adaln_mlp",normed,size_t(n)*hidden,layer);
        project(b.fc1, d1, normed, n, fused);
        prof.tick("mlp.fc1", stream.get());
        // Gate first: our checkpoints use the original `mlp.fc1` naming, whose
        // first half goes through the SiLU (spec 4.4).
        if (block_attention_mode == AttentionMode::kExact) {
          cuda::launch_swiglu_exact(fused, act, n, cfg.ffn_dim, stream.get());
        } else {
          cuda::launch_swiglu(fused, act, n, cfg.ffn_dim, stream.get());
        }
        prof.tick("mlp.swiglu", stream.get());
        project(b.fc2, d2, act, n, branch);
        diagnose("mlp",branch,size_t(n)*hidden,layer);
        prof.tick("mlp.fc2", stream.get());
        if (mod_base != nullptr) {
          cuda::launch_add_gated(x + off, branch, gate_mlp, adaln_idx + start, n, hidden,
                                 stream.get());
        } else {
          cuda::launch_add_rows_bf16(x + off, branch, static_cast<size_t>(n) * hidden,
                                     stream.get());
        }
        prof.tick("mlp.residual", stream.get());
        diagnose("mlp_residual",x+off,size_t(n)*hidden,layer);
      }
    }
    if (layer >= 0) {
      finish_block_capture(x);
      capture_graph_boundary(x, layer);
    }
    emit_stage("ffn", x, rows, hidden);
  }
}  // namespace slopfab::dit
