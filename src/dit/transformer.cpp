// H3 transformer lifecycle and forward orchestration.
#include "transformer_impl.h"

namespace slopfab::dit {



// ---------------------------------------------------------------------------

Transformer::Transformer() : impl_(new Impl()) {}
Transformer::~Transformer() = default;

const TransformerConfig& Transformer::config() const { return impl_->cfg; }
size_t Transformer::weight_bytes() const {
  return impl_->arena_bytes + impl_->lora.weight_bytes() +
         (impl_->streamer ? impl_->streamer->device_bytes() : 0);
}
size_t Transformer::offloaded_blocks() const { return impl_->streamer ? impl_->streamer->count : 0; }
size_t Transformer::offloaded_host_bytes() const { return impl_->streamer ? impl_->streamer->host_bytes : 0; }
void Transformer::set_adaln_lookup(AdaLNLookup mode) { impl_->lookup = mode; }
AdaLNLookup Transformer::adaln_lookup() const { return impl_->lookup; }

void Transformer::set_attention_band(int frames) {
  const int requested = frames > 0 ? frames : 0;
  if (impl_->attention_configuration_locked && requested != impl_->attn_band) {
    throw std::runtime_error(
        "transformer: attention band cannot change after attention preparation");
  }
  impl_->attn_band = requested;
}
int Transformer::attention_band() const { return impl_->attn_band; }
void Transformer::set_attention_mode(AttentionMode mode) {
  if (!attention_mode_supported(DeviceBackend::kCuda, mode)) {
    throw std::invalid_argument("transformer: invalid CUDA attention mode");
  }
  if (impl_->attention_configuration_locked && mode != impl_->attention_mode) {
    throw std::runtime_error(
        "transformer: attention mode cannot change after attention preparation");
  }
  if (mode == AttentionMode::kExact && !cuda::deterministic_h3_attention_available()) {
    throw std::runtime_error(
        "transformer: exact attention is unavailable on this CUDA tuple");
  }
  impl_->attention_mode = mode;
}
AttentionMode Transformer::attention_mode() const { return impl_->attention_mode; }

void Transformer::set_row_chunk(int rows) {
  if (rows <= 0) throw std::invalid_argument("transformer: row chunk must be positive");
  if (impl_->attention_configuration_locked && rows != impl_->row_chunk)
    throw std::runtime_error("transformer: set row chunk before preparation");
  impl_->row_chunk = rows;
}

void Transformer::set_query_chunking(bool enabled) {
  if (impl_->attention_configuration_locked && enabled != impl_->query_chunking)
    throw std::runtime_error("transformer: set query chunking before preparation");
  impl_->query_chunking = enabled;
}

size_t Transformer::workspace_bytes() const { return impl_->ws.capacity(); }
void Transformer::set_sol_schedule(const SolSchedule& schedule) { impl_->sol_schedule=schedule; }
void Transformer::set_denoise_step(int step) { impl_->denoise_step = step; }

void Transformer::set_block_cache(const BlockCacheConfig& config, int num_steps) {
  Impl& s = *impl_;
  s.require_loaded("set_block_cache");
  s.block_cache = BlockCache(config, num_steps);
  // Resolved against the loaded stack, so this needs `load` to have run. The
  // number of blocks is the one input the config cannot supply and getting it
  // from `cfg.num_layers` instead would resolve a span against the *configured*
  // depth, which the ref2va architectures do not share.
  s.bc_span = resolve_block_span(config, static_cast<int>(s.blocks.size()));
  s.bc_have_delta = false;
  // The forward pass skips the span by assigning `bc_span.end - 1` to a
  // `size_t` induction variable, so an `end` of 0 would wrap to SIZE_MAX and
  // the `++` would land on 0 — an infinite loop rather than a fault.
  // `resolve_block_span` cannot return that today, because `valid()` requires
  // `end > begin >= 0`. Asserted here, where it costs nothing, because the
  // guard that makes it true is fourteen lines away from the arithmetic that
  // depends on it.
  if (s.bc_span.valid() && s.bc_span.end < 1) {
    throw std::runtime_error("transformer: resolved block cache span ends before block 1");
  }
  if (!s.block_cache.enabled() || !s.bc_span.valid()) {
    // Enabling and then disabling between requests must not leave 844 MB of
    // device memory pinned for a feature that is off.
    s.bc_delta.reset();
  }
}

const BlockCacheConfig& Transformer::block_cache_config() const {
  return impl_->block_cache.config();
}

BlockSpan Transformer::block_cache_span() const { return impl_->bc_span; }

int Transformer::num_blocks() const { return static_cast<int>(impl_->blocks.size()); }

int Transformer::block_cache_computed() const { return impl_->block_cache.computed(); }

int Transformer::block_cache_reused() const { return impl_->block_cache.reused(); }
std::array<float, AdaLNTable::kRank> Transformer::adaln_code(float t) const {
  if (!is_pruned_table_architecture(impl_->architecture)) {
    throw std::runtime_error("transformer: rank-8 adaln_code is unavailable for full-AdaLN architecture");
  }
  return impl_->table.lookup(t, impl_->lookup);
}

void Transformer::unload() {
  impl_->streamer.reset();
  impl_->lora = cuda::LoraRunner();
  impl_->blocks.clear();
  impl_->refiner.clear();
  impl_->arena.reset();
  impl_->arena_bytes = 0;
  impl_->has_sequence = false;
  impl_->attention_configuration_locked = false;
  impl_->attention_routes = {};
  impl_->vsa_rows.reset(); impl_->vsa_sizes.reset(); impl_->vsa_row_tiles.reset();
  impl_->vsa_compressed.reset(); impl_->vsa_config = {};
}

size_t Transformer::activation_bytes(const SequenceLayout& layout) const {
  const TransformerConfig& cfg = impl_->cfg;
  const Carve c = plan_carve(cfg, layout, impl_->row_chunk, impl_->attention_mode,
                             stack_uses_convrot(impl_->blocks),
                             impl_->compact_queries(impl_->attention_mode), impl_->is_vsa());
  const int seq = layout.total_rows();

  size_t total = c.total + impl_->lora.scratch_bytes();
  if (impl_->is_vsa()) {
    const auto tiles = build_vsa_tiles(layout);
    total += align_up(tiles.sizes.size() * cfg.inner_dim() * sizeof(__nv_bfloat16));
    total += align_up(tiles.rows.size() * sizeof(int32_t)) +
             align_up(tiles.sizes.size() * sizeof(int32_t)) +
             align_up(tiles.row_tiles.size() * sizeof(int32_t));
  }
  if (impl_->architecture == TransformerArchitecture::kRef2VAFullAdaLN &&
      !impl_->blocks.empty()) {
    size_t full_scratch = cuda::linear_workspace_bytes(
        impl_->blocks.front().full_adaln, 2, ComputeType::kF32);
    full_scratch = std::max(full_scratch, cuda::linear_workspace_bytes(
        impl_->final_full_adaln, 2, ComputeType::kF32));
    if (full_scratch > c.scratch) total += align_up(full_scratch) - align_up(c.scratch);
  }
  total += align_up(static_cast<size_t>(seq) * cfg.hidden_size * sizeof(__nv_bfloat16));  // hidden
  // The block cache's delta, same shape as `hidden`: 844 MB at the production
  // geometry.
  //
  // Load-time residency planning adds this separately when the requested
  // cache has not yet been configured on the loaded model.
  if (impl_->block_cache.enabled() && impl_->bc_span.valid()) {
    total += align_up(static_cast<size_t>(seq) * cfg.hidden_size * sizeof(__nv_bfloat16));
  }
  total += 2 * align_up(static_cast<size_t>(seq) * 96 * sizeof(float));                   // rope
  total += 3 * align_up(static_cast<size_t>(seq) * sizeof(int32_t));                      // indices
  // Modulation at the t2va worst case of two distinct timesteps (spec 7.5).
  total += align_up(static_cast<size_t>(cfg.num_layers) * kNumParams * 2 * kNumModalities *
                    cfg.hidden_size * sizeof(float));
  const size_t video_rows = static_cast<size_t>(layout.num_condition_video) + layout.num_video_rows;
  total += 2 * align_up(video_rows * cfg.video_patch_dim() * sizeof(float));
  total += 2 * align_up(static_cast<size_t>(layout.num_audio_rows) * cfg.audio_in_channels *
                        sizeof(float));
  return total;
}

size_t Transformer::debug_attention_scratch_bytes(const SequenceLayout& layout) const {
  return plan_carve(impl_->cfg, layout, impl_->row_chunk, impl_->attention_mode).attention_scratch;
}

Transformer::DebugAttentionRoutes Transformer::debug_attention_routes() const {
  return impl_->attention_routes;
}

std::vector<float> Transformer::debug_modulation(int block_index,
                                                 const std::vector<float>& timesteps) {
  Impl& s = *impl_;
  s.require_loaded("debug_modulation");
  if (block_index < 0 || block_index >= static_cast<int>(s.blocks.size())) {
    throw std::runtime_error("transformer: block index " + std::to_string(block_index) +
                             " out of range");
  }
  s.build_modulation(timesteps);
  const size_t per_block = s.block_mod_stride();
  std::vector<float> out(per_block);
  SLOPFAB_CUDA_CHECK(cudaMemcpyAsync(out.data(), s.mod.get() + block_index * per_block,
                                    per_block * sizeof(float), cudaMemcpyDeviceToHost,
                                    s.stream.get()));
  SLOPFAB_CUDA_CHECK(cudaStreamSynchronize(s.stream.get()));
  return out;
}

std::vector<Transformer::DebugStage> Transformer::debug_text_stages(const float* prompt_embeds,
                                                                   int num_tokens) {
  Impl& s = *impl_;
  std::vector<DebugStage> stages;
  s.stage_hook = [&](const char* label, const __nv_bfloat16* x, int rows, int dim) {
    DebugStage stage;
    stage.label = label;
    stage.rows = rows;
    stage.dim = dim;
    const size_t n = static_cast<size_t>(rows) * dim;
    std::vector<uint16_t> bits(n);
    SLOPFAB_CUDA_CHECK(
        cudaMemcpy(bits.data(), x, n * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    stage.data.resize(n);
    for (size_t i = 0; i < n; ++i) stage.data[i] = bf16_to_f32(bits[i]);
    stages.push_back(std::move(stage));
  };
  try {
    prepare_text(prompt_embeds, num_tokens);
  } catch (...) {
    s.stage_hook = nullptr;
    throw;
  }
  s.stage_hook = nullptr;
  return stages;
}

std::vector<float> Transformer::debug_text_cache() const {
  Impl& s = *impl_;
  if (s.num_text == 0 || s.text_cache.size() == 0) return {};
  std::vector<uint16_t> bits(s.text_cache.size());
  SLOPFAB_CUDA_CHECK(cudaMemcpy(bits.data(), s.text_cache.get(),
                               bits.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost));
  std::vector<float> out(bits.size());
  for (size_t i = 0; i < bits.size(); ++i) out[i] = bf16_to_f32(bits[i]);
  return out;
}

// ---------------------------------------------------------------------------

void Transformer::forward(const float* video_latents, const float* audio_latents,
                          const RowTimesteps& row_timesteps, float* video_velocity,
                          float* audio_velocity) {
  cuda::StageMemorySpan memory("denoising.forward");
  Impl& s = *impl_;
  s.require_loaded("forward");
  if (!s.has_sequence) throw std::runtime_error("transformer: forward before prepare_sequence");

  const TransformerConfig& cfg = s.cfg;
  const int seq = s.layout.total_rows();
  const int hidden = cfg.hidden_size;
  const int patch = cfg.video_patch_dim();
  const int audio_dim = cfg.audio_in_channels;
  const int chunk = s.carve.chunk;
  const int video_rows = static_cast<int>(s.indices.video.size());
  const int audio_rows = static_cast<int>(s.indices.audio.size());

  if (static_cast<int>(row_timesteps.adaln.size()) != seq) {
    throw std::runtime_error("transformer: row timesteps do not cover the sequence");
  }

  cuda::StepProfiler& prof = cuda::StepProfiler::instance();
  const std::chrono::steady_clock::time_point t_enter = std::chrono::steady_clock::now();
  prof.begin_step(s.stream.get());
  if (s.streamer) s.streamer->prefetch(s.streamer->first);

  // Modulation for this step's distinct timesteps, and the per-row indices
  // into it. `torch.unique(sorted=True)` sorts ascending, so which of the video
  // and audio timesteps is index 0 flips over the schedule (spec 7.5) — the
  // indices have to be re-uploaded every step, not cached.
  s.build_modulation(row_timesteps.unique);
  prof.tick("mod.expand", s.stream.get());
  s.d_adaln.copy_from_host(row_timesteps.adaln.data(), row_timesteps.adaln.size(),
                           s.stream.get());
  // The final layer selects on `timestep_indices` alone, with no modality
  // dependence (spec 3.2), so each head needs its own rows' timestep index in
  // gathered order.
  s.host_ts.assign(static_cast<size_t>(video_rows + audio_rows), 0);
  for (int i = 0; i < video_rows; ++i) {
    s.host_ts[static_cast<size_t>(i)] =
        row_timesteps.indices[static_cast<size_t>(s.indices.video[i])];
  }
  for (int i = 0; i < audio_rows; ++i) {
    s.host_ts[static_cast<size_t>(video_rows + i)] =
        row_timesteps.indices[static_cast<size_t>(s.indices.audio[i])];
  }
  if (video_rows > 0) {
    s.d_ts_video.copy_from_host(s.host_ts.data(), video_rows, s.stream.get());
  }
  if (audio_rows > 0) {
    s.d_ts_audio.copy_from_host(s.host_ts.data() + video_rows, audio_rows, s.stream.get());
  }
  prof.tick("mod.index_h2d", s.stream.get());
  s.begin_transformer_forward_capture(video_latents, audio_latents,
                                      video_rows, audio_rows);

  s.ws.clear();
  Workspace& ws = s.ws;
  const size_t qkv_n = s.carve.qkv;
  __nv_bfloat16* q = ws.alloc_n<__nv_bfloat16>(s.carve.query);
  __nv_bfloat16* k = ws.alloc_n<__nv_bfloat16>(qkv_n);
  __nv_bfloat16* v = ws.alloc_n<__nv_bfloat16>(qkv_n);
  __nv_bfloat16* attn_out = ws.alloc_n<__nv_bfloat16>(s.carve.query);
  __nv_bfloat16* normed = ws.alloc_n<__nv_bfloat16>(s.carve.normed);
  __nv_bfloat16* fused = ws.alloc_n<__nv_bfloat16>(s.carve.fused);
  __nv_bfloat16* act = ws.alloc_n<__nv_bfloat16>(s.carve.act);
  __nv_bfloat16* branch = ws.alloc_n<__nv_bfloat16>(s.carve.normed);
  float* fa = ws.alloc_n<float>(s.carve.fbuf);
  float* fb = ws.alloc_n<float>(s.carve.fbuf);

  __nv_bfloat16* x = s.hidden.get();
  // Both index builders partition [0, S) exactly — `build_indices` by
  // construction from the layout's half-open regions, `build_ref2va_packed_
  // sequence` by walking one cursor from 0 to S — so the scatters below cover
  // every row and the zero is dead stores. At the production geometry it is a
  // 844 MB memset per step.
  //
  // Kept, behind SLOPFAB_TENSOR_DIAG, as insurance against a layout that ever
  // stops being a permutation. Note what it does and does not buy: `hidden` is
  // allocated once in `prepare_sequence` and reused, so from the second step an
  // uncovered row would hold the previous step's residual — finite, plausible,
  // and invisible to a non-finite check. Zeroing makes such a row read as zeros
  // instead, which is what a bisect can actually see. The safety here rests on
  // the partition, not on the zero.
  if (s.tensor_diag) {
    s.hidden.zero(s.stream.get());
    prof.tick("hidden.zero", s.stream.get());
  }

  if (video_rows > 0) {
    s.d_video_rows.copy_from_host(video_latents, static_cast<size_t>(video_rows) * patch,
                                  s.stream.get());
  }
  if (audio_rows > 0) {
    s.d_audio_rows.copy_from_host(audio_latents, static_cast<size_t>(audio_rows) * audio_dim,
                                  s.stream.get());
  }
  prof.tick("latent_h2d", s.stream.get());

  // proj_in / audio_proj_in run in fp32: they are fp32 tensors in the
  // checkpoint and the reference aligns the activation with the parameter
  // dtype at each of them (spec 9.1).
  auto project_in = [&](const QuantWeight& w, const float* src, int in_dim, int rows,
                        const int32_t* index) {
    for (int start = 0; start < rows; start += chunk) {
      const int n = std::min(chunk, rows - start);
      if (s.attention_mode == AttentionMode::kExact) {
        if (w.format != QuantFormat::kF32 || w.bias_format != QuantFormat::kF32 ||
            w.pre_quant_scale != nullptr || w.convrot) {
          throw std::runtime_error(
              "transformer: exact input projection requires plain fp32 weight and bias");
        }
        cuda::launch_deterministic_scalar_gemm_nt(
            src, w.data, w.bias, fa, static_cast<uint32_t>(n),
            static_cast<uint32_t>(hidden), static_cast<uint32_t>(in_dim),
            DenseGemmMode::kFloat32, DenseGemmBias::kFloat32,
            static_cast<uint32_t>(start), 0, s.stream.get());
      } else {
        s.linear.forward_f32(w, src + static_cast<size_t>(start) * in_dim, n, fa, ws);
      }
      cuda::launch_narrow_to_bf16(fa, normed, static_cast<size_t>(n) * hidden, s.stream.get());
      cuda::launch_scatter_rows(normed, index + start, x, n, hidden, s.stream.get());
    }
  };
  project_in(s.video_in, s.d_video_rows.get(), patch, video_rows, s.d_video_idx.get());
  project_in(s.audio_in, s.d_audio_rows.get(), audio_dim, audio_rows, s.d_audio_idx.get());
  if (s.num_text > 0) {
    cuda::launch_scatter_rows(s.text_cache.get(), s.d_text_idx.get(), x, s.num_text, hidden,
                              s.stream.get());
  }
  prof.tick("proj_in", s.stream.get());
  s.capture_transformer_packed_input(x);

  const size_t per_block = s.block_mod_stride();
  const size_t stream_n = static_cast<size_t>(seq) * hidden;
  const bool bc_on = s.block_cache.enabled() && s.bc_span.valid();
  // The ordering requirement — `set_block_cache` before `prepare_sequence` — is
  // otherwise enforced only by comments at the header and the call site.
  // Getting it wrong leaves a null buffer with `bc_on` true, and the way that
  // surfaces depends on which step happens to run first. Checked so it is
  // always the same error, and one that names the cause.
  if (bc_on && s.bc_delta.size() != stream_n) {
    throw std::runtime_error(
        "transformer: block cache buffer does not match the sequence; set_block_cache must be "
        "called before prepare_sequence");
  }
  // Decided once per forward, not per block: `should_compute` counts its calls
  // for reporting, so asking it inside the loop would inflate the tally by the
  // block count and make a reused step look like fifty.
  const bool bc_compute =
      !bc_on || s.block_cache.should_compute(s.denoise_step, s.bc_have_delta);

  for (size_t b = 0; b < s.blocks.size(); ++b) {
    if (bc_on && static_cast<int>(b) == s.bc_span.begin) {
      if (bc_compute) {
        // The stream as it enters the span, parked in the delta buffer until
        // the subtract below turns it into the delta. `run_block` works in
        // place, so without this copy the delta would be `x - x`.
        SLOPFAB_CUDA_CHECK(cudaMemcpyAsync(s.bc_delta.get(), x,
                                          stream_n * sizeof(__nv_bfloat16),
                                          cudaMemcpyDeviceToDevice, s.stream.get()));
        // Cleared for the duration: between here and the capture the buffer
        // holds the "before" state, and a `forward` that threw in the middle of
        // the span would otherwise leave it looking like a usable delta.
        s.bc_have_delta = false;
        prof.tick("block_cache.snapshot", s.stream.get());
      } else {
        // The reuse, and the whole point of the feature: the span's blocks are
        // never launched. `bc_have_delta` is guaranteed true here because
        // `should_compute` forces a compute while it is false.
        cuda::launch_add_bf16(x, s.bc_delta.get(), stream_n, s.stream.get());
        prof.tick("block_cache.reuse", s.stream.get());
        b = static_cast<size_t>(s.bc_span.end) - 1;  // the ++ lands on `end`
        continue;
      }
    }

    const AttentionMode block_mode =
        is_sol_attention(s.attention_mode) &&
                !s.sol_schedule.active(s.denoise_step,static_cast<int>(b))
            ? AttentionMode::kFlash2
            : s.attention_mode;
    BlockWeights active = s.blocks[b];
    if (s.streamer && s.streamer->contains(b)) {
      s.streamer->acquire(b);
      prof.tick("offload.wait", s.stream.get());
      const auto& host = s.streamer->blocks[b].host;
      active = relocate_block(active, host.get(), host.size(), s.streamer->device(b));
    }
    s.run_block(active, s.mod.get() + b * per_block, seq, x, s.d_adaln.get(),
                s.rope_cos.get(), s.rope_sin.get(), q, k, v, attn_out, normed, fused, act, branch,
                block_mode, static_cast<int>(b));

    if (bc_on && bc_compute && static_cast<int>(b) == s.bc_span.end - 1) {
      // In place over the "before" state: `out` aliases `b`, which the kernel
      // permits and the buffer comment above explains.
      cuda::launch_sub_bf16(x, s.bc_delta.get(), s.bc_delta.get(), stream_n, s.stream.get());
      s.bc_have_delta = true;
      prof.tick("block_cache.capture", s.stream.get());
    }
  }
  s.capture_transformer_main_final(x);

  // Both heads run over every row in the reference and are selected afterwards.
  // Gathering first is mathematically identical, because `norm_out` is per-row,
  // and it saves S * 5376 * 128 flops per step (spec 1.5).
  const size_t final_stride = static_cast<size_t>(s.mod_timesteps) * hidden;
  const float* final_shift = s.final_mod.get();
  const float* final_scale = s.final_mod.get() + final_stride;

  auto run_head = [&](const QuantWeight& w, int out_dim, int rows, const int32_t* index,
                      const int32_t* ts_index, float* dst) {
    for (int start = 0; start < rows; start += chunk) {
      const int n = std::min(chunk, rows - start);
      cuda::launch_gather_rows(x, index + start, normed, n, hidden, s.stream.get());
      cuda::launch_widen_bf16(normed, fa, static_cast<size_t>(n) * hidden, s.stream.get());
      cuda::launch_rmsnorm_modulate_f32(fa, s.final_norm, final_scale, final_shift,
                                        ts_index + start, fb, n, hidden, cfg.norm_eps,
                                        s.stream.get());
      if (s.attention_mode == AttentionMode::kExact) {
        if (w.format != QuantFormat::kF32 || w.bias_format != QuantFormat::kF32 ||
            w.pre_quant_scale != nullptr || w.convrot) {
          throw std::runtime_error(
              "transformer: exact output head requires plain fp32 weight and bias");
        }
        cuda::launch_deterministic_scalar_gemm_nt(
            fb, w.data, w.bias, dst, static_cast<uint32_t>(n),
            static_cast<uint32_t>(out_dim), static_cast<uint32_t>(hidden),
            DenseGemmMode::kFloat32, DenseGemmBias::kFloat32,
            0, static_cast<uint32_t>(start), s.stream.get());
      } else {
        s.linear.forward_f32(w, fb, n, dst + static_cast<size_t>(start) * out_dim, ws);
      }
    }
  };
  run_head(s.video_out, patch, video_rows, s.d_video_idx.get(), s.d_ts_video.get(),
           s.d_video_head.get());
  run_head(s.audio_out, audio_dim, audio_rows, s.d_audio_idx.get(), s.d_ts_audio.get(),
           s.d_audio_head.get());
  prof.tick("final_layer", s.stream.get());

  if (video_rows > 0) {
    s.d_video_head.copy_to_host(video_velocity, static_cast<size_t>(video_rows) * patch,
                                s.stream.get());
  }
  if (audio_rows > 0) {
    s.d_audio_head.copy_to_host(audio_velocity, static_cast<size_t>(audio_rows) * audio_dim,
                                s.stream.get());
  }
  prof.tick("velocity_d2h", s.stream.get());

  // The one synchronise in the step. How much of it is spent blocked here is
  // the whole answer to "was the host ever the bottleneck": if the host had
  // been the slow side it would arrive late and wait for nothing.
  const std::chrono::steady_clock::time_point t_issued = std::chrono::steady_clock::now();
  SLOPFAB_CUDA_CHECK(cudaStreamSynchronize(s.stream.get()));
  s.finish_transformer_capture(video_velocity, audio_velocity);
  const std::chrono::steady_clock::time_point t_exit = std::chrono::steady_clock::now();
  prof.end_step();
  prof.sample_memory();
  prof.add_step_wall(std::chrono::duration<double, std::milli>(t_exit - t_enter).count(),
                     std::chrono::duration<double, std::milli>(t_issued - t_enter).count(),
                     std::chrono::duration<double, std::milli>(t_exit - t_issued).count());
}


}  // namespace slopfab::dit
