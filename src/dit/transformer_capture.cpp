#include "transformer_impl.h"

namespace slopfab::dit {

void Transformer::Impl::diagnose(const char* stage,const __nv_bfloat16* p,size_t n,int layer) {
    if(!tensor_diag)return;
    SLOPFAB_CUDA_CHECK(cudaMemsetAsync(d_tensor_diag.get(),0,sizeof(cuda::TensorScan),stream.get()));
    cuda::launch_tensor_scan(p,n,d_tensor_diag.get(),stream.get());
    cuda::TensorScan h{};d_tensor_diag.copy_to_host(&h,1,stream.get());
    SLOPFAB_CUDA_CHECK(cudaStreamSynchronize(stream.get()));
    float mx=0;std::memcpy(&mx,&h.max_bits,sizeof(mx));
    std::fprintf(stderr,"slopfab tensor step=%d layer=%d stage=%s nonfinite=%llu max=%.7g\n",
                 denoise_step,layer,stage,h.nonfinite,mx);
    if(h.nonfinite)throw std::runtime_error("transformer: first non-finite tensor at "+std::string(stage));
  }

uint64_t Transformer::Impl::fnv64_append(uint64_t hash, const void* data, size_t bytes) {
    const auto* cursor = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < bytes; ++i) {
      hash ^= cursor[i]; hash *= 1099511628211ull;
    }
    return hash;
  }

uint64_t Transformer::Impl::capture_hash(const __nv_bfloat16* data, size_t elements,
                        uint64_t seed) {
    std::vector<uint16_t> host(elements);
    SLOPFAB_CUDA_CHECK(cudaMemcpyAsync(host.data(), data, elements * 2,
                                      cudaMemcpyDeviceToHost, stream.get()));
    SLOPFAB_CUDA_CHECK(cudaStreamSynchronize(stream.get()));
    return fnv64_append(seed, host.data(), host.size() * 2);
  }

void Transformer::Impl::begin_block_capture(const __nv_bfloat16* residual,
                           const int32_t* selectors, const float* cosine,
                           const float* sine, int rows, int layer,
                           AttentionMode mode) {
    block_capture.active = false;
    if (block_capture_done || block_capture_path.empty() ||
        denoise_step != block_capture_step || layer != block_capture_layer)
      return;
    if (mode != AttentionMode::kExact)
      throw std::runtime_error("transformer: H3 block capture requires exact attention");
    if (mod_timesteps <= 0 || host_code.size() !=
        static_cast<size_t>(mod_timesteps) * cfg.adaln_rank)
      throw std::runtime_error("transformer: H3 block capture lacks rank-8 timestep code");
    ActiveBlockCapture next;
    std::memcpy(next.header.magic, "VFH3BLK\0", 8);
    next.header.version = 1;
    next.header.header_bytes = sizeof(H3BlockCaptureHeader);
    next.header.sequence = static_cast<uint32_t>(rows);
    next.header.hidden = static_cast<uint32_t>(cfg.hidden_size);
    next.header.heads = static_cast<uint32_t>(cfg.num_attention_heads);
    next.header.head_dim = static_cast<uint32_t>(cfg.attention_head_dim);
    next.header.ffn = static_cast<uint32_t>(cfg.ffn_dim);
    next.header.timesteps = static_cast<uint32_t>(mod_timesteps);
    next.header.modalities = kNumModalities;
    next.header.adaln_rank = static_cast<uint32_t>(cfg.adaln_rank);
    next.header.denoise_step = denoise_step;
    next.header.layer = layer;
    next.header.range_values = static_cast<uint32_t>(host_band.size());
    next.header.residual_elements = static_cast<uint64_t>(rows) * cfg.hidden_size;
    next.header.qkv_elements = static_cast<uint64_t>(rows) * cfg.inner_dim();
    next.header.rope_elements = static_cast<uint64_t>(rows) * 96;
    next.input.resize(static_cast<size_t>(next.header.residual_elements));
    next.selectors.resize(rows);
    next.code = host_code;
    next.cosine.resize(static_cast<size_t>(next.header.rope_elements));
    next.sine.resize(static_cast<size_t>(next.header.rope_elements));
    next.ranges = host_band;
    SLOPFAB_CUDA_CHECK(cudaMemcpyAsync(next.input.data(), residual,
        next.input.size() * 2, cudaMemcpyDeviceToHost, stream.get()));
    SLOPFAB_CUDA_CHECK(cudaMemcpyAsync(next.selectors.data(), selectors,
        next.selectors.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, stream.get()));
    SLOPFAB_CUDA_CHECK(cudaMemcpyAsync(next.cosine.data(), cosine,
        next.cosine.size() * sizeof(float), cudaMemcpyDeviceToHost, stream.get()));
    SLOPFAB_CUDA_CHECK(cudaMemcpyAsync(next.sine.data(), sine,
        next.sine.size() * sizeof(float), cudaMemcpyDeviceToHost, stream.get()));
    SLOPFAB_CUDA_CHECK(cudaStreamSynchronize(stream.get()));
    next.header.input_fnv64 = fnv64_append(1469598103934665603ull,
        next.input.data(), next.input.size() * 2);
    next.active = true;
    block_capture = std::move(next);
  }

void Transformer::Impl::capture_block_qkv(const __nv_bfloat16* q, const __nv_bfloat16* k,
                         const __nv_bfloat16* v) {
    if (!block_capture.active) return;
    uint64_t hash = capture_hash(q, block_capture.header.qkv_elements);
    hash = capture_hash(k, block_capture.header.qkv_elements, hash);
    block_capture.header.qkv_fnv64 =
        capture_hash(v, block_capture.header.qkv_elements, hash);
  }

void Transformer::Impl::capture_block_attention(const __nv_bfloat16* attention) {
    if (block_capture.active)
      block_capture.header.attention_fnv64 = capture_hash(
          attention, block_capture.header.qkv_elements);
  }

void Transformer::Impl::capture_block_attention_residual(const __nv_bfloat16* residual) {
    if (block_capture.active)
      block_capture.header.attention_residual_fnv64 = capture_hash(
          residual, block_capture.header.residual_elements);
  }

void Transformer::Impl::finish_block_capture(const __nv_bfloat16* residual) {
    if (!block_capture.active) return;
    std::vector<uint16_t> final(
        static_cast<size_t>(block_capture.header.residual_elements));
    SLOPFAB_CUDA_CHECK(cudaMemcpyAsync(final.data(), residual, final.size() * 2,
                                      cudaMemcpyDeviceToHost, stream.get()));
    SLOPFAB_CUDA_CHECK(cudaStreamSynchronize(stream.get()));
    block_capture.header.final_fnv64 = fnv64_append(1469598103934665603ull,
        final.data(), final.size() * 2);
    std::ofstream output(block_capture_path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error(
        "transformer: cannot create H3 block capture: " + block_capture_path);
    output.write(reinterpret_cast<const char*>(&block_capture.header),
                 sizeof(block_capture.header));
    auto write = [&](const auto& values) {
      output.write(reinterpret_cast<const char*>(values.data()),
          static_cast<std::streamsize>(values.size() * sizeof(values[0])));
    };
    write(block_capture.input); write(block_capture.selectors);
    write(block_capture.code); write(block_capture.cosine);
    write(block_capture.sine); write(block_capture.ranges); write(final);
    if (!output) throw std::runtime_error(
        "transformer: failed writing H3 block capture: " + block_capture_path);
    block_capture.active = false; block_capture_done = true;
    std::fprintf(stderr, "slopfab: captured exact H3 block step %d layer %d to %s\n",
                 denoise_step, block_capture_layer, block_capture_path.c_str());
    const char* stop = std::getenv("SLOPFAB_H3_BLOCK_CAPTURE_EXIT");
    if (stop && stop[0] == '1')
      throw std::runtime_error("transformer: stopped after requested H3 block capture");
  }

void Transformer::Impl::begin_graph_capture(const __nv_bfloat16* residual,
                           const int32_t* selectors, const float* cosine,
                           const float* sine, int rows, int layer,
                           AttentionMode mode) {
    if (graph_capture_done || graph_capture_path.empty() ||
        denoise_step != graph_capture_step || layer != 0) return;
    if (mode != AttentionMode::kExact || blocks.size() != kH3MainCaptureLayers)
      throw std::runtime_error(
          "transformer: H3 graph capture requires exact 50-layer main stack");
    if (mod_timesteps <= 0 || host_code.size() !=
        static_cast<size_t>(mod_timesteps) * cfg.adaln_rank)
      throw std::runtime_error(
          "transformer: H3 graph capture lacks rank-8 timestep code");
    ActiveGraphCapture next;
    std::memcpy(next.header.magic, "VFH3GRF\0", 8);
    next.header.version = 1;
    next.header.header_bytes = sizeof(H3MainGraphCaptureHeader);
    next.header.sequence = static_cast<uint32_t>(rows);
    next.header.hidden = static_cast<uint32_t>(cfg.hidden_size);
    next.header.heads = static_cast<uint32_t>(cfg.num_attention_heads);
    next.header.head_dim = static_cast<uint32_t>(cfg.attention_head_dim);
    next.header.ffn = static_cast<uint32_t>(cfg.ffn_dim);
    next.header.timesteps = static_cast<uint32_t>(mod_timesteps);
    next.header.modalities = kNumModalities;
    next.header.adaln_rank = static_cast<uint32_t>(cfg.adaln_rank);
    next.header.layers = static_cast<uint32_t>(blocks.size());
    next.header.range_values = static_cast<uint32_t>(host_band.size());
    next.header.denoise_step = denoise_step;
    next.header.residual_elements = static_cast<uint64_t>(rows) * cfg.hidden_size;
    next.header.rope_elements = static_cast<uint64_t>(rows) * 96;
    next.input.resize(static_cast<size_t>(next.header.residual_elements));
    next.selectors.resize(rows);
    next.code = host_code;
    next.cosine.resize(static_cast<size_t>(next.header.rope_elements));
    next.sine.resize(static_cast<size_t>(next.header.rope_elements));
    next.ranges = host_band;
    SLOPFAB_CUDA_CHECK(cudaMemcpyAsync(next.input.data(), residual,
        next.input.size() * 2, cudaMemcpyDeviceToHost, stream.get()));
    SLOPFAB_CUDA_CHECK(cudaMemcpyAsync(next.selectors.data(), selectors,
        next.selectors.size() * sizeof(int32_t), cudaMemcpyDeviceToHost,
        stream.get()));
    SLOPFAB_CUDA_CHECK(cudaMemcpyAsync(next.cosine.data(), cosine,
        next.cosine.size() * sizeof(float), cudaMemcpyDeviceToHost,
        stream.get()));
    SLOPFAB_CUDA_CHECK(cudaMemcpyAsync(next.sine.data(), sine,
        next.sine.size() * sizeof(float), cudaMemcpyDeviceToHost,
        stream.get()));
    SLOPFAB_CUDA_CHECK(cudaStreamSynchronize(stream.get()));
    next.header.input_fnv64 = fnv64_append(1469598103934665603ull,
        next.input.data(), next.input.size() * 2);
    next.active = true;
    graph_capture = std::move(next);
  }

void Transformer::Impl::capture_graph_boundary(const __nv_bfloat16* residual, int layer) {
    if (!graph_capture.active) return;
    if (layer < 0 || static_cast<uint32_t>(layer) >= graph_capture.header.layers)
      throw std::runtime_error("transformer: H3 graph capture layer escaped stack");
    graph_capture.header.boundary_fnv64[layer] = capture_hash(
        residual, graph_capture.header.residual_elements);
    if (static_cast<uint32_t>(layer + 1) != graph_capture.header.layers) return;
    std::vector<uint16_t> final(
        static_cast<size_t>(graph_capture.header.residual_elements));
    SLOPFAB_CUDA_CHECK(cudaMemcpyAsync(final.data(), residual, final.size() * 2,
                                      cudaMemcpyDeviceToHost, stream.get()));
    SLOPFAB_CUDA_CHECK(cudaStreamSynchronize(stream.get()));
    graph_capture.header.final_fnv64 = fnv64_append(1469598103934665603ull,
        final.data(), final.size() * 2);
    std::ofstream output(graph_capture_path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error(
        "transformer: cannot create H3 graph capture: " + graph_capture_path);
    output.write(reinterpret_cast<const char*>(&graph_capture.header),
                 sizeof(graph_capture.header));
    auto write = [&](const auto& values) {
      output.write(reinterpret_cast<const char*>(values.data()),
          static_cast<std::streamsize>(values.size() * sizeof(values[0])));
    };
    write(graph_capture.input); write(graph_capture.selectors);
    write(graph_capture.code); write(graph_capture.cosine);
    write(graph_capture.sine); write(graph_capture.ranges); write(final);
    if (!output) throw std::runtime_error(
        "transformer: failed writing H3 graph capture: " + graph_capture_path);
    graph_capture.active = false;
    graph_capture_done = true;
    std::fprintf(stderr, "slopfab: captured exact H3 50-layer graph step %d to %s\n",
                 denoise_step, graph_capture_path.c_str());
    const char* stop = std::getenv("SLOPFAB_H3_GRAPH_CAPTURE_EXIT");
    if (stop && stop[0] == '1')
      throw std::runtime_error(
          "transformer: stopped after requested H3 graph capture");
  }

void Transformer::Impl::begin_transformer_text_capture(const float* prompt, int rows) {
    if (transformer_capture_done || transformer_capture_path.empty()) return;
    if (attention_mode != AttentionMode::kExact || blocks.size() != 50 ||
        refiner.size() != 2)
      throw std::runtime_error(
          "transformer: full capture requires exact 2-refiner/50-main graph");
    ActiveTransformerCapture next;
    std::memcpy(next.header.magic, "VFH3FWD\0", 8);
    next.header.version = 1;
    next.header.header_bytes = sizeof(H3TransformerCaptureHeader);
    next.header.hidden = static_cast<uint32_t>(cfg.hidden_size);
    next.header.heads = static_cast<uint32_t>(cfg.num_attention_heads);
    next.header.head_dim = static_cast<uint32_t>(cfg.attention_head_dim);
    next.header.ffn = static_cast<uint32_t>(cfg.ffn_dim);
    next.header.modalities = kNumModalities;
    next.header.adaln_rank = static_cast<uint32_t>(cfg.adaln_rank);
    next.header.layers = static_cast<uint32_t>(blocks.size());
    next.header.text_rows = static_cast<uint32_t>(rows);
    next.header.text_dim = static_cast<uint32_t>(cfg.text_dim);
    next.header.video_dim = static_cast<uint32_t>(cfg.video_patch_dim());
    next.header.audio_dim = static_cast<uint32_t>(cfg.audio_in_channels);
    next.header.refiner_layers = static_cast<uint32_t>(refiner.size());
    next.header.prompt_elements = static_cast<uint64_t>(rows) * cfg.text_dim;
    next.header.text_elements = static_cast<uint64_t>(rows) * cfg.hidden_size;
    next.prompt.assign(prompt, prompt + next.header.prompt_elements);
    next.text_active = true;
    transformer_capture = std::move(next);
  }

void Transformer::Impl::capture_transformer_text_stage(const __nv_bfloat16* values,
                                      int rows, int dim) {
    if (!transformer_capture.text_active) return;
    if (rows != static_cast<int>(transformer_capture.header.text_rows) ||
        dim != cfg.hidden_size || transformer_capture.text_stage >= 6)
      throw std::runtime_error("transformer: unexpected full-capture text stage");
    transformer_capture.header.text_boundary_fnv64[
        transformer_capture.text_stage++] = capture_hash(
            values, static_cast<size_t>(rows) * dim);
  }

void Transformer::Impl::begin_transformer_forward_capture(const float* video_latents,
                                         const float* audio_latents,
                                         int video_rows, int audio_rows) {
    auto& cap = transformer_capture;
    cap.forward_active = false;
    if (transformer_capture_done || transformer_capture_path.empty() ||
        denoise_step != transformer_capture_step) return;
    if (cap.text_stage != 6 || cap.text_active)
      throw std::runtime_error(
          "transformer: full capture is missing completed text preparation");
    if (layout.num_condition_video != 0 || layout.num_condition_audio != 0 ||
        layout.total_rows() != num_text + video_rows + audio_rows)
      throw std::runtime_error(
          "transformer: full capture currently requires canonical t2va packing");
    cap.header.sequence = static_cast<uint32_t>(layout.total_rows());
    cap.header.timesteps = static_cast<uint32_t>(mod_timesteps);
    cap.header.video_rows = static_cast<uint32_t>(video_rows);
    cap.header.audio_rows = static_cast<uint32_t>(audio_rows);
    cap.header.range_values = static_cast<uint32_t>(host_band.size());
    cap.header.denoise_step = denoise_step;
    cap.header.video_elements = static_cast<uint64_t>(video_rows) *
        cfg.video_patch_dim();
    cap.header.audio_elements = static_cast<uint64_t>(audio_rows) *
        cfg.audio_in_channels;
    cap.header.rope_elements = static_cast<uint64_t>(layout.total_rows()) * 96;
    cap.header.packed_elements = static_cast<uint64_t>(layout.total_rows()) *
        cfg.hidden_size;
    cap.video.assign(video_latents, video_latents + cap.header.video_elements);
    cap.audio.assign(audio_latents, audio_latents + cap.header.audio_elements);
    cap.selectors = std::vector<int32_t>();
    cap.code = host_code;
    cap.cosine = host_rope_cos;
    cap.sine = host_rope_sin;
    cap.ranges = host_band;
    cap.video_ts.assign(host_ts.begin(), host_ts.begin() + video_rows);
    cap.audio_ts.assign(host_ts.begin() + video_rows, host_ts.end());
    cap.selectors.resize(layout.total_rows());
    SLOPFAB_CUDA_CHECK(cudaMemcpyAsync(cap.selectors.data(), d_adaln.get(),
        cap.selectors.size() * sizeof(int32_t), cudaMemcpyDeviceToHost,
        stream.get()));
    cap.text_cache.resize(static_cast<size_t>(cap.header.text_elements));
    SLOPFAB_CUDA_CHECK(cudaMemcpyAsync(cap.text_cache.data(), text_cache.get(),
        cap.text_cache.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost,
        stream.get()));
    SLOPFAB_CUDA_CHECK(cudaStreamSynchronize(stream.get()));
    cap.forward_active = true;
  }

void Transformer::Impl::capture_transformer_packed_input(const __nv_bfloat16* values) {
    auto& cap = transformer_capture;
    if (!cap.forward_active) return;
    cap.packed_input.resize(static_cast<size_t>(cap.header.packed_elements));
    SLOPFAB_CUDA_CHECK(cudaMemcpyAsync(cap.packed_input.data(), values,
        cap.packed_input.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost,
        stream.get()));
    SLOPFAB_CUDA_CHECK(cudaStreamSynchronize(stream.get()));
    cap.header.packed_input_fnv64 = fnv64_append(1469598103934665603ull,
        cap.packed_input.data(), cap.packed_input.size() * sizeof(uint16_t));
  }

void Transformer::Impl::capture_transformer_main_final(const __nv_bfloat16* values) {
    auto& cap = transformer_capture;
    if (!cap.forward_active) return;
    cap.main_final.resize(static_cast<size_t>(cap.header.packed_elements));
    SLOPFAB_CUDA_CHECK(cudaMemcpyAsync(cap.main_final.data(), values,
        cap.main_final.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost,
        stream.get()));
    SLOPFAB_CUDA_CHECK(cudaStreamSynchronize(stream.get()));
    cap.header.main_final_fnv64 = fnv64_append(1469598103934665603ull,
        cap.main_final.data(), cap.main_final.size() * sizeof(uint16_t));
  }

void Transformer::Impl::finish_transformer_capture(const float* video_velocity,
                                  const float* audio_velocity) {
    auto& cap = transformer_capture;
    if (!cap.forward_active) return;
    cap.video_output.assign(video_velocity,
        video_velocity + cap.header.video_elements);
    cap.audio_output.assign(audio_velocity,
        audio_velocity + cap.header.audio_elements);
    cap.header.video_output_fnv64 = fnv64_append(1469598103934665603ull,
        cap.video_output.data(), cap.video_output.size() * sizeof(float));
    cap.header.audio_output_fnv64 = fnv64_append(1469598103934665603ull,
        cap.audio_output.data(), cap.audio_output.size() * sizeof(float));
    std::ofstream output(transformer_capture_path,
                         std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error(
        "transformer: cannot create full H3 capture: " +
        transformer_capture_path);
    output.write(reinterpret_cast<const char*>(&cap.header), sizeof(cap.header));
    auto write = [&](const auto& values) {
      output.write(reinterpret_cast<const char*>(values.data()),
          static_cast<std::streamsize>(values.size() * sizeof(values[0])));
    };
    write(cap.prompt); write(cap.video); write(cap.audio);
    write(cap.selectors); write(cap.code); write(cap.cosine); write(cap.sine);
    write(cap.ranges); write(cap.video_ts); write(cap.audio_ts);
    write(cap.text_cache); write(cap.packed_input); write(cap.main_final);
    write(cap.video_output); write(cap.audio_output);
    if (!output) throw std::runtime_error(
        "transformer: failed writing full H3 capture: " +
        transformer_capture_path);
    cap.forward_active = false;
    transformer_capture_done = true;
    std::fprintf(stderr, "slopfab: captured exact H3 transformer step %d to %s\n",
                 denoise_step, transformer_capture_path.c_str());
    const char* stop = std::getenv("SLOPFAB_H3_TRANSFORMER_CAPTURE_EXIT");
    if (stop && stop[0] == '1')
      throw std::runtime_error(
          "transformer: stopped after requested full H3 capture");
  }

void Transformer::Impl::capture_sol_inputs(const __nv_bfloat16* q, const __nv_bfloat16* k,
                          const __nv_bfloat16* v, int rows, int layer) {
    if (sol_capture_done) return;
    if (denoise_step != sol_capture_step || layer != sol_capture_layer) return;

    const uint64_t n = static_cast<uint64_t>(rows) * cfg.num_attention_heads *
                       cfg.attention_head_dim;
    std::vector<uint16_t> host(static_cast<size_t>(n) * 3);
    SLOPFAB_CUDA_CHECK(cudaMemcpyAsync(host.data(), q, n * sizeof(uint16_t),
                                      cudaMemcpyDeviceToHost, stream.get()));
    SLOPFAB_CUDA_CHECK(cudaMemcpyAsync(host.data() + n, k, n * sizeof(uint16_t),
                                      cudaMemcpyDeviceToHost, stream.get()));
    SLOPFAB_CUDA_CHECK(cudaMemcpyAsync(host.data() + 2 * n, v, n * sizeof(uint16_t),
                                      cudaMemcpyDeviceToHost, stream.get()));
    SLOPFAB_CUDA_CHECK(cudaStreamSynchronize(stream.get()));
    SolCaptureHeader header{{'V','F','S','O','L','Q','K','V'}, 1, sizeof(SolCaptureHeader),
                            static_cast<uint32_t>(rows),
                            static_cast<uint32_t>(cfg.num_attention_heads),
                            static_cast<uint32_t>(cfg.attention_head_dim),
                            static_cast<uint32_t>(layout.video_start()), denoise_step, layer, n,
                            {0, 0}};
    std::ofstream out(sol_capture_path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("transformer: cannot create Sol capture: " + sol_capture_path);
    out.write(reinterpret_cast<const char*>(&header), sizeof(header));
    out.write(reinterpret_cast<const char*>(host.data()),
              static_cast<std::streamsize>(host.size() * sizeof(uint16_t)));
    if (!out) throw std::runtime_error("transformer: failed writing Sol capture: " + sol_capture_path);
    sol_capture_done = true;
    std::fprintf(stderr, "slopfab: captured Sol Q/K/V step %d layer %d to %s\n",
                 denoise_step, layer, sol_capture_path.c_str());
    const char* stop = std::getenv("SLOPFAB_SOL_CAPTURE_EXIT");
    if (stop != nullptr && stop[0] == '1')
      throw std::runtime_error("transformer: stopped after requested Sol capture");
  }
}  // namespace slopfab::dit
