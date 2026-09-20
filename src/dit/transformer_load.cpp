#include "transformer_impl.h"

namespace slopfab::dit {

void Transformer::load(const SafeTensors& checkpoint, const TransformerConfig& config,
                       const LoraAdapters* loras, const TransformerLoadOptions& options) {
  cuda::StageMemorySpan memory("transformer.weights");
  Impl& s = *impl_;
  // Issued first, before anything else in this function, because it is
  // asynchronous: the plan walk and the arena allocation below run while the
  // OS is already reading the file. Every byte of this checkpoint is about to
  // be consumed, which is the condition `prefetch` documents. Best-effort — a
  // failure just means the old demand-fault path, which is what this did
  // before and is 3x slower on a cold cache.
  checkpoint.prefetch();
  unload();
  s.cfg = config;
  s.model = resolve_model_descriptor(checkpoint);
  s.architecture = s.model.compatibility_architecture;
  if (s.model.modulation == ModulationImplementation::kTable)
    validate_adaln_table_config(config.adaln_rank, config.adaln_table_rows);
  if (s.architecture == TransformerArchitecture::kUnknown) {
    throw std::runtime_error("transformer: checkpoint architecture is unknown");
  }
  const TransformerQuantization checkpoint_quant = detect_transformer_quantization(checkpoint);
  if (s.architecture == TransformerArchitecture::kRef2VAFullAdaLN &&
      checkpoint_quant == TransformerQuantization::kBitsAndBytesNF4) {
    throw std::runtime_error(
        "transformer: full-AdaLN NF4 Ref2VA is disabled because it produces invalid tiled "
        "video; use the pruned FP8 Ref2VA checkpoint");
  }
  if (s.architecture == TransformerArchitecture::kRef2VAFullAdaLN &&
      checkpoint_quant != TransformerQuantization::kFloat8 &&
      checkpoint_quant != TransformerQuantization::kBitsAndBytesNF4) {
    throw std::runtime_error(
        "transformer: full-AdaLN execution requires an FP8 or bitsandbytes NF4 Ref2VA "
        "checkpoint");
  }
  const bool full_adaln = s.architecture == TransformerArchitecture::kRef2VAFullAdaLN;
  const bool interleaved_qkv = transformer_qkv_is_interleaved(checkpoint);

  const int hidden = config.hidden_size;
  const int inner = config.inner_dim();
  const int ffn = config.ffn_dim;
  const int head_dim = config.attention_head_dim;
  const int patch = config.video_patch_dim();
  const int adaln_out = kNumParams * kNumModalities * hidden;
  const int final_adaln_out = kFinalParams * hidden;

  Plan plan(checkpoint);

  auto plan_nf4 = [&](const std::string& name, int out_features, int in_features) {
    const NF4State state = read_nf4_state(checkpoint, name);
    if (state.shape != std::vector<int64_t>{out_features, in_features}) {
      throw std::runtime_error("transformer: '" + nf4_state_name(name) + "' declares shape " +
                               shape_string(state.shape) + ", expected " +
                               shape_string({out_features, in_features}));
    }
    const int64_t elements = static_cast<int64_t>(out_features) * in_features;
    const int64_t blocks = (elements + state.block_size - 1) / state.block_size;
    const int64_t nested = (blocks + state.nested_block_size - 1) / state.nested_block_size;
    plan.require(name + ".weight", {elements / 2, 1}, Store::kVerbatim);
    plan.require(name + ".weight.absmax", {blocks}, Store::kVerbatim);
    plan.require(name + ".weight.quant_map", {16}, Store::kVerbatim);
    plan.require(name + ".weight.nested_absmax", {nested}, Store::kVerbatim);
    plan.require(name + ".weight.nested_quant_map", {256}, Store::kVerbatim);
    plan.require(nf4_state_name(name),
                 {static_cast<int64_t>(checkpoint.at(nf4_state_name(name)).nbytes)},
                 Store::kVerbatim);
  };

  // Spec 8.3 — the 32 top-level tensors.
  plan.require("video_patch_proj.weight", {hidden, patch}, Store::kAsF32);
  plan.require("video_patch_proj.bias", {hidden}, Store::kAsF32);
  plan.require("audio_patch_proj.weight", {hidden, config.audio_in_channels}, Store::kAsF32);
  plan.require("audio_patch_proj.bias", {hidden}, Store::kAsF32);
  plan.require("condition_proj.weight", {hidden, config.text_dim}, Store::kVerbatim);
  plan.require("condition_proj.bias", {hidden}, Store::kAsF32);
  // Claimed but not uploaded: the lookup runs on the host — 8 floats per
  // denoising step — and `AdaLNTable::load` validates its shape and finiteness.
  plan.optional("adaln_t_table");
  if (full_adaln) {
    plan.optional("time_embedder.proj_in.weight");
    plan.optional("time_embedder.proj_in.bias");
    plan.optional("time_embedder.proj_out.weight");
    plan.optional("time_embedder.proj_out.bias");
  }
  // Recomputed in fp64 and rounded, which convert.py:104-107 says is bitwise
  // equal to the stored tensor. Claimed only so the count check balances.
  plan.optional("rope.inv_freq");

  plan.require("final_layer.norm.weight", {hidden}, Store::kAsBF16);
  if (full_adaln) {
    const std::string final_adaln = "final_layer.adaln_proj.linear";
    if (is_nf4(checkpoint, final_adaln)) {
      plan_nf4(final_adaln, final_adaln_out, config.timestep_embed_dim);
    } else {
      plan.require(final_adaln + ".weight",
                   {final_adaln_out, config.timestep_embed_dim}, Store::kVerbatim);
      plan.optional_scalar(final_adaln + ".weight_scale");
    }
    plan.optional("final_layer.adaln_proj.linear.input_scale");
    plan.optional("final_layer.adaln_proj.linear.comfy_quant");
    plan.require("final_layer.adaln_proj.linear.bias", {final_adaln_out}, Store::kAsBF16);
  } else {
    plan.require("final_layer.adaln_proj.linear.weight", {final_adaln_out, AdaLNTable::kRank},
                 Store::kAsF32);
    plan.require("final_layer.adaln_proj.linear.bias", {final_adaln_out}, Store::kAsF32);
  }
  plan.require("final_layer.video_out.weight", {patch, hidden}, Store::kAsF32);
  plan.require("final_layer.video_out.bias", {patch}, Store::kAsF32);
  plan.require("final_layer.audio_out.weight", {config.audio_in_channels, hidden}, Store::kAsF32);
  plan.require("final_layer.audio_out.bias", {config.audio_in_channels}, Store::kAsF32);
  plan.require("token_refiner.final_norm.weight", {hidden}, Store::kAsBF16);

  // A block's tensor set, shared by the 50 main blocks and the 2 refiner
  // blocks; the refiner differs only by having no `adaln_proj` (spec 6).
  // One quantised projection. nvfp4 changes both the weight's shape — half as
  // many bytes on the contraction axis — and the weight_scale's, from a
  // per-tensor scalar to one e4m3 byte per 16 contracted elements, and adds the
  // second-level `weight_scale_2`. Which of the two it is comes from the file.
  auto plan_linear = [&](const std::string& name, int out_features, int in_features) {
    if (is_nf4(checkpoint, name)) {
      plan_nf4(name, out_features, in_features);
    } else if (is_nvfp4(checkpoint, name, in_features)) {
      plan.require(name + ".weight", {out_features, in_features / 2}, Store::kVerbatim);
      plan.require(name + ".weight_scale",
                   {out_features, in_features / static_cast<int>(cuda::kNVFP4BlockSize)},
                   Store::kVerbatim);
      plan.optional_scalar(name + ".weight_scale_2");
    } else if (const TensorView* weight = checkpoint.find(name + ".weight");
               weight != nullptr && weight->dtype == DType::kI8) {
      plan.require(name + ".weight", {out_features, in_features}, Store::kVerbatim);
      // ComfyUI calls this tensorwise INT8, but the checkpoint stores one F32
      // dequantisation scale per output row as [out, 1].
      plan.require(name + ".weight_scale", {out_features, 1}, Store::kAsF32);
    } else {
      plan.require(name + ".weight", {out_features, in_features}, Store::kVerbatim);
      plan.optional_scalar(name + ".weight_scale");
    }
    plan.optional(name + ".input_scale");
    plan.optional(name + ".comfy_quant");
  };

  auto plan_block = [&](const std::string& prefix, bool with_adaln) {
    plan.require(prefix + "norm1.weight", {hidden}, Store::kAsBF16);
    plan.require(prefix + "norm2.weight", {hidden}, Store::kAsBF16);
    plan_linear(prefix + "attn.qkv_proj", 3 * inner, hidden);
    if (interleaved_qkv)
      validate_interleaved_qkv(checkpoint.at(prefix + "attn.qkv_proj.weight"), head_dim);
    plan.require(prefix + "attn.q_norm.weight", {head_dim}, Store::kAsBF16);
    plan.require(prefix + "attn.k_norm.weight", {head_dim}, Store::kAsBF16);
    plan_linear(prefix + "attn.out_proj", hidden, inner);
    if (with_adaln && s.is_vsa()) plan_linear(prefix + "attn.to_gate_compress", inner, hidden);
    plan_linear(prefix + "mlp.fc1", 2 * ffn, hidden);
    plan_linear(prefix + "mlp.fc2", hidden, ffn);
    if (with_adaln) {
      if (full_adaln) {
        plan_linear(prefix + "adaln_proj.linear", adaln_out, config.timestep_embed_dim);
        plan.require(prefix + "adaln_proj.linear.bias", {adaln_out}, Store::kAsBF16);
      } else {
        plan.require(prefix + "adaln_proj.linear.weight", {adaln_out, AdaLNTable::kRank},
                     Store::kAsF32);
        plan.require(prefix + "adaln_proj.linear.bias", {adaln_out}, Store::kAsF32);
      }
    }
  };

  for (int i = 0; i < config.num_refiner_layers; ++i) {
    plan_block("token_refiner.blocks." + std::to_string(i) + ".", /*with_adaln=*/false);
  }
  for (int i = 0; i < config.num_layers; ++i) {
    plan_block("blocks." + std::to_string(i) + ".", /*with_adaln=*/true);
  }
  plan.finish();

  // --- upload ---------------------------------------------------------------

  std::vector<size_t> block_bytes(config.num_layers, 0);
  size_t fixed_bytes = 0, adapter_rank = 0, adapter_out = 0;
  for (const auto& item : plan.records()) {
    const int block = streamable_block(item.first);
    if (block >= 0) block_bytes.at(block) += align_up(item.second.bytes);
    else fixed_bytes += align_up(item.second.bytes);
  }
  auto count_adapters = [&](const std::string& prefix, size_t& bytes) {
    visit_block_loras(loras, prefix, config,
        [&](int, const std::vector<LoraFactors>& factors, int, int out) {
          for (const auto& f : factors) {
            const size_t n = align_up(f.a.size() * 2) + align_up(size_t(out) * f.rank * 2);
            bytes += n;
            adapter_rank = std::max(adapter_rank, size_t(f.rank));
            adapter_out = std::max(adapter_out, size_t(out));
          }
        });
  };
  for (int i = 0; i < config.num_layers; ++i)
    count_adapters("blocks." + std::to_string(i) + ".", block_bytes[i]);
  for (int i = 0; i < config.num_refiner_layers; ++i)
    count_adapters("token_refiner.blocks." + std::to_string(i) + ".", fixed_bytes);

  int forced = options.offload_blocks;
  if (const char* value = std::getenv("SLOPFAB_DIT_OFFLOAD_BLOCKS")) {
    size_t end = 0; forced = std::stoi(value, &end);
    if (value[end] != '\0') throw std::invalid_argument("invalid SLOPFAB_DIT_OFFLOAD_BLOCKS");
  }
  size_t cap = options.device_budget_bytes;
  if (const char* value = std::getenv("SLOPFAB_DIT_VRAM_GIB")) {
    size_t end = 0; const double gib = std::stod(value, &end);
    if (value[end] != '\0' || !std::isfinite(gib) || gib <= 0 || gib > 1048576)
      throw std::invalid_argument("invalid SLOPFAB_DIT_VRAM_GIB");
    cap = static_cast<size_t>(gib * 1024 * 1024 * 1024);
  }
  size_t reserve = options.headroom_bytes + size_t(256) * (adapter_rank + adapter_out) * 2;
  if (options.layout) {
    const auto& layout = *options.layout;
    // Conservative ConvRot scratch even for archives that do not use it.
    const Carve main = plan_carve(config, layout, s.row_chunk, s.attention_mode,
                                  true, s.compact_queries(s.attention_mode), s.is_vsa());
    const Carve plain = plan_carve(config, layout, s.row_chunk, s.attention_mode,
                                   false, s.compact_queries(s.attention_mode), s.is_vsa());
    size_t main_bytes = activation_bytes(layout) + main.total - plain.total;
    // Account for cached text and up to four distinct row timesteps, rather
    // than the two used by the original text-to-video estimator.
    main_bytes += size_t(layout.num_text) * config.hidden_size * 2;
    main_bytes += size_t(config.num_layers) * 6 * 2 * 3 * config.hidden_size * 4;
    if (options.block_cache) main_bytes += size_t(layout.total_rows()) * config.hidden_size * 2;
    if (full_adaln) {
      QuantWeight probe; probe.format = QuantFormat::kF8E4M3;
      probe.in_features = config.timestep_embed_dim; probe.out_features = adaln_out;
      const size_t scratch = cuda::linear_workspace_bytes(probe, 4, ComputeType::kF32);
      if (scratch > main.scratch) main_bytes += scratch - main.scratch;
    }
    SequenceLayout text_layout; text_layout.num_text = layout.num_text;
    const Carve text = plan_carve(config, text_layout, s.row_chunk,
        s.attention_mode == AttentionMode::kExact ? AttentionMode::kExact : AttentionMode::kFlash2,
        true, s.compact_queries(s.attention_mode == AttentionMode::kExact
            ? AttentionMode::kExact : AttentionMode::kFlash2));
    const size_t text_bytes = text.total + size_t(layout.num_text) *
        (config.text_dim * 6 + config.hidden_size * 2);
    reserve += std::max(main_bytes, text_bytes);
  }
  size_t budget = std::numeric_limits<size_t>::max();
  if (options.layout || cap || forced >= 0) {
    size_t free = 0, total = 0;
    SLOPFAB_CUDA_CHECK(cudaMemGetInfo(&free, &total));
    if (cap) free = std::min(free, cap > total - free ? cap - (total - free) : 0);
    if (free <= reserve)
      throw std::runtime_error("transformer offload: activations and safety headroom exhaust the VRAM budget");
    budget = free - reserve;
  }
  const BlockOffloadPlan residency = plan_block_offload(block_bytes, fixed_bytes, budget, forced);
  if (residency.count) {
    if (!s.block_capture_path.empty() || !s.graph_capture_path.empty() ||
        !s.transformer_capture_path.empty() || !s.sol_capture_path.empty())
      throw std::runtime_error("transformer offload: full-tensor capture requires resident weights");
    s.streamer = std::make_unique<BlockStreamer>(s.stream.get());
    s.streamer->initialize(residency, block_bytes);
  }
  size_t resident_records = 0;
  for (const auto& item : plan.records()) {
    const int block = streamable_block(item.first);
    if (!s.streamer || block < 0 || !s.streamer->contains(block))
      resident_records += align_up(item.second.bytes);
  }
  s.arena.allocate(resident_records);
  s.arena_bytes = resident_records;
  uint8_t* base = s.arena.get();
  std::map<std::string, uint8_t*> destinations;
  size_t resident_cursor = 0;
  for (const auto& item : plan.records()) {
    const int block = streamable_block(item.first);
    if (s.streamer && block >= 0 && s.streamer->contains(block)) {
      auto& staged = s.streamer->blocks[block];
      destinations[item.first] = staged.host.get() + staged.cursor;
      staged.cursor += align_up(item.second.bytes);
    } else {
      destinations[item.first] = base + (s.streamer ? resident_cursor : item.second.offset);
      resident_cursor += align_up(item.second.bytes);
    }
  }
  if (residency.count || cuda::StepProfiler::instance().enabled()) {
    constexpr double gib = 1024.0 * 1024 * 1024;
    std::printf("offload     blocks [%zu,%zu) of %d; weights+adapters %.3f GiB GPU, "
                "%.3f GiB pinned CPU, %zu transfer slots (%.3f GiB); activation/headroom reserve %.3f GiB\n",
                residency.first, residency.first + residency.count, config.num_layers,
                residency.device_bytes / gib, residency.host_bytes / gib, residency.slots(),
                residency.slots() * residency.slot_bytes / gib, reserve / gib);
    std::fflush(stdout);
  }

  {
    // Both declared before the uploader so that they outlive it. ~Uploader
    // synchronises the stream, so anything the stream might still be reading —
    // the registered mapping, and the scratch buffers the staged path copies
    // out of — has to still exist when that sync runs.
    cuda::RegisteredMapping lock(checkpoint.mapping_base(), checkpoint.file_size());
    // Success is the normal path and stays quiet; a failure silently costs
    // seconds of staged memcpy, so it says so rather than looking like a
    // mysterious regression later.
    if (!lock.registered()) {
      std::fprintf(stderr,
                   "slopfab: could not page-lock the transformer mapping; uploading via the "
                   "staged path, which is slower\n");
    }
    std::vector<float> wide;
    std::vector<uint16_t> narrow;
    std::vector<uint8_t> reordered;
    // Landing pad for the fp16 records that are widened on the device rather
    // than on the host. On the real checkpoint the largest is
    // `blocks.N.adaln_proj.linear.weight`, 774144 fp16 values, so 1.5 MB of
    // VRAM, and it is freed with this scope well before the first forward pass.
    //
    // Sized to the maximum up front rather than grown on demand. Growing would
    // mean `DeviceBuffer::allocate` -> `reset` -> `cudaFree` on a pointer that
    // an already-enqueued `launch_widen_f16` may still be reading, and stream
    // ordering does not make that safe — it is safe today only because
    // `cudaFree` is an implicit device-wide sync point, which is a property of
    // the legacy allocator and not something this code should depend on. A
    // future move to `cudaFreeAsync` would turn it into a use-after-free that
    // no test could catch. One pass over the records costs nothing and removes
    // the question.
    size_t widen_max = 0;
    for (const auto& kv : plan.records()) {
      const Record& r = kv.second;
      if (r.store == Store::kAsF32 && r.view->dtype == DType::kF16) {
        widen_max = std::max(widen_max, static_cast<size_t>(r.view->numel()));
      }
    }
    cuda::DeviceBuffer<uint16_t> widen_src;
    if (widen_max != 0) widen_src.allocate(widen_max);
    Uploader up(s.stream.get(), &lock);
    for (const auto& kv : plan.records()) {
      const Record& r = kv.second;
      uint8_t* dst = destinations.at(kv.first);
      const int block = streamable_block(kv.first);
      const bool staged = s.streamer && block >= 0 && s.streamer->contains(block);
      auto copy = [&](const void* source, size_t bytes, bool mapping) {
        if (staged) std::memcpy(dst, source, bytes);
        else up.copy(dst, source, bytes, mapping);
      };
      if (loras && !full_adaln) {
        const bool weight = kv.first.size() > 7 && kv.first.compare(kv.first.size() - 7, 7, ".weight") == 0;
        const bool bias = kv.first.size() > 5 && kv.first.compare(kv.first.size() - 5, 5, ".bias") == 0;
        const std::string name = kv.first.substr(0, kv.first.size() - (weight ? 7 : bias ? 5 : 0));
        if ((weight || bias) && loras->has_adaln(name)) {
          wide = weight ? loras->merged_adaln_weight(checkpoint, name)
                        : loras->merged_adaln_bias(checkpoint, name);
          copy(wide.data(), wide.size() * sizeof(float), /*from_mapping=*/false);
          continue;
        }
      }
      if (loras && (kv.first == "video_patch_proj.weight" ||
                    kv.first == "final_layer.video_out.weight")) {
        const std::string name = kv.first.substr(0, kv.first.size() - 7);
        if (loras->find(name)) {
          wide = loras->merged_endpoint_weight(checkpoint, name);
          copy(wide.data(), wide.size() * sizeof(float), /*from_mapping=*/false);
          continue;
        }
      }
      if (interleaved_qkv &&
          (kv.first.find(".attn.qkv_proj.weight") != std::string::npos) &&
          r.view->shape.size() == 2) {
        reordered = deinterleave_qkv_rows(*r.view, head_dim);
        // These records retain their disk dtype (INT8 weights / F32 scales).
        if (reordered.size() != r.bytes)
          throw std::runtime_error("transformer: interleaved QKV upload dtype mismatch");
        copy(reordered.data(), reordered.size(), /*from_mapping=*/false);
        continue;
      }
      switch (r.store) {
        case Store::kVerbatim:
          copy(r.view->data, r.bytes, /*from_mapping=*/true);
          break;
        case Store::kAsF32:
          // The arena wants fp32 here, and the two dtypes that actually occur
          // reach it without a host conversion loop at all.
          //
          // fp32 on disk is already the arena's format, so `to_f32` was
          // copying it element by element into a scratch vector in order to
          // upload an identical copy. It is a straight DMA out of the mapping.
          //
          // fp16 -> fp32 is exact, so sending the fp16 bytes and widening on
          // the device produces the same arena bytes while halving the traffic
          // and dropping the single-threaded host loop — 43.6 M elements of
          // AdaLN projection on the real checkpoint. `ViTDecoder::load` has
          // done this since the video VAE work; `test_widen_f16` compares the
          // device kernel against `f16_to_f32` over every finite fp16 bit
          // pattern.
          //
          // Note what this does *not* do: the arena still stores these records
          // as fp32, so it saves no device memory. That is not an oversight,
          // it is the thing that was verified — `SLOPFAB_ARENA_HASH=1` over
          // fl2va_pruned_fp8_scaled.safetensors gives 21045398272 bytes, 730
          // records, fnv1a 190cdce19da29c2d both before and after this change.
          // An arena that got smaller would be a different arena.
          if (r.view->dtype == DType::kF32) {
            copy(r.view->data, r.bytes, /*from_mapping=*/true);
          } else if (r.view->dtype == DType::kF16 && !staged) {
            const auto count = static_cast<size_t>(r.view->numel());
            // Reuse across records needs no synchronise: both halves are on
            // `s.stream`, so the next record's copy into `widen_src` is
            // ordered after this widen has finished reading it. The buffer is
            // never reallocated here — see the sizing loop above for why that
            // distinction matters.
            up.copy(widen_src.get(), r.view->data, count * sizeof(uint16_t),
                    /*from_mapping=*/true);
            cuda::launch_widen_f16(widen_src.get(), reinterpret_cast<float*>(dst), count,
                                   s.stream.get());
          } else {
            to_f32(*r.view, wide);
            copy(wide.data(), wide.size() * sizeof(float), /*from_mapping=*/false);
          }
          break;
        case Store::kAsBF16:
          to_f32(*r.view, wide);
          narrow.resize(wide.size());
          for (size_t i = 0; i < wide.size(); ++i) narrow[i] = f32_to_bf16(wide[i]);
          copy(narrow.data(), narrow.size() * sizeof(uint16_t), /*from_mapping=*/false);
          break;
      }
    }
  }
  SLOPFAB_CUDA_CHECK(cudaStreamSynchronize(s.stream.get()));

  // A hash of the finished weight arena, off unless asked for. This exists so
  // that a change to *how* the file is read can be shown to have left *what*
  // was read alone: any reordering, buffering or prefetch change must produce
  // the same 12.5 GB byte for byte, and a single number either matches or it
  // does not. Reading it back costs one D2H of the arena — about half a second
  // — which is why it is behind an environment variable rather than always on.
  if (const char* want = std::getenv("SLOPFAB_ARENA_HASH"); want != nullptr && want[0] == '1') {
    constexpr size_t kChunk = 64u << 20;
    std::vector<uint64_t> host(kChunk / sizeof(uint64_t));
    uint64_t h = 1469598103934665603ull;  // FNV-1a offset basis
    size_t left = s.arena_bytes;
    const uint8_t* src = base;
    while (left > 0) {
      const size_t n = std::min(left, kChunk);
      SLOPFAB_CUDA_CHECK(cudaMemcpy(host.data(), src, n, cudaMemcpyDeviceToHost));
      // Whole words only; the arena's records are 256-byte aligned, so the
      // trailing partial word can only be padding this loop never reaches.
      const size_t words = n / sizeof(uint64_t);
      for (size_t i = 0; i < words; ++i) h = (h ^ host[i]) * 1099511628211ull;
      src += n;
      left -= n;
    }
    std::printf("arena %zu bytes, %zu records, fnv1a %016llx\n", s.arena_bytes,
                plan.records().size(), static_cast<unsigned long long>(h));
    std::fflush(stdout);
  }

  // --- wire up --------------------------------------------------------------

  auto ptr = [&](const std::string& name) -> const void* {
    const auto it = plan.records().find(name);
    if (it == plan.records().end()) return nullptr;
    return destinations.at(name);
  };
  auto bf = [&](const std::string& name) {
    return static_cast<const __nv_bfloat16*>(ptr(name));
  };
  auto f32 = [&](const std::string& name) { return static_cast<const float*>(ptr(name)); };

  // A scalar `input_scale` is a host value: it selects a code path rather than
  // feeding a kernel. Zero means absent, which for an fp8 weight is the
  // checkpoint asserting the layer must run at full precision (spec 8.2).
  auto host_scalar = [&](const std::string& name) -> float {
    const TensorView* v = checkpoint.find(name);
    if (v == nullptr) return 0.0f;
    std::vector<float> tmp;
    to_f32(*v, tmp);
    return tmp.empty() ? 0.0f : tmp[0];
  };

  auto quant = [&](const std::string& name, int out_features, int in_features) {
    QuantWeight w;
    const TensorView& v = checkpoint.at(name + ".weight");
    w.out_features = out_features;
    w.in_features = in_features;
    w.data = ptr(name + ".weight");

    if (is_nf4(checkpoint, name)) {
      const NF4State state = read_nf4_state(checkpoint, name);
      w.format = QuantFormat::kNF4;
      w.nf4_absmax = static_cast<const uint8_t*>(ptr(name + ".weight.absmax"));
      w.nf4_quant_map = f32(name + ".weight.quant_map");
      w.nf4_nested_absmax = f32(name + ".weight.nested_absmax");
      w.nf4_nested_quant_map = f32(name + ".weight.nested_quant_map");
      w.nf4_block_size = state.block_size;
      w.nf4_nested_block_size = state.nested_block_size;
      w.nf4_nested_offset = state.nested_offset;
    } else if (is_nvfp4(checkpoint, name, in_features)) {
      w.format = QuantFormat::kNVFP4;
      // Raw e4m3 bytes in a 128x4 tiling, not floats and not row-major, so this
      // deliberately does not go through `f32` — reinterpreting them as float
      // would read a quarter of the array and scale by nonsense.
      w.block_scale = static_cast<const uint8_t*>(ptr(name + ".weight_scale"));
      if (checkpoint.find(name + ".weight_scale_2") == nullptr) {
        throw std::runtime_error("transformer: '" + name +
                                 ".weight' is nvfp4 but ships no weight_scale_2; the default 1.0 "
                                 "would be wrong by about 700x rather than visibly broken");
      }
      w.global_scale = host_scalar(name + ".weight_scale_2");
    } else {
      w.format = format_of(v.dtype, name + ".weight");
      w.weight_scale = f32(name + ".weight_scale");
      if (w.format == QuantFormat::kF8E4M3 && w.weight_scale == nullptr) {
        throw std::runtime_error("transformer: '" + name +
                                 ".weight' is fp8 but ships no weight_scale");
      }
      if (w.format == QuantFormat::kI8 && w.weight_scale == nullptr) {
        throw std::runtime_error("transformer: '" + name +
                                 ".weight' is int8 but ships no per-channel weight_scale");
      }
      w.per_channel_scale = w.format == QuantFormat::kI8;
    }

    w.input_scale = host_scalar(name + ".input_scale");
    // The file decides, not a heuristic on which scales are present. No layer
    // of the nvfp4 transformer sets it and 50 of the fp8 transformer's do —
    // exactly `mlp.fc2`, which also ships no input_scale (spec 8.2).
    const QuantTag tag = read_comfy_quant(checkpoint, name);
    w.full_precision = tag.full_precision;
    if (tag.convrot) {
      int power = 1;
      while (power < tag.convrot_group) power *= 4;
      if (tag.format != "int8_tensorwise" || w.format != QuantFormat::kI8 ||
          power != tag.convrot_group) {
        throw std::runtime_error("transformer: '" + name +
                                 "' has incompatible ConvRot metadata");
      }
      // The quantiser deliberately leaves a non-aligned contraction axis
      // unrotated, so apply the same per-tensor rule at inference time.
      w.convrot = in_features % tag.convrot_group == 0;
      w.convrot_group = tag.convrot_group;
    }
    return w;
  };

  auto build_block = [&](const std::string& prefix, bool with_adaln) {
    BlockWeights b;
    b.norm1 = bf(prefix + "norm1.weight");
    b.norm2 = bf(prefix + "norm2.weight");
    b.q_norm = bf(prefix + "attn.q_norm.weight");
    b.k_norm = bf(prefix + "attn.k_norm.weight");

    // The upload canonicalizes QKV to contiguous [Wq; Wk; Wv].
    // Three views over one allocation, sharing both scales.
    //
    // For nvfp4 **both** arrays have to be sliced. Advancing `data` and leaving
    // `block_scale` at the base gives k and v the wrong scales entirely, which
    // is finite, well shaped and wrong — the exact failure this format invites.
    const QuantWeight fused = quant(prefix + "attn.qkv_proj", 3 * inner, hidden);
    b.wq = fused;
    b.wq.out_features = inner;
    b.wk = b.wq;
    b.wv = b.wq;

    if (fused.format == QuantFormat::kNF4) {
      const size_t elements_third = static_cast<size_t>(inner) * hidden;
      if (elements_third % fused.nf4_block_size != 0 ||
          (elements_third / fused.nf4_block_size) % fused.nf4_nested_block_size != 0) {
        throw std::runtime_error("transformer: qkv thirds do not align to NF4 nested blocks");
      }
      const size_t blocks_third = elements_third / fused.nf4_block_size;
      const size_t nested_third = blocks_third / fused.nf4_nested_block_size;
      b.wk.data = static_cast<const uint8_t*>(fused.data) + elements_third / 2;
      b.wv.data = static_cast<const uint8_t*>(fused.data) + elements_third;
      b.wk.nf4_absmax = fused.nf4_absmax + blocks_third;
      b.wv.nf4_absmax = fused.nf4_absmax + 2 * blocks_third;
      b.wk.nf4_nested_absmax = fused.nf4_nested_absmax + nested_third;
      b.wv.nf4_nested_absmax = fused.nf4_nested_absmax + 2 * nested_third;
    } else if (fused.format == QuantFormat::kNVFP4) {
      // Slicing the block scales on a byte offset is only correct because each
      // third is a whole number of the 128-row tiles they are stored in: the
      // tile index runs row-major, so rows [inner, 2*inner) begin exactly at
      // tile (inner/128) and nowhere else. With `inner` not a multiple of 128
      // the thirds would start mid-tile and the offset would silently address
      // another row's scales.
      if (inner % 128 != 0) {
        throw std::runtime_error(
            "transformer: inner_dim " + std::to_string(inner) +
            " is not a multiple of 128, so the qkv thirds do not fall on nvfp4 block-scale tile "
            "boundaries and cannot be sliced by offset");
      }
      const size_t data_third = static_cast<size_t>(inner) * hidden / 2;
      const size_t scale_third =
          static_cast<size_t>(inner) * hidden / static_cast<size_t>(cuda::kNVFP4BlockSize);
      b.wk.data = static_cast<const uint8_t*>(fused.data) + data_third;
      b.wv.data = static_cast<const uint8_t*>(fused.data) + 2 * data_third;
      b.wk.block_scale = fused.block_scale + scale_third;
      b.wv.block_scale = fused.block_scale + 2 * scale_third;
    } else {
      const size_t third = static_cast<size_t>(inner) * hidden * format_bytes(fused.format);
      b.wk.data = static_cast<const uint8_t*>(fused.data) + third;
      b.wv.data = static_cast<const uint8_t*>(fused.data) + 2 * third;
      if (fused.per_channel_scale) {
        b.wk.weight_scale = fused.weight_scale + inner;
        b.wv.weight_scale = fused.weight_scale + 2 * inner;
      }
    }

    b.out_proj = quant(prefix + "attn.out_proj", hidden, inner);
    if (with_adaln && s.is_vsa()) b.compress_gate = quant(prefix + "attn.to_gate_compress", inner, hidden);
    b.fc1 = quant(prefix + "mlp.fc1", 2 * ffn, hidden);
    b.fc2 = quant(prefix + "mlp.fc2", hidden, ffn);
    const int staged_block = streamable_block(prefix);
    const bool staged = s.streamer && staged_block >= 0 && s.streamer->contains(staged_block);
    const QuantWeight* projections[] = {&b.wq, &b.wk, &b.wv, &b.out_proj, &b.fc1, &b.fc2};
    visit_block_loras(loras, prefix, config,
        [&](int projection, const std::vector<LoraFactors>& factors, int offset, int out) {
          const void* key = projections[projection]->data;
          if (staged) {
            auto& storage = s.streamer->blocks[staged_block];
            size_t need = 0;
            for (const auto& f : factors)
              need += align_up(f.a.size() * 2) + align_up(size_t(out) * f.rank * 2);
            if (need > storage.host.size() - storage.cursor)
              throw std::runtime_error("transformer offload: adapter staging exceeds its plan");
            s.lora.attach(key, factors, offset, out, storage.host.get(),
                          s.streamer->device(staged_block), &storage.cursor);
          } else {
            s.lora.attach(key, factors, offset, out);
          }
        });

    if (with_adaln) {
      if (full_adaln) {
        b.full_adaln = quant(prefix + "adaln_proj.linear", adaln_out,
                             config.timestep_embed_dim);
        b.full_adaln.bias = ptr(prefix + "adaln_proj.linear.bias");
        b.full_adaln.bias_format = QuantFormat::kBF16;
      } else {
        b.adaln_w = f32(prefix + "adaln_proj.linear.weight");
        b.adaln_b = f32(prefix + "adaln_proj.linear.bias");
      }
    }
    return b;
  };

  s.refiner.clear();
  for (int i = 0; i < config.num_refiner_layers; ++i) {
    s.refiner.push_back(build_block("token_refiner.blocks." + std::to_string(i) + ".", false));
  }
  s.blocks.clear();
  for (int i = 0; i < config.num_layers; ++i) {
    s.blocks.push_back(build_block("blocks." + std::to_string(i) + ".", true));
  }
  if (s.streamer) for (size_t i = s.streamer->first; i < s.blocks.size(); ++i)
    if (s.streamer->blocks[i].cursor != s.streamer->blocks[i].host.size())
      throw std::runtime_error("transformer offload: staged block size does not match its plan");

  s.refiner_final_norm = bf("token_refiner.final_norm.weight");
  s.final_norm = bf("final_layer.norm.weight");
  if (full_adaln) {
    s.final_full_adaln = quant("final_layer.adaln_proj.linear", final_adaln_out,
                               config.timestep_embed_dim);
    s.final_full_adaln.bias = ptr("final_layer.adaln_proj.linear.bias");
    s.final_full_adaln.bias_format = QuantFormat::kBF16;
  } else {
    s.final_adaln_w = f32("final_layer.adaln_proj.linear.weight");
    s.final_adaln_b = f32("final_layer.adaln_proj.linear.bias");
  }

  s.condition_proj = quant("condition_proj", hidden, config.text_dim);
  s.condition_proj.bias = ptr("condition_proj.bias");
  s.condition_proj.bias_format = QuantFormat::kF32;

  auto fp32_layer = [&](const std::string& name, int out_features, int in_features) {
    QuantWeight w;
    w.format = QuantFormat::kF32;
    w.data = ptr(name + ".weight");
    w.out_features = out_features;
    w.in_features = in_features;
    w.bias = ptr(name + ".bias");
    w.bias_format = QuantFormat::kF32;
    return w;
  };
  s.video_in = fp32_layer("video_patch_proj", hidden, patch);
  s.audio_in = fp32_layer("audio_patch_proj", hidden, config.audio_in_channels);
  s.video_out = fp32_layer("final_layer.video_out", patch, hidden);
  s.audio_out = fp32_layer("final_layer.audio_out", config.audio_in_channels, hidden);

  if (full_adaln) {
    s.timestep_embedding.load(checkpoint, config.timestep_freq_dim,
                              config.timestep_hidden_dim, config.timestep_embed_dim);
  } else {
    s.table.load(checkpoint);
  }
}

// ---------------------------------------------------------------------------


}  // namespace slopfab::dit
