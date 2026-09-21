#include "encoder_internal.cuh"
#include "slopfab/text/backend_capabilities.h"

namespace slopfab::text {
using namespace encoder_detail;

// --- Encoder ------------------------------------------------------------------

struct Encoder::Impl {
  EncoderConfig cfg;
  EncoderStats stats;
  Residency mode = Residency::kResident;
  bool loaded = false;

  const SafeTensors* checkpoint = nullptr;
  const TensorView* embed = nullptr;
  // Non-null only for the nvfp4 build, whose embedding table is I8 with a
  // per-row F32 scale rather than BF16.
  const TensorView* embed_scale = nullptr;
  LayerLayout layout;
  // One set of seven `weight_scale_2` values per layer. Host floats, read once
  // at load: 350 scalars is not worth a device allocation or a re-read per
  // encode.
  std::vector<LayerGlobalScales> globals;

  // kResident: one blob per layer. kStreaming: two, ping-ponged.
  std::vector<DeviceBuffer<uint8_t>> resident;
  DeviceBuffer<uint8_t> ping[2];
  PinnedBuffer<uint8_t> staging[2];
  // Set when the whole checkpoint mapping is page-locked, which lets each
  // weight DMA straight out of it instead of being memcpy'd into a pinned
  // staging buffer first. See `try_register_mapping`.
  bool mapping_registered = false;
  const void* registered_base = nullptr;

  cublasHandle_t cublas = nullptr;
  cudaStream_t compute = nullptr;
  cudaStream_t transfer = nullptr;
  cudaEvent_t upload_done[2] = {nullptr, nullptr};
  cudaEvent_t compute_done[2] = {nullptr, nullptr};

  Workspace ws;
  slopfab::cuda::LinearRunner linear;
  QwenVisionEncoder vision;
  const std::vector<QwenPixelValues>* pending_images = nullptr;

  void open_device() {
    if (cublas != nullptr)
      return;
    SLOPFAB_CUBLAS_CHECK(slopfab::cuda::cublas_create(&cublas));
    // Non-blocking rather than the legacy default stream: the streaming path
    // needs the upload stream to run concurrently with compute, and the legacy
    // default stream serialises against every other blocking stream.
    SLOPFAB_CUDA_CHECK(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
    SLOPFAB_CUDA_CHECK(cudaStreamCreateWithFlags(&transfer, cudaStreamNonBlocking));
    for (int i = 0; i < 2; ++i) {
      SLOPFAB_CUDA_CHECK(cudaEventCreateWithFlags(&upload_done[i], cudaEventDisableTiming));
      SLOPFAB_CUDA_CHECK(cudaEventCreateWithFlags(&compute_done[i], cudaEventDisableTiming));
    }
    linear.init(cublas, compute);
  }

  void close_device() {
    for (int i = 0; i < 2; ++i) {
      if (upload_done[i] != nullptr)
        cudaEventDestroy(upload_done[i]);
      if (compute_done[i] != nullptr)
        cudaEventDestroy(compute_done[i]);
      upload_done[i] = nullptr;
      compute_done[i] = nullptr;
    }
    if (transfer != nullptr)
      cudaStreamDestroy(transfer);
    if (compute != nullptr)
      cudaStreamDestroy(compute);
    if (cublas != nullptr)
      slopfab::cuda::cublas_destroy(cublas);
    transfer = nullptr;
    compute = nullptr;
    cublas = nullptr;
  }

  // Page-locks the entire checkpoint mapping so weights can be uploaded
  // without a host-side copy.
  //
  // The staging path this replaces was, measured, 96% of a streaming encode:
  // 57-62 ms per layer of single-threaded `memcpy` against 9.6 ms of DMA. Most
  // of that was not memcpy bandwidth but soft page faults on the 27 GB mapping
  // (~119k pages per layer). Registering the range up front pays those faults
  // once and lifts H2D from 8.7 GB/s pageable to ~42 GB/s.
  //
  // Best-effort by design: registering tens of gigabytes can fail on a machine
  // short of physical memory or lockable pages, and that is not a reason to
  // refuse to run. On failure the staging path still works, just slower.
  void try_register_mapping() {
    mapping_registered = false;
    registered_base = nullptr;
    if (checkpoint == nullptr)
      return;
    const void* base = checkpoint->mapping_base();
    if (base == nullptr)
      return;

    // cudaHostRegister wants a page-aligned range. The mapping base is already
    // allocation-granularity aligned, and a file mapping always covers whole
    // pages, so rounding the length up stays inside it.
    constexpr size_t kPage = 4096;
    const size_t bytes = (checkpoint->file_size() + kPage - 1) / kPage * kPage;

    const cudaError_t status =
        cudaHostRegister(const_cast<void*>(base), bytes, cudaHostRegisterReadOnly);
    if (status == cudaSuccess) {
      mapping_registered = true;
      registered_base = base;
    } else {
      // Clear the sticky error so the next real call is not misattributed.
      cudaGetLastError();
    }
  }

  void unregister_mapping() {
    if (!mapping_registered || registered_base == nullptr)
      return;
    cudaHostUnregister(const_cast<void*>(registered_base));
    mapping_registered = false;
    registered_base = nullptr;
  }

  // Packs layer `layer` into pinned slot `slot` and starts its upload to
  // `dst`. Pageable memory would make cudaMemcpyAsync synchronous and force a
  // staging copy inside the driver, which is exactly what the pinned buffers
  // are for (spec section 7.2).
  void stage_upload(int layer, int slot, uint8_t* dst) {
    // The previous upload out of this pinned buffer must have landed before it
    // is overwritten.
    if (mapping_registered) {
      // No host copy at all: eighteen DMAs straight out of the page-locked
      // mapping into the layer arena. Nothing is written on the host, so the
      // pinned slot is unused and there is nothing to wait for beyond the
      // previous upload into this arena, which the caller's ping-pong event
      // already orders.
      SLOPFAB_CUDA_CHECK(cudaEventSynchronize(upload_done[slot]));
      upload_layer_direct(*checkpoint, cfg, layer, layout, dst, transfer);
      SLOPFAB_CUDA_CHECK(cudaEventRecord(upload_done[slot], transfer));
      return;
    }

    // The previous upload out of this pinned buffer must have landed before it
    // is overwritten.
    SLOPFAB_CUDA_CHECK(cudaEventSynchronize(upload_done[slot]));
    pack_layer(*checkpoint, cfg, layer, layout, staging[slot].get());
    SLOPFAB_CUDA_CHECK(cudaMemcpyAsync(dst, staging[slot].get(), layout.total_bytes,
                                       cudaMemcpyHostToDevice, transfer));
    SLOPFAB_CUDA_CHECK(cudaEventRecord(upload_done[slot], transfer));
  }

  void free_weights() {
    unregister_mapping();
    resident.clear();
    for (int i = 0; i < 2; ++i) {
      ping[i].reset();
      staging[i].reset();
    }
  }

  // Returns the encoder to a reusable unloaded state even after an asynchronous
  // upload or a partially-created CUDA object failed. Cleanup is deliberately
  // best-effort/noexcept: the original load error is the useful one to report.
  void reset_runtime_state(bool close_handles) noexcept {
    if (transfer != nullptr)
      (void)cudaStreamSynchronize(transfer);
    if (compute != nullptr)
      (void)cudaStreamSynchronize(compute);
    vision.unload();
    free_weights();
    ws = Workspace();
    loaded = false;
    checkpoint = nullptr;
    embed = nullptr;
    embed_scale = nullptr;
    globals.clear();
    pending_images = nullptr;
    stats = EncoderStats();
    if (close_handles)
      close_device();
    // cudaStreamSynchronize and resource destruction may each report the same
    // deferred upload failure. Consume it only after all owned work is gone so
    // a subsequent encoder does not misattribute the old error to cublasCreate.
    (void)cudaGetLastError();
  }
};

Encoder::Encoder() : impl_(new Impl()) {
}

Encoder::~Encoder() {
  unload();
  impl_->close_device();
}

const EncoderConfig& Encoder::config() const {
  return impl_->cfg;
}

size_t Encoder::weight_bytes() const {
  return impl_->stats.weight_bytes;
}

Residency Encoder::residency() const {
  return impl_->mode;
}

WeightFormat Encoder::format() const {
  return impl_->cfg.format;
}

const EncoderStats& Encoder::stats() const {
  return impl_->stats;
}

void Encoder::unload() {
  impl_->reset_runtime_state(false);
}

void Encoder::load(const SafeTensors& checkpoint, const EncoderConfig& config) {
  // See the note on `SafeTensors::prefetch`: this loader consumes the whole
  // file, so it asks for it up front rather than one page fault at a time.
  // Issued before anything else because it is asynchronous — validation and
  // the scale reads below run while the OS is already reading the file. It
  // matters more here than anywhere else: `try_register_mapping` below calls
  // `cudaHostRegister` over the whole mapping, which must fault every page
  // resident synchronously, one outstanding request at a time.
  checkpoint.prefetch();
  unload();
  const auto t0 = std::chrono::steady_clock::now();

  Impl& s = *impl_;
  try {
    s.cfg = config;
    if (config.arithmetic == EncoderArithmetic::kExact &&
        !supports_exact_text_layer({config.max_prompt_tokens, config.hidden_size,
                                    config.num_attention_heads, config.num_key_value_heads,
                                    config.head_dim, config.intermediate_size,
                                    config.rms_norm_eps}))
      throw std::invalid_argument("CUDA text encoder: unsupported exact layer capability");
    // Detection is per file and there is no flag: the two builds differ in tensor
    // count, dtype and shape, so validation catches a mismatch rather than
    // letting it become wrong numbers. An explicitly set format is checked
    // against the file rather than trusted.
    s.cfg.format = detect_weight_format(checkpoint);
    if (config.format != WeightFormat::kAuto && config.format != s.cfg.format) {
      throw std::runtime_error(
          "text encoder: the checkpoint declares the other weight format; detection is per file");
    }
    validate_checkpoint(checkpoint, s.cfg);
    s.checkpoint = &checkpoint;
    s.embed = &checkpoint.at("model.embed_tokens.weight");
    s.embed_scale = checkpoint.find("model.embed_tokens.weight_scale");
    s.layout = make_layer_layout(s.cfg);
    s.globals.resize(static_cast<size_t>(s.cfg.num_layers));
    for (int i = 0; i < s.cfg.num_layers; ++i) {
      s.globals[static_cast<size_t>(i)] = read_global_scales(checkpoint, s.cfg, i);
    }
    const size_t layer_bytes = s.layout.total_bytes;
    const size_t weight_bytes = layer_bytes * static_cast<size_t>(s.cfg.num_layers);

    Residency mode = config.residency;
    if (mode == Residency::kAuto || mode == Residency::kResident) {
      size_t free_bytes = 0;
      size_t total_bytes = 0;
      SLOPFAB_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
      const size_t required = resident_request_bytes(s.cfg, weight_bytes, total_bytes);
      if (free_bytes < required) {
        if (mode == Residency::kResident) {
          throw std::runtime_error(
              "text encoder: resident mode requires " +
              std::to_string(required / (size_t(1) << 20)) +
              " MiB free for weights, the maximum request, and the driver reserve; only " +
              std::to_string(free_bytes / (size_t(1) << 20)) + " MiB is free");
        }
        mode = Residency::kStreaming;
      } else {
        mode = Residency::kResident;
      }
    }
    s.open_device();

    // Page-lock the mapping if we can, which removes the host copy from every
    // upload in both modes. Best effort: if it fails we fall back to staging.
    s.try_register_mapping();

    // The staging buffers are only needed when registration failed. Allocating
    // 930 MB of pinned memory that nothing will ever touch would be a waste of
    // exactly the resource that made registration fail in the first place.
    if (!s.mapping_registered) {
      s.staging[0].allocate(layer_bytes);
      s.staging[1].allocate(layer_bytes);
      s.stats.host_pinned_bytes = 2 * layer_bytes;
    } else {
      s.stats.host_pinned_bytes = 0;
    }

    if (mode == Residency::kResident) {
      try {
        s.resident.resize(static_cast<size_t>(config.num_layers));
        for (int i = 0; i < config.num_layers; ++i)
          s.resident[static_cast<size_t>(i)].allocate(layer_bytes);
      } catch (const std::exception&) {
        s.resident.clear();
        if (config.residency != Residency::kAuto)
          throw;
        // Only kAuto is allowed to change its mind: an explicit kResident that
        // does not fit is a sizing error the caller wants to hear about.
        std::printf(
            "  text encoder: %.2f GB of layer weights did not fit; falling back to streaming\n",
            static_cast<double>(weight_bytes) / (1 << 30));
        mode = Residency::kStreaming;
      }
    }

    if (mode == Residency::kResident) {
      for (int i = 0; i < config.num_layers; ++i) {
        s.stage_upload(i, i % 2, s.resident[static_cast<size_t>(i)].get());
      }
      SLOPFAB_CUDA_CHECK(cudaStreamSynchronize(s.transfer));
      for (int i = 0; i < 2; ++i)
        s.staging[i].reset();
      s.stats.host_pinned_bytes = 0;
      s.stats.weight_bytes = weight_bytes;
    } else {
      // Two device buffers, ping-ponged: layer i+1 uploads while layer i
      // computes. 0.98 GB instead of 24.39 GB.
      for (int i = 0; i < 2; ++i)
        s.ping[i].allocate(layer_bytes);
      s.stats.weight_bytes = 2 * layer_bytes;
    }

    s.mode = mode;
    s.stats.load_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    s.loaded = true;
  } catch (...) {
    // A resident load queues many uploads before the final synchronization.
    // Roll every allocation, registration, event, stream and handle back now;
    // waiting for the Encoder destructor leaves the CUDA thread in a failed
    // state and can make an unrelated streaming encoder's cublasCreate fail.
    s.reset_runtime_state(true);
    throw;
  }
}

PromptEmbedding Encoder::encode(const std::vector<int32_t>& token_ids, EncoderTrace* trace) {
  Impl& s = *impl_;
  require(s.loaded, "encode: load() has not been called");
  require(!token_ids.empty(),
          "encode: the prompt tokenised to zero tokens. The reference does not handle an empty "
          "prompt and neither does this port");
  require(static_cast<int>(token_ids.size()) <= s.cfg.max_prompt_tokens,
          "encode: " + std::to_string(token_ids.size()) + " tokens exceeds max_prompt_tokens " +
              std::to_string(s.cfg.max_prompt_tokens) +
              ". Truncating would be silently observable to the user, so this is an error");

  const auto t0 = std::chrono::steady_clock::now();
  const int L = static_cast<int>(token_ids.size());
  const int hidden = s.cfg.hidden_size;
  const size_t stream_elems = static_cast<size_t>(L) * hidden;

  // hidden_states[0] is the embedding lookup, unscaled and with no positional
  // add. Gathered on the host: only L of 151936 rows are ever read.
  std::vector<uint16_t> host_embed;
  gather_embedding_rows(*s.embed, s.embed_scale, token_ids, host_embed);
  QwenVisionEmbedding visual;
  QwenMultimodalPlan mm_plan;
  if (s.pending_images && !s.pending_images->empty()) {
    s.vision.load(*s.checkpoint);
    visual = s.cfg.arithmetic == EncoderArithmetic::kExact
                 ? s.vision.encode_exact(*s.pending_images)
                 : s.vision.encode(*s.pending_images);
    std::vector<QwenImageGrid> grids;
    grids.reserve(s.pending_images->size());
    for (const auto& im : *s.pending_images)
      grids.push_back(im.grid);
    mm_plan = qwen3vl_multimodal_plan(token_ids, grids);
    require(static_cast<int>(mm_plan.image_rows.size()) == visual.tokens,
            "encode: visual output count disagrees with image-pad tokens");
    for (int i = 0; i < visual.tokens; ++i)
      std::memcpy(host_embed.data() + static_cast<size_t>(mm_plan.image_rows[i]) * hidden,
                  visual.main.data() + static_cast<size_t>(i) * hidden,
                  static_cast<size_t>(hidden) * sizeof(uint16_t));
    s.vision.unload();
  }
  DeviceBuffer<uint16_t> x(stream_elems);
  DeviceBuffer<uint16_t> trace_device;
  if (trace != nullptr) {
    require(s.cfg.arithmetic == EncoderArithmetic::kExact,
            "encode: boundary trace requires exact conditioner arithmetic");
    trace_device.allocate(static_cast<size_t>(s.cfg.num_layers) * stream_elems);
  }
  x.copy_from_host(host_embed.data(), host_embed.size(), s.compute);
  __nv_bfloat16* xp = reinterpret_cast<__nv_bfloat16*>(x.get());

  // One cos/sin pair for the whole request, shared by all 50 layers and by both
  // q and k.
  const std::vector<float> inv_freq = rope_inv_freq(s.cfg.head_dim, s.cfg.rope_theta);
  std::vector<float> cos_host;
  std::vector<float> sin_host;
  if (visual.tokens)
    qwen3vl_decoder_rope_tables(mm_plan, L, cos_host, sin_host, s.cfg.head_dim, s.cfg.rope_theta);
  else
    build_rope_tables(L, inv_freq, cos_host, sin_host);
  DeviceBuffer<float> cos(cos_host.size());
  DeviceBuffer<float> sin(sin_host.size());
  cos.copy_from_host(cos_host.data(), cos_host.size(), s.compute);
  sin.copy_from_host(sin_host.data(), sin_host.size(), s.compute);
  DeviceBuffer<int32_t> image_rows;
  DeviceBuffer<uint16_t> deep[3];
  if (visual.tokens) {
    image_rows.allocate(mm_plan.image_rows.size());
    image_rows.copy_from_host(mm_plan.image_rows.data(), mm_plan.image_rows.size(), s.compute);
    for (int j = 0; j < 3; ++j) {
      deep[j].allocate(visual.deepstack[j].size());
      deep[j].copy_from_host(visual.deepstack[j].data(), visual.deepstack[j].size(), s.compute);
    }
  }
  auto inject = [&](int layer) {
    if (!visual.tokens)
      return;
    const int j = qwen3vl_deepstack_slot(layer);
    if (j >= 0)
      cuda::launch_scatter_add_rows(reinterpret_cast<__nv_bfloat16*>(deep[j].get()),
                                    image_rows.get(), xp, visual.tokens, hidden, s.compute);
  };

  LayerDims dims;
  dims.format = s.cfg.format;
  dims.num_tokens = L;
  dims.hidden = hidden;
  dims.num_heads = s.cfg.num_attention_heads;
  dims.num_kv_heads = s.cfg.num_key_value_heads;
  dims.head_dim = s.cfg.head_dim;
  dims.intermediate = s.cfg.intermediate_size;
  dims.rms_norm_eps = s.cfg.rms_norm_eps;
  if (s.cfg.arithmetic == EncoderArithmetic::kShipped)
    s.ws.reserve(layer_workspace_bytes(dims));

  auto forward_layer = [&](const LayerWeights& weights) {
    if (s.cfg.arithmetic == EncoderArithmetic::kExact) {
      s.ws.reserve(exact_layer_workspace_bytes(weights, dims));
      encoder_layer_forward_exact(s.compute, weights, dims, cos.get(), sin.get(), xp, s.ws);
    } else {
      encoder_layer_forward(s.cublas, s.compute, s.linear, weights, dims, cos.get(), sin.get(), xp,
                            s.ws);
    }
  };

  const int N = s.cfg.num_layers;
  auto capture_layer = [&](int layer) {
    if (trace_device.get() == nullptr)
      return;
    SLOPFAB_CUDA_CHECK(
        cudaMemcpyAsync(trace_device.get() + static_cast<size_t>(layer) * stream_elems, x.get(),
                        stream_elems * sizeof(uint16_t), cudaMemcpyDeviceToDevice, s.compute));
  };
  if (s.mode == Residency::kResident) {
    for (int i = 0; i < N; ++i) {
      const LayerWeights w =
          layer_weights_from_blob(s.resident[static_cast<size_t>(i)].get(), s.layout, s.cfg,
                                  s.globals[static_cast<size_t>(i)]);
      forward_layer(w);
      inject(i);
      capture_layer(i);
    }
  } else {
    s.stage_upload(0, 0, s.ping[0].get());
    for (int i = 0; i < N; ++i) {
      const int slot = i % 2;
      SLOPFAB_CUDA_CHECK(cudaStreamWaitEvent(s.compute, s.upload_done[slot], 0));
      const LayerWeights w = layer_weights_from_blob(s.ping[slot].get(), s.layout, s.cfg,
                                                     s.globals[static_cast<size_t>(i)]);
      forward_layer(w);
      inject(i);
      capture_layer(i);
      SLOPFAB_CUDA_CHECK(cudaEventRecord(s.compute_done[slot], s.compute));

      if (i + 1 < N) {
        const int next = (i + 1) % 2;
        // The buffer layer i+1 lands in is the one layer i-1 computed from, so
        // that compute must finish first. Without this the upload would race
        // ahead and rewrite weights mid-GEMM — silently, and only under load.
        if (i >= 1)
          SLOPFAB_CUDA_CHECK(cudaStreamWaitEvent(s.transfer, s.compute_done[next], 0));
        s.stage_upload(i + 1, next, s.ping[next].get());
      }
    }
  }

  // fp32 on the host. NO final norm and NO lm_head: the wanted tensor is the
  // raw output of the last layer present (spec section 1.4).
  DeviceBuffer<float> out(stream_elems);
  slopfab::cuda::launch_widen_bf16(xp, out.get(), stream_elems, s.compute);

  PromptEmbedding result;
  result.num_tokens = L;
  result.hidden_size = hidden;
  result.data.resize(stream_elems);
  result.modality_tags.assign(static_cast<size_t>(L), 1);
  if (visual.tokens) {
    size_t at = 0;
    for (const auto& image : *s.pending_images) {
      const size_t count = image.grid.merged_token_count();
      const int first = mm_plan.image_rows[at];
      for (size_t j = 0; j < count + 2; ++j)
        result.modality_tags[static_cast<size_t>(first - 1) + j] = 0;
      at += count;
    }
  }
  out.copy_to_host(result.data.data(), stream_elems, s.compute);
  std::vector<uint16_t> trace_host;
  if (trace_device.get() != nullptr) {
    trace_host.resize(static_cast<size_t>(N) * stream_elems);
    trace_device.copy_to_host(trace_host.data(), trace_host.size(), s.compute);
  }
  SLOPFAB_CUDA_CHECK(cudaStreamSynchronize(s.compute));
  if (trace != nullptr) {
    trace->num_tokens = L;
    trace->hidden_size = hidden;
    trace->layer_residual_bf16 = std::move(trace_host);
  }

  s.stats.workspace_bytes = s.ws.capacity();
  s.stats.activation_bytes = x.nbytes() + cos.nbytes() + sin.nbytes() + out.nbytes();
  s.stats.peak_device_bytes =
      s.stats.weight_bytes + s.stats.workspace_bytes + s.stats.activation_bytes;
  s.stats.last_num_tokens = L;
  s.stats.last_encode_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return result;
}

PromptEmbedding Encoder::encode(const std::vector<int32_t>& token_ids,
                                const std::vector<QwenPixelValues>& images, EncoderTrace* trace) {
  Impl& s = *impl_;
  require(!images.empty(), "encode: multimodal overload requires at least one image");
  s.pending_images = &images;
  try {
    PromptEmbedding out = encode(token_ids, trace);
    s.pending_images = nullptr;
    return out;
  } catch (...) {
    s.pending_images = nullptr;
    throw;
  }
}

PromptEmbedding Encoder::encode(const Tokenizer& tokenizer, const std::string& prompt) {
  require(!prompt.empty(), "encode: the prompt is empty");
  // add_special_tokens=False: no BOS, no EOS, no chat template. One extra
  // leading token shifts every RoPE position and, because attention is causal,
  // changes every row (spec section 1.2).
  return encode(tokenizer.encode(prompt));
}

} // namespace slopfab::text
