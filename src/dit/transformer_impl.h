#pragma once
#include "transformer_support.h"

namespace slopfab::dit {
struct Transformer::Impl {
  TransformerConfig cfg;
  ModelDescriptor model;
  TransformerArchitecture architecture = TransformerArchitecture::kUnknown;
  AdaLNLookup lookup = AdaLNLookup::kLinear;
  AdaLNTable table;
  FullAdaLNTimestepEmbedding timestep_embedding;

  cuda::Stream stream;
  cublasHandle_t blas = nullptr;
  cuda::LinearRunner linear;
  cuda::LoraRunner lora;
  Workspace ws;

  DeviceBuffer<uint8_t> arena;
  std::unique_ptr<BlockStreamer> streamer;
  size_t arena_bytes = 0;

  std::vector<BlockWeights> blocks;
  std::vector<BlockWeights> refiner;
  const __nv_bfloat16* refiner_final_norm = nullptr;
  const __nv_bfloat16* final_norm = nullptr;
  const float* final_adaln_w = nullptr;
  const float* final_adaln_b = nullptr;
  QuantWeight final_full_adaln;
  QuantWeight condition_proj, video_in, audio_in, video_out, audio_out;

  // Per request.
  SequenceLayout layout;
  PackedIndices indices;
  bool has_sequence = false;
  bool attention_configuration_locked = false;
  Transformer::DebugAttentionRoutes attention_routes;
  int num_text = 0;
  Carve carve;
  // Frame-banded attention. `attn_band` is a request-level setting; `d_band`
  // holds the per-query-tile key ranges the kernel reads, built once in
  // `prepare_sequence` and empty when banding is off.
  int attn_band = 0;
  AttentionMode attention_mode = AttentionMode::kFlash2;
  int row_chunk = kRowChunk;
  bool query_chunking = true;
  bool is_vsa() const { return model.compressed_attention; }
  bool compact_queries(AttentionMode mode) const {
    return !is_vsa() && query_chunking && mode == AttentionMode::kFlash2 &&
           (cfg.attention_head_dim == 64 || cfg.attention_head_dim == 128) &&
           row_chunk % cuda::attention_fused_query_tile() == 0 &&
           block_capture_path.empty() && graph_capture_path.empty() &&
           transformer_capture_path.empty() && sol_capture_path.empty();
  }
  int denoise_step = -1;
  SolSchedule sol_schedule;
  std::string sol_capture_path;
  int sol_capture_step = 0;
  int sol_capture_layer = 0;
  bool sol_capture_done = false;
  std::string block_capture_path;
  int block_capture_step = 0;
  int block_capture_layer = 0;
  bool block_capture_done = false;
  struct ActiveBlockCapture {
    H3BlockCaptureHeader header{};
    std::vector<uint16_t> input;
    std::vector<int32_t> selectors;
    std::vector<float> code, cosine, sine;
    std::vector<int32_t> ranges;
    bool active = false;
  } block_capture;
  std::string graph_capture_path;
  int graph_capture_step = 0;
  bool graph_capture_done = false;
  struct ActiveGraphCapture {
    H3MainGraphCaptureHeader header{};
    std::vector<uint16_t> input;
    std::vector<int32_t> selectors;
    std::vector<float> code, cosine, sine;
    std::vector<int32_t> ranges;
    bool active = false;
  } graph_capture;
  std::string transformer_capture_path;
  int transformer_capture_step = 0;
  bool transformer_capture_done = false;
  struct ActiveTransformerCapture {
    H3TransformerCaptureHeader header{};
    std::vector<float> prompt, video, audio, code, cosine, sine;
    std::vector<int32_t> selectors, ranges, video_ts, audio_ts;
    std::vector<uint16_t> text_cache, packed_input, main_final;
    std::vector<float> video_output, audio_output;
    uint32_t text_stage = 0;
    bool text_active = false;
    bool forward_active = false;
  } transformer_capture;
  bool tensor_diag = false;
  bool sol_pipeline_diag = false;
  DeviceBuffer<cuda::TensorScan> d_tensor_diag;

  void diagnose(const char* stage,const __nv_bfloat16* p,size_t n,int layer);

  static uint64_t fnv64_append(uint64_t hash, const void* data, size_t bytes);

  uint64_t capture_hash(const __nv_bfloat16* data, size_t elements,
                        uint64_t seed = 1469598103934665603ull);

  void begin_block_capture(const __nv_bfloat16* residual,
                           const int32_t* selectors, const float* cosine,
                           const float* sine, int rows, int layer,
                           AttentionMode mode);

  void capture_block_qkv(const __nv_bfloat16* q, const __nv_bfloat16* k,
                         const __nv_bfloat16* v);

  void capture_block_attention(const __nv_bfloat16* attention);

  void capture_block_attention_residual(const __nv_bfloat16* residual);

  void finish_block_capture(const __nv_bfloat16* residual);

  void begin_graph_capture(const __nv_bfloat16* residual,
                           const int32_t* selectors, const float* cosine,
                           const float* sine, int rows, int layer,
                           AttentionMode mode);

  void capture_graph_boundary(const __nv_bfloat16* residual, int layer);

  void begin_transformer_text_capture(const float* prompt, int rows);

  void capture_transformer_text_stage(const __nv_bfloat16* values,
                                      int rows, int dim);

  void begin_transformer_forward_capture(const float* video_latents,
                                         const float* audio_latents,
                                         int video_rows, int audio_rows);

  void capture_transformer_packed_input(const __nv_bfloat16* values);

  void capture_transformer_main_final(const __nv_bfloat16* values);

  void finish_transformer_capture(const float* video_velocity,
                                  const float* audio_velocity);

  void capture_sol_inputs(const __nv_bfloat16* q, const __nv_bfloat16* k,
                          const __nv_bfloat16* v, int rows, int layer);
  DeviceBuffer<int32_t> d_band;
  DeviceBuffer<int32_t> vsa_rows, vsa_sizes, vsa_row_tiles;
  DeviceBuffer<__nv_bfloat16> vsa_compressed;
  cuda::VsaConfig vsa_config;
  std::vector<int32_t> host_band;
  DeviceBuffer<float> rope_cos, rope_sin;
  std::vector<float> host_rope_cos, host_rope_sin;
  DeviceBuffer<int32_t> d_text_idx, d_audio_idx, d_video_idx;
  DeviceBuffer<__nv_bfloat16> text_cache;
  DeviceBuffer<__nv_bfloat16> hidden;

  // Block-span residual cache (dit/block_cache.h). **One** buffer the size of
  // the residual stream — 844 MB at the production geometry — allocated only
  // when the feature is on.
  //
  // It serves two roles in sequence rather than two buffers in parallel. On a
  // computed step it first holds `x` as it entered the span, because the span
  // mutates the stream in place and the delta needs both ends; then
  // `launch_sub_bf16(x, bc_delta, bc_delta)` overwrites it with the delta
  // itself. That in-place subtract is safe because the kernel is elementwise at
  // a single index, and it is worth the subtlety: a second buffer here is 844
  // MB on a card that already holds 19.6 GiB of weights plus 4.5 GB of q/k/v,
  // and is the most likely place a production-geometry run runs out.
  //
  // `bc_have_delta` says whether the buffer holds a captured span or the
  // "before" state / uninitialised memory. Adding either to the residual stream
  // is the one way this feature reaches a NaN rather than merely a drift.
  BlockCache block_cache;
  BlockSpan bc_span;
  DeviceBuffer<__nv_bfloat16> bc_delta;
  bool bc_have_delta = false;
  DeviceBuffer<float> d_video_rows, d_audio_rows;  // latents in, velocities out
  DeviceBuffer<float> d_video_head, d_audio_head;

  // Per step.
  DeviceBuffer<float> d_code;
  DeviceBuffer<float> mod;        // [layers][6][T*3][hidden]
  DeviceBuffer<float> final_mod;  // [2][T][hidden]
  DeviceBuffer<int32_t> d_adaln, d_ts_video, d_ts_audio;
  int mod_timesteps = 0;

  // Staging for the two async host->device copies whose source would otherwise
  // be a local: cudaMemcpyAsync from pageable memory is not guaranteed to have
  // consumed the buffer by the time it returns. What actually makes these safe
  // is the synchronise at the end of `forward`; if step pipelining is ever
  // added and that sync removed, these must become PinnedBuffers, because the
  // driver's inline-copy threshold for small transfers is not a guarantee.
  std::vector<float> host_code;
  std::vector<int32_t> host_ts;

  Impl() {
    const char* full_queries = std::getenv("SLOPFAB_DIT_FULL_QUERIES");
    query_chunking = !(full_queries != nullptr && full_queries[0] == '1');
    SLOPFAB_CUBLAS_CHECK(cuda::cublas_create(&blas));
    linear.init(blas, stream.get());

    // The native nvfp4 GEMM is 2.6-4.1x the dequantise-then-cuBLAS path and is
    // off unless asked for, because it quantises activations to E2M1 and that
    // costs ~9% rms per layer against a bf16-activation reference — a property
    // of the format, not a defect (see the README). Whether that survives 50
    // blocks and 29 steps is an end-to-end question, and this switch exists so
    // it can be answered by generating the same seed both ways rather than
    // argued about. Same shape as SLOPFAB_CUBLAS_PEDANTIC in vit_decoder.cu.
    const char* native = std::getenv("SLOPFAB_NATIVE_NVFP4");
    if (native != nullptr && native[0] == '1') linear.set_native(true);
    const char* capture = std::getenv("SLOPFAB_SOL_CAPTURE");
    const char* block_capture_env = std::getenv("SLOPFAB_H3_BLOCK_CAPTURE");
    const char* graph_capture_env = std::getenv("SLOPFAB_H3_GRAPH_CAPTURE");
    const char* transformer_capture_env =
        std::getenv("SLOPFAB_H3_TRANSFORMER_CAPTURE");
    const char* diag = std::getenv("SLOPFAB_TENSOR_DIAG");
    tensor_diag=diag!=nullptr&&diag[0]=='1';
    if(tensor_diag)d_tensor_diag.allocate(1);
    const char* sol_pipe=std::getenv("SLOPFAB_SOL_PIPELINE");
    sol_pipeline_diag=sol_pipe!=nullptr&&sol_pipe[0]=='1';
    if (capture != nullptr && *capture != '\0') {
      sol_capture_path = capture;
      const char* step = std::getenv("SLOPFAB_SOL_CAPTURE_STEP");
      const char* layer = std::getenv("SLOPFAB_SOL_CAPTURE_LAYER");
      if (step != nullptr && *step != '\0') sol_capture_step = std::atoi(step);
      if (layer != nullptr && *layer != '\0') sol_capture_layer = std::atoi(layer);
    }
    if (block_capture_env && *block_capture_env) {
      block_capture_path = block_capture_env;
      const char* step = std::getenv("SLOPFAB_H3_BLOCK_CAPTURE_STEP");
      const char* layer = std::getenv("SLOPFAB_H3_BLOCK_CAPTURE_LAYER");
      if (step && *step) block_capture_step = std::atoi(step);
      if (layer && *layer) block_capture_layer = std::atoi(layer);
    }
    if (graph_capture_env && *graph_capture_env) {
      graph_capture_path = graph_capture_env;
      const char* step = std::getenv("SLOPFAB_H3_GRAPH_CAPTURE_STEP");
      if (step && *step) graph_capture_step = std::atoi(step);
    }
    if (transformer_capture_env && *transformer_capture_env) {
      transformer_capture_path = transformer_capture_env;
      const char* step = std::getenv("SLOPFAB_H3_TRANSFORMER_CAPTURE_STEP");
      if (step && *step) transformer_capture_step = std::atoi(step);
    }
  }
  ~Impl() {
    if (blas != nullptr) cuda::cublas_destroy(blas);
  }

  bool loaded() const { return !blocks.empty(); }

  void require_loaded(const char* what) const {
    if (!loaded()) throw std::runtime_error(std::string("transformer: ") + what +
                                            " called before load()");
  }

  // --- modulation ----------------------------------------------------------

  size_t block_mod_stride() const {
    return static_cast<size_t>(kNumParams) * mod_timesteps * kNumModalities * cfg.hidden_size;
  }
  size_t mod_row_stride() const {
    return static_cast<size_t>(mod_timesteps) * kNumModalities * cfg.hidden_size;
  }

  // Evaluates c(t) for each distinct timestep and expands it through all 51
  // rank-8 projections. 38.7 MB in fp32 at T = 2, which is the whole per-step
  // AdaLN cost — the pruned checkpoint replaced 13 billion parameters with
  // this (spec 3.4).
  void build_modulation(const std::vector<float>& timesteps);

  // --- one block -----------------------------------------------------------

  // Debug capture. Null in production and therefore free; when set, it is
  // called with the residual stream at each labelled point so a bisect can
  // find the first stage that diverges from a CPU reference. Without this the
  // block is opaque from outside, which is exactly why four plausible
  // explanations for the known numerical gap could each be eliminated without
  // the number moving.
  std::function<void(const char*, const __nv_bfloat16*, int, int)> stage_hook;

  void emit_stage(const char* label, const __nv_bfloat16* x, int rows, int dim) {
    (void)label;
    capture_transformer_text_stage(x, rows, dim);
    if (!stage_hook) return;
    SLOPFAB_CUDA_CHECK(cudaStreamSynchronize(stream.get()));
    stage_hook(label, x, rows, dim);
  }

  void run_block(const BlockWeights& b, const float* mod_base, int rows, __nv_bfloat16* x,
                 const int32_t* adaln_idx, const float* cos, const float* sin, __nv_bfloat16* q,
                 __nv_bfloat16* k, __nv_bfloat16* v, __nv_bfloat16* attn_out,
                 __nv_bfloat16* normed, __nv_bfloat16* fused, __nv_bfloat16* act,
                 __nv_bfloat16* branch, AttentionMode block_attention_mode, int layer = -1);
};
}  // namespace slopfab::dit
