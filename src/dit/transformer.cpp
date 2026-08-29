// The H3 omni transformer forward pass. See the header for the four hazards;
// this file is the arithmetic, the memory plan and the reasons behind both.
//
// **The memory plan is the design.** 19.5 GiB of weights are resident on a
// 32 GB card, leaving roughly 12 GB for everything else, and at seq = 37710 the
// naive activation buffers do not fit comfortably:
//
//     hidden [S, 5376]  bf16  =  405 MB
//     qkv    [S, 21504] bf16  = 1.62 GB
//     ffn    [S, 28672] bf16  = 2.16 GB
//
// Every one of `qkv_proj`, the FFN, the two input projections and the two
// output heads is strictly row-wise, so they are processed `kRowChunk` rows at
// a time — exactly equivalent, and it cuts the transient buffers by four to
// five times. Attention is the exception: it needs the whole sequence on the
// key axis, so `q`, `k` and `v` are materialised in full and only the query
// axis is blocked, which `attention_forward` already does internally.
//
// The workspace is reserved once at the high-water mark in `prepare_sequence`
// and never grows inside the loop.

#include "vidfab/dit/transformer.h"
#include "vidfab/dit/block_capture.h"
#include "vidfab/dit/rope.h"

#include <functional>
#include <fstream>

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "vidfab/cuda/attention.cuh"
#include "vidfab/cuda/deterministic_attention.cuh"
#include "vidfab/cuda/deterministic_gemm.cuh"
#include "vidfab/cuda/device.h"
#include "vidfab/cuda/diagnostics.cuh"
#include "vidfab/cuda/gemm.cuh"
#include "vidfab/cuda/linear.cuh"
#include "vidfab/cuda/nn_kernels.cuh"
#include "vidfab/cuda/profile.h"
#include "vidfab/cuda/vae_kernels.cuh"
#include "vidfab/cuda/workspace.cuh"
#include "vidfab/dtype.h"
#include "vidfab/json.h"
#include "vidfab/sol_capture.h"
#include "vidfab/tensor_convert.h"

namespace vidfab::cuda {

// Defined in src/cuda/dit_kernels.cu. Declared here rather than in a header
// because they have exactly one caller each; a signature drift is a link
// error, which is the failure mode you want.
void launch_adaln_expand(const float* w, const float* bias, const float* code, float* out,
                         int num_t, int num_modality, int num_param, int channels, int rank,
                         cudaStream_t stream);
void launch_add_rows_bf16(__nv_bfloat16* x, const __nv_bfloat16* branch, size_t n,
                          cudaStream_t stream);

}  // namespace vidfab::cuda

namespace vidfab::dit {
namespace {

using cuda::AttentionBackend;
using cuda::AttentionConfig;
using cuda::ComputeType;
using cuda::DeviceBuffer;
using cuda::QuantFormat;
using cuda::QuantWeight;
using cuda::Workspace;

// Rows per pass through the row-wise stages. 8192 keeps the fused FFN
// projection at 470 MB instead of 2.16 GB while still handing cuBLAS a GEMM
// large enough to reach peak.
constexpr int kRowChunk = 8192;

// Spec 3.2: six modulation parameters, three modalities.
constexpr int kNumParams = 6;
constexpr int kNumModalities = 3;
// Spec 8.3: the final layer emits [shift; scale] and has no modality axis.
constexpr int kFinalParams = 2;

inline size_t align_up(size_t n) { return (n + 255) / 256 * 256; }

QuantFormat format_of(DType dt, const std::string& name) {
  switch (dt) {
    case DType::kF32:
      return QuantFormat::kF32;
    case DType::kBF16:
      return QuantFormat::kBF16;
    case DType::kF16:
      return QuantFormat::kF16;
    case DType::kF8E4M3:
      return QuantFormat::kF8E4M3;
    case DType::kI8:
      return QuantFormat::kI8;
    default:
      throw std::runtime_error("transformer: tensor '" + name + "' has dtype " +
                               dtype_name(dt) + ", which no linear layer can consume");
  }
}

size_t format_bytes(QuantFormat f) {
  switch (f) {
    case QuantFormat::kF32:
      return 4;
    case QuantFormat::kF16:
    case QuantFormat::kBF16:
      return 2;
    case QuantFormat::kF8E4M3:
    case QuantFormat::kI8:
      return 1;
    case QuantFormat::kNVFP4:
    case QuantFormat::kNF4:
      // The qkv split below is the only caller, and for nvfp4 it has to slice
      // the block scales as well as the nibbles. Throwing keeps a half-sized
      // offset from being computed silently.
      throw std::runtime_error("transformer: nvfp4 is not a whole-byte format");
  }
  return 0;
}

struct NF4State {
  int block_size = 0;
  int nested_block_size = 0;
  float nested_offset = 0.0f;
  std::vector<int64_t> shape;
};

std::string nf4_state_name(const std::string& name) {
  return name + ".weight.quant_state.bitsandbytes__nf4";
}

bool is_nf4(const SafeTensors& st, const std::string& name) {
  return st.find(nf4_state_name(name)) != nullptr;
}

NF4State read_nf4_state(const SafeTensors& st, const std::string& name) {
  const std::string state_name = nf4_state_name(name);
  const TensorView* v = st.find(state_name);
  if (v == nullptr || v->dtype != DType::kU8 || v->shape.size() != 1) {
    throw std::runtime_error("transformer: '" + state_name +
                             "' must be a rank-1 U8 JSON tensor");
  }
  std::string text(static_cast<const char*>(v->data), v->nbytes);
  json::Value root;
  try {
    root = json::parse(text);
  } catch (const std::exception& e) {
    throw std::runtime_error("transformer: '" + state_name + "' is not valid JSON (" +
                             e.what() + ")");
  }
  auto require = [&](const char* key) -> const json::Value& {
    const json::Value* value = root.find(key);
    if (value == nullptr) {
      throw std::runtime_error("transformer: '" + state_name + "' is missing '" + key + "'");
    }
    return *value;
  };
  if (require("quant_type").as_string() != "nf4" ||
      require("dtype").as_string() != "bfloat16" ||
      require("nested_dtype").as_string() != "float32") {
    throw std::runtime_error("transformer: '" + state_name +
                             "' has an unsupported bitsandbytes NF4 contract");
  }
  NF4State state;
  state.block_size = static_cast<int>(require("blocksize").as_int());
  state.nested_block_size = static_cast<int>(require("nested_blocksize").as_int());
  state.nested_offset = static_cast<float>(require("nested_offset").as_number());
  for (const json::Value& dim : require("shape").as_array()) state.shape.push_back(dim.as_int());
  if (state.block_size != 64 || state.nested_block_size != 256 ||
      !std::isfinite(state.nested_offset)) {
    throw std::runtime_error("transformer: '" + state_name +
                             "' requires unsupported NF4 block sizes or offset");
  }
  return state;
}

// True when `name` is stored as nvfp4. Structural, from the file itself, rather
// than from a flag or from which checkpoint the caller thinks it opened: a U8
// `.weight` alongside an e4m3 `.weight_scale` carrying one scale per 16
// contracted elements. Both halves are needed — U8 alone would also match a
// byte blob, and it is the scale's trailing dimension that pins the block size
// this port implements. The two bf16 refiner blocks fall out of it for free.
bool is_nvfp4(const SafeTensors& st, const std::string& name, int in_features) {
  const TensorView* w = st.find(name + ".weight");
  const TensorView* s = st.find(name + ".weight_scale");
  if (w == nullptr || s == nullptr) return false;
  if (w->dtype != DType::kU8 || s->dtype != DType::kF8E4M3) return false;
  return s->shape.size() == 2 &&
         s->shape[1] == in_features / static_cast<int>(cuda::kNVFP4BlockSize);
}

// What a `comfy_quant` blob says about its tensor.
struct QuantTag {
  std::string format;             // empty when the tensor carries no blob
  bool full_precision = false;
};

// The blob is the checkpoint's own statement of what its bytes mean, so it is
// parsed rather than pattern-matched. A substring search cannot tell a format
// it does not implement from one it does — it just fails to find its needle and
// carries on — and this is the file's only description of layouts that are
// otherwise indistinguishable by inspection.
QuantTag read_comfy_quant(const SafeTensors& st, const std::string& name) {
  QuantTag tag;
  const TensorView* v = st.find(name + ".comfy_quant");
  if (v == nullptr) return tag;

  std::string text(static_cast<const char*>(v->data), v->nbytes);
  // ComfyUI writes the blob as a byte tensor, which may be NUL-padded to a
  // whole number of elements.
  while (!text.empty() && (text.back() == '\0' || text.back() == ' ' || text.back() == '\n')) {
    text.pop_back();
  }
  if (text.empty()) return tag;

  json::Value root;
  try {
    root = json::parse(text);
  } catch (const std::exception& e) {
    throw std::runtime_error("transformer: '" + name + ".comfy_quant' is not valid JSON (" +
                             e.what() + "): " + text);
  }
  if (const json::Value* f = root.find("format"); f != nullptr) tag.format = f->as_string();
  if (const json::Value* p = root.find("full_precision_matrix_mult"); p != nullptr) {
    tag.full_precision = p->as_bool();
  }

  // Anything else is a layout this port has not been shown, and guessing at one
  // yields finite plausible output rather than a failure.
  if (tag.format != "nvfp4" && tag.format != "float8_e4m3fn") {
    throw std::runtime_error("transformer: '" + name + "' declares quant format '" + tag.format +
                             "', which this port does not implement");
  }
  return tag;
}

std::string shape_string(const std::vector<int64_t>& s) {
  std::string out = "[";
  for (size_t i = 0; i < s.size(); ++i) {
    if (i != 0) out += ", ";
    out += std::to_string(s[i]);
  }
  return out + "]";
}

// --- staged host -> device upload -------------------------------------------
//
// 21 GB through pageable memory is the difference between a three second load
// and a forty second one: a pageable cudaMemcpy stages through a driver-owned
// bounce buffer one chunk at a time with no overlap. Two pinned buffers and two
// events let the next memcpy from the mapping run while the previous DMA is in
// flight.
class Uploader {
 public:
  Uploader(cudaStream_t stream, const cuda::RegisteredMapping* lock)
      : stream_(stream), lock_(lock) {
    for (int i = 0; i < 2; ++i) {
      slot_[i].allocate(kStageBytes);
      VIDFAB_CUDA_CHECK(cudaEventCreateWithFlags(&event_[i], cudaEventDisableTiming));
      // Recorded once so the first wait on each slot is a no-op rather than a
      // wait on an event that was never recorded (which is legal but reads as
      // an accident).
      VIDFAB_CUDA_CHECK(cudaEventRecord(event_[i], stream_));
    }
  }

  ~Uploader() {
    // Best-effort: throwing from a destructor would terminate, and the caller
    // synchronises again before touching any of the uploaded memory.
    cudaStreamSynchronize(stream_);
    for (int i = 0; i < 2; ++i) cudaEventDestroy(event_[i]);
  }

  Uploader(const Uploader&) = delete;
  Uploader& operator=(const Uploader&) = delete;

  // `from_mapping` says the source is the checkpoint mapping itself, which
  // only the caller can know: the alternative sources here are short-lived
  // `std::vector` scratch buffers, and where the heap puts those relative to a
  // 12 GB mapping is luck. Taking the direct path for one of those would DMA
  // out of pageable memory and, worse, return before the scratch buffer is
  // rewritten by the next record. So provenance is passed in, and the range
  // check is only a second opinion that must also agree.
  void copy(void* dst, const void* src, size_t bytes, bool from_mapping) {
    // Source already page-locked: hand the whole range to the DMA engine and
    // return. Nothing is written on the host, so there is no staging slot to
    // wait for, and stream order keeps this correctly sequenced against the
    // staged copies around it.
    if (from_mapping && lock_ != nullptr && lock_->contains(src, bytes)) {
      VIDFAB_CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream_));
      return;
    }

    const uint8_t* s = static_cast<const uint8_t*>(src);
    uint8_t* d = static_cast<uint8_t*>(dst);
    while (bytes > 0) {
      const size_t n = std::min(bytes, kStageBytes);
      VIDFAB_CUDA_CHECK(cudaEventSynchronize(event_[cur_]));
      std::memcpy(slot_[cur_].get(), s, n);
      VIDFAB_CUDA_CHECK(
          cudaMemcpyAsync(d, slot_[cur_].get(), n, cudaMemcpyHostToDevice, stream_));
      VIDFAB_CUDA_CHECK(cudaEventRecord(event_[cur_], stream_));
      cur_ ^= 1;
      s += n;
      d += n;
      bytes -= n;
    }
  }

 private:
  static constexpr size_t kStageBytes = 32u << 20;
  cudaStream_t stream_;
  const cuda::RegisteredMapping* lock_ = nullptr;
  cuda::PinnedBuffer<uint8_t> slot_[2];
  cudaEvent_t event_[2] = {nullptr, nullptr};
  int cur_ = 0;
};

// How a checkpoint tensor reaches the device.
enum class Store {
  // Byte-for-byte. fp8 projection weights, bf16 refiner weights, fp32 scales.
  kVerbatim,
  // Widened to fp32 on the host, then narrowed to bf16. Norm weights, which
  // every kernel here consumes as bf16 regardless of how they were stored.
  kAsBF16,
  // Widened to fp32. The rank-8 AdaLN projections (F16 on disk) and the four
  // fp32 tensors, all of which the spec requires be evaluated in fp32 (9.1).
  kAsF32,
};

struct Record {
  const TensorView* view = nullptr;
  Store store = Store::kVerbatim;
  size_t offset = 0;
  size_t bytes = 0;
};

// Collects the expected tensor list, validating every shape as it goes, and
// accounts for exactly one arena allocation.
class Plan {
 public:
  explicit Plan(const SafeTensors& st) : st_(st) {}

  const TensorView& require(const std::string& name, const std::vector<int64_t>& shape,
                            Store store) {
    const TensorView* v = st_.find(name);
    if (v == nullptr) {
      throw std::runtime_error("transformer: checkpoint is missing '" + name + "' " +
                               shape_string(shape));
    }
    if (v->shape != shape) {
      throw std::runtime_error("transformer: '" + name + "' has shape " +
                               shape_string(v->shape) + ", expected " + shape_string(shape));
    }
    add(name, *v, store);
    return *v;
  }

  // Present-or-not tensors: `.input_scale` is absent exactly on `mlp.fc2`
  // (spec 8.2), `.comfy_quant` is a ComfyUI provenance tag we only need to
  // account for, and `rope.inv_freq` is recomputed rather than read.
  const TensorView* optional(const std::string& name) {
    const TensorView* v = st_.find(name);
    if (v == nullptr) return nullptr;
    consumed_.insert(name);
    return v;
  }

  // A per-tensor scale. Written as a rank-0 tensor in our checkpoint; `[1]` is
  // accepted too because nothing downstream can tell the difference and a
  // requantiser that emits one rather than the other is not wrong.
  const TensorView* optional_scalar(const std::string& name) {
    const TensorView* v = st_.find(name);
    if (v == nullptr) return nullptr;
    if (v->numel() != 1 || v->shape.size() > 1) {
      throw std::runtime_error("transformer: '" + name + "' has shape " +
                               shape_string(v->shape) + ", expected a scalar");
    }
    add(name, *v, Store::kVerbatim);
    return v;
  }

  // Every tensor in the file must have been claimed by the walk above. A
  // checkpoint with extra keys is a different model, and finding that out now
  // beats finding it out from a wrong video.
  void finish() {
    if (consumed_.size() == st_.tensor_count()) return;
    std::string extra;
    int shown = 0;
    for (const auto& kv : st_.tensors()) {
      if (consumed_.count(kv.first) != 0) continue;
      if (shown++ != 0) extra += ", ";
      if (shown > 6) {
        extra += "...";
        break;
      }
      extra += kv.first;
    }
    throw std::runtime_error("transformer: checkpoint has " + std::to_string(st_.tensor_count()) +
                             " tensors but only " + std::to_string(consumed_.size()) +
                             " are part of the model; unclaimed: " + extra);
  }

  const std::map<std::string, Record>& records() const { return records_; }
  size_t arena_bytes() const { return total_; }

 private:
  void add(const std::string& name, const TensorView& v, Store store) {
    Record r;
    r.view = &v;
    r.store = store;
    switch (store) {
      case Store::kVerbatim:
        r.bytes = v.nbytes;
        break;
      case Store::kAsBF16:
        r.bytes = static_cast<size_t>(v.numel()) * 2;
        break;
      case Store::kAsF32:
        r.bytes = static_cast<size_t>(v.numel()) * 4;
        break;
    }
    r.offset = total_;
    total_ += align_up(r.bytes);
    records_[name] = r;
    consumed_.insert(name);
  }

  const SafeTensors& st_;
  std::map<std::string, Record> records_;
  std::set<std::string> consumed_;
  size_t total_ = 0;
};

// One attention + FFN pair. The main blocks and the refiner blocks differ only
// in whether `adaln_w` is set: the refiner has no AdaLN, no RoPE and no
// timestep dependence (spec 6).
struct BlockWeights {
  const __nv_bfloat16* norm1 = nullptr;
  const __nv_bfloat16* norm2 = nullptr;
  const __nv_bfloat16* q_norm = nullptr;
  const __nv_bfloat16* k_norm = nullptr;
  QuantWeight wq, wk, wv, out_proj, fc1, fc2;
  const float* adaln_w = nullptr;
  const float* adaln_b = nullptr;
  QuantWeight full_adaln;
};

// Sizes of the transient buffers `forward` carves, all in one place so that
// `activation_bytes` and the carve cannot drift apart.
struct Carve {
  int chunk = 0;
  size_t qkv = 0;       // one of q/k/v, elements
  size_t normed = 0;
  size_t fused = 0;
  size_t act = 0;
  size_t fbuf = 0;      // elements of one fp32 [chunk, hidden] buffer
  size_t attention_scratch = 0;
  size_t scratch = 0;   // bytes left for LinearRunner and attention
  size_t total = 0;     // bytes
};

// `chunk_override` exists for the token refiner, which runs the whole text
// stream in one pass: its attention is over all L rows anyway, so chunking the
// row-wise stages around it would buy nothing and complicate the carve.
size_t attention_scratch_for_mode(const TransformerConfig& cfg, int sequence,
                                  AttentionMode mode) {
  if (mode == AttentionMode::kExact) return 0;
  AttentionConfig acfg;
  acfg.seq_len = std::max(sequence, 1);
  acfg.num_heads = cfg.num_attention_heads;
  acfg.head_dim = cfg.attention_head_dim;
  AttentionBackend backend = AttentionBackend::kFused;
  if (mode == AttentionMode::kNone) backend = AttentionBackend::kBlocked;
  if (mode == AttentionMode::kSage2) backend = AttentionBackend::kSage2;
  if (is_sol_attention(mode)) backend = AttentionBackend::kSol;
  return cuda::attention_workspace_bytes(acfg, backend);
}

Carve plan_carve(const TransformerConfig& cfg, const SequenceLayout& layout,
                 int chunk_override = 0,
                 AttentionMode attention_mode = AttentionMode::kFlash2) {
  const int seq = layout.total_rows();
  const int hidden = cfg.hidden_size;
  const int inner = cfg.inner_dim();

  Carve c;
  c.chunk = chunk_override > 0 ? chunk_override : std::min(kRowChunk, std::max(seq, 1));
  c.qkv = static_cast<size_t>(seq) * inner;
  c.normed = static_cast<size_t>(c.chunk) * hidden;
  c.fused = static_cast<size_t>(c.chunk) * 2 * cfg.ffn_dim;
  c.act = static_cast<size_t>(c.chunk) * cfg.ffn_dim;
  c.fbuf = static_cast<size_t>(c.chunk) * hidden;

  size_t bytes = 0;
  bytes += 4 * align_up(c.qkv * sizeof(__nv_bfloat16));  // q, k, v, attn_out
  bytes += align_up(c.normed * sizeof(__nv_bfloat16));
  bytes += align_up(c.fused * sizeof(__nv_bfloat16));
  bytes += align_up(c.act * sizeof(__nv_bfloat16));
  bytes += align_up(c.normed * sizeof(__nv_bfloat16));  // branch
  bytes += 2 * align_up(c.fbuf * sizeof(float));        // fp32 in/out of the final norm

  // The largest set of weights any one row-chunk loop has to hold dequantised
  // at once, plus attention's score tile. All are taken through
  // Workspace::Scope, so they overlap rather than accumulate.
  //
  // `run_block` hoists the dequantisation out of its three loops, so the unit
  // is the loop, not the GEMM: qkv holds its three thirds together — which is
  // exactly the fused `qkv_proj` this already sized for — and the FFN loop
  // holds fc1 and fc2 together, which is the one place the requirement grew.
  size_t scratch = 0;
  {
    QuantWeight probe;
    probe.format = QuantFormat::kF8E4M3;
    const auto dense = [&](int out, int in) {
      probe.out_features = out;
      probe.in_features = in;
      return cuda::linear_dense_weight_bytes(probe);
    };
    const auto act_ws = [&](int out, int in) {
      probe.out_features = out;
      probe.in_features = in;
      return cuda::linear_activation_workspace_bytes(probe, c.chunk, ComputeType::kBF16);
    };
    // qkv: three [inner, hidden] thirds live at once.
    scratch = std::max(scratch, 3 * dense(inner, hidden) + act_ws(inner, hidden));
    // out_proj on its own.
    scratch = std::max(scratch, dense(hidden, inner) + act_ws(hidden, inner));
    // fc1 and fc2 live at once.
    scratch = std::max(scratch, dense(2 * cfg.ffn_dim, hidden) + dense(hidden, cfg.ffn_dim) +
                                    std::max(act_ws(2 * cfg.ffn_dim, hidden),
                                             act_ws(hidden, cfg.ffn_dim)));
    // The fp32 heads widen their weight through bf16, so both copies are live.
    // They still go through `forward_f32`, one weight at a time.
    probe.format = QuantFormat::kBF16;
    probe.out_features = hidden;
    probe.in_features = cfg.text_dim;
    scratch = std::max(scratch, cuda::linear_workspace_bytes(probe, c.chunk, ComputeType::kF32));
  }
  c.attention_scratch = attention_scratch_for_mode(cfg, seq, attention_mode);
  scratch = std::max(scratch, c.attention_scratch);
  c.scratch = scratch;
  c.total = bytes + align_up(scratch);
  return c;
}

}  // namespace

// ---------------------------------------------------------------------------

struct Transformer::Impl {
  TransformerConfig cfg;
  TransformerArchitecture architecture = TransformerArchitecture::kUnknown;
  AdaLNLookup lookup = AdaLNLookup::kLinear;
  AdaLNTable table;
  FullAdaLNTimestepEmbedding timestep_embedding;

  cuda::Stream stream;
  cublasHandle_t blas = nullptr;
  cuda::LinearRunner linear;
  Workspace ws;

  DeviceBuffer<uint8_t> arena;
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
  bool tensor_diag = false;
  bool sol_pipeline_diag = false;
  DeviceBuffer<cuda::TensorScan> d_tensor_diag;

  void diagnose(const char* stage,const __nv_bfloat16* p,size_t n,int layer){
    if(!tensor_diag)return;
    VIDFAB_CUDA_CHECK(cudaMemsetAsync(d_tensor_diag.get(),0,sizeof(cuda::TensorScan),stream.get()));
    cuda::launch_tensor_scan(p,n,d_tensor_diag.get(),stream.get());
    cuda::TensorScan h{};d_tensor_diag.copy_to_host(&h,1,stream.get());
    VIDFAB_CUDA_CHECK(cudaStreamSynchronize(stream.get()));
    float mx=0;std::memcpy(&mx,&h.max_bits,sizeof(mx));
    std::fprintf(stderr,"vidfab tensor step=%d layer=%d stage=%s nonfinite=%llu max=%.7g\n",
                 denoise_step,layer,stage,h.nonfinite,mx);
    if(h.nonfinite)throw std::runtime_error("transformer: first non-finite tensor at "+std::string(stage));
  }

  static uint64_t fnv64_append(uint64_t hash, const void* data, size_t bytes) {
    const auto* cursor = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < bytes; ++i) {
      hash ^= cursor[i]; hash *= 1099511628211ull;
    }
    return hash;
  }

  uint64_t capture_hash(const __nv_bfloat16* data, size_t elements,
                        uint64_t seed = 1469598103934665603ull) {
    std::vector<uint16_t> host(elements);
    VIDFAB_CUDA_CHECK(cudaMemcpyAsync(host.data(), data, elements * 2,
                                      cudaMemcpyDeviceToHost, stream.get()));
    VIDFAB_CUDA_CHECK(cudaStreamSynchronize(stream.get()));
    return fnv64_append(seed, host.data(), host.size() * 2);
  }

  void begin_block_capture(const __nv_bfloat16* residual,
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
    VIDFAB_CUDA_CHECK(cudaMemcpyAsync(next.input.data(), residual,
        next.input.size() * 2, cudaMemcpyDeviceToHost, stream.get()));
    VIDFAB_CUDA_CHECK(cudaMemcpyAsync(next.selectors.data(), selectors,
        next.selectors.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, stream.get()));
    VIDFAB_CUDA_CHECK(cudaMemcpyAsync(next.cosine.data(), cosine,
        next.cosine.size() * sizeof(float), cudaMemcpyDeviceToHost, stream.get()));
    VIDFAB_CUDA_CHECK(cudaMemcpyAsync(next.sine.data(), sine,
        next.sine.size() * sizeof(float), cudaMemcpyDeviceToHost, stream.get()));
    VIDFAB_CUDA_CHECK(cudaStreamSynchronize(stream.get()));
    next.header.input_fnv64 = fnv64_append(1469598103934665603ull,
        next.input.data(), next.input.size() * 2);
    next.active = true;
    block_capture = std::move(next);
  }

  void capture_block_qkv(const __nv_bfloat16* q, const __nv_bfloat16* k,
                         const __nv_bfloat16* v) {
    if (!block_capture.active) return;
    uint64_t hash = capture_hash(q, block_capture.header.qkv_elements);
    hash = capture_hash(k, block_capture.header.qkv_elements, hash);
    block_capture.header.qkv_fnv64 =
        capture_hash(v, block_capture.header.qkv_elements, hash);
  }

  void capture_block_attention(const __nv_bfloat16* attention) {
    if (block_capture.active)
      block_capture.header.attention_fnv64 = capture_hash(
          attention, block_capture.header.qkv_elements);
  }

  void capture_block_attention_residual(const __nv_bfloat16* residual) {
    if (block_capture.active)
      block_capture.header.attention_residual_fnv64 = capture_hash(
          residual, block_capture.header.residual_elements);
  }

  void finish_block_capture(const __nv_bfloat16* residual) {
    if (!block_capture.active) return;
    std::vector<uint16_t> final(
        static_cast<size_t>(block_capture.header.residual_elements));
    VIDFAB_CUDA_CHECK(cudaMemcpyAsync(final.data(), residual, final.size() * 2,
                                      cudaMemcpyDeviceToHost, stream.get()));
    VIDFAB_CUDA_CHECK(cudaStreamSynchronize(stream.get()));
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
    std::fprintf(stderr, "vidfab: captured exact H3 block step %d layer %d to %s\n",
                 denoise_step, block_capture_layer, block_capture_path.c_str());
    const char* stop = std::getenv("VIDFAB_H3_BLOCK_CAPTURE_EXIT");
    if (stop && stop[0] == '1')
      throw std::runtime_error("transformer: stopped after requested H3 block capture");
  }

  void capture_sol_inputs(const __nv_bfloat16* q, const __nv_bfloat16* k,
                          const __nv_bfloat16* v, int rows, int layer) {
    if (sol_capture_done) return;
    if (denoise_step != sol_capture_step || layer != sol_capture_layer) return;

    const uint64_t n = static_cast<uint64_t>(rows) * cfg.num_attention_heads *
                       cfg.attention_head_dim;
    std::vector<uint16_t> host(static_cast<size_t>(n) * 3);
    VIDFAB_CUDA_CHECK(cudaMemcpyAsync(host.data(), q, n * sizeof(uint16_t),
                                      cudaMemcpyDeviceToHost, stream.get()));
    VIDFAB_CUDA_CHECK(cudaMemcpyAsync(host.data() + n, k, n * sizeof(uint16_t),
                                      cudaMemcpyDeviceToHost, stream.get()));
    VIDFAB_CUDA_CHECK(cudaMemcpyAsync(host.data() + 2 * n, v, n * sizeof(uint16_t),
                                      cudaMemcpyDeviceToHost, stream.get()));
    VIDFAB_CUDA_CHECK(cudaStreamSynchronize(stream.get()));
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
    std::fprintf(stderr, "vidfab: captured Sol Q/K/V step %d layer %d to %s\n",
                 denoise_step, layer, sol_capture_path.c_str());
    const char* stop = std::getenv("VIDFAB_SOL_CAPTURE_EXIT");
    if (stop != nullptr && stop[0] == '1')
      throw std::runtime_error("transformer: stopped after requested Sol capture");
  }
  DeviceBuffer<int32_t> d_band;
  std::vector<int32_t> host_band;
  DeviceBuffer<float> rope_cos, rope_sin;
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
    VIDFAB_CUBLAS_CHECK(cublasCreate(&blas));
    linear.init(blas, stream.get());

    // The native nvfp4 GEMM is 2.6-4.1x the dequantise-then-cuBLAS path and is
    // off unless asked for, because it quantises activations to E2M1 and that
    // costs ~9% rms per layer against a bf16-activation reference — a property
    // of the format, not a defect (see the README). Whether that survives 50
    // blocks and 29 steps is an end-to-end question, and this switch exists so
    // it can be answered by generating the same seed both ways rather than
    // argued about. Same shape as VIDFAB_CUBLAS_PEDANTIC in vit_decoder.cu.
    const char* native = std::getenv("VIDFAB_NATIVE_NVFP4");
    if (native != nullptr && native[0] == '1') linear.set_native(true);
    const char* capture = std::getenv("VIDFAB_SOL_CAPTURE");
    const char* block_capture_env = std::getenv("VIDFAB_H3_BLOCK_CAPTURE");
    const char* diag = std::getenv("VIDFAB_TENSOR_DIAG");
    tensor_diag=diag!=nullptr&&diag[0]=='1';
    if(tensor_diag)d_tensor_diag.allocate(1);
    const char* sol_pipe=std::getenv("VIDFAB_SOL_PIPELINE");
    sol_pipeline_diag=sol_pipe!=nullptr&&sol_pipe[0]=='1';
    if (capture != nullptr && *capture != '\0') {
      sol_capture_path = capture;
      const char* step = std::getenv("VIDFAB_SOL_CAPTURE_STEP");
      const char* layer = std::getenv("VIDFAB_SOL_CAPTURE_LAYER");
      if (step != nullptr && *step != '\0') sol_capture_step = std::atoi(step);
      if (layer != nullptr && *layer != '\0') sol_capture_layer = std::atoi(layer);
    }
    if (block_capture_env && *block_capture_env) {
      block_capture_path = block_capture_env;
      const char* step = std::getenv("VIDFAB_H3_BLOCK_CAPTURE_STEP");
      const char* layer = std::getenv("VIDFAB_H3_BLOCK_CAPTURE_LAYER");
      if (step && *step) block_capture_step = std::atoi(step);
      if (layer && *layer) block_capture_layer = std::atoi(layer);
    }
  }
  ~Impl() {
    if (blas != nullptr) cublasDestroy(blas);
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
  void build_modulation(const std::vector<float>& timesteps) {
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

  // --- one block -----------------------------------------------------------

  // Debug capture. Null in production and therefore free; when set, it is
  // called with the residual stream at each labelled point so a bisect can
  // find the first stage that diverges from a CPU reference. Without this the
  // block is opaque from outside, which is exactly why four plausible
  // explanations for the known numerical gap could each be eliminated without
  // the number moving.
  std::function<void(const char*, const __nv_bfloat16*, int, int)> stage_hook;

  void emit_stage(const char* label, const __nv_bfloat16* x, int rows, int dim) {
    if (!stage_hook) return;
    VIDFAB_CUDA_CHECK(cudaStreamSynchronize(stream.get()));
    stage_hook(label, x, rows, dim);
  }

  void run_block(const BlockWeights& b, const float* mod_base, int rows, __nv_bfloat16* x,
                 const int32_t* adaln_idx, const float* cos, const float* sin, __nv_bfloat16* q,
                 __nv_bfloat16* k, __nv_bfloat16* v, __nv_bfloat16* attn_out,
                 __nv_bfloat16* normed, __nv_bfloat16* fused, __nv_bfloat16* act,
                 __nv_bfloat16* branch, AttentionMode block_attention_mode, int layer = -1) {
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
    auto project = [&](const QuantWeight& weight,
                       const __nv_bfloat16* dense,
                       const __nv_bfloat16* input, int count,
                       __nv_bfloat16* output) {
      if (block_attention_mode != AttentionMode::kExact) {
        linear.forward_prepared(weight, dense, input, count, output, ws);
        return;
      }
      const bool transformed = weight.pre_quant_scale != nullptr ||
          (weight.convrot && weight.convrot_group > 0 &&
           weight.in_features % weight.convrot_group == 0);
      if (weight.bias != nullptr || transformed)
        throw std::runtime_error(
            "transformer: exact H3 projection requires bias-free untransformed weights");
      const uint32_t tiled = static_cast<uint32_t>(count) / 64 * 64;
      if (tiled != 0) {
        cuda::launch_deterministic_bf16_gemm_nt(
            input, dense, nullptr, output, tiled,
            static_cast<uint32_t>(weight.out_features),
            static_cast<uint32_t>(weight.in_features), DenseGemmBias::kNone,
            0, 0, stream.get());
      }
      if (tiled != static_cast<uint32_t>(count)) {
        cuda::launch_deterministic_scalar_gemm_nt(
            input, dense, nullptr, output,
            static_cast<uint32_t>(count) - tiled,
            static_cast<uint32_t>(weight.out_features),
            static_cast<uint32_t>(weight.in_features),
            DenseGemmMode::kBFloat16, DenseGemmBias::kNone,
            tiled, tiled, stream.get());
      }
    };

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
    if (block_attention_mode == AttentionMode::kExact) {
      cuda::launch_deterministic_h3_attention(
          stream.get(), q, k, v, attn_out, acfg.band_ranges,
          static_cast<uint32_t>(rows), static_cast<uint32_t>(cfg.num_attention_heads),
          static_cast<uint32_t>(cfg.attention_head_dim), acfg.effective_scale());
      if (layer < 0) ++attention_routes.exact_refiner_full;
      else if (acfg.band_ranges == nullptr) ++attention_routes.exact_main_full;
      else ++attention_routes.exact_main_banded;
    } else {
      cuda::attention_forward(blas, stream.get(), q, k, v, attn_out, acfg, backend, ws);
      if (layer < 0) ++attention_routes.generic_refiner;
      else ++attention_routes.generic_main;
    }
    if (layer >= 0) capture_block_attention(attn_out);
    diagnose("attention",attn_out,size_t(rows)*inner,layer);
    const char* label = block_attention_mode == AttentionMode::kExact ? "attn.exact" :
                        backend == AttentionBackend::kFused ? "attn.flash2" :
                        backend == AttentionBackend::kSage2 ? "attn.sage2" :
                        backend == AttentionBackend::kSol ?
                          (block_attention_mode==AttentionMode::kSolExperimental?
                            "attn.sol.experimental":"attn.sol") : "attn.none";
    prof.tick(label, stream.get());

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
    if (layer >= 0) finish_block_capture(x);
    emit_stage("ffn", x, rows, hidden);
  }
};

// ---------------------------------------------------------------------------

Transformer::Transformer() : impl_(new Impl()) {}
Transformer::~Transformer() = default;

const TransformerConfig& Transformer::config() const { return impl_->cfg; }
size_t Transformer::weight_bytes() const { return impl_->arena_bytes; }
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
  impl_->blocks.clear();
  impl_->refiner.clear();
  impl_->arena.reset();
  impl_->arena_bytes = 0;
  impl_->has_sequence = false;
  impl_->attention_configuration_locked = false;
  impl_->attention_routes = {};
}

void Transformer::load(const SafeTensors& checkpoint, const TransformerConfig& config) {
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
  s.architecture = detect_transformer_architecture(checkpoint);
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
    plan.require(prefix + "attn.q_norm.weight", {head_dim}, Store::kAsBF16);
    plan.require(prefix + "attn.k_norm.weight", {head_dim}, Store::kAsBF16);
    plan_linear(prefix + "attn.out_proj", hidden, inner);
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

  s.arena.allocate(plan.arena_bytes());
  s.arena_bytes = plan.arena_bytes();
  uint8_t* base = s.arena.get();

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
                   "vidfab: could not page-lock the transformer mapping; uploading via the "
                   "staged path, which is slower\n");
    }
    std::vector<float> wide;
    std::vector<uint16_t> narrow;
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
      uint8_t* dst = base + r.offset;
      switch (r.store) {
        case Store::kVerbatim:
          up.copy(dst, r.view->data, r.bytes, /*from_mapping=*/true);
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
          // it is the thing that was verified — `VIDFAB_ARENA_HASH=1` over
          // fl2va_pruned_fp8_scaled.safetensors gives 21045398272 bytes, 730
          // records, fnv1a 190cdce19da29c2d both before and after this change.
          // An arena that got smaller would be a different arena.
          if (r.view->dtype == DType::kF32) {
            up.copy(dst, r.view->data, r.bytes, /*from_mapping=*/true);
          } else if (r.view->dtype == DType::kF16) {
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
            up.copy(dst, wide.data(), wide.size() * sizeof(float), /*from_mapping=*/false);
          }
          break;
        case Store::kAsBF16:
          to_f32(*r.view, wide);
          narrow.resize(wide.size());
          for (size_t i = 0; i < wide.size(); ++i) narrow[i] = f32_to_bf16(wide[i]);
          up.copy(dst, narrow.data(), narrow.size() * sizeof(uint16_t), /*from_mapping=*/false);
          break;
      }
    }
  }
  VIDFAB_CUDA_CHECK(cudaStreamSynchronize(s.stream.get()));

  // A hash of the finished weight arena, off unless asked for. This exists so
  // that a change to *how* the file is read can be shown to have left *what*
  // was read alone: any reordering, buffering or prefetch change must produce
  // the same 12.5 GB byte for byte, and a single number either matches or it
  // does not. Reading it back costs one D2H of the arena — about half a second
  // — which is why it is behind an environment variable rather than always on.
  if (const char* want = std::getenv("VIDFAB_ARENA_HASH"); want != nullptr && want[0] == '1') {
    constexpr size_t kChunk = 64u << 20;
    std::vector<uint64_t> host(kChunk / sizeof(uint64_t));
    uint64_t h = 1469598103934665603ull;  // FNV-1a offset basis
    size_t left = s.arena_bytes;
    const uint8_t* src = base;
    while (left > 0) {
      const size_t n = std::min(left, kChunk);
      VIDFAB_CUDA_CHECK(cudaMemcpy(host.data(), src, n, cudaMemcpyDeviceToHost));
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
    return base + it->second.offset;
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
    }

    w.input_scale = host_scalar(name + ".input_scale");
    // The file decides, not a heuristic on which scales are present. No layer
    // of the nvfp4 transformer sets it and 50 of the fp8 transformer's do —
    // exactly `mlp.fc2`, which also ships no input_scale (spec 8.2).
    w.full_precision = read_comfy_quant(checkpoint, name).full_precision;
    return w;
  };

  auto build_block = [&](const std::string& prefix, bool with_adaln) {
    BlockWeights b;
    b.norm1 = bf(prefix + "norm1.weight");
    b.norm2 = bf(prefix + "norm2.weight");
    b.q_norm = bf(prefix + "attn.q_norm.weight");
    b.k_norm = bf(prefix + "attn.k_norm.weight");

    // Spec 8.1: `qkv_proj.weight` is contiguous [Wq; Wk; Wv], already
    // de-interleaved. Three views over one allocation, sharing both scales.
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
    }

    b.out_proj = quant(prefix + "attn.out_proj", hidden, inner);
    b.fc1 = quant(prefix + "mlp.fc1", 2 * ffn, hidden);
    b.fc2 = quant(prefix + "mlp.fc2", hidden, ffn);
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

size_t Transformer::activation_bytes(const SequenceLayout& layout) const {
  const TransformerConfig& cfg = impl_->cfg;
  const Carve c = plan_carve(cfg, layout, 0, impl_->attention_mode);
  const int seq = layout.total_rows();

  size_t total = c.total;
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
  // **This number is reported, not enforced.** Nothing in the generate path
  // calls this — the only caller in the tree is tests/test_transformer.cu — so
  // enabling the block cache on a marginal-fit configuration still fails as a
  // `cudaMalloc` in `prepare_sequence` after the 19.6 GiB load, and the error
  // does not mention the block cache. Counted here anyway so the accounting is
  // right if a preflight check is ever added.
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
  return plan_carve(impl_->cfg, layout, 0, impl_->attention_mode).attention_scratch;
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
  VIDFAB_CUDA_CHECK(cudaMemcpyAsync(out.data(), s.mod.get() + block_index * per_block,
                                    per_block * sizeof(float), cudaMemcpyDeviceToHost,
                                    s.stream.get()));
  VIDFAB_CUDA_CHECK(cudaStreamSynchronize(s.stream.get()));
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
    VIDFAB_CUDA_CHECK(
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
  VIDFAB_CUDA_CHECK(cudaMemcpy(bits.data(), s.text_cache.get(),
                               bits.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost));
  std::vector<float> out(bits.size());
  for (size_t i = 0; i < bits.size(); ++i) out[i] = bf16_to_f32(bits[i]);
  return out;
}

// ---------------------------------------------------------------------------

void Transformer::prepare_text(const float* prompt_embeds, int num_tokens) {
  Impl& s = *impl_;
  s.require_loaded("prepare_text");
  if (num_tokens < 0) throw std::runtime_error("transformer: negative token count");

  s.num_text = num_tokens;
  if (num_tokens == 0) {
    s.text_cache.reset();
    return;
  }

  const int hidden = s.cfg.hidden_size;
  const int inner = s.cfg.inner_dim();
  const size_t rows = static_cast<size_t>(num_tokens);

  // The refiner is O(L^2) with L in the low thousands and, unlike everything
  // else here, has no timestep dependence — so it runs once per request rather
  // than once per step, which the reference only fails to do because its
  // forward is stateless (spec 6).
  SequenceLayout text_only;
  text_only.num_text = num_tokens;
  const Carve text_carve = plan_carve(s.cfg, text_only, /*chunk_override=*/num_tokens,
                                      s.attention_mode);

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
  s.linear.forward(s.condition_proj, xin, num_tokens, x, ws);
  s.emit_stage("condition_proj", x, num_tokens, hidden);

  __nv_bfloat16* q = ws.alloc_n<__nv_bfloat16>(rows * inner);
  __nv_bfloat16* k = ws.alloc_n<__nv_bfloat16>(rows * inner);
  __nv_bfloat16* v = ws.alloc_n<__nv_bfloat16>(rows * inner);
  __nv_bfloat16* attn_out = ws.alloc_n<__nv_bfloat16>(rows * inner);
  __nv_bfloat16* normed = ws.alloc_n<__nv_bfloat16>(rows * hidden);
  __nv_bfloat16* fused = ws.alloc_n<__nv_bfloat16>(rows * 2 * s.cfg.ffn_dim);
  __nv_bfloat16* act = ws.alloc_n<__nv_bfloat16>(rows * s.cfg.ffn_dim);
  __nv_bfloat16* branch = ws.alloc_n<__nv_bfloat16>(rows * hidden);

  const Carve saved = s.carve;
  s.carve.chunk = num_tokens;
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

  cuda::launch_rmsnorm(x, s.refiner_final_norm, normed, num_tokens, hidden, s.cfg.norm_eps,
                       s.stream.get());
  VIDFAB_CUDA_CHECK(cudaMemcpyAsync(x, normed, rows * hidden * sizeof(__nv_bfloat16),
                                    cudaMemcpyDeviceToDevice, s.stream.get()));
  VIDFAB_CUDA_CHECK(cudaStreamSynchronize(s.stream.get()));
  s.emit_stage("final_norm", x, num_tokens, hidden);
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

  s.layout = layout;
  s.indices = indices;
  s.carve = plan_carve(s.cfg, layout, 0, s.attention_mode);
  if (s.architecture == TransformerArchitecture::kRef2VAFullAdaLN) {
    const size_t old_scratch = s.carve.scratch;
    s.carve.scratch = std::max(s.carve.scratch, cuda::linear_workspace_bytes(
        s.blocks.front().full_adaln, 2, ComputeType::kF32));
    s.carve.scratch = std::max(s.carve.scratch, cuda::linear_workspace_bytes(
        s.final_full_adaln, 2, ComputeType::kF32));
    s.carve.total += align_up(s.carve.scratch) - align_up(old_scratch);
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
  s.rope_cos.allocate(rope.cosine.size());
  s.rope_sin.allocate(rope.sine.size());
  s.rope_cos.copy_from_host(rope.cosine.data(), rope.cosine.size(), s.stream.get());
  s.rope_sin.copy_from_host(rope.sine.data(), rope.sine.size(), s.stream.get());
  VIDFAB_CUDA_CHECK(cudaStreamSynchronize(s.stream.get()));

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
  VIDFAB_CUDA_CHECK(cudaStreamSynchronize(s.stream.get()));
  s.attention_configuration_locked = true;
}

// ---------------------------------------------------------------------------

void Transformer::forward(const float* video_latents, const float* audio_latents,
                          const RowTimesteps& row_timesteps, float* video_velocity,
                          float* audio_velocity) {
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

  s.ws.clear();
  Workspace& ws = s.ws;
  const size_t qkv_n = s.carve.qkv;
  __nv_bfloat16* q = ws.alloc_n<__nv_bfloat16>(qkv_n);
  __nv_bfloat16* k = ws.alloc_n<__nv_bfloat16>(qkv_n);
  __nv_bfloat16* v = ws.alloc_n<__nv_bfloat16>(qkv_n);
  __nv_bfloat16* attn_out = ws.alloc_n<__nv_bfloat16>(qkv_n);
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
  // Kept, behind VIDFAB_TENSOR_DIAG, as insurance against a layout that ever
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
      s.linear.forward_f32(w, src + static_cast<size_t>(start) * in_dim, n, fa, ws);
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
        VIDFAB_CUDA_CHECK(cudaMemcpyAsync(s.bc_delta.get(), x,
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
    s.run_block(s.blocks[b], s.mod.get() + b * per_block, seq, x, s.d_adaln.get(),
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
      s.linear.forward_f32(w, fb, n, dst + static_cast<size_t>(start) * out_dim, ws);
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
  VIDFAB_CUDA_CHECK(cudaStreamSynchronize(s.stream.get()));
  const std::chrono::steady_clock::time_point t_exit = std::chrono::steady_clock::now();
  prof.end_step();
  prof.sample_memory();
  prof.add_step_wall(std::chrono::duration<double, std::milli>(t_exit - t_enter).count(),
                     std::chrono::duration<double, std::milli>(t_issued - t_enter).count(),
                     std::chrono::duration<double, std::milli>(t_exit - t_issued).count());
}

}  // namespace vidfab::dit
