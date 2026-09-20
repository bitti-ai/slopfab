#pragma once
#include "slopfab/dit/transformer.h"
#include "slopfab/cuda/lora.cuh"
#include "slopfab/dit/block_capture.h"
#include "slopfab/dit/graph_capture.h"
#include "slopfab/dit/rope.h"
#include "slopfab/dit/offload.h"
#include "slopfab/dit/vsa.h"
#include "slopfab/cuda/vsa_attention.cuh"

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
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "slopfab/cuda/attention.cuh"
#include "slopfab/cuda/deterministic_attention.cuh"
#include "slopfab/cuda/deterministic_gemm.cuh"
#include "slopfab/cuda/device.h"
#include "slopfab/cuda/diagnostics.cuh"
#include "slopfab/cuda/gemm.cuh"
#include "slopfab/cuda/linear.cuh"
#include "slopfab/cuda/nn_kernels.cuh"
#include "slopfab/cuda/profile.h"
#include "slopfab/cuda/vae_kernels.cuh"
#include "slopfab/cuda/workspace.cuh"
#include "slopfab/dtype.h"
#include "slopfab/json.h"
#include "slopfab/nf4.h"
#include "weight_metadata.h"
#include "slopfab/sol_capture.h"
#include "slopfab/tensor_convert.h"

namespace slopfab::cuda {

// Defined in src/cuda/dit_kernels.cu. Declared here rather than in a header
// because they have exactly one caller each; a signature drift is a link
// error, which is the failure mode you want.
void launch_adaln_expand(const float* w, const float* bias, const float* code, float* out,
                         int num_t, int num_modality, int num_param, int channels, int rank,
                         cudaStream_t stream);
void launch_add_rows_bf16(__nv_bfloat16* x, const __nv_bfloat16* branch, size_t n,
                          cudaStream_t stream);

}  // namespace slopfab::cuda

namespace slopfab::dit {
namespace detail {

using cuda::AttentionBackend;
using cuda::AttentionConfig;
using cuda::ComputeType;
using cuda::DeviceBuffer;
using cuda::QuantFormat;
using cuda::QuantWeight;
using cuda::Workspace;

// Bound row-wise scratch independently of the video and conditioning length.
constexpr int kRowChunk = 2048;

// Spec 3.2: six modulation parameters, three modalities.
constexpr int kNumParams = 6;
constexpr int kNumModalities = 3;
// Spec 8.3: the final layer emits [shift; scale] and has no modality axis.
constexpr int kFinalParams = 2;

inline size_t align_up(size_t n) { return (n + 255) / 256 * 256; }

inline QuantFormat format_of(DType dt, const std::string& name) {
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

inline size_t format_bytes(QuantFormat f) {
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

inline std::string nf4_state_name(const std::string& name) {
  return name + ".weight.quant_state.bitsandbytes__nf4";
}
inline bool is_nf4(const SafeTensors& st, const std::string& name) {
  return slopfab::is_nf4_weight(st, name + ".weight");
}
inline NF4State read_nf4_state(const SafeTensors& st, const std::string& name) {
  return slopfab::read_nf4_state(st, name + ".weight", "transformer", true);
}

// True when `name` is stored as nvfp4. Structural, from the file itself, rather
// than from a flag or from which checkpoint the caller thinks it opened: a U8
// `.weight` alongside an e4m3 `.weight_scale` carrying one scale per 16
// contracted elements. Both halves are needed — U8 alone would also match a
// byte blob, and it is the scale's trailing dimension that pins the block size
// this port implements. The two bf16 refiner blocks fall out of it for free.
inline bool is_nvfp4(const SafeTensors& st, const std::string& name, int in_features) {
  const TensorView* w = st.find(name + ".weight");
  const TensorView* s = st.find(name + ".weight_scale");
  if (w == nullptr || s == nullptr) return false;
  if (w->dtype != DType::kU8 || s->dtype != DType::kF8E4M3) return false;
  return s->shape.size() == 2 &&
         s->shape[1] == in_features / static_cast<int>(cuda::kNVFP4BlockSize);
}

using detail::read_comfy_quant;
using detail::QuantTag;

inline std::string shape_string(const std::vector<int64_t>& s) {
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
      SLOPFAB_CUDA_CHECK(cudaEventCreateWithFlags(&event_[i], cudaEventDisableTiming));
      // Recorded once so the first wait on each slot is a no-op rather than a
      // wait on an event that was never recorded (which is legal but reads as
      // an accident).
      SLOPFAB_CUDA_CHECK(cudaEventRecord(event_[i], stream_));
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
      SLOPFAB_CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream_));
      return;
    }

    const uint8_t* s = static_cast<const uint8_t*>(src);
    uint8_t* d = static_cast<uint8_t*>(dst);
    while (bytes > 0) {
      const size_t n = std::min(bytes, kStageBytes);
      SLOPFAB_CUDA_CHECK(cudaEventSynchronize(event_[cur_]));
      std::memcpy(slot_[cur_].get(), s, n);
      SLOPFAB_CUDA_CHECK(
          cudaMemcpyAsync(d, slot_[cur_].get(), n, cudaMemcpyHostToDevice, stream_));
      SLOPFAB_CUDA_CHECK(cudaEventRecord(event_[cur_], stream_));
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
    consumed_.insert(v->name);
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
    consumed_.insert(v.name);
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
  QuantWeight wq, wk, wv, out_proj, fc1, fc2, compress_gate;
  const float* adaln_w = nullptr;
  const float* adaln_b = nullptr;
  QuantWeight full_adaln;
};

// Owns CPU copies independently of the checkpoint mapping. Two device slots
// alternate; transfer waits for prior compute before overwriting a slot.
class BlockStreamer {
 public:
  struct Block { cuda::PinnedBuffer<uint8_t> host; size_t cursor = 0; };
  explicit BlockStreamer(cudaStream_t compute) : compute_(compute) {}
  ~BlockStreamer() {
    cudaStreamSynchronize(compute_);
    if (copy_) cudaStreamSynchronize(copy_->get());
    for (int i = 0; i < 2; ++i) {
      if (ready_[i]) cudaEventDestroy(ready_[i]);
      if (consumed_[i]) cudaEventDestroy(consumed_[i]);
    }
  }
  void initialize(const BlockOffloadPlan& plan, const std::vector<size_t>& sizes) {
    first = plan.first; count = plan.count; host_bytes = plan.host_bytes;
    copy_ = std::make_unique<cuda::Stream>();
    blocks.resize(sizes.size());
    for (size_t i = first; i < sizes.size(); ++i) blocks[i].host.allocate(sizes[i]);
    for (size_t i = 0; i < plan.slots(); ++i) {
      slots_[i].allocate(plan.slot_bytes);
      SLOPFAB_CUDA_CHECK(cudaEventCreateWithFlags(&ready_[i], cudaEventDisableTiming));
      SLOPFAB_CUDA_CHECK(cudaEventCreateWithFlags(&consumed_[i], cudaEventDisableTiming));
    }
  }
  bool contains(size_t block) const { return block >= first && block < first + count; }
  uint8_t* device(size_t block) { return slots_[(block - first) % 2].get(); }
  size_t device_bytes() const { return slots_[0].nbytes() + slots_[1].nbytes(); }
  void prefetch(size_t block) {
    if (!contains(block)) return;
    const size_t slot = (block - first) % 2;
    if (queued_[slot] == block) return;
    SLOPFAB_CUDA_CHECK(cudaEventRecord(consumed_[slot], compute_));
    SLOPFAB_CUDA_CHECK(cudaStreamWaitEvent(copy_->get(), consumed_[slot], 0));
    SLOPFAB_CUDA_CHECK(cudaMemcpyAsync(device(block), blocks[block].host.get(),
        blocks[block].host.size(), cudaMemcpyHostToDevice, copy_->get()));
    SLOPFAB_CUDA_CHECK(cudaEventRecord(ready_[slot], copy_->get()));
    queued_[slot] = block;
  }
  void acquire(size_t block) {
    prefetch(block);
    SLOPFAB_CUDA_CHECK(cudaStreamWaitEvent(compute_, ready_[(block - first) % 2], 0));
    prefetch(block + 1);
  }
  size_t first = 0, count = 0, host_bytes = 0;
  std::vector<Block> blocks;
 private:
  cudaStream_t compute_;
  std::unique_ptr<cuda::Stream> copy_;
  DeviceBuffer<uint8_t> slots_[2];
  cudaEvent_t ready_[2]{}, consumed_[2]{};
  size_t queued_[2] = {size_t(-1), size_t(-1)};
};

// AdaLN is consumed for every block before the block loop. Keep it resident.
inline int streamable_block(const std::string& name) {
  if (name.compare(0, 7, "blocks.") != 0) return -1;
  const size_t end = name.find('.', 7);
  if (end == std::string::npos || name.compare(end + 1, 6, "adaln_") == 0) return -1;
  return std::stoi(name.substr(7, end - 7));
}

template <typename Fn>
void visit_block_loras(const LoraAdapters* loras, const std::string& prefix,
                       const TransformerConfig& cfg, Fn&& fn) {
  if (!loras) return;
  struct Target { const char* name; int projection, offset, out; };
  const int inner = cfg.inner_dim();
  const Target targets[] = {
      {"attn.qkv_proj", 0, 0, inner}, {"attn.qkv_proj", 1, inner, inner},
      {"attn.qkv_proj", 2, 2 * inner, inner},
      {"attn.to_q", 0, 0, inner}, {"attn.to_k", 1, 0, inner}, {"attn.to_v", 2, 0, inner},
      {"attn.out_proj", 3, 0, cfg.hidden_size}, {"mlp.fc1", 4, 0, 2 * cfg.ffn_dim},
      {"mlp.fc2", 5, 0, cfg.hidden_size}};
  for (const auto& t : targets)
    if (const auto* factors = loras->find(prefix + t.name)) fn(t.projection, *factors, t.offset, t.out);
}

inline BlockWeights relocate_block(BlockWeights b, const uint8_t* host, size_t bytes, uint8_t* device) {
  auto relocate = [&](auto& ptr) {
    const auto address = reinterpret_cast<uintptr_t>(ptr);
    const auto base = reinterpret_cast<uintptr_t>(host);
    if (ptr && address >= base && address - base < bytes)
      ptr = reinterpret_cast<std::remove_reference_t<decltype(ptr)>>(device + address - base);
  };
  relocate(b.norm1); relocate(b.norm2); relocate(b.q_norm); relocate(b.k_norm);
  for (QuantWeight* w : {&b.wq, &b.wk, &b.wv, &b.out_proj, &b.fc1, &b.fc2, &b.compress_gate}) {
    relocate(w->data); relocate(w->weight_scale); relocate(w->block_scale);
    relocate(w->pre_quant_scale); relocate(w->nf4_absmax); relocate(w->nf4_quant_map);
    relocate(w->nf4_nested_quant_map); relocate(w->nf4_nested_absmax); relocate(w->bias);
  }
  return b;
}

// Sizes of the transient buffers `forward` carves, all in one place so that
// `activation_bytes` and the carve cannot drift apart.
struct Carve {
  int chunk = 0;
  bool chunked_attention = false;
  size_t query = 0;     // Q and attention output; compact on Flash2
  size_t qkv = 0;       // one of q/k/v, elements
  size_t normed = 0;
  size_t fused = 0;
  size_t act = 0;
  size_t fbuf = 0;      // elements of one fp32 [chunk, hidden] buffer
  size_t attention_scratch = 0;
  size_t scratch = 0;   // bytes left for LinearRunner and attention
  size_t total = 0;     // bytes
};

inline bool stack_uses_convrot(const std::vector<BlockWeights>& blocks) {
  for (const BlockWeights& block : blocks) {
    if (block.wq.convrot || block.wk.convrot || block.wv.convrot ||
        block.out_proj.convrot || block.fc1.convrot || block.fc2.convrot ||
        block.full_adaln.convrot || block.compress_gate.convrot) {
      return true;
    }
  }
  return false;
}

inline size_t attention_scratch_for_mode(const TransformerConfig& cfg, int sequence,
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
  return cuda::AttentionPlan::compile(acfg, cfg.num_attention_heads, backend).workspace_bytes();
}

inline Carve plan_carve(const TransformerConfig& cfg, const SequenceLayout& layout,
                 int row_chunk = kRowChunk,
                 AttentionMode attention_mode = AttentionMode::kFlash2,
                 bool reserve_convrot = false, bool query_chunking = false, bool vsa = false) {
  const int seq = layout.total_rows();
  const int hidden = cfg.hidden_size;
  const int inner = cfg.inner_dim();

  Carve c;
  c.chunk = std::min(row_chunk, std::max(seq, 1));
  c.qkv = static_cast<size_t>(seq) * inner;
  c.chunked_attention = query_chunking && seq > c.chunk;
  c.query = static_cast<size_t>(c.chunked_attention ? c.chunk : seq) * inner;
  c.normed = static_cast<size_t>(c.chunk) * hidden;
  c.fused = static_cast<size_t>(c.chunk) * std::max(2 * cfg.ffn_dim, vsa ? inner : 0);
  c.act = static_cast<size_t>(c.chunk) * cfg.ffn_dim;
  c.fbuf = static_cast<size_t>(c.chunk) * hidden;

  size_t bytes = 0;
  bytes += 2 * align_up(c.qkv * sizeof(__nv_bfloat16));  // k, v
  bytes += 2 * align_up(c.query * sizeof(__nv_bfloat16));  // q, attn_out
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
    // The dense dequantisation footprint is format-independent for every
    // non-BF16 block weight, but ConvRot also needs one [rows, in] activation
    // buffer. Use group 1 in this sizing-only probe so mixed/custom shapes are
    // conservatively covered whenever any block in the stack rotates.
    probe.convrot = reserve_convrot;
    probe.convrot_group = 1;
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
    // Chunked attention keeps Q and output projection weights together.
    scratch = std::max(scratch, dense(inner, hidden) + dense(hidden, inner) +
                      std::max(act_ws(inner, hidden), act_ws(hidden, inner)));
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
  c.attention_scratch = vsa
      ? cuda::vsa_attention_workspace_bytes(static_cast<int>(build_vsa_tiles(layout).sizes.size()),
                                            cfg.num_attention_heads, cfg.attention_head_dim)
      : attention_scratch_for_mode(cfg, seq, attention_mode);
  scratch = std::max(scratch, c.attention_scratch);
  c.scratch = scratch;
  c.total = bytes + align_up(scratch);
  return c;
}

}  // namespace detail
using namespace detail;

// ---------------------------------------------------------------------------


}  // namespace slopfab::dit
