#include "slopfab/text/qwen_vision.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <unordered_map>

#include "slopfab/cuda/device.h"
#include "slopfab/cuda/gemm.cuh"
#include "slopfab/cuda/linear.cuh"
#include "slopfab/cuda/qwen_vision.cuh"

namespace slopfab::text {
using cuda::DeviceBuffer;

namespace {
// Activation widths. Tied to the public config rather than re-typed, because
// the buffers below are carved out of one arena end to end: a width that is too
// small here is no longer a short allocation, it is a write into the next
// buffer. Only the ones the header actually declares can be checked this way —
// see the note on `kPatchDim`.
constexpr QwenVisionConfig kCfg{};
constexpr int kHidden = kCfg.hidden_size;             // 1152
constexpr int kIntermediate = kCfg.intermediate_size; // 4304
constexpr int kOutDim = kCfg.output_size;             // 5120
constexpr int kMerge = kCfg.merge_size;               // 2, so four rows merge
constexpr int kQkvDim = 3 * kHidden;                  // 3456
constexpr int kMergedDim = kMerge * kMerge * kHidden; // 4608
// 3 channels x 2 temporal x 16 x 16 spatial, per the `QwenPixelValues::rows`
// contract in qwen_vision.h. Not derivable from QwenVisionConfig.
constexpr int kPatchDim = 3 * 2 * 16 * 16; // 1536
static_assert(kHidden == 1152 && kIntermediate == 4304 && kOutDim == 5120 && kMerge == 2,
              "QwenVisionConfig moved; the carved activation widths must move with it");

cuda::QuantWeight dense(const DeviceBuffer<uint16_t>& w, int out, int in,
                        const DeviceBuffer<uint16_t>* b = nullptr) {
  cuda::QuantWeight q;
  q.format = cuda::QuantFormat::kBF16;
  q.data = w.get();
  q.out_features = out;
  q.in_features = in;
  if (b) {
    q.bias = b->get();
    q.bias_format = cuda::QuantFormat::kBF16;
  }
  return q;
}

// Every activation one image needs, in one place so the sizing pass and the
// carving pass cannot disagree about what exists.
struct VisionScratch {
  __nv_bfloat16 *pixels, *x, *normed, *qkv, *q, *k, *v, *branch, *mlp;
  __nv_bfloat16 *merger_normed, *merged, *merger_hidden, *out, *deep[3];
  int32_t* pos_index;
  float *rope_cos, *rope_sin;
};

// A bump allocator that either measures or carves, so `carve_vision` is written
// once and run twice. `Workspace::alloc` rounds to 256, so the measuring pass
// has to round identically or the reserve comes up short by up to 255 bytes a
// buffer.
struct Carver {
  cuda::Workspace* ws = nullptr; // null while measuring
  size_t bytes = 0;

  template <typename T> T* take(size_t n) {
    if (ws != nullptr)
      return ws->alloc_n<T>(n);
    bytes += (n * sizeof(T) + 255u) & ~size_t(255u);
    return nullptr;
  }
};

VisionScratch carve_vision(Carver& c, int rows, int groups, size_t pos_count, size_t rope_count) {
  const size_t r = static_cast<size_t>(rows), g = static_cast<size_t>(groups);
  VisionScratch s{};
  s.pixels = c.take<__nv_bfloat16>(r * kPatchDim);
  s.x = c.take<__nv_bfloat16>(r * kHidden);
  s.normed = c.take<__nv_bfloat16>(r * kHidden);
  s.qkv = c.take<__nv_bfloat16>(r * kQkvDim);
  s.q = c.take<__nv_bfloat16>(r * kHidden);
  s.k = c.take<__nv_bfloat16>(r * kHidden);
  s.v = c.take<__nv_bfloat16>(r * kHidden);
  s.branch = c.take<__nv_bfloat16>(r * kHidden);
  s.mlp = c.take<__nv_bfloat16>(r * kIntermediate);
  s.merger_normed = c.take<__nv_bfloat16>(r * kHidden);
  s.merged = c.take<__nv_bfloat16>(g * kMergedDim);
  s.merger_hidden = c.take<__nv_bfloat16>(g * kMergedDim);
  s.out = c.take<__nv_bfloat16>(g * kOutDim);
  for (int i = 0; i < 3; ++i)
    s.deep[i] = c.take<__nv_bfloat16>(g * kOutDim);
  s.pos_index = c.take<int32_t>(pos_count);
  s.rope_cos = c.take<float>(rope_count);
  s.rope_sin = c.take<float>(rope_count);
  return s;
}
}

struct QwenVisionEncoder::Impl {
  bool loaded = false;
  std::string prefix;
  std::unordered_map<std::string, DeviceBuffer<uint16_t>> tensors;
  cublasHandle_t handle = nullptr;
  cudaStream_t stream = nullptr;
  cuda::LinearRunner linear;
  // Two arenas, deliberately. `ws` is handed down to the tower, and
  // qwen_vision_attention reserves it from inside that call; a grow there frees
  // the whole buffer and resets the cursor. `acts` holds this file's per-image
  // activations, which have to outlive that call, so they live somewhere the
  // callee cannot reach. Sharing one arena would work only for as long as the
  // reserve here stayed >= the one the tower computes from its own copies of
  // the head geometry, and those are separate constants in qwen_vision.cu.
  cuda::Workspace ws, acts;

  DeviceBuffer<uint16_t>& at(const std::string& suffix) {
    return tensors.at(prefix + suffix);
  }

  void open() {
    if (handle)
      return;
    if (slopfab::cuda::cublas_create(&handle) != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("Qwen vision: cublasCreate failed");
    SLOPFAB_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    linear.init(handle, stream);
  }

  void close() {
    if (stream)
      cudaStreamDestroy(stream);
    if (handle)
      slopfab::cuda::cublas_destroy(handle);
    stream = nullptr;
    handle = nullptr;
  }

  cuda::QwenVisionBlockWeights block(int i) {
    const std::string p = "blocks." + std::to_string(i) + ".";
    cuda::QwenVisionBlockWeights b;
    b.norm1_weight = reinterpret_cast<__nv_bfloat16*>(at(p + "norm1.weight").get());
    b.norm1_bias = reinterpret_cast<__nv_bfloat16*>(at(p + "norm1.bias").get());
    b.norm2_weight = reinterpret_cast<__nv_bfloat16*>(at(p + "norm2.weight").get());
    b.norm2_bias = reinterpret_cast<__nv_bfloat16*>(at(p + "norm2.bias").get());
    b.qkv = dense(at(p + "attn.qkv.weight"), 3456, 1152, &at(p + "attn.qkv.bias"));
    b.attention_out = dense(at(p + "attn.proj.weight"), 1152, 1152, &at(p + "attn.proj.bias"));
    b.mlp_fc1 = dense(at(p + "mlp.linear_fc1.weight"), 4304, 1152, &at(p + "mlp.linear_fc1.bias"));
    b.mlp_fc2 = dense(at(p + "mlp.linear_fc2.weight"), 1152, 4304, &at(p + "mlp.linear_fc2.bias"));
    return b;
  }

  cuda::QwenVisionMergerWeights merger(const std::string& p, bool main) {
    cuda::QwenVisionMergerWeights m;
    m.norm_weight = reinterpret_cast<__nv_bfloat16*>(at(p + "norm.weight").get());
    m.norm_bias = reinterpret_cast<__nv_bfloat16*>(at(p + "norm.bias").get());
    m.fc1 = dense(at(p + "linear_fc1.weight"), 4608, 4608, &at(p + "linear_fc1.bias"));
    m.fc2 = dense(at(p + "linear_fc2.weight"), 5120, 4608, &at(p + "linear_fc2.bias"));
    m.norm_before_merge = main;
    return m;
  }

  QwenVisionEmbedding run(const std::vector<QwenPixelValues>& images, bool exact,
                          QwenVisionTrace* trace);
};

QwenVisionEncoder::QwenVisionEncoder() : impl_(new Impl) {
}

QwenVisionEncoder::~QwenVisionEncoder() {
  unload();
  impl_->close();
}

void QwenVisionEncoder::unload() {
  if (impl_->stream)
    cudaStreamSynchronize(impl_->stream);
  impl_->tensors.clear();
  impl_->ws = cuda::Workspace();
  impl_->acts = cuda::Workspace();
  impl_->loaded = false;
}

void QwenVisionEncoder::load(const SafeTensors& st) {
  unload();
  auto c = load_qwen3vl_vision_checkpoint(st);
  auto& s = *impl_;
  s.prefix = c.prefix;
  // The tower is 1.19 GB of a 27 GB conditioner, so the whole-file `prefetch()`
  // the other loaders use would pull 22x the bytes this one reads. Hinting just
  // the tower's extent turns ~290k serialised demand faults into one async
  // read. Issued before `open()` so the OS is already reading while the device
  // context comes up. Advisory: the loop below is correct without it.
  const void* extent = nullptr;
  size_t extent_bytes = 0;
  st.prefix_extent(s.prefix, &extent, &extent_bytes);
  st.prefetch_range(extent, extent_bytes);
  s.open();
  for (const auto& kv : st.tensors())
    if (kv.first.rfind(s.prefix, 0) == 0) {
      const TensorView& t = kv.second;
      DeviceBuffer<uint16_t> d(t.nbytes / 2);
      d.copy_from_host(static_cast<const uint16_t*>(t.data), t.nbytes / 2, s.stream);
      s.tensors.emplace(kv.first, std::move(d));
    }
  SLOPFAB_CUDA_CHECK(cudaStreamSynchronize(s.stream));
  s.loaded = true;
}

QwenVisionEmbedding QwenVisionEncoder::Impl::run(const std::vector<QwenPixelValues>& images,
                                                 bool exact, QwenVisionTrace* trace) {
  auto& s = *this;
  if (!s.loaded)
    throw std::runtime_error("Qwen vision: load first");
  QwenVisionEmbedding result;
  if (trace)
    *trace = {};
  std::vector<cuda::QwenVisionBlockWeights> blocks;
  for (int i = 0; i < 27; ++i)
    blocks.push_back(s.block(i));
  auto main = s.merger("merger.", true);
  cuda::QwenVisionMergerWeights deep[3];
  for (int i = 0; i < 3; ++i)
    deep[i] = s.merger("deepstack_merger_list." + std::to_string(i) + ".", false);
  // Fifteen DeviceBuffers used to be constructed and destroyed per image, and a
  // cudaFree synchronises the whole device (see the note on vit_decoder.cu's
  // scratch). They come out of `s.acts` instead, which keeps its high-water mark
  // for the life of the encoder -- bounded, because `unload()` is called right
  // after `encode` in encoder_kernels.cu and replaces both arenas.
  //
  // `s.acts` is not the arena handed to the tower. It cannot be: qwen_vision.cu
  // reserves that one from inside the call, sized from its own copies of the
  // head geometry, and a grow would free every pointer carved here and reset the
  // cursor -- freed memory, plausible numbers, no crash. Keeping the two
  // separate makes that structurally impossible rather than true-by-arithmetic.
  for (const auto& image : images) {
    const size_t patch_count = image.grid.patch_count();
    if (image.grid.temporal <= 0 || image.grid.height <= 0 || image.grid.width <= 0 ||
        (image.grid.height & 1) != 0 || (image.grid.width & 1) != 0 || patch_count == 0 ||
        patch_count > 16384 || patch_count % 4 != 0 || image.rows.size() != patch_count * kPatchDim)
      throw std::runtime_error("Qwen vision: pixel row mismatch");
    const int rows = static_cast<int>(patch_count), groups = rows / 4;
    std::vector<uint16_t> hp(image.rows.size());
    for (size_t i = 0; i < hp.size(); ++i)
      hp[i] = f32_to_bf16(image.rows[i]);
    auto positions = qwen3vl_vision_positions(image.grid);
    std::vector<float> hc, hs;
    qwen3vl_vision_rope_tables(positions, hc, hs);

    Carver measure;
    carve_vision(measure, rows, groups, positions.learned.size(), hc.size());
    s.acts.reserve(measure.bytes);
    cuda::Workspace::Scope scope(s.acts);
    Carver carver;
    carver.ws = &s.acts;
    const VisionScratch b = carve_vision(carver, rows, groups, positions.learned.size(), hc.size());

    SLOPFAB_CUDA_CHECK(cudaMemcpyAsync(b.pixels, hp.data(), hp.size() * sizeof(uint16_t),
                                       cudaMemcpyHostToDevice, s.stream));
    SLOPFAB_CUDA_CHECK(cudaMemcpyAsync(b.pos_index, positions.learned.data(),
                                       positions.learned.size() * sizeof(int32_t),
                                       cudaMemcpyHostToDevice, s.stream));
    SLOPFAB_CUDA_CHECK(cudaMemcpyAsync(b.rope_cos, hc.data(), hc.size() * sizeof(float),
                                       cudaMemcpyHostToDevice, s.stream));
    SLOPFAB_CUDA_CHECK(cudaMemcpyAsync(b.rope_sin, hs.data(), hs.size() * sizeof(float),
                                       cudaMemcpyHostToDevice, s.stream));
    auto patch =
        dense(s.at("patch_embed.proj.weight"), kHidden, kPatchDim, &s.at("patch_embed.proj.bias"));
    if (exact) {
      cuda::qwen_vision_patch_embed_exact(
          s.stream, patch, b.pixels,
          reinterpret_cast<__nv_bfloat16*>(s.at("pos_embed.weight").get()), b.pos_index, b.x, rows);
    } else {
      cuda::qwen_vision_patch_embed(
          s.linear, patch, b.pixels,
          reinterpret_cast<__nv_bfloat16*>(s.at("pos_embed.weight").get()), b.pos_index, b.x, rows,
          s.ws, s.stream);
    }
    cuda::QwenVisionBlockScratch bs{b.normed, b.qkv, b.q, b.k, b.v, b.branch, b.mlp};
    __nv_bfloat16* dp[3];
    for (int i = 0; i < 3; ++i)
      dp[i] = b.deep[i];
    DeviceBuffer<uint16_t> trace_device;
    if (exact) {
      if (trace)
        trace_device.allocate(static_cast<size_t>(27) * rows * kHidden);
      int deep_slot = 0;
      for (int layer = 0; layer < 27; ++layer) {
        cuda::qwen_vision_block_forward_exact(s.stream, blocks[layer], b.rope_cos, b.rope_sin, b.x,
                                              rows, bs);
        if (trace) {
          SLOPFAB_CUDA_CHECK(
              cudaMemcpyAsync(trace_device.get() + static_cast<size_t>(layer) * rows * kHidden, b.x,
                              static_cast<size_t>(rows) * kHidden * sizeof(uint16_t),
                              cudaMemcpyDeviceToDevice, s.stream));
        }
        if (layer == 8 || layer == 16 || layer == 24) {
          cuda::qwen_vision_merger_forward_exact(s.stream, deep[deep_slot], b.x, b.merger_normed,
                                                 b.merged, b.merger_hidden, b.deep[deep_slot],
                                                 rows);
          ++deep_slot;
        }
      }
      cuda::qwen_vision_merger_forward_exact(s.stream, main, b.x, b.merger_normed, b.merged,
                                             b.merger_hidden, b.out, rows);
    } else {
      cuda::qwen_vision_tower_forward(s.handle, s.stream, s.linear, blocks.data(), main, deep,
                                      b.rope_cos, b.rope_sin, b.x, rows, bs, b.merger_normed,
                                      b.merged, b.merger_hidden, b.out, dp, s.ws);
    }
    const size_t out_n = static_cast<size_t>(groups) * kOutDim;
    const size_t old = result.main.size();
    result.main.resize(old + out_n);
    SLOPFAB_CUDA_CHECK(cudaMemcpyAsync(result.main.data() + old, b.out, out_n * sizeof(uint16_t),
                                       cudaMemcpyDeviceToHost, s.stream));
    for (int i = 0; i < 3; ++i) {
      size_t o = result.deepstack[i].size();
      result.deepstack[i].resize(o + out_n);
      SLOPFAB_CUDA_CHECK(cudaMemcpyAsync(result.deepstack[i].data() + o, b.deep[i],
                                         out_n * sizeof(uint16_t), cudaMemcpyDeviceToHost,
                                         s.stream));
    }
    if (trace) {
      const size_t old_trace = trace->block_residuals.size();
      const size_t trace_count = static_cast<size_t>(27) * rows * kHidden;
      trace->block_residuals.resize(old_trace + trace_count);
      SLOPFAB_CUDA_CHECK(cudaMemcpyAsync(trace->block_residuals.data() + old_trace,
                                         trace_device.get(), trace_count * sizeof(uint16_t),
                                         cudaMemcpyDeviceToHost, s.stream));
      trace->tokens += rows;
    }
    result.tokens += groups;
    SLOPFAB_CUDA_CHECK(cudaStreamSynchronize(s.stream));
  }
  return result;
}

QwenVisionEmbedding QwenVisionEncoder::encode(const std::vector<QwenPixelValues>& images) {
  return impl_->run(images, false, nullptr);
}

QwenVisionEmbedding QwenVisionEncoder::encode_exact(const std::vector<QwenPixelValues>& images,
                                                    QwenVisionTrace* trace) {
  return impl_->run(images, true, trace);
}
} // namespace slopfab::text
