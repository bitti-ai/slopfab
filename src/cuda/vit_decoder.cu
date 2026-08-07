#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "vidfab/cuda/device.h"
#include "vidfab/cuda/gemm.cuh"
#include "vidfab/cuda/vae_kernels.cuh"
#include "vidfab/tensor_convert.h"
#include "vidfab/vae/vit_decoder.h"

namespace vidfab::vae {
namespace {

void cublas_check(cublasStatus_t status, const char* expr, int line) {
  if (status == CUBLAS_STATUS_SUCCESS) return;
  throw std::runtime_error("cublas: status " + std::to_string(static_cast<int>(status)) + " at " +
                           std::string(expr) + " (vit_decoder.cu:" + std::to_string(line) + ")");
}
#define CUBLAS_CHECK(expr) cublas_check((expr), #expr, __LINE__)

using cuda::DeviceBuffer;

// Staging for the weight upload path. fp16 tensors go across PCIe as fp16 and
// are widened on the device: half the bytes on the bus, and no scalar
// conversion loop on the host. A pinned bounce buffer is used because the
// safetensors mapping is pageable, and a pageable async copy silently
// synchronises.
class WeightUploader {
 public:
  explicit WeightUploader(cudaStream_t stream) : stream_(stream) {
    staging_.allocate(kStagingElems);
    raw_.allocate(kStagingElems * sizeof(uint16_t));
  }

  DeviceBuffer<float> upload(const SafeTensors& ckpt, const std::string& name,
                             size_t expected_elems) {
    const TensorView& view = ckpt.at(name);
    const auto count = static_cast<size_t>(view.numel());
    if (expected_elems != 0 && count != expected_elems) {
      throw std::runtime_error("vae: tensor '" + name + "' has " + std::to_string(count) +
                               " elements, expected " + std::to_string(expected_elems));
    }

    DeviceBuffer<float> out(count);
    if (view.dtype == DType::kF16) {
      // Chunked so a single tensor larger than the staging buffer still works.
      const auto* src = static_cast<const uint8_t*>(view.data);
      size_t done = 0;
      while (done < count) {
        const size_t n = std::min(count - done, kStagingElems);
        std::memcpy(raw_.get(), src + done * sizeof(uint16_t), n * sizeof(uint16_t));
        VIDFAB_CUDA_CHECK(cudaMemcpyAsync(staging_.get(), raw_.get(), n * sizeof(uint16_t),
                                          cudaMemcpyHostToDevice, stream_));
        cuda::launch_widen_f16(staging_.get(), out.get() + done, n, stream_);
        // The staging buffer is reused next iteration, so the copy and widen
        // must complete before the next memcpy overwrites it.
        VIDFAB_CUDA_CHECK(cudaStreamSynchronize(stream_));
        done += n;
      }
    } else {
      // Rare in this checkpoint; fall back to host conversion.
      const std::vector<float> host = to_f32(view);
      out.copy_from_host(host.data(), host.size(), stream_);
      VIDFAB_CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
    return out;
  }

 private:
  // 64 Mi elements = 128 MiB of fp16 per hop.
  static constexpr size_t kStagingElems = 64ull << 20;
  cudaStream_t stream_;
  DeviceBuffer<uint16_t> staging_;
  cuda::PinnedBuffer<uint8_t> raw_;
};

struct BlockWeights {
  DeviceBuffer<float> norm1;      // [dim]
  DeviceBuffer<float> norm2;      // [dim]
  DeviceBuffer<float> scale1;     // [dim]
  DeviceBuffer<float> scale2;     // [dim]
  DeviceBuffer<float> qkv_w;      // [3*dim, dim]
  DeviceBuffer<float> qkv_b;      // [3*dim]
  DeviceBuffer<float> out_w;      // [dim, dim]
  DeviceBuffer<float> out_b;      // [dim]
  DeviceBuffer<float> w1;         // [2*ffn_inner, dim]
  DeviceBuffer<float> w1_b;       // [2*ffn_inner]
  DeviceBuffer<float> w2;         // [dim, ffn_inner]
  DeviceBuffer<float> w2_b;       // [dim]
};

}  // namespace

struct ViTDecoder::Impl {
  ViTConfig cfg;
  cublasHandle_t blas = nullptr;
  cuda::Stream stream;
  size_t weight_bytes = 0;

  std::vector<BlockWeights> blocks;
  DeviceBuffer<float> x_embed_w;      // [dim, in_channels]
  DeviceBuffer<float> x_embed_b;      // [dim]
  DeviceBuffer<float> register_tokens;  // [num_register, dim]
  DeviceBuffer<float> norm_out_w;
  DeviceBuffer<float> norm_out_b;
  DeviceBuffer<float> proj_out_w;     // [patch_dim, dim]
  DeviceBuffer<float> proj_out_b;     // [patch_dim]
  DeviceBuffer<float> post_quant_w;   // [in_channels, in_channels]
  DeviceBuffer<float> post_quant_b;   // [in_channels]

  // Scratch, resized on demand for the current window size. Nothing is ever
  // freed mid-run: cudaFree synchronises the whole device, which would make
  // any future stream overlap impossible.
  int cap_seq = 0;
  int cap_batch = 0;
  // RoPE tables depend only on the window extents, which repeat across every
  // tile and chunk, so they are rebuilt only when the shape actually changes.
  int rope_T = -1;
  int rope_H = -1;
  int rope_W = -1;
  DeviceBuffer<float> d_tokens;   // [S, dim]
  DeviceBuffer<float> d_normed;   // [S, dim]
  DeviceBuffer<float> d_qkv;      // [S, 3*dim]
  DeviceBuffer<float> d_q, d_k, d_v;  // [H, S, D]
  DeviceBuffer<float> d_scores;   // [H, S, S]
  DeviceBuffer<float> d_merged;   // [S, dim]
  DeviceBuffer<float> d_proj;     // [S, dim] or [S, patch_dim]
  DeviceBuffer<float> d_ffn;      // [S, 2*ffn_inner]
  DeviceBuffer<float> d_act;      // [S, ffn_inner]
  DeviceBuffer<float> d_cos, d_sin;  // [S, rope_dim]
  DeviceBuffer<float> d_latent;     // [in_channels, T*H*W]
  DeviceBuffer<float> d_patch;      // [N, in_channels] packed tokens
  DeviceBuffer<float> d_quantised;  // [N, in_channels] after post_quant_conv
  DeviceBuffer<float> d_pixels;     // [3, T*4, H*16, W*16]
  size_t cap_pixels = 0;
  cuda::PinnedBuffer<float> pinned_out;  // staging for the D2H of decoded pixels

  ~Impl() {
    if (blas != nullptr) cublasDestroy(blas);
  }

  void gemm_nt(const float* A, const float* B, float* C, int M, int N, int K) {
    cuda::gemm_nt(blas, A, B, C, M, N, K);
  }

  void gemm_nn(const float* A, const float* B, float* C, int M, int N, int K) {
    cuda::gemm_nn(blas, A, B, C, M, N, K);
  }

  void gemm_nt_batched(const float* A, const float* B, float* C, int M, int N, int K, int batch,
                       long long strideA, long long strideB, long long strideC) {
    cuda::gemm_nt_batched(blas, A, B, C, M, N, K, batch, strideA, strideB, strideC);
  }

  void gemm_nn_batched(const float* A, const float* B, float* C, int M, int N, int K, int batch,
                       long long strideA, long long strideB, long long strideC) {
    cuda::gemm_nn_batched(blas, A, B, C, M, N, K, batch, strideA, strideB, strideC);
  }

  void ensure_scratch(int seq, int num_patches, int batch) {
    // Three buffers below are sized from num_patches while the early-out tests
    // seq. That is only safe because the two move together; assert it rather
    // than rely on the caller.
    if (seq != num_patches + cfg.num_suffix) {
      throw std::runtime_error("vae: ensure_scratch called with inconsistent seq/num_patches");
    }
    if (seq <= cap_seq && batch <= cap_batch) return;
    const int dim = cfg.dim;
    const int hd = cfg.head_dim;
    const size_t s = static_cast<size_t>(seq) * batch;
    const int ch = cfg.in_channels;

    d_latent.allocate(static_cast<size_t>(batch) * ch * num_patches);
    d_patch.allocate(static_cast<size_t>(batch) * num_patches * ch);
    d_quantised.allocate(static_cast<size_t>(batch) * num_patches * ch);
    d_tokens.allocate(s * dim);
    d_normed.allocate(s * dim);
    d_qkv.allocate(s * 3 * dim);
    // Attention is deliberately serialized by document. Replicating the
    // quadratic score buffer for every spatial tile would erase batching's
    // memory advantage; token-wise activations above remain batched.
    d_q.allocate(static_cast<size_t>(seq) * cfg.heads * hd);
    d_k.allocate(static_cast<size_t>(seq) * cfg.heads * hd);
    d_v.allocate(static_cast<size_t>(seq) * cfg.heads * hd);
    d_scores.allocate(static_cast<size_t>(cfg.heads) * seq * seq);
    d_merged.allocate(s * dim);
    d_proj.allocate(s * static_cast<size_t>(cfg.patch_dim()));
    d_ffn.allocate(s * 2 * cfg.ffn_inner);
    d_act.allocate(s * cfg.ffn_inner);
    d_cos.allocate(static_cast<size_t>(seq) * cfg.rope_dim);
    d_sin.allocate(static_cast<size_t>(seq) * cfg.rope_dim);

    // Only after every allocation has succeeded: if one throws, a retry at the
    // same seq must rebuild rather than run on undersized buffers.
    cap_seq = seq;
    cap_batch = batch;
    rope_T = rope_H = rope_W = -1;  // tables live in the reallocated buffers
  }

  // Builds the RoPE cos/sin tables on the host.
  //
  // Coordinates are length-normalised per axis to (-1, 1) — they depend on the
  // extents of the tensor entering the ViT, not on any global frame index. The
  // 24 unique angles are ordered T(8), H(8), W(8) and then duplicated to 48.
  void build_rope(int T, int H, int W, int seq, int num_patches) {
    if (T == rope_T && H == rope_H && W == rope_W) return;
    const int rope_dim = cfg.rope_dim;
    const int half = rope_dim / 2;  // 24 unique angles
    const int per_axis = half / 3;  // 8 frequencies per axis

    std::vector<float> inv_freq(per_axis);
    for (int f = 0; f < per_axis; ++f) {
      inv_freq[f] = 1.0f / std::pow(cfg.rope_theta,
                                    static_cast<float>(f) / static_cast<float>(per_axis));
    }

    auto axis_coord = [](int index, int extent) {
      const float c = (static_cast<float>(index) + 0.5f) / static_cast<float>(extent);
      return 2.0f * c - 1.0f;
    };

    std::vector<float> cos_tab(static_cast<size_t>(seq) * rope_dim, 1.0f);
    std::vector<float> sin_tab(static_cast<size_t>(seq) * rope_dim, 0.0f);

    const double two_pi = 6.283185307179586476925286766559;
    for (int t = 0; t < T; ++t) {
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          const size_t token = (static_cast<size_t>(t) * H + h) * W + w;
          if (token >= static_cast<size_t>(num_patches)) continue;
          const float coord[3] = {axis_coord(t, T), axis_coord(h, H), axis_coord(w, W)};
          for (int axis = 0; axis < 3; ++axis) {
            for (int f = 0; f < per_axis; ++f) {
              const int j = axis * per_axis + f;
              const double angle = two_pi * static_cast<double>(coord[axis]) *
                                   static_cast<double>(inv_freq[f]);
              const float c = static_cast<float>(std::cos(angle));
              const float s = static_cast<float>(std::sin(angle));
              cos_tab[token * rope_dim + j] = c;
              sin_tab[token * rope_dim + j] = s;
              // tile(2): the second half repeats the first.
              cos_tab[token * rope_dim + j + half] = c;
              sin_tab[token * rope_dim + j + half] = s;
            }
          }
        }
      }
    }
    d_cos.copy_from_host(cos_tab.data(), cos_tab.size(), stream);
    d_sin.copy_from_host(sin_tab.data(), sin_tab.size(), stream);
    stream.synchronize();  // host vectors die at scope exit
    rope_T = T;
    rope_H = H;
    rope_W = W;
  }

  void run_block(const BlockWeights& b, int seq, int num_patches, int batch) {
    const int dim = cfg.dim;
    const int hd = cfg.head_dim;
    const int heads = cfg.heads;
    cudaStream_t s = stream.get();

    // --- attention ---
    const int rows = seq * batch;
    cuda::launch_rmsnorm(d_tokens.get(), b.norm1.get(), d_normed.get(), rows, dim, cfg.eps, s);
    gemm_nt(d_normed.get(), b.qkv_w.get(), d_qkv.get(), rows, 3 * dim, dim);

    // The qkv bias is applied inside the split kernel, which already reads
    // every element of d_qkv once.
    const long long head_stride = static_cast<long long>(seq) * hd;
    const long long score_stride = static_cast<long long>(seq) * seq;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    for (int doc = 0; doc < batch; ++doc) {
      const size_t row0 = static_cast<size_t>(doc) * seq;
      cuda::launch_split_qkv_norm_rope(d_qkv.get() + row0 * 3 * dim, b.qkv_b.get(), d_cos.get(),
                                       d_sin.get(), d_q.get(), d_k.get(), d_v.get(), seq, heads,
                                       hd, cfg.rope_dim, num_patches, cfg.eps, s);
      gemm_nt_batched(d_q.get(), d_k.get(), d_scores.get(), seq, seq, hd, heads, head_stride,
                      head_stride, score_stride);
      cuda::launch_softmax_rows(d_scores.get(), heads * seq, seq, scale, s);
      cuda::gemm_nn_batched_ld(blas, d_scores.get(), d_v.get(), d_merged.get() + row0 * dim, seq,
                               hd, seq, heads, score_stride, head_stride, /*strideC=*/hd,
                               /*ldc=*/heads * hd);
    }

    gemm_nt(d_merged.get(), b.out_w.get(), d_normed.get(), rows, dim, dim);
    cuda::launch_layerscale_residual(d_tokens.get(), d_normed.get(), b.out_b.get(), b.scale1.get(),
                                     rows, dim, s);

    // --- feed forward ---
    cuda::launch_rmsnorm(d_tokens.get(), b.norm2.get(), d_normed.get(), rows, dim, cfg.eps, s);
    gemm_nt(d_normed.get(), b.w1.get(), d_ffn.get(), rows, 2 * cfg.ffn_inner, dim);
    cuda::launch_swiglu(d_ffn.get(), b.w1_b.get(), d_act.get(), rows, cfg.ffn_inner, s);
    gemm_nt(d_act.get(), b.w2.get(), d_normed.get(), rows, dim, cfg.ffn_inner);
    cuda::launch_layerscale_residual(d_tokens.get(), d_normed.get(), b.w2_b.get(), b.scale2.get(),
                                     rows, dim, s);
  }
};

ViTDecoder::ViTDecoder() : impl_(std::make_unique<Impl>()) {}
ViTDecoder::~ViTDecoder() = default;

const ViTConfig& ViTDecoder::config() const { return impl_->cfg; }
size_t ViTDecoder::weight_bytes() const { return impl_->weight_bytes; }

void ViTDecoder::load(const SafeTensors& ckpt, const ViTConfig& config) {
  Impl& d = *impl_;
  // See the note on `SafeTensors::prefetch`: this loader consumes the whole
  // file, so it asks for it up front rather than one page fault at a time.
  ckpt.prefetch();
  d.cfg = config;
  CUBLAS_CHECK(cublasCreate(&d.blas));
  CUBLAS_CHECK(cublasSetStream(d.blas, d.stream.get()));
  // DEFAULT already refuses to drop mantissa bits for an fp32 compute type —
  // TF32 requires an explicitly TF32 math mode or compute type, neither of
  // which we ask for. PEDANTIC additionally forbids optimisations that do not
  // affect precision, and NVIDIA documents it as slower. Keep it available for
  // the correctness harness via VIDFAB_CUBLAS_PEDANTIC=1.
  const char* pedantic = std::getenv("VIDFAB_CUBLAS_PEDANTIC");
  const bool want_pedantic = pedantic != nullptr && pedantic[0] == '1';
  CUBLAS_CHECK(cublasSetMathMode(d.blas,
                                 want_pedantic ? CUBLAS_PEDANTIC_MATH : CUBLAS_DEFAULT_MATH));

  const int dim = config.dim;
  const int inner = config.ffn_inner;
  const int ch = config.in_channels;

  // Shape constraints the kernels rely on. Checked once here rather than in the
  // launcher, which runs 36 times per window.
  if (config.head_dim != 64) {
    throw std::runtime_error("vae: split_qkv_norm_rope requires head_dim 64, got " +
                             std::to_string(config.head_dim));
  }
  if (config.rope_dim % 2 != 0 || config.rope_dim > config.head_dim) {
    throw std::runtime_error("vae: rope_dim must be even and <= head_dim");
  }
  if (config.heads * config.head_dim != dim) {
    throw std::runtime_error("vae: heads * head_dim must equal dim");
  }

  WeightUploader uploader(d.stream.get());

  d.x_embed_w = uploader.upload(ckpt, "decoder.x_embedder.weight", static_cast<size_t>(dim) * ch);
  d.x_embed_b = uploader.upload(ckpt, "decoder.x_embedder.bias", dim);
  d.register_tokens = uploader.upload(ckpt, "decoder.register_tokens",
                                      static_cast<size_t>(config.num_register) * dim);
  d.norm_out_w = uploader.upload(ckpt, "decoder.norm_out.weight", dim);
  d.norm_out_b = uploader.upload(ckpt, "decoder.norm_out.bias", dim);
  d.proj_out_w = uploader.upload(ckpt, "decoder.proj_out.weight",
                                 static_cast<size_t>(config.patch_dim()) * dim);
  d.proj_out_b = uploader.upload(ckpt, "decoder.proj_out.bias", config.patch_dim());
  d.post_quant_w = uploader.upload(ckpt, "post_quant_conv.weight", static_cast<size_t>(ch) * ch);
  d.post_quant_b = uploader.upload(ckpt, "post_quant_conv.bias", ch);

  d.blocks.resize(config.num_layers);
  for (int i = 0; i < config.num_layers; ++i) {
    const std::string p = "decoder.transformer_blocks." + std::to_string(i) + ".";
    BlockWeights& b = d.blocks[i];
    b.norm1 = uploader.upload(ckpt, p + "norm1.weight", dim);
    b.norm2 = uploader.upload(ckpt, p + "norm2.weight", dim);
    b.scale1 = uploader.upload(ckpt, p + "scale1", dim);
    b.scale2 = uploader.upload(ckpt, p + "scale2", dim);
    b.qkv_w = uploader.upload(ckpt, p + "attn.to_qkv.weight", static_cast<size_t>(3) * dim * dim);
    b.qkv_b = uploader.upload(ckpt, p + "attn.to_qkv.bias", static_cast<size_t>(3) * dim);
    b.out_w = uploader.upload(ckpt, p + "attn.to_out.weight", static_cast<size_t>(dim) * dim);
    b.out_b = uploader.upload(ckpt, p + "attn.to_out.bias", dim);
    b.w1 = uploader.upload(ckpt, p + "ff.w1.weight", static_cast<size_t>(2) * inner * dim);
    b.w1_b = uploader.upload(ckpt, p + "ff.w1.bias", static_cast<size_t>(2) * inner);
    b.w2 = uploader.upload(ckpt, p + "ff.w2.weight", static_cast<size_t>(dim) * inner);
    b.w2_b = uploader.upload(ckpt, p + "ff.w2.bias", dim);
  }

  size_t total = d.x_embed_w.nbytes() + d.x_embed_b.nbytes() + d.register_tokens.nbytes() +
                 d.norm_out_w.nbytes() + d.norm_out_b.nbytes() + d.proj_out_w.nbytes() +
                 d.proj_out_b.nbytes() + d.post_quant_w.nbytes() + d.post_quant_b.nbytes();
  for (const BlockWeights& b : d.blocks) {
    total += b.norm1.nbytes() + b.norm2.nbytes() + b.scale1.nbytes() + b.scale2.nbytes() +
             b.qkv_w.nbytes() + b.qkv_b.nbytes() + b.out_w.nbytes() + b.out_b.nbytes() +
             b.w1.nbytes() + b.w1_b.nbytes() + b.w2.nbytes() + b.w2_b.nbytes();
  }
  d.weight_bytes = total;
  d.stream.synchronize();
}

void ViTDecoder::forward_window(const float* z, int T, int H, int W, std::vector<float>& out) {
  std::vector<std::vector<float>> batch_out;
  forward_windows(z, 1, T, H, W, batch_out);
  out = std::move(batch_out.front());
}

void ViTDecoder::forward_windows(const float* z, int batch, int T, int H, int W,
                                 std::vector<std::vector<float>>& out) {
  Impl& d = *impl_;
  if (d.blocks.empty()) throw std::runtime_error("vae: decoder weights not loaded");
  if (batch <= 0) throw std::runtime_error("vae: window batch must be positive");

  const ViTConfig& cfg = d.cfg;
  const int ch = cfg.in_channels;
  const int num_patches = T * H * W;
  const int seq = num_patches + cfg.num_suffix;
  const int dim = cfg.dim;
  cudaStream_t s = d.stream.get();

  d.ensure_scratch(seq, num_patches, batch);
  d.build_rope(T, H, W, seq, num_patches);

  // Latent arrives channel-first [C, T, H, W]; the ViT wants one token per
  // voxel, channel-last, in (t, h, w) row-major order. The transpose runs on
  // the device — it used to round-trip the same bytes back to the host.
  const size_t voxels = static_cast<size_t>(num_patches);
  d.d_latent.copy_from_host(z, static_cast<size_t>(batch) * ch * voxels, s);
  for (int doc = 0; doc < batch; ++doc) {
    const size_t latent0 = static_cast<size_t>(doc) * ch * voxels;
    const size_t patch0 = static_cast<size_t>(doc) * num_patches;
    const size_t token0 = static_cast<size_t>(doc) * seq;
    cuda::launch_transpose_cn_to_nc(d.d_latent.get() + latent0,
                                    d.d_patch.get() + patch0 * ch, ch,
                                    static_cast<int>(voxels), s);
    d.gemm_nt(d.d_patch.get() + patch0 * ch, d.post_quant_w.get(),
              d.d_quantised.get() + patch0 * ch, num_patches, ch, ch);
    cuda::launch_add_bias(d.d_quantised.get() + patch0 * ch, d.post_quant_b.get(), num_patches,
                          ch, s);
    d.gemm_nt(d.d_quantised.get() + patch0 * ch, d.x_embed_w.get(),
              d.d_tokens.get() + token0 * dim, num_patches, dim, ch);
    cuda::launch_add_bias(d.d_tokens.get() + token0 * dim, d.x_embed_b.get(), num_patches, dim, s);
    VIDFAB_CUDA_CHECK(cudaMemcpyAsync(
        d.d_tokens.get() + (token0 + num_patches) * dim, d.register_tokens.get(),
        static_cast<size_t>(cfg.num_register) * dim * sizeof(float), cudaMemcpyDeviceToDevice, s));
    VIDFAB_CUDA_CHECK(cudaMemsetAsync(
        d.d_tokens.get() + (token0 + num_patches + cfg.num_register) * dim, 0,
        static_cast<size_t>(dim) * sizeof(float), s));
  }

  for (int i = 0; i < cfg.num_layers; ++i)
    d.run_block(d.blocks[i], seq, num_patches, batch);

  const size_t pixels = static_cast<size_t>(cfg.out_channels) * (T * cfg.patch_t) *
                        (H * cfg.patch) * (W * cfg.patch);
  if (pixels > d.cap_pixels) {
    d.d_pixels.allocate(pixels);
    d.pinned_out.allocate(pixels);
    d.cap_pixels = pixels;
  }
  const int patch_dim = cfg.patch_dim();
  out.resize(static_cast<size_t>(batch));
  for (int doc = 0; doc < batch; ++doc) {
    const size_t token0 = static_cast<size_t>(doc) * seq;
    // The suffix rows between documents mean the final projection is issued
    // per document. It runs once per decode, unlike the 216 projections in
    // the transformer body, and preserves the exact singleton arithmetic.
    cuda::launch_layernorm(d.d_tokens.get() + token0 * dim, d.norm_out_w.get(),
                           d.norm_out_b.get(), d.d_normed.get(), num_patches, dim, cfg.eps, s);
    d.gemm_nt(d.d_normed.get(), d.proj_out_w.get(), d.d_proj.get(), num_patches, patch_dim, dim);
    cuda::launch_add_bias(d.d_proj.get(), d.proj_out_b.get(), num_patches, patch_dim, s);
    cuda::launch_depth_to_space(d.d_proj.get(), d.d_pixels.get(), T, H, W, cfg.out_channels,
                                cfg.patch_t, cfg.patch, s);
    d.d_pixels.copy_to_host(d.pinned_out.get(), pixels, s);
    d.stream.synchronize();
    out[static_cast<size_t>(doc)].resize(pixels);
    std::memcpy(out[static_cast<size_t>(doc)].data(), d.pinned_out.get(), pixels * sizeof(float));
  }
}

}  // namespace vidfab::vae
