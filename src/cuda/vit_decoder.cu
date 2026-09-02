#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "vidfab/cuda/device.h"
#include "vidfab/cuda/attention.cuh"
#include "vidfab/cuda/deterministic_gemm.cuh"
#include "vidfab/cuda/gemm.cuh"
#include "vidfab/cuda/linear.cuh"
#include "vidfab/cuda/nf4_weight.cuh"
#include "vidfab/cuda/profile.h"
#include "vidfab/cuda/vae_kernels.cuh"
#include "vidfab/cuda/vae_vit_block.h"
#include "vidfab/nf4.h"
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
// conversion loop on the host.
//
// When the caller managed to page-lock the mapping, the fp16 bytes are read
// straight out of it. Otherwise a pinned bounce buffer is used, because a
// pageable async copy silently synchronises.
class WeightUploader {
 public:
  WeightUploader(cudaStream_t stream, const cuda::RegisteredMapping& mapping)
      : stream_(stream), mapping_(mapping) {
    staging_.allocate(kStagingElems);
    // `raw_`, the pinned bounce buffer, is allocated lazily: it only exists
    // for the path where the source is not page-locked, and when registration
    // succeeds that path is never taken.
  }

  // The direct path leaves DMAs in flight out of the caller's mapping, which
  // the caller is about to unregister. It synchronises before that happens;
  // this is here so the ordering stays safe if it ever stops.
  ~WeightUploader() { cudaStreamSynchronize(stream_); }

  WeightUploader(const WeightUploader&) = delete;
  WeightUploader& operator=(const WeightUploader&) = delete;

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
      const bool direct = mapping_.contains(view.data, count * sizeof(uint16_t));
      size_t done = 0;
      while (done < count) {
        const size_t n = std::min(count - done, kStagingElems);
        if (direct) {
          // Straight out of the page-locked mapping: no host copy, and no
          // synchronise either. The only reused buffer left is `staging_`,
          // which lives on the device, so stream order already guarantees the
          // widen of one hop finishes before the next hop overwrites it.
          VIDFAB_CUDA_CHECK(cudaMemcpyAsync(staging_.get(), src + done * sizeof(uint16_t),
                                            n * sizeof(uint16_t), cudaMemcpyHostToDevice,
                                            stream_));
          cuda::launch_widen_f16(staging_.get(), out.get() + done, n, stream_);
        } else {
          // Allocated on first use rather than in the constructor, so the
          // common registered path never pays for 128 MiB of pinned memory it
          // will not touch.
          if (raw_.get() == nullptr) raw_.allocate(kStagingElems * sizeof(uint16_t));
          std::memcpy(raw_.get(), src + done * sizeof(uint16_t), n * sizeof(uint16_t));
          VIDFAB_CUDA_CHECK(cudaMemcpyAsync(staging_.get(), raw_.get(), n * sizeof(uint16_t),
                                            cudaMemcpyHostToDevice, stream_));
          cuda::launch_widen_f16(staging_.get(), out.get() + done, n, stream_);
          // The pinned buffer is reused next iteration, so the copy and widen
          // must complete before the next memcpy overwrites it.
          VIDFAB_CUDA_CHECK(cudaStreamSynchronize(stream_));
        }
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
  const cuda::RegisteredMapping& mapping_;
  DeviceBuffer<uint16_t> staging_;
  cuda::PinnedBuffer<uint8_t> raw_;
};

struct BlockWeights {
  DeviceBuffer<float> norm1;      // [dim]
  DeviceBuffer<float> norm2;      // [dim]
  DeviceBuffer<float> scale1;     // [dim]
  DeviceBuffer<float> scale2;     // [dim]
  cuda::F16Weight qkv_w;          // [3*dim, dim]
  DeviceBuffer<float> qkv_b;      // [3*dim]
  cuda::F16Weight out_w;          // [dim, dim]
  DeviceBuffer<float> out_b;      // [dim]
  cuda::F16Weight w1;             // [2*ffn_inner, dim]
  DeviceBuffer<float> w1_b;       // [2*ffn_inner]
  cuda::F16Weight w2;             // [dim, ffn_inner]
  DeviceBuffer<float> w2_b;       // [dim]
};

}  // namespace

struct ViTDecoder::Impl {
  ViTConfig cfg;
  cublasHandle_t blas = nullptr;
  cuda::Stream stream;
  size_t weight_bytes = 0;

  std::vector<BlockWeights> blocks;
  std::unique_ptr<cuda::ExactViTBlockGraph> exact_blocks;
  bool loaded = false;
  cuda::F16Weight x_embed_w;          // [dim, in_channels]
  DeviceBuffer<float> x_embed_b;      // [dim]
  DeviceBuffer<float> register_tokens;  // [num_register, dim]
  DeviceBuffer<float> norm_out_w;
  DeviceBuffer<float> norm_out_b;
  cuda::F16Weight proj_out_w;         // [patch_dim, dim]
  DeviceBuffer<float> proj_out_b;     // [patch_dim]
  cuda::F16Weight post_quant_w;       // [in_channels, in_channels]
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
  DeviceBuffer<__nv_bfloat16> d_q_bf16, d_k_bf16, d_v_bf16;  // [S, H, D]
  DeviceBuffer<__nv_bfloat16> d_attn_bf16;                     // [S, H, D]
  cuda::Workspace attention_ws;
  DeviceBuffer<float> d_proj;     // [S, dim] or [S, patch_dim]
  DeviceBuffer<float> d_ffn;      // [S, 2*ffn_inner]
  DeviceBuffer<__half> d_gemm_in; // narrowed input for tensor-core linears
  DeviceBuffer<__half> d_weight;  // active NF4 matrix expansion
  size_t cap_weight = 0;
  DeviceBuffer<float> d_cos, d_sin;  // [S, rope_dim]
  DeviceBuffer<float> d_latent;     // [in_channels, T*H*W]
  DeviceBuffer<float> d_patch;      // [N, in_channels] packed tokens
  DeviceBuffer<float> d_quantised;  // [N, in_channels] after post_quant_conv
  DeviceBuffer<float> d_pixels;     // [3, T*4, H*16, W*16]
  size_t cap_pixels = 0;
  DeviceBuffer<float> d_denorm_input, d_denorm_output, d_denorm_mean,
      d_denorm_std;
  uint64_t cap_denorm_voxels = 0;
  cuda::PinnedBuffer<float> pinned_out;  // staging for the D2H of decoded pixels

  // Page-locking of the *caller's* tile buffers. The decode hoists one buffer
  // per tile and hands the same one back every chunk, so registering it once
  // lets the D2H land the tile in its final home instead of copying it into
  // `pinned_out` and then memcpy'ing it out again — two crossings of host
  // memory per tile, 1.2 GiB per decode at the shipped geometry.
  //
  // Best-effort, exactly like cuda::RegisteredMapping: if the registration
  // fails the staged path still runs and is only slower. Registration is not
  // tracked by the buffer, so the *caller* must call
  // release_host_registrations() before those buffers are freed, and this class
  // must drop a registration before a resize reallocates underneath it.
  struct HostRegistration {
    void* base = nullptr;
    size_t bytes = 0;
  };
  std::vector<HostRegistration> host_regs;
  bool warned_no_page_lock = false;

  ~Impl() {
    release_host_regs();
    if (blas != nullptr) vidfab::cuda::cublas_destroy(blas);
  }

  void erase_registration(size_t index) {
    cudaHostUnregister(host_regs[index].base);
    // Symmetric with the register side: never leave a sticky error behind for
    // the next unrelated call to be blamed for.
    cudaGetLastError();
    host_regs.erase(host_regs.begin() + static_cast<ptrdiff_t>(index));
  }

  void unregister_host(void* p) {
    if (p == nullptr) return;
    for (size_t i = 0; i < host_regs.size(); ++i) {
      if (host_regs[i].base == p) {
        erase_registration(i);
        return;
      }
    }
  }

  void release_host_regs() {
    while (!host_regs.empty()) erase_registration(host_regs.size() - 1);
  }

  // True when `[p, p + bytes)` is page-locked on return.
  bool ensure_registered(void* p, size_t bytes) {
    if (p == nullptr || bytes == 0) return false;
    for (size_t i = 0; i < host_regs.size(); ++i) {
      if (host_regs[i].base != p) continue;
      if (host_regs[i].bytes == bytes) return true;
      // Same address, different length: the old lock covers the wrong range.
      erase_registration(i);
      break;
    }
    const cudaError_t rc = cudaHostRegister(p, bytes, cudaHostRegisterDefault);
    if (rc != cudaSuccess) {
      cudaGetLastError();
      // Said once, not once per tile per chunk. Without it the decode silently
      // reverts to staging every tile through pinned memory and copying it out
      // — the exact cost this path exists to remove — and the only evidence is
      // a phase timing nobody is looking at. cudaErrorHostMemoryAlreadyRegistered
      // matters just as much as an out-of-memory here: it means some other
      // registration already covers this range, and that tile stays staged for
      // the whole run.
      if (!warned_no_page_lock) {
        warned_no_page_lock = true;
        std::fprintf(stderr,
                     "vidfab: could not page-lock a video vae output tile (%s); decoded tiles "
                     "are being staged through pinned memory and copied, which is slower\n",
                     cudaGetErrorName(rc));
      }
      return false;
    }
    host_regs.push_back({p, bytes});
    return true;
  }

  void gemm_nt(const float* A, const cuda::F16Weight& weight, float* C, int M, int N, int K) {
    cuda::launch_narrow_f16(A, d_gemm_in.get(), static_cast<size_t>(M) * K, stream.get());
    gemm_nt_prepared(d_gemm_in.get(), weight, C, M, N, K);
  }

  // The producer has already rounded the fp32 activation to the exact fp16
  // operand the old launch_narrow_f16 pass produced.
  void gemm_nt_prepared(const __half* A, const cuda::F16Weight& weight, float* C,
                        int M, int N, int K) {
    const __half* B = weight.materialize(d_weight.get(), cap_weight, stream.get());
    if (cfg.transformer_mode == ViTTransformerMode::kExact) {
      cuda::launch_deterministic_scalar_gemm_nt(
          A, B, nullptr, C, static_cast<uint32_t>(M),
          static_cast<uint32_t>(N), static_cast<uint32_t>(K),
          DenseGemmMode::kFloat16Vae, DenseGemmBias::kNone, 0, 0,
          stream.get());
      return;
    }
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(vidfab::cuda::cublas_gemm_ex(blas, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_16F,
                              K, A, CUDA_R_16F, K, &beta, C, CUDA_R_32F, N,
                              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
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
    if (cfg.transformer_mode == ViTTransformerMode::kExact) {
      // The exact graph owns one shape-specific block arena shared by all 36
      // layers. Keep only embedding/output scratch here; retaining the shipped
      // block arena as well would waste hundreds of MiB.
      d_proj.allocate(s * static_cast<size_t>(cfg.patch_dim()));
      d_gemm_in.allocate(s * static_cast<size_t>(dim));
      d_cos.allocate(static_cast<size_t>(seq) * cfg.rope_dim);
      d_sin.allocate(static_cast<size_t>(seq) * cfg.rope_dim);
      cap_seq = seq;
      cap_batch = batch;
      rope_T = rope_H = rope_W = -1;
      return;
    }
    d_qkv.allocate(s * 3 * dim);
    // Attention is deliberately serialized by document. Replicating the
    // quadratic score buffer for every spatial tile would erase batching's
    // memory advantage; token-wise activations above remain batched.
    d_q_bf16.allocate(static_cast<size_t>(seq) * cfg.heads * hd);
    d_k_bf16.allocate(static_cast<size_t>(seq) * cfg.heads * hd);
    d_v_bf16.allocate(static_cast<size_t>(seq) * cfg.heads * hd);
    d_attn_bf16.allocate(static_cast<size_t>(seq) * cfg.heads * hd);
    d_proj.allocate(s * static_cast<size_t>(cfg.patch_dim()));
    d_ffn.allocate(s * 2 * cfg.ffn_inner);
    d_gemm_in.allocate(s * static_cast<size_t>(std::max(cfg.ffn_inner, cfg.dim)));
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
    if (seq != num_patches + cfg.num_suffix)
      throw std::runtime_error("vae: invalid RoPE sequence");
    vae::ViTRopeTables tables = vae::build_vit_rope_tables(
        static_cast<uint32_t>(T), static_cast<uint32_t>(H),
        static_cast<uint32_t>(W), static_cast<uint32_t>(cfg.num_suffix),
        static_cast<uint32_t>(cfg.rope_dim), cfg.rope_theta);
    d_cos.copy_from_host(tables.cosine.data(), tables.cosine.size(), stream);
    d_sin.copy_from_host(tables.sine.data(), tables.sine.size(), stream);
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
    cuda::launch_rmsnorm_f16(d_tokens.get(), b.norm1.get(), d_gemm_in.get(), rows, dim,
                             cfg.eps, s);
    gemm_nt_prepared(d_gemm_in.get(), b.qkv_w, d_qkv.get(), rows, 3 * dim, dim);

    // The qkv bias is applied inside the split kernel, which already reads
    // every element of d_qkv once.
    cuda::AttentionConfig attn_cfg;
    attn_cfg.seq_len = seq;
    attn_cfg.num_heads = heads;
    attn_cfg.head_dim = hd;
    const cuda::AttentionBackend attn_backend = cuda::attention_preferred_backend(attn_cfg);
    attention_ws.reserve(cuda::attention_workspace_bytes(attn_cfg, attn_backend));
    for (int doc = 0; doc < batch; ++doc) {
      const size_t row0 = static_cast<size_t>(doc) * seq;
      cuda::launch_split_qkv_norm_rope_bf16(
          d_qkv.get() + row0 * 3 * dim, b.qkv_b.get(), d_cos.get(), d_sin.get(),
          d_q_bf16.get(), d_k_bf16.get(), d_v_bf16.get(), seq, heads, hd,
          cfg.rope_dim, num_patches, cfg.eps, s);
      attention_ws.clear();
      cuda::attention_forward(blas, s, d_q_bf16.get(), d_k_bf16.get(), d_v_bf16.get(),
                              d_attn_bf16.get(), attn_cfg, attn_backend, attention_ws);
      cuda::launch_bf16_to_f16(d_attn_bf16.get(), d_gemm_in.get() + row0 * dim,
                               static_cast<size_t>(seq) * dim, s);
    }

    gemm_nt_prepared(d_gemm_in.get(), b.out_w, d_normed.get(), rows, dim, dim);
    cuda::launch_layerscale_residual(d_tokens.get(), d_normed.get(), b.out_b.get(), b.scale1.get(),
                                     rows, dim, s);

    // --- feed forward ---
    cuda::launch_rmsnorm_f16(d_tokens.get(), b.norm2.get(), d_gemm_in.get(), rows, dim,
                             cfg.eps, s);
    gemm_nt_prepared(d_gemm_in.get(), b.w1, d_ffn.get(), rows, 2 * cfg.ffn_inner, dim);
    cuda::launch_swiglu_f16(d_ffn.get(), b.w1_b.get(), d_gemm_in.get(), rows,
                            cfg.ffn_inner, s);
    gemm_nt_prepared(d_gemm_in.get(), b.w2, d_normed.get(), rows, dim, cfg.ffn_inner);
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
  CUBLAS_CHECK(vidfab::cuda::cublas_create(&d.blas));
  CUBLAS_CHECK(vidfab::cuda::cublas_set_stream(d.blas, d.stream.get()));
  // DEFAULT already refuses to drop mantissa bits for an fp32 compute type —
  // TF32 requires an explicitly TF32 math mode or compute type, neither of
  // which we ask for. PEDANTIC additionally forbids optimisations that do not
  // affect precision, and NVIDIA documents it as slower. Keep it available for
  // the correctness harness via VIDFAB_CUBLAS_PEDANTIC=1.
  const char* pedantic = std::getenv("VIDFAB_CUBLAS_PEDANTIC");
  const bool want_pedantic = pedantic != nullptr && pedantic[0] == '1';
  CUBLAS_CHECK(vidfab::cuda::cublas_set_math_mode(d.blas,
                                 want_pedantic ? CUBLAS_PEDANTIC_MATH : CUBLAS_DEFAULT_MATH));

  const int dim = config.dim;
  const int inner = config.ffn_inner;
  const int ch = config.in_channels;
  d.loaded = false;
  d.blocks.clear();
  d.exact_blocks.reset();

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

  // Page-locks the checkpoint mapping for the whole of the load below, and is
  // declared here rather than inside the uploader because the uploader is not
  // its main beneficiary. Every large tensor — the qkv, out, ff and embedding
  // weights, which are nearly all of the 4.85 GB — is read by
  // `F16Weight::load`, which copies straight out of `view.data`. From a
  // pageable mapping that copy is synchronous and staged through the driver at
  // about 4.5 GB/s; out of a registered one it is a real DMA at about 44 GB/s.
  // Destroyed after `uploader` and after the synchronise at the end of this
  // function, so nothing is still reading the mapping when it is unregistered.
  const cuda::RegisteredMapping mapping(ckpt.mapping_base(), ckpt.file_size());
  if (!mapping.registered()) {
    std::fprintf(stderr,
                 "vidfab: could not page-lock the video vae mapping; uploading via the staged "
                 "path, which is slower\n");
  }
  WeightUploader uploader(d.stream.get(), mapping);

  d.x_embed_w.load(ckpt, "decoder.x_embedder.weight", static_cast<size_t>(dim) * ch,
                   d.stream.get(), "video vae",
                   config.transformer_mode == ViTTransformerMode::kExact);
  d.x_embed_b = uploader.upload(ckpt, "decoder.x_embedder.bias", dim);
  d.register_tokens = uploader.upload(ckpt, "decoder.register_tokens",
                                      static_cast<size_t>(config.num_register) * dim);
  d.norm_out_w = uploader.upload(ckpt, "decoder.norm_out.weight", dim);
  d.norm_out_b = uploader.upload(ckpt, "decoder.norm_out.bias", dim);
  d.proj_out_w.load(ckpt, "decoder.proj_out.weight",
                    static_cast<size_t>(config.patch_dim()) * dim, d.stream.get(), "video vae",
                    config.transformer_mode == ViTTransformerMode::kExact);
  d.proj_out_b = uploader.upload(ckpt, "decoder.proj_out.bias", config.patch_dim());
  d.post_quant_w.load(ckpt, "post_quant_conv.weight", static_cast<size_t>(ch) * ch,
                      d.stream.get(), "video vae",
                      config.transformer_mode == ViTTransformerMode::kExact);
  d.post_quant_b = uploader.upload(ckpt, "post_quant_conv.bias", ch);

  if (config.transformer_mode == ViTTransformerMode::kExact) {
    vae::ViTBlockConfig exact_config;
    // Weights are shape-independent. Start with the suffix-only shape and let
    // forward_windows select/cache real ragged tile shapes without reloading.
    exact_config.sequence = static_cast<uint32_t>(config.num_suffix);
    exact_config.num_patches = 0;
    exact_config.dim = static_cast<uint32_t>(config.dim);
    exact_config.heads = static_cast<uint32_t>(config.heads);
    exact_config.head_dim = static_cast<uint32_t>(config.head_dim);
    exact_config.ffn_inner = static_cast<uint32_t>(config.ffn_inner);
    exact_config.rope_dim = static_cast<uint32_t>(config.rope_dim);
    exact_config.epsilon = config.eps;
    auto graph = std::make_unique<cuda::ExactViTBlockGraph>(
        cuda::ExactViTBlockGraph::create(exact_config,
                                         static_cast<uint32_t>(config.num_layers)));
    graph->load(ckpt);
    d.exact_blocks = std::move(graph);
  } else {
    d.blocks.resize(config.num_layers);
    for (int i = 0; i < config.num_layers; ++i) {
      const std::string p =
          "decoder.transformer_blocks." + std::to_string(i) + ".";
      BlockWeights& b = d.blocks[i];
      b.norm1 = uploader.upload(ckpt, p + "norm1.weight", dim);
      b.norm2 = uploader.upload(ckpt, p + "norm2.weight", dim);
      b.scale1 = uploader.upload(ckpt, p + "scale1", dim);
      b.scale2 = uploader.upload(ckpt, p + "scale2", dim);
      b.qkv_w.load(ckpt, p + "attn.to_qkv.weight",
                   static_cast<size_t>(3) * dim * dim, d.stream.get(),
                   "video vae");
      b.qkv_b = uploader.upload(ckpt, p + "attn.to_qkv.bias",
                                static_cast<size_t>(3) * dim);
      b.out_w.load(ckpt, p + "attn.to_out.weight",
                   static_cast<size_t>(dim) * dim, d.stream.get(),
                   "video vae");
      b.out_b = uploader.upload(ckpt, p + "attn.to_out.bias", dim);
      b.w1.load(ckpt, p + "ff.w1.weight",
                static_cast<size_t>(2) * inner * dim, d.stream.get(),
                "video vae");
      b.w1_b = uploader.upload(ckpt, p + "ff.w1.bias",
                               static_cast<size_t>(2) * inner);
      b.w2.load(ckpt, p + "ff.w2.weight",
                static_cast<size_t>(dim) * inner, d.stream.get(), "video vae");
      b.w2_b = uploader.upload(ckpt, p + "ff.w2.bias", dim);
    }
  }

  size_t max_weight = std::max({d.x_embed_w.elements(), d.proj_out_w.elements(),
                                d.post_quant_w.elements()});
  size_t total = d.x_embed_w.stored_bytes() + d.x_embed_b.nbytes() + d.register_tokens.nbytes() +
                 d.norm_out_w.nbytes() + d.norm_out_b.nbytes() + d.proj_out_w.stored_bytes() +
                 d.proj_out_b.nbytes() + d.post_quant_w.stored_bytes() + d.post_quant_b.nbytes();
  for (const BlockWeights& b : d.blocks) {
    total += b.norm1.nbytes() + b.norm2.nbytes() + b.scale1.nbytes() + b.scale2.nbytes() +
             b.qkv_w.stored_bytes() + b.qkv_b.nbytes() + b.out_w.stored_bytes() + b.out_b.nbytes() +
             b.w1.stored_bytes() + b.w1_b.nbytes() + b.w2.stored_bytes() + b.w2_b.nbytes();
    max_weight = std::max({max_weight, b.qkv_w.elements(), b.out_w.elements(), b.w1.elements(),
                           b.w2.elements()});
  }
  if (d.exact_blocks) total += d.exact_blocks->persistent_bytes();
  d.cap_weight = max_weight;
  d.d_weight.allocate(max_weight);
  d.weight_bytes = total;
  d.stream.synchronize();
  d.loaded = true;
}

void ViTDecoder::forward_window(const float* z, int T, int H, int W, std::vector<float>& out) {
  std::vector<std::vector<float>> batch_out(1);
  // Declared after `batch_out` so it destructs first. forward_windows can throw
  // after it has page-locked the slot — every VIDFAB_CUDA_CHECK in it can, and
  // a device out-of-memory really does — and without this `batch_out` would die
  // still locked, leaving an entry pointing at freed memory for ~Impl to
  // unregister later.
  HostRegistrationScope registration_scope(*this);
  const size_t slot = 0;
  forward_windows(z, 1, T, H, W, batch_out, &slot);
  // On the normal path the lock still has to go before *ownership* does: the
  // move hands the buffer to a vector this class cannot see. The guard above
  // then finds nothing left to release.
  release_host_registrations();
  out = std::move(batch_out.front());
}

void ViTDecoder::forward_windows(const float* z, int batch, int T, int H, int W,
                                 std::vector<std::vector<float>>& out, const size_t* slots) {
  Impl& d = *impl_;
  if (!d.loaded) throw std::runtime_error("vae: decoder weights not loaded");
  if (batch <= 0) throw std::runtime_error("vae: window batch must be positive");

  const ViTConfig& cfg = d.cfg;
  const int ch = cfg.in_channels;
  const int num_patches = T * H * W;
  const int seq = num_patches + cfg.num_suffix;
  const int dim = cfg.dim;
  cudaStream_t s = d.stream.get();
  if (cfg.transformer_mode == ViTTransformerMode::kExact)
    d.exact_blocks->prepare_shape(static_cast<uint32_t>(seq),
                                  static_cast<uint32_t>(num_patches));

  cuda::PhaseSpan s_prep("forward: prepare");
  d.ensure_scratch(seq, num_patches, batch);
  d.build_rope(T, H, W, seq, num_patches);
  s_prep.stop();

  // Latent arrives channel-first [C, T, H, W]; the ViT wants one token per
  // voxel, channel-last, in (t, h, w) row-major order. The transpose runs on
  // the device — it used to round-trip the same bytes back to the host.
  const size_t voxels = static_cast<size_t>(num_patches);
  {
  cuda::PhaseSpan s_embed("forward: issue embed");
  cuda::PhaseGpuSpan g_embed("forward: embed", s);
  d.d_latent.copy_from_host(z, static_cast<size_t>(batch) * ch * voxels, s);
  for (int doc = 0; doc < batch; ++doc) {
    const size_t latent0 = static_cast<size_t>(doc) * ch * voxels;
    const size_t patch0 = static_cast<size_t>(doc) * num_patches;
    const size_t token0 = static_cast<size_t>(doc) * seq;
    cuda::launch_transpose_cn_to_nc(d.d_latent.get() + latent0,
                                    d.d_patch.get() + patch0 * ch, ch,
                                    static_cast<int>(voxels), s);
    d.gemm_nt(d.d_patch.get() + patch0 * ch, d.post_quant_w,
              d.d_quantised.get() + patch0 * ch, num_patches, ch, ch);
    cuda::launch_add_bias(d.d_quantised.get() + patch0 * ch, d.post_quant_b.get(), num_patches,
                          ch, s);
    d.gemm_nt(d.d_quantised.get() + patch0 * ch, d.x_embed_w,
              d.d_tokens.get() + token0 * dim, num_patches, dim, ch);
    cuda::launch_add_bias(d.d_tokens.get() + token0 * dim, d.x_embed_b.get(), num_patches, dim, s);
    VIDFAB_CUDA_CHECK(cudaMemcpyAsync(
        d.d_tokens.get() + (token0 + num_patches) * dim, d.register_tokens.get(),
        static_cast<size_t>(cfg.num_register) * dim * sizeof(float), cudaMemcpyDeviceToDevice, s));
    VIDFAB_CUDA_CHECK(cudaMemsetAsync(
        d.d_tokens.get() + (token0 + num_patches + cfg.num_register) * dim, 0,
        static_cast<size_t>(dim) * sizeof(float), s));
  }
  }

  {
  cuda::PhaseSpan s_blocks("forward: issue blocks");
  cuda::PhaseGpuSpan g_blocks("forward: blocks", s);
  if (cfg.transformer_mode == ViTTransformerMode::kExact) {
    for (int doc = 0; doc < batch; ++doc) {
      d.exact_blocks->forward_device(
          d.d_tokens.get() + static_cast<size_t>(doc) * seq * dim,
          d.d_cos.get(), d.d_sin.get(), s);
    }
  } else {
    for (int i = 0; i < cfg.num_layers; ++i)
      d.run_block(d.blocks[i], seq, num_patches, batch);
  }
  }

  const size_t pixels = static_cast<size_t>(cfg.out_channels) * (T * cfg.patch_t) *
                        (H * cfg.patch) * (W * cfg.patch);
  cuda::PhaseSpan s_grow("forward: grow output");
  if (pixels > d.cap_pixels) {
    d.d_pixels.allocate(pixels);
    d.cap_pixels = pixels;
  }
  const int patch_dim = cfg.patch_dim();
  s_grow.stop();
  for (int doc = 0; doc < batch; ++doc) {
    const size_t token0 = static_cast<size_t>(doc) * seq;
    // Resolve the destination first: when it can be page-locked the D2H writes
    // straight into it and the staging copy below disappears entirely. The
    // registration has to be dropped *before* a resize that reallocates, while
    // the block it locks is still alive.
    // Timed: page-locking is the one substantial host cost this path adds, and
    // it is paid on the first chunk only. Left outside every span it would be
    // the single largest new cost in the stage and invisible to the profiler
    // this whole effort is steered by — including the case where it fails and
    // every tile silently reverts to being staged and copied.
    cuda::PhaseSpan s_lock("forward: page-lock output");
    std::vector<float>& dst = out[slots[static_cast<size_t>(doc)]];
    if (pixels > dst.capacity()) d.unregister_host(dst.data());
    dst.resize(pixels);
    const bool landed = d.ensure_registered(dst.data(), dst.capacity() * sizeof(float));
    if (!landed && d.pinned_out.size() < pixels) {
      // Allocated only when the direct landing is unavailable — a machine short
      // of lockable pages, or one whose GPU is already exhausted. Allocating it
      // up front instead would hold 22 MiB of pinned host memory for the life of
      // the decoder on every run that never needs it.
      d.pinned_out.allocate(pixels);
    }
    float* host_dst = landed ? dst.data() : d.pinned_out.get();
    s_lock.stop();
    // The suffix rows between documents mean the final projection is issued
    // per document. It runs once per decode, unlike the 216 projections in
    // the transformer body, and preserves the exact singleton arithmetic.
    {
      cuda::PhaseSpan s_proj("forward: issue project");
      cuda::PhaseGpuSpan g_proj("forward: project + D2H", s);
      if (cfg.transformer_mode == ViTTransformerMode::kExact) {
        cuda::launch_layernorm(d.d_tokens.get() + token0 * dim, d.norm_out_w.get(),
                               d.norm_out_b.get(), d.d_normed.get(), num_patches, dim, cfg.eps, s);
        d.gemm_nt(d.d_normed.get(), d.proj_out_w, d.d_proj.get(), num_patches, patch_dim, dim);
      } else {
        cuda::launch_layernorm_f16(d.d_tokens.get() + token0 * dim, d.norm_out_w.get(),
                                   d.norm_out_b.get(), d.d_gemm_in.get(), num_patches, dim,
                                   cfg.eps, s);
        d.gemm_nt_prepared(d.d_gemm_in.get(), d.proj_out_w, d.d_proj.get(), num_patches,
                           patch_dim, dim);
      }
      cuda::launch_add_bias(d.d_proj.get(), d.proj_out_b.get(), num_patches, patch_dim, s);
      cuda::launch_depth_to_space(d.d_proj.get(), d.d_pixels.get(), T, H, W, cfg.out_channels,
                                  cfg.patch_t, cfg.patch, s);
      d.d_pixels.copy_to_host(host_dst, pixels, s);
    }
    cuda::PhaseSpan s_sync("forward: sync wait");
    d.stream.synchronize();
    s_sync.stop();
    // Every event pair recorded above is now readable, and the stream is idle,
    // so draining them here costs nothing and adds no synchronise of its own.
    cuda::PhaseProfiler::instance().flush_gpu();
    cuda::PhaseProfiler::instance().sample_memory();
    if (!landed) {
      cuda::PhaseSpan s_copy("forward: output copy");
      std::memcpy(dst.data(), d.pinned_out.get(), pixels * sizeof(float));
      s_copy.stop();
    }
  }
}

void ViTDecoder::release_host_registrations() { impl_->release_host_regs(); }

void ViTDecoder::denormalize_latents(
    const float* normalized, int channels, uint64_t voxels,
    const std::vector<float>& mean, const std::vector<float>& std_dev,
    std::vector<float>& output) {
  if (!normalized || channels <= 0 || mean.size() != size_t(channels) ||
      std_dev.size() != size_t(channels)) {
    throw std::invalid_argument("vae: invalid latent denormalization input");
  }
  output.resize(static_cast<size_t>(channels) * voxels);
  if (impl_->cfg.transformer_mode == ViTTransformerMode::kExact) {
    if (voxels > impl_->cap_denorm_voxels) {
      impl_->d_denorm_input.allocate(static_cast<size_t>(channels) * voxels);
      impl_->d_denorm_output.allocate(static_cast<size_t>(channels) * voxels);
      impl_->d_denorm_mean.allocate(channels);
      impl_->d_denorm_std.allocate(channels);
      impl_->cap_denorm_voxels = voxels;
    }
    impl_->d_denorm_input.copy_from_host(
        normalized, static_cast<size_t>(channels) * voxels, impl_->stream.get());
    impl_->d_denorm_mean.copy_from_host(mean.data(), mean.size(),
                                        impl_->stream.get());
    impl_->d_denorm_std.copy_from_host(std_dev.data(), std_dev.size(),
                                       impl_->stream.get());
    cuda::launch_latent_denorm(
        impl_->d_denorm_input.get(), impl_->d_denorm_mean.get(),
        impl_->d_denorm_std.get(), impl_->d_denorm_output.get(), channels,
        static_cast<uint32_t>(voxels), impl_->stream.get());
    impl_->d_denorm_output.copy_to_host(
        output.data(), output.size(), impl_->stream.get());
    impl_->stream.synchronize();
    return;
  }
  for (int channel = 0; channel < channels; ++channel) {
    const size_t base = static_cast<size_t>(channel) * voxels;
    for (uint64_t i = 0; i < voxels; ++i)
      output[base + i] = normalized[base + i] * std_dev[channel] + mean[channel];
  }
}

}  // namespace vidfab::vae
