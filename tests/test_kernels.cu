// GPU kernel tests against independent CPU references.
//
// The failure mode these exist to catch is silent: a wrong QKV de-interleave,
// a wrong depth-to-space ordering or a transposed GEMM all produce finite,
// plausibly-scaled output that only looks wrong once it reaches the eye. Each
// kernel is therefore checked against a straightforward CPU implementation
// written from the spec rather than from the kernel.

#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

#include <algorithm>

#include "harness.h"
#include "vidfab/cuda/device.h"
#include "vidfab/cuda/gemm.cuh"
#include "vidfab/cuda/vae_kernels.cuh"
#include "vidfab/dtype.h"

namespace {

// Deterministic pseudo-random fill; avoids <random> so results are identical
// across standard library versions.
std::vector<float> make_data(size_t n, uint32_t seed, float scale = 1.0f) {
  std::vector<float> v(n);
  uint32_t s = seed | 1u;
  for (size_t i = 0; i < n; ++i) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    v[i] = ((static_cast<float>(s & 0xFFFFFFu) / 16777216.0f) - 0.5f) * 2.0f * scale;
  }
  return v;
}

using vidfab::cuda::DeviceBuffer;

DeviceBuffer<float> to_device(const std::vector<float>& host) {
  DeviceBuffer<float> d(host.size());
  d.copy_from_host(host.data(), host.size());
  return d;
}

std::vector<float> to_host(const DeviceBuffer<float>& d) {
  std::vector<float> h(d.size());
  d.copy_to_host(h.data(), h.size());
  return h;
}

// --- CPU references ---------------------------------------------------------

std::vector<float> cpu_rmsnorm(const std::vector<float>& x, const std::vector<float>& w, int rows,
                               int dim, float eps) {
  std::vector<float> out(x.size());
  for (int r = 0; r < rows; ++r) {
    double sum_sq = 0.0;
    for (int i = 0; i < dim; ++i) {
      const double v = x[static_cast<size_t>(r) * dim + i];
      sum_sq += v * v;
    }
    const double inv = 1.0 / std::sqrt(sum_sq / dim + eps);
    for (int i = 0; i < dim; ++i) {
      out[static_cast<size_t>(r) * dim + i] =
          static_cast<float>(x[static_cast<size_t>(r) * dim + i] * inv * w[i]);
    }
  }
  return out;
}

std::vector<float> cpu_layernorm(const std::vector<float>& x, const std::vector<float>& w,
                                 const std::vector<float>& b, int rows, int dim, float eps) {
  std::vector<float> out(x.size());
  for (int r = 0; r < rows; ++r) {
    double mean = 0.0;
    for (int i = 0; i < dim; ++i) mean += x[static_cast<size_t>(r) * dim + i];
    mean /= dim;
    double var = 0.0;
    for (int i = 0; i < dim; ++i) {
      const double d = x[static_cast<size_t>(r) * dim + i] - mean;
      var += d * d;
    }
    var /= dim;
    const double inv = 1.0 / std::sqrt(var + eps);
    for (int i = 0; i < dim; ++i) {
      out[static_cast<size_t>(r) * dim + i] =
          static_cast<float>((x[static_cast<size_t>(r) * dim + i] - mean) * inv * w[i] + b[i]);
    }
  }
  return out;
}

std::vector<float> cpu_matmul_nt(const std::vector<float>& A, const std::vector<float>& B, int M,
                                 int N, int K) {
  std::vector<float> C(static_cast<size_t>(M) * N);
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      double acc = 0.0;
      for (int k = 0; k < K; ++k) {
        acc += static_cast<double>(A[static_cast<size_t>(m) * K + k]) *
               B[static_cast<size_t>(n) * K + k];
      }
      C[static_cast<size_t>(m) * N + n] = static_cast<float>(acc);
    }
  }
  return C;
}

std::vector<float> cpu_matmul_nn(const std::vector<float>& A, const std::vector<float>& B, int M,
                                 int N, int K) {
  std::vector<float> C(static_cast<size_t>(M) * N);
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      double acc = 0.0;
      for (int k = 0; k < K; ++k) {
        acc += static_cast<double>(A[static_cast<size_t>(m) * K + k]) *
               B[static_cast<size_t>(k) * N + n];
      }
      C[static_cast<size_t>(m) * N + n] = static_cast<float>(acc);
    }
  }
  return C;
}

// --- tests ------------------------------------------------------------------

void test_norms() {
  TEST("norms");
  const int rows = 37;
  const int dim = 2048;
  const float eps = 1e-5f;

  const std::vector<float> x = make_data(static_cast<size_t>(rows) * dim, 11u, 3.0f);
  const std::vector<float> w = make_data(dim, 22u, 1.0f);
  const std::vector<float> b = make_data(dim, 33u, 0.5f);

  DeviceBuffer<float> dx = to_device(x);
  DeviceBuffer<float> dw = to_device(w);
  DeviceBuffer<float> db = to_device(b);
  DeviceBuffer<float> dout(x.size());
  DeviceBuffer<uint16_t> dlegacy_f16(x.size());
  DeviceBuffer<uint16_t> dfused_f16(x.size());

  vidfab::cuda::launch_rmsnorm(dx.get(), dw.get(), dout.get(), rows, dim, eps, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK_CLOSE(cpu_rmsnorm(x, w, rows, dim, eps), to_host(dout), 1e-4, "rmsnorm");
  vidfab::cuda::launch_narrow_f16(dout.get(), dlegacy_f16.get(), x.size(), nullptr);
  vidfab::cuda::launch_rmsnorm_f16(dx.get(), dw.get(), dfused_f16.get(), rows, dim, eps,
                                   nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<uint16_t> legacy_f16(x.size()), fused_f16(x.size());
  dlegacy_f16.copy_to_host(legacy_f16.data(), legacy_f16.size());
  dfused_f16.copy_to_host(fused_f16.data(), fused_f16.size());
  CHECK(legacy_f16 == fused_f16);

  vidfab::cuda::launch_layernorm(dx.get(), dw.get(), db.get(), dout.get(), rows, dim, eps,
                                 nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK_CLOSE(cpu_layernorm(x, w, b, rows, dim, eps), to_host(dout), 1e-4, "layernorm");
  vidfab::cuda::launch_narrow_f16(dout.get(), dlegacy_f16.get(), x.size(), nullptr);
  vidfab::cuda::launch_layernorm_f16(dx.get(), dw.get(), db.get(), dfused_f16.get(), rows, dim,
                                     eps, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  dlegacy_f16.copy_to_host(legacy_f16.data(), legacy_f16.size());
  dfused_f16.copy_to_host(fused_f16.data(), fused_f16.size());
  CHECK(legacy_f16 == fused_f16);
}

void test_gemm() {
  TEST("gemm");
  cublasHandle_t h = nullptr;
  VIDFAB_CUBLAS_CHECK(vidfab::cuda::cublas_create(&h));
  VIDFAB_CUBLAS_CHECK(vidfab::cuda::cublas_set_math_mode(h, CUBLAS_PEDANTIC_MATH));

  // Non-square and mutually distinct dimensions, so a transposed wrapper
  // cannot accidentally pass.
  const int M = 29;
  const int N = 53;
  const int K = 41;

  const std::vector<float> A = make_data(static_cast<size_t>(M) * K, 101u);
  const std::vector<float> Bnt = make_data(static_cast<size_t>(N) * K, 202u);
  const std::vector<float> Bnn = make_data(static_cast<size_t>(K) * N, 303u);

  DeviceBuffer<float> dA = to_device(A);
  DeviceBuffer<float> dBnt = to_device(Bnt);
  DeviceBuffer<float> dBnn = to_device(Bnn);
  DeviceBuffer<float> dC(static_cast<size_t>(M) * N);

  vidfab::cuda::gemm_nt(h, dA.get(), dBnt.get(), dC.get(), M, N, K);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK_CLOSE(cpu_matmul_nt(A, Bnt, M, N, K), to_host(dC), 1e-3, "gemm_nt");

  vidfab::cuda::gemm_nn(h, dA.get(), dBnn.get(), dC.get(), M, N, K);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK_CLOSE(cpu_matmul_nn(A, Bnn, M, N, K), to_host(dC), 1e-3, "gemm_nn");

  // Batched variants with distinct per-batch data.
  const int batch = 3;
  std::vector<float> Ab(static_cast<size_t>(batch) * M * K);
  std::vector<float> Bb(static_cast<size_t>(batch) * N * K);
  for (int b = 0; b < batch; ++b) {
    const std::vector<float> a = make_data(static_cast<size_t>(M) * K, 401u + b);
    const std::vector<float> bb = make_data(static_cast<size_t>(N) * K, 501u + b);
    std::copy(a.begin(), a.end(), Ab.begin() + static_cast<long long>(b) * M * K);
    std::copy(bb.begin(), bb.end(), Bb.begin() + static_cast<long long>(b) * N * K);
  }
  DeviceBuffer<float> dAb = to_device(Ab);
  DeviceBuffer<float> dBb = to_device(Bb);
  DeviceBuffer<float> dCb(static_cast<size_t>(batch) * M * N);

  vidfab::cuda::gemm_nt_batched(h, dAb.get(), dBb.get(), dCb.get(), M, N, K, batch,
                                static_cast<long long>(M) * K, static_cast<long long>(N) * K,
                                static_cast<long long>(M) * N);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  const std::vector<float> got = to_host(dCb);
  for (int b = 0; b < batch; ++b) {
    const std::vector<float> a(Ab.begin() + static_cast<long long>(b) * M * K,
                               Ab.begin() + static_cast<long long>(b + 1) * M * K);
    const std::vector<float> bb(Bb.begin() + static_cast<long long>(b) * N * K,
                                Bb.begin() + static_cast<long long>(b + 1) * N * K);
    const std::vector<float> want = cpu_matmul_nt(a, bb, M, N, K);
    const std::vector<float> slice(got.begin() + static_cast<long long>(b) * M * N,
                                   got.begin() + static_cast<long long>(b + 1) * M * N);
    CHECK_CLOSE(want, slice, 1e-3, ("gemm_nt_batched b" + std::to_string(b)).c_str());
  }
  vidfab::cuda::cublas_destroy(h);
}

void test_swiglu() {
  TEST("swiglu");
  const int rows = 17;
  const int inner = 512;
  const std::vector<float> in = make_data(static_cast<size_t>(rows) * 2 * inner, 77u, 4.0f);

  std::vector<float> want(static_cast<size_t>(rows) * inner);
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < inner; ++c) {
      // gate is the FIRST half; value the second.
      const float gate = in[static_cast<size_t>(r) * 2 * inner + c];
      const float value = in[static_cast<size_t>(r) * 2 * inner + inner + c];
      want[static_cast<size_t>(r) * inner + c] = (gate / (1.0f + std::exp(-gate))) * value;
    }
  }

  DeviceBuffer<float> din = to_device(in);
  DeviceBuffer<float> dout(want.size());
  DeviceBuffer<uint16_t> dlegacy_f16(want.size());
  DeviceBuffer<uint16_t> dfused_f16(want.size());
  vidfab::cuda::launch_swiglu(din.get(), nullptr, dout.get(), rows, inner, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK_CLOSE(want, to_host(dout), 1e-5, "swiglu");

  // With the w1 bias folded in, the result must match applying the bias first.
  const std::vector<float> bias = make_data(static_cast<size_t>(2) * inner, 78u, 1.5f);
  std::vector<float> want_biased(static_cast<size_t>(rows) * inner);
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < inner; ++c) {
      const float gate = in[static_cast<size_t>(r) * 2 * inner + c] + bias[c];
      const float value = in[static_cast<size_t>(r) * 2 * inner + inner + c] + bias[inner + c];
      want_biased[static_cast<size_t>(r) * inner + c] =
          (gate / (1.0f + std::exp(-gate))) * value;
    }
  }
  DeviceBuffer<float> dbias = to_device(bias);
  vidfab::cuda::launch_swiglu(din.get(), dbias.get(), dout.get(), rows, inner, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK_CLOSE(want_biased, to_host(dout), 1e-5, "swiglu with fused bias");
  vidfab::cuda::launch_narrow_f16(dout.get(), dlegacy_f16.get(), want.size(), nullptr);
  vidfab::cuda::launch_swiglu_f16(din.get(), dbias.get(), dfused_f16.get(), rows, inner, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<uint16_t> legacy_f16(want.size()), fused_f16(want.size());
  dlegacy_f16.copy_to_host(legacy_f16.data(), legacy_f16.size());
  dfused_f16.copy_to_host(fused_f16.data(), fused_f16.size());
  CHECK(legacy_f16 == fused_f16);
}

void test_softmax() {
  TEST("softmax");
  const int rows = 13;
  const int cols = 257;
  const float scale = 0.125f;
  const std::vector<float> x = make_data(static_cast<size_t>(rows) * cols, 91u, 6.0f);

  std::vector<float> want(x.size());
  for (int r = 0; r < rows; ++r) {
    double m = -1e30;
    for (int c = 0; c < cols; ++c) m = std::max(m, static_cast<double>(x[r * cols + c]) * scale);
    double sum = 0.0;
    for (int c = 0; c < cols; ++c) sum += std::exp(x[r * cols + c] * scale - m);
    for (int c = 0; c < cols; ++c) {
      want[r * cols + c] = static_cast<float>(std::exp(x[r * cols + c] * scale - m) / sum);
    }
  }

  DeviceBuffer<float> d = to_device(x);
  vidfab::cuda::launch_softmax_rows(d.get(), rows, cols, scale, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  const std::vector<float> got = to_host(d);
  CHECK_CLOSE(want, got, 1e-6, "softmax values");

  // Every row must sum to one.
  for (int r = 0; r < rows; ++r) {
    double sum = 0.0;
    for (int c = 0; c < cols; ++c) sum += got[r * cols + c];
    CHECK_MSG(std::fabs(sum - 1.0) <= 1e-5, "softmax row %d sums to %.9f", r, sum);
  }
}

// The ordering test that matters: index = c*1024 + pt*256 + ph*16 + pw, with
// c outermost. A PixelShuffle-style (c, ph, pw) or a (pt, ph, pw, c) ordering
// both produce structured-looking but wrong images.
void test_depth_to_space() {
  TEST("depth_to_space");
  const int T = 2;
  const int H = 3;
  const int W = 2;
  const int channels = 3;
  const int patch_t = 4;
  const int patch = 16;
  const int patch_dim = channels * patch_t * patch * patch;
  const int tokens = T * H * W;

  const std::vector<float> in = make_data(static_cast<size_t>(tokens) * patch_dim, 555u);

  const int out_T = T * patch_t;
  const int out_H = H * patch;
  const int out_W = W * patch;
  std::vector<float> want(static_cast<size_t>(channels) * out_T * out_H * out_W);
  for (int c = 0; c < channels; ++c) {
    for (int ot = 0; ot < out_T; ++ot) {
      for (int oh = 0; oh < out_H; ++oh) {
        for (int ow = 0; ow < out_W; ++ow) {
          const int t = ot / patch_t;
          const int pt = ot % patch_t;
          const int hh = oh / patch;
          const int ph = oh % patch;
          const int ww = ow / patch;
          const int pw = ow % patch;
          const size_t token = (static_cast<size_t>(t) * H + hh) * W + ww;
          const int offset = ((c * patch_t + pt) * patch + ph) * patch + pw;
          const size_t dst =
              ((static_cast<size_t>(c) * out_T + ot) * out_H + oh) * out_W + ow;
          want[dst] = in[token * patch_dim + offset];
        }
      }
    }
  }

  DeviceBuffer<float> din = to_device(in);
  DeviceBuffer<float> dout(want.size());
  vidfab::cuda::launch_depth_to_space(din.get(), dout.get(), T, H, W, channels, patch_t, patch,
                                      nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK_CLOSE(want, to_host(dout), 0.0, "depth_to_space");

  const std::vector<float> bias = make_data(patch_dim, 556u, 0.25f);
  DeviceBuffer<float> dbias = to_device(bias);
  DeviceBuffer<float> dlegacy_tokens(in.size());
  DeviceBuffer<float> dlegacy_out(want.size());
  DeviceBuffer<float> dfused_out(want.size());
  VIDFAB_CUDA_CHECK(cudaMemcpy(dlegacy_tokens.get(), din.get(), in.size() * sizeof(float),
                               cudaMemcpyDeviceToDevice));
  vidfab::cuda::launch_add_bias(dlegacy_tokens.get(), dbias.get(), tokens, patch_dim, nullptr);
  vidfab::cuda::launch_depth_to_space(dlegacy_tokens.get(), dlegacy_out.get(), T, H, W,
                                      channels, patch_t, patch, nullptr);
  vidfab::cuda::launch_depth_to_space_bias(din.get(), dbias.get(), dfused_out.get(), T, H, W,
                                           channels, patch_t, patch, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  const std::vector<float> legacy = to_host(dlegacy_out);
  const std::vector<float> fused = to_host(dfused_out);
  CHECK(std::memcmp(legacy.data(), fused.data(), legacy.size() * sizeof(float)) == 0);
}

// The other silent-failure candidate: QKV is interleaved per head as
// r = h*192 + s, and RoPE uses GPT-NeoX half-split pairing (d, d+24) over the
// first 48 of 64 head dims, with suffix tokens left unrotated.
void test_qkv_rope() {
  TEST("qkv_norm_rope");
  const int heads = 4;
  const int head_dim = 64;
  const int rope_dim = 48;
  const int num_patches = 9;
  const int seq = num_patches + 5;
  const float eps = 1e-5f;

  const std::vector<float> qkv =
      make_data(static_cast<size_t>(seq) * heads * 3 * head_dim, 909u, 2.0f);
  const std::vector<float> cos_t = make_data(static_cast<size_t>(seq) * rope_dim, 31u, 1.0f);
  const std::vector<float> sin_t = make_data(static_cast<size_t>(seq) * rope_dim, 47u, 1.0f);

  const size_t per = static_cast<size_t>(heads) * seq * head_dim;
  std::vector<float> want_q(per);
  std::vector<float> want_k(per);
  std::vector<float> want_v(per);

  const int triple = 3 * head_dim;
  const int half = rope_dim / 2;
  for (int t = 0; t < seq; ++t) {
    for (int h = 0; h < heads; ++h) {
      const size_t row = static_cast<size_t>(t) * heads * triple + static_cast<size_t>(h) * triple;
      const size_t out = (static_cast<size_t>(h) * seq + t) * head_dim;

      for (int d = 0; d < head_dim; ++d) want_v[out + d] = qkv[row + 2 * head_dim + d];

      for (int which = 0; which < 2; ++which) {
        const size_t src = row + static_cast<size_t>(which) * head_dim;
        std::vector<float>& dst = (which == 0) ? want_q : want_k;

        double sum_sq = 0.0;
        for (int d = 0; d < head_dim; ++d) {
          const double v = qkv[src + d];
          sum_sq += v * v;
        }
        const double inv = 1.0 / std::sqrt(sum_sq / head_dim + eps);

        std::vector<float> normed(head_dim);
        for (int d = 0; d < head_dim; ++d) {
          normed[d] = static_cast<float>(qkv[src + d] * inv);
        }
        for (int d = 0; d < head_dim; ++d) {
          if (t >= num_patches || d >= rope_dim) {
            dst[out + d] = normed[d];
            continue;
          }
          const float c = cos_t[static_cast<size_t>(t) * rope_dim + d];
          const float s = sin_t[static_cast<size_t>(t) * rope_dim + d];
          const float partner = (d < half) ? -normed[d + half] : normed[d - half];
          dst[out + d] = normed[d] * c + partner * s;
        }
      }
    }
  }

  DeviceBuffer<float> dqkv = to_device(qkv);
  DeviceBuffer<float> dcos = to_device(cos_t);
  DeviceBuffer<float> dsin = to_device(sin_t);
  DeviceBuffer<float> dq(per);
  DeviceBuffer<float> dk(per);
  DeviceBuffer<float> dv(per);

  vidfab::cuda::launch_split_qkv_norm_rope(dqkv.get(), nullptr, dcos.get(), dsin.get(), dq.get(),
                                           dk.get(), dv.get(), seq, heads, head_dim, rope_dim,
                                           num_patches, eps, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK_CLOSE(want_q, to_host(dq), 1e-5, "rope q");
  CHECK_CLOSE(want_k, to_host(dk), 1e-5, "rope k");
  CHECK_CLOSE(want_v, to_host(dv), 0.0, "v passthrough");

  // With the to_qkv bias fused in, the result must match adding it beforehand.
  const std::vector<float> qkv_bias =
      make_data(static_cast<size_t>(heads) * 3 * head_dim, 313u, 0.7f);
  std::vector<float> qkv_biased = qkv;
  for (int t = 0; t < seq; ++t) {
    for (size_t j = 0; j < qkv_bias.size(); ++j) {
      qkv_biased[static_cast<size_t>(t) * heads * triple + j] += qkv_bias[j];
    }
  }
  DeviceBuffer<float> dqkvb = to_device(qkv_biased);
  DeviceBuffer<float> dbias = to_device(qkv_bias);
  DeviceBuffer<float> dq_ref(per), dk_ref(per), dv_ref(per);
  DeviceBuffer<float> dq_fused(per), dk_fused(per), dv_fused(per);

  vidfab::cuda::launch_split_qkv_norm_rope(dqkvb.get(), nullptr, dcos.get(), dsin.get(),
                                           dq_ref.get(), dk_ref.get(), dv_ref.get(), seq, heads,
                                           head_dim, rope_dim, num_patches, eps, nullptr);
  vidfab::cuda::launch_split_qkv_norm_rope(dqkv.get(), dbias.get(), dcos.get(), dsin.get(),
                                           dq_fused.get(), dk_fused.get(), dv_fused.get(), seq,
                                           heads, head_dim, rope_dim, num_patches, eps, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK_CLOSE(to_host(dq_ref), to_host(dq_fused), 1e-6, "fused qkv bias q");
  CHECK_CLOSE(to_host(dk_ref), to_host(dk_fused), 1e-6, "fused qkv bias k");
  CHECK_CLOSE(to_host(dv_ref), to_host(dv_fused), 0.0, "fused qkv bias v");

  // The shipped decoder writes token-major BF16 directly. It must retain the
  // exact old rounding point: fp32 split/norm/RoPE, then layout+narrow.
  DeviceBuffer<uint16_t> dq_legacy(per), dk_legacy(per), dv_legacy(per);
  DeviceBuffer<uint16_t> dq_direct(per), dk_direct(per), dv_direct(per);
  vidfab::cuda::launch_heads_to_tokens_bf16(
      dq_fused.get(), reinterpret_cast<__nv_bfloat16*>(dq_legacy.get()), seq, heads,
      head_dim, nullptr);
  vidfab::cuda::launch_heads_to_tokens_bf16(
      dk_fused.get(), reinterpret_cast<__nv_bfloat16*>(dk_legacy.get()), seq, heads,
      head_dim, nullptr);
  vidfab::cuda::launch_heads_to_tokens_bf16(
      dv_fused.get(), reinterpret_cast<__nv_bfloat16*>(dv_legacy.get()), seq, heads,
      head_dim, nullptr);
  vidfab::cuda::launch_split_qkv_norm_rope_bf16(
      dqkv.get(), dbias.get(), dcos.get(), dsin.get(),
      reinterpret_cast<__nv_bfloat16*>(dq_direct.get()),
      reinterpret_cast<__nv_bfloat16*>(dk_direct.get()),
      reinterpret_cast<__nv_bfloat16*>(dv_direct.get()), seq, heads, head_dim,
      rope_dim, num_patches, eps, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<uint16_t> q_legacy(per), k_legacy(per), v_legacy(per);
  std::vector<uint16_t> q_direct(per), k_direct(per), v_direct(per);
  dq_legacy.copy_to_host(q_legacy.data(), per);
  dk_legacy.copy_to_host(k_legacy.data(), per);
  dv_legacy.copy_to_host(v_legacy.data(), per);
  dq_direct.copy_to_host(q_direct.data(), per);
  dk_direct.copy_to_host(k_direct.data(), per);
  dv_direct.copy_to_host(v_direct.data(), per);
  CHECK(q_legacy == q_direct);
  CHECK(k_legacy == k_direct);
  CHECK(v_legacy == v_direct);
}

// The AV GEMM writes token-major directly by using ldc = heads*head_dim and a
// per-head column offset. This is the only wrapper whose layout is not
// otherwise exercised, and a mistake here yields plausible wrong numbers, so it
// is tested at the shape production actually uses.
void test_gemm_scatter() {
  TEST("gemm_nn_batched_ld");
  cublasHandle_t h = nullptr;
  VIDFAB_CUBLAS_CHECK(vidfab::cuda::cublas_create(&h));

  const int seq = 1797;  // production: 7*16*16 patches + 5 suffix tokens
  const int heads = 32;
  const int head_dim = 64;

  // P[head][seq][seq] @ V[head][seq][head_dim] -> out[seq][heads*head_dim]
  // Small values keep the 1797-term dot products well conditioned.
  const std::vector<float> P =
      make_data(static_cast<size_t>(heads) * seq * seq, 61u, 0.05f);
  const std::vector<float> V =
      make_data(static_cast<size_t>(heads) * seq * head_dim, 62u, 1.0f);

  DeviceBuffer<float> dP = to_device(P);
  DeviceBuffer<float> dV = to_device(V);
  DeviceBuffer<float> dout(static_cast<size_t>(seq) * heads * head_dim);
  dout.zero();

  vidfab::cuda::gemm_nn_batched_ld(h, dP.get(), dV.get(), dout.get(), seq, head_dim, seq, heads,
                                   static_cast<long long>(seq) * seq,
                                   static_cast<long long>(seq) * head_dim,
                                   /*strideC=*/head_dim, /*ldc=*/heads * head_dim);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  const std::vector<float> got = to_host(dout);

  // Spot-check a scattering of (token, head, dim) triples against a CPU dot
  // product. Checking all 3.7 M outputs against an O(seq) reference would take
  // minutes; these positions cover the first/last head and token extremes.
  const int tokens[] = {0, 1, 897, 1795, 1796};
  const int hs[] = {0, 1, 17, 31};
  const int ds[] = {0, 33, 63};
  int checked = 0;
  double worst = 0.0;
  for (int t : tokens) {
    for (int hh : hs) {
      for (int dd : ds) {
        double acc = 0.0;
        for (int k = 0; k < seq; ++k) {
          acc += static_cast<double>(P[(static_cast<size_t>(hh) * seq + t) * seq + k]) *
                 V[(static_cast<size_t>(hh) * seq + k) * head_dim + dd];
        }
        const double actual = got[(static_cast<size_t>(t) * heads + hh) * head_dim + dd];
        worst = std::max(worst, std::fabs(acc - actual));
        ++checked;
      }
    }
  }
  // 1797-term fp32 dot products; 2e-3 is comfortably inside accumulation noise.
  CHECK_MSG(worst <= 2e-3, "gemm_nn_batched_ld: worst abs err %.3e over %d probes", worst,
            checked);

  // Full coverage without an O(seq) CPU reference: run the same multiply
  // through the plain batched wrapper into a contiguous [H][S][D] buffer,
  // permute on the host, and compare every element. The two calls may select
  // different cuBLAS kernels so this is not bit-exact, but a layout error is
  // O(result magnitude), not O(1e-4).
  DeviceBuffer<float> dplain(static_cast<size_t>(heads) * seq * head_dim);
  vidfab::cuda::gemm_nn_batched(h, dP.get(), dV.get(), dplain.get(), seq, head_dim, seq, heads,
                                static_cast<long long>(seq) * seq,
                                static_cast<long long>(seq) * head_dim,
                                static_cast<long long>(seq) * head_dim);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  const std::vector<float> plain = to_host(dplain);

  std::vector<float> permuted(plain.size());
  for (int t = 0; t < seq; ++t) {
    for (int hh = 0; hh < heads; ++hh) {
      for (int dd = 0; dd < head_dim; ++dd) {
        permuted[(static_cast<size_t>(t) * heads + hh) * head_dim + dd] =
            plain[(static_cast<size_t>(hh) * seq + t) * head_dim + dd];
      }
    }
  }
  CHECK_CLOSE(permuted, got, 1e-4, "gemm_nn_batched_ld full scatter layout");

  vidfab::cuda::cublas_destroy(h);
}

// The weight path now widens fp16 on the device instead of on the host. That
// changed the input to every parameter in the network, so the two conversions
// are compared directly rather than assumed equivalent.
void test_widen_f16() {
  TEST("widen_f16");
  // Cover every fp16 bit pattern: normals, subnormals, both zeros, and the
  // boundaries. Infinities and NaNs are excluded because the host path emits a
  // signalling NaN payload where the hardware quiets it, which is irrelevant
  // for real weights but would fail an exact comparison.
  std::vector<uint16_t> patterns;
  for (uint32_t bits = 0; bits <= 0xFFFFu; ++bits) {
    const uint16_t exp = static_cast<uint16_t>((bits >> 10) & 0x1Fu);
    if (exp == 0x1F) continue;  // inf / nan
    patterns.push_back(static_cast<uint16_t>(bits));
  }

  // Host reference: the same routine the loader used before.
  std::vector<float> want(patterns.size());
  for (size_t i = 0; i < patterns.size(); ++i) {
    want[i] = vidfab::f16_to_f32(patterns[i]);
  }

  DeviceBuffer<uint16_t> draw(patterns.size());
  draw.copy_from_host(patterns.data(), patterns.size());
  DeviceBuffer<float> dout(patterns.size());
  vidfab::cuda::launch_widen_f16(draw.get(), dout.get(), patterns.size(), nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  const std::vector<float> got = to_host(dout);

  // Exact equality: these are lossless widenings, not approximations.
  size_t mismatches = 0;
  size_t first = 0;
  for (size_t i = 0; i < want.size(); ++i) {
    if (std::memcmp(&want[i], &got[i], sizeof(float)) != 0) {
      if (mismatches == 0) first = i;
      ++mismatches;
    }
  }
  CHECK_MSG(mismatches == 0, "widen_f16: %zu of %zu patterns differ; first at 0x%04X (%.9g vs %.9g)",
            mismatches, want.size(), patterns[first], want[first], got[first]);
  if (mismatches == 0) {
    std::printf("  all %zu finite fp16 bit patterns widen identically\n", patterns.size());
  }
}

void test_narrow_f16() {
  TEST("narrow_f16");
  std::vector<float> input = make_data(65537, 0x51a7u, 64.0f);
  input.insert(input.end(), {0.0f, -0.0f, 65504.0f, -65504.0f, 0x1p-24f, -0x1p-24f});

  std::vector<uint16_t> want(input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    const __half h = __float2half_rn(input[i]);
    std::memcpy(&want[i], &h, sizeof(uint16_t));
  }
  DeviceBuffer<float> src = to_device(input);
  DeviceBuffer<uint16_t> dst(input.size());
  vidfab::cuda::launch_narrow_f16(src.get(), dst.get(), input.size(), nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<uint16_t> got(input.size());
  dst.copy_to_host(got.data(), got.size());
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());

  size_t mismatches = 0;
  for (size_t i = 0; i < want.size(); ++i) mismatches += want[i] != got[i];
  CHECK_MSG(mismatches == 0, "narrow_f16: %zu of %zu round-to-nearest-even results differ",
            mismatches, want.size());
}

void test_bf16_to_f16() {
  TEST("bf16_to_f16");
  std::vector<uint16_t> packed_input;
  for (uint32_t bits = 0; bits <= 0xFFFFu; ++bits) {
    if ((bits & 0x7F80u) == 0x7F80u) continue;  // exclude inf/nan payload details
    packed_input.push_back(static_cast<uint16_t>(bits));
  }
  CHECK(packed_input.size() % 8 == 0);

  auto check_conversion = [](const std::vector<uint16_t>& input) {
    std::vector<uint16_t> want(input.size());
    for (size_t i = 0; i < input.size(); ++i)
      want[i] = vidfab::f32_to_f16(vidfab::bf16_to_f32(input[i]));
    DeviceBuffer<uint16_t> src(input.size()), dst(input.size());
    src.copy_from_host(input.data(), input.size());
    vidfab::cuda::launch_bf16_to_f16(
        reinterpret_cast<const __nv_bfloat16*>(src.get()), dst.get(), input.size(), nullptr);
    VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<uint16_t> got(input.size());
    dst.copy_to_host(got.data(), got.size());
    return want == got;
  };

  // cudaMalloc supplies the required alignment and this count is divisible by
  // eight, so exhaustive finite BF16 coverage goes through the uint4 kernel.
  CHECK(check_conversion(packed_input));

  // A non-multiple count selects the scalar fallback, independently covering
  // values from both signs and throughout the BF16 exponent range.
  const std::vector<uint16_t> scalar_input{
      0x0000, 0x8000, 0x0001, 0x807F, 0x0080, 0x8080, 0x3F80, 0xBF80, 0x3FA0,
      0xC020, 0x3800, 0xB800, 0x477F, 0xC77F, 0x7F7F, 0xFF7F, 0x4049};
  CHECK(scalar_input.size() % 8 != 0);
  CHECK(check_conversion(scalar_input));
}

void test_heads_to_tokens_bf16() {
  TEST("heads_to_tokens_bf16");
  const int seq = 11, heads = 3, dim = 64;
  const std::vector<float> src = make_data(static_cast<size_t>(seq) * heads * dim, 0xa771u, 8.0f);
  DeviceBuffer<float> dsrc = to_device(src);
  DeviceBuffer<__nv_bfloat16> ddst(src.size());
  vidfab::cuda::launch_heads_to_tokens_bf16(dsrc.get(), ddst.get(), seq, heads, dim, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<__nv_bfloat16> got(src.size());
  ddst.copy_to_host(got.data(), got.size());
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());

  size_t mismatches = 0;
  for (int token = 0; token < seq; ++token) {
    for (int h = 0; h < heads; ++h) {
      for (int d = 0; d < dim; ++d) {
        const size_t dst = (static_cast<size_t>(token) * heads + h) * dim + d;
        const size_t from = (static_cast<size_t>(h) * seq + token) * dim + d;
        const __nv_bfloat16 want = __float2bfloat16_rn(src[from]);
        uint16_t wb = 0, gb = 0;
        std::memcpy(&wb, &want, sizeof(wb));
        std::memcpy(&gb, &got[dst], sizeof(gb));
        mismatches += wb != gb;
      }
    }
  }
  CHECK_MSG(mismatches == 0, "heads_to_tokens_bf16: %zu layout/rounding mismatches", mismatches);
}

void test_transpose() {
  TEST("transpose_cn_to_nc");
  const int channels = 24;
  const int voxels = 7 * 16 * 16;
  const std::vector<float> src = make_data(static_cast<size_t>(channels) * voxels, 71u);
  std::vector<float> want(src.size());
  for (int n = 0; n < voxels; ++n) {
    for (int c = 0; c < channels; ++c) {
      want[static_cast<size_t>(n) * channels + c] = src[static_cast<size_t>(c) * voxels + n];
    }
  }
  DeviceBuffer<float> dsrc = to_device(src);
  DeviceBuffer<float> ddst(src.size());
  vidfab::cuda::launch_transpose_cn_to_nc(dsrc.get(), ddst.get(), channels, voxels, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK_CLOSE(want, to_host(ddst), 0.0, "transpose_cn_to_nc");
}

void test_misc() {
  TEST("misc");
  const int head_dim = 8;
  (void)head_dim;

  // layerscale_residual: x += y * scale, scale broadcast over columns
  const int rows = 7;
  const int cols = 16;
  const std::vector<float> x = make_data(static_cast<size_t>(rows) * cols, 21u);
  const std::vector<float> y = make_data(static_cast<size_t>(rows) * cols, 22u);
  const std::vector<float> scale = make_data(cols, 23u);
  std::vector<float> want_res(x.size());
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      want_res[static_cast<size_t>(r) * cols + c] =
          x[static_cast<size_t>(r) * cols + c] + y[static_cast<size_t>(r) * cols + c] * scale[c];
    }
  }
  DeviceBuffer<float> dx = to_device(x);
  DeviceBuffer<float> dy = to_device(y);
  DeviceBuffer<float> ds = to_device(scale);
  vidfab::cuda::launch_layerscale_residual(dx.get(), dy.get(), nullptr, ds.get(), rows, cols,
                                           nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK_CLOSE(want_res, to_host(dx), 1e-6, "layerscale_residual");

  // Fused bias variant: x += (y + bias) * scale
  const std::vector<float> lsbias = make_data(cols, 24u);
  std::vector<float> want_fused(x.size());
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      const size_t i = static_cast<size_t>(r) * cols + c;
      want_fused[i] = x[i] + (y[i] + lsbias[c]) * scale[c];
    }
  }
  DeviceBuffer<float> dx2 = to_device(x);
  DeviceBuffer<float> dlb = to_device(lsbias);
  vidfab::cuda::launch_layerscale_residual(dx2.get(), dy.get(), dlb.get(), ds.get(), rows, cols,
                                           nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK_CLOSE(want_fused, to_host(dx2), 1e-6, "layerscale_residual with fused bias");

  // add_bias
  const std::vector<float> base = make_data(static_cast<size_t>(rows) * cols, 31u);
  const std::vector<float> bias = make_data(cols, 32u);
  std::vector<float> want_bias(base.size());
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      want_bias[static_cast<size_t>(r) * cols + c] = base[static_cast<size_t>(r) * cols + c] +
                                                     bias[c];
    }
  }
  DeviceBuffer<float> dbase = to_device(base);
  DeviceBuffer<float> dbias = to_device(bias);
  vidfab::cuda::launch_add_bias(dbase.get(), dbias.get(), rows, cols, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK_CLOSE(want_bias, to_host(dbase), 1e-6, "add_bias");

  // latent_denorm: out = z*std + mean, per channel over [C][voxels]
  const int channels = 4;
  const int voxels = 9;
  const std::vector<float> z = make_data(static_cast<size_t>(channels) * voxels, 41u);
  const std::vector<float> mean = make_data(channels, 42u);
  const std::vector<float> sd = make_data(channels, 43u);
  std::vector<float> want_dn(z.size());
  for (int c = 0; c < channels; ++c) {
    for (int i = 0; i < voxels; ++i) {
      want_dn[static_cast<size_t>(c) * voxels + i] =
          z[static_cast<size_t>(c) * voxels + i] * sd[c] + mean[c];
    }
  }
  DeviceBuffer<float> dz = to_device(z);
  DeviceBuffer<float> dm = to_device(mean);
  DeviceBuffer<float> dsd = to_device(sd);
  DeviceBuffer<float> ddn(z.size());
  vidfab::cuda::launch_latent_denorm(dz.get(), dm.get(), dsd.get(), ddn.get(), channels, voxels,
                                     nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK_CLOSE(want_dn, to_host(ddn), 1e-6, "latent_denorm");
}

// `RegisteredMapping::contains` decides whether a pointer may be used as a
// DMA source. Saying yes about a pointer that is not in the range is not a
// missed optimisation, it is silently wrong weights: the caller then copies
// asynchronously out of pageable scratch memory that is rewritten immediately
// afterwards. The first version of it compared `n <= (base + bytes) - p`,
// which for any `p` past the end computes a negative difference, converts it
// to a huge size_t, and answers yes to precisely the pointers it exists to
// reject. Whether that happened depended on where the heap landed relative to
// the mapping, so it reproduced roughly one launch in ten.
void test_registered_mapping_contains() {
  // cudaHostRegister wants a page-aligned base, so the region is carved out of
  // a larger buffer and aligned by hand. It is placed one `kBytes` block in,
  // so there is addressable memory both below and above it to probe with.
  constexpr size_t kPage = 4096;
  constexpr size_t kBytes = 64 * 1024;
  std::vector<unsigned char> backing(kPage + kBytes * 3);
  const auto raw = reinterpret_cast<uintptr_t>(backing.data());
  unsigned char* const aligned =
      reinterpret_cast<unsigned char*>((raw + kPage - 1) & ~static_cast<uintptr_t>(kPage - 1));
  unsigned char* const region = aligned + kBytes;

  vidfab::cuda::RegisteredMapping mapping(region, kBytes);
  if (!mapping.registered()) {
    // Registration is best-effort, and the predicate is defined to answer no
    // when there is no range. Still worth asserting rather than skipping.
    CHECK(!mapping.contains(region, 1));
    return;
  }

  // Inside.
  CHECK(mapping.contains(region, kBytes));
  CHECK(mapping.contains(region, 1));
  CHECK(mapping.contains(region + kBytes - 1, 1));
  CHECK(mapping.contains(region + kBytes / 2, kBytes / 2));

  // Runs off the end.
  CHECK(!mapping.contains(region, kBytes + 1));
  CHECK(!mapping.contains(region + kBytes - 1, 2));

  // Below the range.
  CHECK(!mapping.contains(region - 1, 1));
  CHECK(!mapping.contains(aligned, 1));

  // Above the range — the case the original arithmetic got wrong. A scratch
  // buffer that the allocator happened to place after the mapping looks
  // exactly like this.
  CHECK(!mapping.contains(region + kBytes, 1));
  CHECK(!mapping.contains(region + kBytes + 1, 1));
  CHECK(!mapping.contains(region + kBytes, kBytes));
  CHECK(!mapping.contains(backing.data() + backing.size() - 1, 1));
}

const bool registered = ::vidfab::test::register_test("registered_mapping_contains",
                                                      &test_registered_mapping_contains) &&
                        ::vidfab::test::register_test("norms", &test_norms) &&
                        ::vidfab::test::register_test("gemm", &test_gemm) &&
                        ::vidfab::test::register_test("swiglu", &test_swiglu) &&
                        ::vidfab::test::register_test("softmax", &test_softmax) &&
                        ::vidfab::test::register_test("depth_to_space", &test_depth_to_space) &&
                        ::vidfab::test::register_test("qkv_norm_rope", &test_qkv_rope) &&
                        ::vidfab::test::register_test("gemm_nn_batched_ld", &test_gemm_scatter) &&
                        ::vidfab::test::register_test("widen_f16", &test_widen_f16) &&
                        ::vidfab::test::register_test("narrow_f16", &test_narrow_f16) &&
                        ::vidfab::test::register_test("bf16_to_f16", &test_bf16_to_f16) &&
                        ::vidfab::test::register_test("heads_to_tokens_bf16",
                                                     &test_heads_to_tokens_bf16) &&
                        ::vidfab::test::register_test("transpose_cn_to_nc", &test_transpose) &&
                        ::vidfab::test::register_test("misc", &test_misc);

}  // namespace

int main() {
  if (vidfab::cuda::device_count() == 0) {
    std::fprintf(stderr, "no CUDA device visible; skipping kernel tests\n");
    return 0;
  }
  return ::vidfab::test::run_all();
}
