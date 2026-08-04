// GPU kernel tests against independent CPU references.
//
// The failure mode these exist to catch is silent: a wrong QKV de-interleave,
// a wrong depth-to-space ordering or a transposed GEMM all produce finite,
// plausibly-scaled output that only looks wrong once it reaches the eye. Each
// kernel is therefore checked against a straightforward CPU implementation
// written from the spec rather than from the kernel.

#include <cublas_v2.h>

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

#include "vidfab/cuda/device.h"
#include "vidfab/cuda/gemm.cuh"
#include "vidfab/cuda/vae_kernels.cuh"

namespace {

int g_checks = 0;
int g_failures = 0;
const char* g_test = "";

void check_close(const std::vector<float>& expected, const std::vector<float>& actual,
                 double tol, const char* what, int line) {
  ++g_checks;
  if (expected.size() != actual.size()) {
    ++g_failures;
    std::fprintf(stderr, "  FAIL %s:%d  %s: size %zu vs %zu\n", g_test, line, what,
                 expected.size(), actual.size());
    return;
  }
  double worst = 0.0;
  size_t worst_i = 0;
  for (size_t i = 0; i < expected.size(); ++i) {
    const double d = std::fabs(static_cast<double>(expected[i]) - actual[i]);
    if (d > worst) {
      worst = d;
      worst_i = i;
    }
  }
  if (worst > tol) {
    ++g_failures;
    std::fprintf(stderr, "  FAIL %s:%d  %s: max abs err %.3e at %zu (%.6g vs %.6g)\n", g_test,
                 line, what, worst, worst_i, expected[worst_i], actual[worst_i]);
  }
}

#define CHECK_CLOSE(e, a, tol, what) check_close((e), (a), (tol), (what), __LINE__)

#define TEST(name)         \
  g_test = name;           \
  std::printf("test %s\n", name);

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

  vidfab::cuda::launch_rmsnorm(dx.get(), dw.get(), dout.get(), rows, dim, eps, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK_CLOSE(cpu_rmsnorm(x, w, rows, dim, eps), to_host(dout), 1e-4, "rmsnorm");

  vidfab::cuda::launch_layernorm(dx.get(), dw.get(), db.get(), dout.get(), rows, dim, eps,
                                 nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK_CLOSE(cpu_layernorm(x, w, b, rows, dim, eps), to_host(dout), 1e-4, "layernorm");
}

void test_gemm() {
  TEST("gemm");
  cublasHandle_t h = nullptr;
  VIDFAB_CUBLAS_CHECK(cublasCreate(&h));
  VIDFAB_CUBLAS_CHECK(cublasSetMathMode(h, CUBLAS_PEDANTIC_MATH));

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
  cublasDestroy(h);
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
    ++g_checks;
    if (std::fabs(sum - 1.0) > 1e-5) {
      ++g_failures;
      std::fprintf(stderr, "  FAIL softmax row %d sums to %.9f\n", r, sum);
    }
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

  vidfab::cuda::launch_split_qkv_norm_rope(dqkv.get(), dcos.get(), dsin.get(), dq.get(), dk.get(),
                                           dv.get(), seq, heads, head_dim, rope_dim, num_patches,
                                           eps, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK_CLOSE(want_q, to_host(dq), 1e-5, "rope q");
  CHECK_CLOSE(want_k, to_host(dk), 1e-5, "rope k");
  CHECK_CLOSE(want_v, to_host(dv), 0.0, "v passthrough");
}

void test_misc() {
  TEST("misc");
  // merge_heads: [H][S][D] -> [S][H*D]
  const int seq = 11;
  const int heads = 5;
  const int head_dim = 8;
  const std::vector<float> in = make_data(static_cast<size_t>(seq) * heads * head_dim, 1234u);
  std::vector<float> want(in.size());
  for (int s = 0; s < seq; ++s) {
    for (int h = 0; h < heads; ++h) {
      for (int d = 0; d < head_dim; ++d) {
        want[(static_cast<size_t>(s) * heads + h) * head_dim + d] =
            in[(static_cast<size_t>(h) * seq + s) * head_dim + d];
      }
    }
  }
  DeviceBuffer<float> din = to_device(in);
  DeviceBuffer<float> dout(in.size());
  vidfab::cuda::launch_merge_heads(din.get(), dout.get(), seq, heads, head_dim, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK_CLOSE(want, to_host(dout), 0.0, "merge_heads");

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

}  // namespace

int main() {
  if (vidfab::cuda::device_count() == 0) {
    std::fprintf(stderr, "no CUDA device visible; skipping kernel tests\n");
    return 0;
  }
  try {
    test_norms();
    test_gemm();
    test_swiglu();
    test_softmax();
    test_depth_to_space();
    test_qkv_rope();
    test_misc();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "exception: %s\n", e.what());
    return 1;
  }
  std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
