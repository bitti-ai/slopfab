#pragma once

// Private shared fixtures for the encoder suites.
// Tests for the Qwen3-VL text conditioner.
//
// Every reference here is written from docs/text_encoder_spec.md, never from
// the code it checks. Section 9 of that spec is a list of fifteen ways to get
// this module wrong, and *none of them crash*: bidirectional attention, a
// skipped ConvRot, the DiT's 96-of-128 RoPE, `h % 8` grouped-query mapping,
// SiLU on the wrong projection — all produce finite, well-scaled, plausible
// output. So wherever a hazard has a specific wrong form, the test computes
// that form too and asserts the implementation does *not* match it. A test
// that only checks the right answer cannot tell you whether it had any power
// to detect the wrong one.
//
// The checkpoint-dependent cases skip cleanly when
// weights/text_encoder/qwen3vl_32b_int8_convrot.safetensors is absent; it is
// licence-restricted and not committed.

#include <cublas_v2.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "../harness.h"
#include "slopfab/cuda/device.h"
#include "slopfab/cuda/gemm.cuh"
#include "slopfab/cuda/linear.cuh"
#include "slopfab/cuda/nn_kernels.cuh"
#include "slopfab/cuda/workspace.cuh"
#include "slopfab/dtype.h"
#include "slopfab/safetensors.h"
#include "slopfab/safetensors_write.h"
#include "slopfab/tensor_convert.h"
#include "slopfab/text/encoder.h"
#include "slopfab/text/tokenizer.h"


namespace {
namespace {

using slopfab::cuda::DeviceBuffer;
using slopfab::cuda::QuantFormat;
using slopfab::cuda::QuantWeight;
using slopfab::cuda::Workspace;
using slopfab::test::make_data;

// --- host/device plumbing ---------------------------------------------------

struct BfBuf {
  DeviceBuffer<uint16_t> raw;

  explicit BfBuf(size_t n) : raw(n) {}
  explicit BfBuf(const std::vector<float>& host) : raw(host.size()) {
    std::vector<uint16_t> bits(host.size());
    for (size_t i = 0; i < host.size(); ++i) bits[i] = slopfab::f32_to_bf16(host[i]);
    raw.copy_from_host(bits.data(), bits.size());
  }

  __nv_bfloat16* p() { return reinterpret_cast<__nv_bfloat16*>(raw.get()); }

  std::vector<float> host() const {
    std::vector<uint16_t> b(raw.size());
    raw.copy_to_host(b.data(), b.size());
    std::vector<float> out(b.size());
    for (size_t i = 0; i < b.size(); ++i) out[i] = slopfab::bf16_to_f32(b[i]);
    return out;
  }
};

struct CublasScope {
  cublasHandle_t h = nullptr;
  CublasScope() { SLOPFAB_CUBLAS_CHECK(slopfab::cuda::cublas_create(&h)); }
  ~CublasScope() { slopfab::cuda::cublas_destroy(h); }
};

std::vector<float> bf16_round(const std::vector<float>& v) {
  std::vector<float> out(v.size());
  for (size_t i = 0; i < v.size(); ++i) out[i] = slopfab::bf16_to_f32(slopfab::f32_to_bf16(v[i]));
  return out;
}

DeviceBuffer<float> to_device(const std::vector<float>& host) {
  DeviceBuffer<float> d(host.size());
  d.copy_from_host(host.data(), host.size());
  return d;
}

double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
  double worst = 0.0;
  for (size_t i = 0; i < a.size(); ++i) worst = std::max(worst, std::fabs(double(a[i]) - b[i]));
  return worst;
}

double rms(const std::vector<float>& v) {
  double acc = 0.0;
  for (float x : v) acc += double(x) * x;
  return std::sqrt(acc / std::max<size_t>(1, v.size()));
}

// Worst absolute error as a fraction of the reference's RMS.
//
// Elementwise relative error is the wrong yardstick for a bf16 GEMM with
// cancellation: a 512-term dot product whose terms sum to 1.5 and whose result
// is 0.07 carries the *absolute* error of the terms, so the elements nearest
// zero always look catastrophic in relative terms while carrying no
// information. The error that matters is measured against the scale of the
// output as a whole, and it is what the 1e-3 / 1e-2 project tolerance means
// once cancellation is in play.
double rms_relative_error(const std::vector<float>& want, const std::vector<float>& got) {
  return max_abs_diff(want, got) / std::max(1e-30, rms(want));
}

// RMS of the error over RMS of the reference. The max-based measure above is
// dominated by the worst of tens of thousands of samples — about 4 sigma for
// Gaussian rounding noise — so quoting both separates "one element is unlucky"
// from "the whole tensor is wrong".
double rms_error_ratio(const std::vector<float>& want, const std::vector<float>& got) {
  std::vector<float> diff(want.size());
  for (size_t i = 0; i < want.size(); ++i) diff[i] = want[i] - got[i];
  return rms(diff) / std::max(1e-30, rms(want));
}

// The worktree layout puts the repository three levels up, so the search
// reaches further than test_adaln's.
std::string find_relative(const std::string& suffix) {
  for (const char* prefix : {"", "../", "../../", "../../../", "../../../../"}) {
    const std::string p = std::string(prefix) + suffix;
    if (std::filesystem::exists(p)) return p;
  }
  return {};
}

std::string find_checkpoint() {
  return find_relative("weights/text_encoder/qwen3vl_32b_int8_convrot.safetensors");
}

std::string find_nvfp4_checkpoint() {
  return find_relative("weights/text_encoder/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors");
}

std::string find_tokenizer() {
  const std::string p = find_relative("ref/text_encoder/tokenizer.json");
  return p.empty() ? find_relative("ref/FL2VA/text_encoder/tokenizer.json") : p;
}

// The prompt every checkpoint-dependent test conditions on. Long enough that
// the row statistics below are about the model rather than about one sentence.
const char* kPrompt =
    "A slow aerial shot over a rain-slicked city at night, neon signs reflected in the "
    "puddles, a lone figure walking beneath the overpass while a tram passes overhead. "
    "The camera drifts forward and tilts down as the light changes from red to green, "
    "steam rising from a vent, distant sirens, the hum of traffic on a wet road. "
    "Cut to a narrow street market, awnings dripping, a vendor folding a tarp over "
    "crates of fruit while two children run past kicking up spray. The lens racks focus "
    "from the foreground puddle to a bus pulling away, its windows fogged, passengers "
    "silhouetted against the interior light. Overhead wires sway. A cat crosses the "
    "frame and disappears into a doorway. The shot holds on the empty street as the rain "
    "eases and the reflections settle, then pushes in slowly on a single lit window "
    "three floors up, where a figure stands with their back to the glass.";

// Tokenised with no chat template and no special tokens, or a synthetic id run
// when ref/ is absent. One extra leading token would shift every RoPE position
// and, because attention is causal, change every row (spec section 1.2).
std::vector<int32_t> prompt_ids() {
  const std::string tok = find_tokenizer();
  if (tok.empty()) {
    std::printf("  ref/ tokenizer.json not present; using synthetic token ids\n");
    std::vector<int32_t> ids;
    for (int i = 0; i < 200; ++i) ids.push_back(1000 + i);
    return ids;
  }
  slopfab::text::Tokenizer tokenizer;
  tokenizer.load(tok);
  std::vector<int32_t> ids = tokenizer.encode(kPrompt);
  std::printf("  tokenised the prompt to %zu tokens\n", ids.size());
  return ids;
}

// Per-row RMS over the 5120 channels of an encoder output.
std::vector<double> row_rms(const slopfab::text::PromptEmbedding& e) {
  std::vector<double> out(e.num_tokens, 0.0);
  for (int r = 0; r < e.num_tokens; ++r) {
    double acc = 0.0;
    for (int i = 0; i < e.hidden_size; ++i) {
      const double v = e.data[size_t(r) * e.hidden_size + i];
      acc += v * v;
    }
    out[size_t(r)] = std::sqrt(acc / e.hidden_size);
  }
  return out;
}

// --- CPU references ---------------------------------------------------------

// y = w * (x * rsqrt(mean(x^2) + eps)). eps inside the root, added to the mean
// of squares; no mean subtraction, no bias. Accumulated in double, which is at
// least as good as the reference's mandatory fp32 upcast (spec 4.1).
std::vector<float> cpu_rmsnorm(const std::vector<float>& x, const std::vector<float>& w, int rows,
                               int dim, float eps) {
  std::vector<float> out(x.size());
  for (int r = 0; r < rows; ++r) {
    double sum_sq = 0.0;
    for (int i = 0; i < dim; ++i) {
      const double v = x[size_t(r) * dim + i];
      sum_sq += v * v;
    }
    const double inv = 1.0 / std::sqrt(sum_sq / dim + eps);
    for (int i = 0; i < dim; ++i) out[size_t(r) * dim + i] = float(x[size_t(r) * dim + i] * inv * w[i]);
  }
  return out;
}

std::vector<float> cpu_matmul_nt(const std::vector<float>& A, const std::vector<float>& B, int M,
                                 int N, int K) {
  std::vector<float> C(size_t(M) * N);
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      double acc = 0.0;
      for (int k = 0; k < K; ++k) acc += double(A[size_t(m) * K + k]) * B[size_t(n) * K + k];
      C[size_t(m) * N + n] = float(acc);
    }
  }
  return C;
}

// GPT-NeoX half-split rotation. `rotary` lets the test build the *hazard* form
// as well — the H3 DiT rotates 96 of 128 and pairs j with j+48, and porting
// that habit here is spec section 9 item 10.
std::vector<float> cpu_rope(const std::vector<float>& x, const std::vector<float>& cos,
                            const std::vector<float>& sin, int rows, int heads, int head_dim,
                            int rotary) {
  std::vector<float> out = x;
  const int half = rotary / 2;
  for (int r = 0; r < rows; ++r) {
    for (int h = 0; h < heads; ++h) {
      const size_t base = (size_t(r) * heads + h) * head_dim;
      for (int j = 0; j < half; ++j) {
        const float lo = x[base + j];
        const float hi = x[base + j + half];
        const float c = cos[size_t(r) * head_dim + j];
        const float s = sin[size_t(r) * head_dim + j];
        out[base + j] = lo * c - hi * s;
        out[base + j + half] = hi * c + lo * s;
      }
    }
  }
  return out;
}

// Dense attention. `causal` false is the hazard: bidirectional attention
// changes every row but the last and is completely silent (spec section 3).
std::vector<float> cpu_attention(const std::vector<float>& q, const std::vector<float>& k,
                                 const std::vector<float>& v, int seq, int heads, int kv_heads,
                                 int head_dim, float scale, bool causal) {
  const int qld = heads * head_dim;
  const int kvld = kv_heads * head_dim;
  const int group = heads / kv_heads;
  std::vector<float> out(size_t(seq) * qld, 0.0f);
  std::vector<double> p(seq);

  for (int h = 0; h < heads; ++h) {
    // Query head h reads kv head h/group — contiguous blocks. `h % kv_heads`
    // is the plausible wrong answer and no shape check catches it.
    const int kv = h / group;
    for (int i = 0; i < seq; ++i) {
      const int last = causal ? i : seq - 1;
      double m = -1e300;
      for (int j = 0; j <= last; ++j) {
        double dot = 0.0;
        for (int d = 0; d < head_dim; ++d) {
          dot += double(q[size_t(i) * qld + h * head_dim + d]) *
                 k[size_t(j) * kvld + kv * head_dim + d];
        }
        p[j] = dot * scale;
        m = std::max(m, p[j]);
      }
      double sum = 0.0;
      for (int j = 0; j <= last; ++j) {
        p[j] = std::exp(p[j] - m);
        sum += p[j];
      }
      for (int d = 0; d < head_dim; ++d) {
        double acc = 0.0;
        for (int j = 0; j <= last; ++j) acc += p[j] * v[size_t(j) * kvld + kv * head_dim + d];
        out[size_t(i) * qld + h * head_dim + d] = float(acc / sum);
      }
    }
  }
  return out;
}

// --- the regular Hadamard, built the long way -------------------------------
//
// h4 is the *regular* Hadamard matrix (symmetric, constant row sums 2), not the
// Sylvester one; H = kron^4(h4)/16. Built here as an explicit 256x256 matrix
// and applied as a matmul, so it shares no code with the butterfly it checks
// (docs/convrot_notes.md).
std::vector<float> hadamard(int size) {
  const int h4[4][4] = {{1, 1, 1, -1}, {1, 1, -1, 1}, {1, -1, 1, 1}, {-1, 1, 1, 1}};
  std::vector<float> m{1.0f};
  int n = 1;
  while (n < size) {
    std::vector<float> next(size_t(n * 4) * (n * 4));
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        for (int r = 0; r < n; ++r) {
          for (int c = 0; c < n; ++c) {
            next[size_t(i * n + r) * (n * 4) + (j * n + c)] = float(h4[i][j]) * m[size_t(r) * n + c];
          }
        }
      }
    }
    m.swap(next);
    n *= 4;
  }
  const float norm = 1.0f / std::sqrt(float(size));
  for (float& v : m) v *= norm;
  return m;
}

// x_rot[g] = x[g] @ H, group by group along the trailing axis. H is symmetric
// and involutory, so the same routine rotates and de-rotates.
std::vector<float> rotate_rows(const std::vector<float>& x, int rows, int dim, int group,
                               const std::vector<float>& H) {
  std::vector<float> out(x.size());
  const int blocks = dim / group;
  for (int r = 0; r < rows; ++r) {
    for (int b = 0; b < blocks; ++b) {
      const size_t base = size_t(r) * dim + size_t(b) * group;
      for (int c = 0; c < group; ++c) {
        double acc = 0.0;
        for (int i = 0; i < group; ++i) acc += double(x[base + i]) * H[size_t(i) * group + c];
        out[base + c] = float(acc);
      }
    }
  }
  return out;
}

float silu(float z) { return z / (1.0f + std::exp(-z)); }

// --- tests ------------------------------------------------------------------

// --- checkpoint-dependent ----------------------------------------------------

// Runs one encoder over `ids` in one residency mode and reports what it cost.
// Separate from the assertions so that the two checkpoints and the two modes
// are measured by identical code — the residency comparison is the point.
struct EncodeRun {
  bool ok = false;
  slopfab::text::PromptEmbedding out;
  double load_seconds = 0.0;
  double cold_encode = 0.0;
  double warm_encode = 0.0;
  size_t weight_bytes = 0;
  size_t peak_bytes = 0;
  size_t measured_bytes = 0;
};

EncodeRun run_encoder(const slopfab::SafeTensors& st, slopfab::text::Residency mode,
                      const std::vector<int32_t>& ids, const char* label) {
  EncodeRun r;
  size_t free_before = 0;
  size_t total = 0;
  SLOPFAB_CUDA_CHECK(cudaMemGetInfo(&free_before, &total));

  slopfab::text::Encoder encoder;
  slopfab::text::EncoderConfig cfg;
  cfg.residency = mode;
  try {
    encoder.load(st, cfg);
  } catch (const std::exception& e) {
    std::printf("  %s: load failed (%s)\n", label, e.what());
    return r;
  }
  if (encoder.residency() != mode) {
    std::printf("  %s: fell back to the other residency mode; not measuring\n", label);
    return r;
  }

  r.out = encoder.encode(ids);
  r.cold_encode = encoder.stats().last_encode_seconds;
  size_t free_after = 0;
  SLOPFAB_CUDA_CHECK(cudaMemGetInfo(&free_after, &total));

  encoder.encode(ids);
  r.warm_encode = encoder.stats().last_encode_seconds;

  r.load_seconds = encoder.stats().load_seconds;
  r.weight_bytes = encoder.stats().weight_bytes;
  r.peak_bytes = encoder.stats().peak_device_bytes;
  r.measured_bytes = free_before - free_after;
  r.ok = true;
  std::printf(
      "  %-22s load %6.2f s, encode %6.3f s (warm %6.3f s) for %d tokens, weights %6.2f GB, "
      "accounted peak %5.2f GB, measured %5.2f GB\n",
      label, r.load_seconds, r.cold_encode, r.warm_encode, r.out.num_tokens,
      double(r.weight_bytes) / (1 << 30), double(r.peak_bytes) / (1 << 30),
      double(r.measured_bytes) / (1 << 30));
  encoder.unload();
  return r;
}

// The structural checks from docs/text_encoder_spec.md section 1.4. They do not
// depend on which build produced the output, which is exactly why they are
// worth applying to both: they say "this is an unnormalised residual stream",
// and a final norm having crept in is the failure they exist to catch.
void check_residual_stream_shape(const slopfab::text::PromptEmbedding& e, const char* label) {
  size_t nonfinite = 0;
  for (float v : e.data) {
    if (!std::isfinite(v)) ++nonfinite;
  }
  CHECK_MSG(nonfinite == 0, "%s: %zu of %zu output values are not finite", label, nonfinite,
            e.data.size());

  const std::vector<double> rms_rows = row_rms(e);
  const double lo = *std::min_element(rms_rows.begin(), rms_rows.end());
  const double hi = *std::max_element(rms_rows.begin(), rms_rows.end());
  std::printf(
      "  %s: row RMS first %.1f, min %.2f, max %.2f, spread %.1fx (L2 of the last row %.1f)\n",
      label, rms_rows.front(), lo, hi, hi / lo,
      rms_rows.back() * std::sqrt(double(e.hidden_size)));

  // An RMSNorm divides every row by its own scale, so a normalised stream has a
  // nearly constant row RMS. A raw residual does not, and the spread is the
  // discriminator rather than the absolute level.
  CHECK_MSG(lo > 0.5, "%s: row RMS falls to %.4f; the stream has collapsed", label, lo);
  CHECK_MSG(hi < 1e5, "%s: row RMS reaches %.4f, implausibly large", label, hi);
  CHECK_MSG(hi / lo > 3.0,
            "%s: row RMS is nearly constant across rows (min %.3f, max %.3f) - that is what an "
            "RMSNorm output looks like, and no final norm may be applied here",
            label, lo, hi);

  // Token 0 carries an attention-sink massive activation an order of magnitude
  // above every other row. Its presence says the stream has the model's own
  // structure, which well-scaled noise would not.
  const double others_max = *std::max_element(rms_rows.begin() + 1, rms_rows.end());
  CHECK_MSG(rms_rows.front() > 5.0 * others_max,
            "%s: token 0 RMS %.2f is not the attention-sink outlier it should be (next largest "
            "%.2f)",
            label, rms_rows.front(), others_max);
}

}  // namespace

}  // namespace
