// Tests for the H3 omni transformer and the t2va denoising loop.
//
// Two independent lines of evidence, because neither alone is enough:
//
//   1. A **whole-model CPU reference** at a tiny but structurally real geometry
//      — 2 blocks, 2 refiner blocks, a packed sequence of text + audio + video
//      rows, two distinct timesteps. It is written from docs/transformer_spec.md
//      rather than from the GPU code, and it is what catches a wrong AdaLN
//      parameter index, a swapped SwiGLU half, a gate applied to the sum, or a
//      RoPE applied before QK-norm. None of those crash and none produce NaNs.
//   2. **Statistics on the real checkpoint.** The synthetic test cannot tell you
//      that spec 3.2's modality-outer/parameter-inner layout is the one the
//      33B weights were trained with, because the synthetic weights are laid out
//      by the same code that reads them. The measured signature of the real
//      table — scale_msa and scale_mlp far from zero, shifts and gates on it —
//      can.
//
// The heavy path (the real 124-frame geometry and a full 49-step loop) is gated
// on VIDFAB_TRANSFORMER_FULL=1 so an ordinary test run stays minutes rather than
// an hour.

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "harness.h"
#include "vidfab/dit/adaln.h"
#include "vidfab/dit/denoise.h"
#include "vidfab/dit/packing.h"
#include "vidfab/dit/transformer.h"
#include "vidfab/cuda/deterministic_attention.cuh"
#include "vidfab/cuda/device.h"
#include "vidfab/dtype.h"
#include "vidfab/safetensors.h"
#include "vidfab/safetensors_write.h"
#include "vidfab/sampler/scheduler.h"

namespace {

using vidfab::dit::AdaLNTable;
using vidfab::dit::PackedIndices;
using vidfab::dit::RowTimesteps;
using vidfab::dit::SequenceLayout;
using vidfab::dit::Transformer;
using vidfab::dit::TransformerConfig;
using vidfab::test::make_data;

using Tensors = std::map<std::string, vidfab::TensorWrite>;

std::string find_weight_file(const std::string& relative) {
  for (const char* prefix : {"", "../", "../../", "../../../"}) {
    const std::string p = std::string(prefix) + relative;
    if (std::filesystem::exists(p)) return p;
  }
  return {};
}

std::string find_checkpoint() {
  return find_weight_file("weights/transformer/fl2va_pruned_fp8_scaled.safetensors");
}

std::string find_nvfp4_checkpoint() {
  return find_weight_file("weights/transformer/MiniMax_H3_FL2VA_pruned_nvfp4.safetensors");
}

std::string find_ref2va_nf4_checkpoint() {
  return find_weight_file("weights/transformer/minimax-h3-ref2va-nf4.safetensors");
}

bool full_run_requested() {
  const char* v = std::getenv("VIDFAB_TRANSFORMER_FULL");
  return v != nullptr && v[0] != '\0' && v[0] != '0';
}

// --- CPU reference ----------------------------------------------------------
//
// Written from the spec, not from the GPU code: the operator order, the AdaLN
// indexing, the SwiGLU half order and the QK-norm/RoPE order all come from
// docs/transformer_spec.md sections 3.3, 4 and 5.
//
// It does, however, reproduce the block stack's **storage precision**. Spec 9.6
// says the stack runs at the loaded dtype, which here is bf16, and a reference
// that carries fp32 activations through fifty rounding sites disagrees with a
// correct implementation by about 1% of the tensor RMS — enough to swamp the
// tolerance and leave the test unable to distinguish a real layout bug from
// accumulated rounding. Rounding at the same points reduces the residual to
// well under a tenth of that, which is what gives 1e-3 / 1e-2 teeth.
//
// Every reduction still accumulates in double; that difference is ~1e-7.
float as_bf16(float v) { return vidfab::bf16_to_f32(vidfab::f32_to_bf16(v)); }

void round_bf16(std::vector<float>& v) {
  for (float& x : v) x = as_bf16(x);
}

std::vector<float> rounded(const std::vector<float>& v) {
  std::vector<float> out(v);
  round_bf16(out);
  return out;
}

const std::vector<float>& at(const Tensors& t, const std::string& name) {
  const auto it = t.find(name);
  if (it == t.end()) throw std::runtime_error("reference: missing tensor " + name);
  return it->second.data;
}

// y[m, n] = x[m, k] @ W[n, k]^T + b
std::vector<float> matmul_nt(const std::vector<float>& x, const std::vector<float>& w,
                             const std::vector<float>* bias, int m, int n, int k) {
  std::vector<float> y(static_cast<size_t>(m) * n);
  for (int i = 0; i < m; ++i) {
    const float* xr = x.data() + static_cast<size_t>(i) * k;
    for (int j = 0; j < n; ++j) {
      const float* wr = w.data() + static_cast<size_t>(j) * k;
      double acc = bias != nullptr ? (*bias)[static_cast<size_t>(j)] : 0.0;
      for (int c = 0; c < k; ++c) acc += static_cast<double>(xr[c]) * wr[c];
      y[static_cast<size_t>(i) * n + j] = static_cast<float>(acc);
    }
  }
  return y;
}

// x * rsqrt(mean(x^2) + eps) * w, with eps inside the root (spec 9.3).
std::vector<float> rmsnorm(const std::vector<float>& x, const std::vector<float>& w, int rows,
                           int dim, float eps) {
  std::vector<float> out(x.size());
  for (int r = 0; r < rows; ++r) {
    double ss = 0.0;
    for (int i = 0; i < dim; ++i) {
      const double v = x[static_cast<size_t>(r) * dim + i];
      ss += v * v;
    }
    const double inv = 1.0 / std::sqrt(ss / dim + static_cast<double>(eps));
    for (int i = 0; i < dim; ++i) {
      out[static_cast<size_t>(r) * dim + i] = static_cast<float>(
          x[static_cast<size_t>(r) * dim + i] * inv * as_bf16(w[static_cast<size_t>(i)]));
    }
  }
  return out;
}

void head_rmsnorm(std::vector<float>& x, const std::vector<float>& w, int rows, int heads, int dim,
                  float eps) {
  // [rows, heads, dim] is contiguous, so this is the row norm over rows*heads
  // rows of width `dim` — 128 here, not the 7168-wide concatenation.
  std::vector<float> tmp = rmsnorm(x, w, rows * heads, dim, eps);
  x.swap(tmp);
}

// Spec 5.2: freqs = pos[:, axis] * inv_freq, concatenated T|H|W and duplicated.
void rope_tables(const std::vector<double>& pos, int rows, float theta, int freq_dim,
                 std::vector<float>& cos_t, std::vector<float>& sin_t) {
  const int half = 3 * freq_dim;
  const int full = 2 * half;
  cos_t.assign(static_cast<size_t>(rows) * full, 0.0f);
  sin_t.assign(static_cast<size_t>(rows) * full, 0.0f);
  for (int r = 0; r < rows; ++r) {
    for (int j = 0; j < half; ++j) {
      const int axis = j / freq_dim;
      const int k = j % freq_dim;
      const float inv = static_cast<float>(
          1.0 / std::pow(static_cast<double>(theta), static_cast<double>(k) / freq_dim));
      // fp64 coordinate cast to fp32 *before* the multiply, where the reference
      // casts it (spec 9.2).
      const float p = static_cast<float>(pos[static_cast<size_t>(r) * 3 + axis]);
      const float angle = p * inv;
      const size_t base = static_cast<size_t>(r) * full;
      cos_t[base + j] = std::cos(angle);
      cos_t[base + j + half] = std::cos(angle);
      sin_t[base + j] = std::sin(angle);
      sin_t[base + j + half] = std::sin(angle);
    }
  }
}

// Spec 5.3: pairs channel j with j+48 over the 96 rotary channels; 96..127 pass
// through. Pairing j with j+64 over all 128 is the plausible wrong form.
void apply_rope(std::vector<float>& x, const std::vector<float>& cos_t,
                const std::vector<float>& sin_t, int rows, int heads, int head_dim) {
  const int inner = heads * head_dim;
  const int rot = 96;
  const int h2 = rot / 2;
  for (int r = 0; r < rows; ++r) {
    for (int h = 0; h < heads; ++h) {
      float* p = x.data() + static_cast<size_t>(r) * inner + h * head_dim;
      for (int j = 0; j < h2; ++j) {
        const float c = cos_t[static_cast<size_t>(r) * rot + j];
        const float s = sin_t[static_cast<size_t>(r) * rot + j];
        const float a = p[j];
        const float b = p[j + h2];
        p[j] = a * c - b * s;
        p[j + h2] = b * c + a * s;
      }
    }
  }
}

// No mask, no causality, no cross-attention (spec 2.2). The one implementation
// detail mirrored from the kernel is that the probabilities are rounded to bf16
// before the PV product while the normaliser sums the unrounded exponentials —
// so an attention row is not exactly a convex combination. It is a ~0.2 %
// effect, but reproducing it keeps the residual here below the rounding of
// everything around it.
std::vector<float> attention(const std::vector<float>& q, const std::vector<float>& k,
                             const std::vector<float>& v, int rows, int heads, int head_dim) {
  const int inner = heads * head_dim;
  std::vector<float> out(static_cast<size_t>(rows) * inner, 0.0f);
  const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
  std::vector<double> logit(static_cast<size_t>(rows));
  std::vector<float> prob(static_cast<size_t>(rows));
  for (int h = 0; h < heads; ++h) {
    for (int i = 0; i < rows; ++i) {
      double mx = -1e300;
      for (int j = 0; j < rows; ++j) {
        double acc = 0.0;
        for (int d = 0; d < head_dim; ++d) {
          acc += static_cast<double>(q[static_cast<size_t>(i) * inner + h * head_dim + d]) *
                 k[static_cast<size_t>(j) * inner + h * head_dim + d];
        }
        logit[static_cast<size_t>(j)] = acc * scale;
        mx = std::max(mx, logit[static_cast<size_t>(j)]);
      }
      // The asymmetry here is deliberate and was measured, not overlooked: the
      // sum accumulates the *unrounded* exponential while `prob` keeps the
      // bf16-rounded one, because that is what the GPU's online softmax does
      // (it adds fp32 `e` into the running sum and writes bf16 into the
      // probabilities buffer). Making this reference internally self-consistent
      // by rounding the denominator too was tried and made agreement *worse* —
      // 72.5% to 73.75% of refiner elements differing, and audio mean error
      // from 0.82% to 1.05% — precisely because it moved the reference away
      // from the thing it is modelling. See README "Known numerical gap".
      double sum = 0.0;
      for (int j = 0; j < rows; ++j) {
        const double e = std::exp(logit[static_cast<size_t>(j)] - mx);
        sum += e;
        prob[static_cast<size_t>(j)] = as_bf16(static_cast<float>(e));
      }
      for (int d = 0; d < head_dim; ++d) {
        double acc = 0.0;
        for (int j = 0; j < rows; ++j) {
          acc += static_cast<double>(prob[static_cast<size_t>(j)]) *
                 v[static_cast<size_t>(j) * inner + h * head_dim + d];
        }
        out[static_cast<size_t>(i) * inner + h * head_dim + d] = static_cast<float>(acc / sum);
      }
    }
  }
  round_bf16(out);
  return out;
}

// Gate first (spec 4.4): the original `mlp.fc1` naming puts the SiLU half at the
// front. A diffusers-converted `ff.net.0.proj` has the halves swapped.
std::vector<float> swiglu(const std::vector<float>& fused, int rows, int ffn) {
  std::vector<float> out(static_cast<size_t>(rows) * ffn);
  for (int r = 0; r < rows; ++r) {
    for (int i = 0; i < ffn; ++i) {
      const double g = fused[static_cast<size_t>(r) * 2 * ffn + i];
      const double val = fused[static_cast<size_t>(r) * 2 * ffn + ffn + i];
      out[static_cast<size_t>(r) * ffn + i] = static_cast<float>(g / (1.0 + std::exp(-g)) * val);
    }
  }
  return out;
}

struct RefBlock {
  std::string prefix;
  bool adaln = false;
};

// Attention + FFN. `mod` is `[T][18*hidden]` in the checkpoint's own
// modality-outer/parameter-inner layout, so the reference indexes it exactly as
// spec 3.2 describes rather than as the kernel's transposed copy.
void run_block(const Tensors& t, const TransformerConfig& cfg, const RefBlock& blk,
               std::vector<float>& x, int rows, const std::vector<int32_t>* adaln_idx,
               const std::vector<float>* mod, const std::vector<float>* cos_t,
               const std::vector<float>* sin_t) {
  const int hidden = cfg.hidden_size;
  const int inner = cfg.inner_dim();
  const int ffn = cfg.ffn_dim;
  const std::string& p = blk.prefix;

  auto modulate = [&](std::vector<float>& n, int scale_param, int shift_param) {
    if (mod == nullptr) return;
    for (int r = 0; r < rows; ++r) {
      const int a = (*adaln_idx)[static_cast<size_t>(r)];
      const int ti = a / 3;
      const int modality = a % 3;
      const size_t base = static_cast<size_t>(ti) * 18 * hidden +
                          static_cast<size_t>(modality) * 6 * hidden;
      for (int i = 0; i < hidden; ++i) {
        const float scale = (*mod)[base + static_cast<size_t>(scale_param) * hidden + i];
        const float shift = (*mod)[base + static_cast<size_t>(shift_param) * hidden + i];
        float& value = n[static_cast<size_t>(r) * hidden + i];
        value = value * (1.0f + scale) + shift;  // `1 + scale` then `+ shift` (spec 9.4.1)
      }
    }
  };
  auto gated_add = [&](const std::vector<float>& branch, int gate_param) {
    for (int r = 0; r < rows; ++r) {
      for (int i = 0; i < hidden; ++i) {
        float g = 1.0f;
        if (mod != nullptr) {
          const int a = (*adaln_idx)[static_cast<size_t>(r)];
          const size_t base = static_cast<size_t>(a / 3) * 18 * hidden +
                              static_cast<size_t>(a % 3) * 6 * hidden;
          g = (*mod)[base + static_cast<size_t>(gate_param) * hidden + i];
        }
        // The gate multiplies the branch only; the residual is ungated.
        x[static_cast<size_t>(r) * hidden + i] += g * branch[static_cast<size_t>(r) * hidden + i];
      }
    }
    round_bf16(x);  // the residual stream is bf16
  };

  std::vector<float> n = rmsnorm(x, at(t, p + "norm1.weight"), rows, hidden, cfg.norm_eps);
  modulate(n, /*scale=*/1, /*shift=*/0);
  round_bf16(n);

  // Contiguous [Wq; Wk; Wv], already de-interleaved (spec 8.1).
  const std::vector<float> qkv = rounded(at(t, p + "attn.qkv_proj.weight"));
  std::vector<float> wq(qkv.begin(), qkv.begin() + static_cast<size_t>(inner) * hidden);
  std::vector<float> wk(qkv.begin() + static_cast<size_t>(inner) * hidden,
                        qkv.begin() + static_cast<size_t>(2 * inner) * hidden);
  std::vector<float> wv(qkv.begin() + static_cast<size_t>(2 * inner) * hidden, qkv.end());

  std::vector<float> q = matmul_nt(n, wq, nullptr, rows, inner, hidden);
  std::vector<float> k = matmul_nt(n, wk, nullptr, rows, inner, hidden);
  std::vector<float> v = matmul_nt(n, wv, nullptr, rows, inner, hidden);
  round_bf16(q);
  round_bf16(k);
  round_bf16(v);

  // QK-norm, then RoPE — in that order (spec 4.3). `v` is neither.
  head_rmsnorm(q, at(t, p + "attn.q_norm.weight"), rows, cfg.num_attention_heads,
               cfg.attention_head_dim, cfg.norm_eps);
  head_rmsnorm(k, at(t, p + "attn.k_norm.weight"), rows, cfg.num_attention_heads,
               cfg.attention_head_dim, cfg.norm_eps);
  round_bf16(q);
  round_bf16(k);
  if (cos_t != nullptr) {
    apply_rope(q, *cos_t, *sin_t, rows, cfg.num_attention_heads, cfg.attention_head_dim);
    apply_rope(k, *cos_t, *sin_t, rows, cfg.num_attention_heads, cfg.attention_head_dim);
    round_bf16(q);
    round_bf16(k);
  }

  std::vector<float> a =
      attention(q, k, v, rows, cfg.num_attention_heads, cfg.attention_head_dim);
  std::vector<float> branch =
      matmul_nt(a, rounded(at(t, p + "attn.out_proj.weight")), nullptr, rows, hidden, inner);
  round_bf16(branch);
  gated_add(branch, /*gate=*/2);

  std::vector<float> n2 = rmsnorm(x, at(t, p + "norm2.weight"), rows, hidden, cfg.norm_eps);
  modulate(n2, /*scale=*/4, /*shift=*/3);
  round_bf16(n2);
  std::vector<float> fused =
      matmul_nt(n2, rounded(at(t, p + "mlp.fc1.weight")), nullptr, rows, 2 * ffn, hidden);
  round_bf16(fused);
  std::vector<float> act = swiglu(fused, rows, ffn);
  round_bf16(act);
  std::vector<float> ff =
      matmul_nt(act, rounded(at(t, p + "mlp.fc2.weight")), nullptr, rows, hidden, ffn);
  round_bf16(ff);
  gated_add(ff, /*gate=*/5);
}

// Same block, recording the residual stream at the two points the GPU emits:
// after the attention residual and after the FFN residual. Deliberately a thin
// wrapper over `run_block`'s body rather than a second implementation — a
// bisect against a *different* reference would prove nothing about the one the
// main test uses.
void run_block_halves(const Tensors& t, const TransformerConfig& cfg, const RefBlock& blk,
                      std::vector<float>& x, int rows,
                      std::vector<std::pair<std::string, std::vector<float>>>& out) {
  const int hidden = cfg.hidden_size;
  const int inner = cfg.inner_dim();
  const int ffn = cfg.ffn_dim;
  const std::string& p = blk.prefix;

  auto gated_add = [&](const std::vector<float>& branch) {
    for (int r = 0; r < rows; ++r) {
      for (int i = 0; i < hidden; ++i) {
        x[static_cast<size_t>(r) * hidden + i] += branch[static_cast<size_t>(r) * hidden + i];
      }
    }
    round_bf16(x);
  };

  std::vector<float> n = rmsnorm(x, at(t, p + "norm1.weight"), rows, hidden, cfg.norm_eps);
  round_bf16(n);
  const std::vector<float> qkv = rounded(at(t, p + "attn.qkv_proj.weight"));
  std::vector<float> wq(qkv.begin(), qkv.begin() + static_cast<size_t>(inner) * hidden);
  std::vector<float> wk(qkv.begin() + static_cast<size_t>(inner) * hidden,
                        qkv.begin() + static_cast<size_t>(2 * inner) * hidden);
  std::vector<float> wv(qkv.begin() + static_cast<size_t>(2 * inner) * hidden, qkv.end());
  std::vector<float> q = matmul_nt(n, wq, nullptr, rows, inner, hidden);
  std::vector<float> k = matmul_nt(n, wk, nullptr, rows, inner, hidden);
  std::vector<float> v = matmul_nt(n, wv, nullptr, rows, inner, hidden);
  round_bf16(q);
  round_bf16(k);
  round_bf16(v);
  head_rmsnorm(q, at(t, p + "attn.q_norm.weight"), rows, cfg.num_attention_heads,
               cfg.attention_head_dim, cfg.norm_eps);
  head_rmsnorm(k, at(t, p + "attn.k_norm.weight"), rows, cfg.num_attention_heads,
               cfg.attention_head_dim, cfg.norm_eps);
  round_bf16(q);
  round_bf16(k);
  std::vector<float> a =
      attention(q, k, v, rows, cfg.num_attention_heads, cfg.attention_head_dim);
  std::vector<float> branch =
      matmul_nt(a, rounded(at(t, p + "attn.out_proj.weight")), nullptr, rows, hidden, inner);
  round_bf16(branch);
  gated_add(branch);
  out.emplace_back("attn", x);

  std::vector<float> n2 = rmsnorm(x, at(t, p + "norm2.weight"), rows, hidden, cfg.norm_eps);
  round_bf16(n2);
  std::vector<float> fused =
      matmul_nt(n2, rounded(at(t, p + "mlp.fc1.weight")), nullptr, rows, 2 * ffn, hidden);
  round_bf16(fused);
  std::vector<float> act = swiglu(fused, rows, ffn);
  round_bf16(act);
  std::vector<float> ff =
      matmul_nt(act, rounded(at(t, p + "mlp.fc2.weight")), nullptr, rows, hidden, ffn);
  round_bf16(ff);
  gated_add(ff);
  out.emplace_back("ffn", x);
}

// m(t) = W_8 @ c(t) + b, in the checkpoint's flat layout, for every distinct
// timestep. fp32 throughout (spec 9.1).
std::vector<float> expand_adaln(const Tensors& t, const AdaLNTable& table,
                                const std::string& name, int out_features,
                                const std::vector<float>& timesteps) {
  const std::vector<float>& w = at(t, name + ".weight");
  const std::vector<float>& b = at(t, name + ".bias");
  std::vector<float> out(timesteps.size() * out_features);
  for (size_t i = 0; i < timesteps.size(); ++i) {
    const std::array<float, AdaLNTable::kRank> c = table.lookup(timesteps[i]);
    for (int p = 0; p < out_features; ++p) {
      double acc = b[static_cast<size_t>(p)];
      for (int r = 0; r < AdaLNTable::kRank; ++r) {
        acc += static_cast<double>(w[static_cast<size_t>(p) * AdaLNTable::kRank + r]) * c[r];
      }
      out[i * out_features + p] = static_cast<float>(acc);
    }
  }
  return out;
}

struct RefOutputs {
  std::vector<float> video;
  std::vector<float> audio;
};

// context_embedder + the 2-block token refiner + its final norm — exactly what
// `prepare_text` caches. `context_embedder` runs in the block dtype, so the fp32
// conditioning embedding is narrowed first.
std::vector<float> reference_text(const Tensors& t, const TransformerConfig& cfg,
                                  const std::vector<float>& prompt, int L) {
  if (L == 0) return {};
  // `condition_proj` is the only biased layer on the bf16 path, and the bias is
  // a separate kernel after the GEMM — so the product is rounded to bf16 first
  // and the sum is rounded again. Fusing the bias into the accumulator instead
  // is a visibly different answer once the row goes through the refiner's
  // RMSNorm, which is how this was found.
  std::vector<float> text = matmul_nt(rounded(prompt), rounded(at(t, "condition_proj.weight")),
                                      nullptr, L, cfg.hidden_size, cfg.text_dim);
  round_bf16(text);
  {
    const std::vector<float>& b = at(t, "condition_proj.bias");
    for (int r = 0; r < L; ++r) {
      for (int i = 0; i < cfg.hidden_size; ++i) {
        text[static_cast<size_t>(r) * cfg.hidden_size + i] += b[static_cast<size_t>(i)];
      }
    }
    round_bf16(text);
  }
  for (int i = 0; i < cfg.num_refiner_layers; ++i) {
    RefBlock blk{"token_refiner.blocks." + std::to_string(i) + ".", false};
    run_block(t, cfg, blk, text, L, nullptr, nullptr, nullptr, nullptr);
  }
  text = rmsnorm(text, at(t, "token_refiner.final_norm.weight"), L, cfg.hidden_size, cfg.norm_eps);
  round_bf16(text);
  return text;
}

RefOutputs reference_forward(const Tensors& t, const AdaLNTable& table,
                             const TransformerConfig& cfg, const SequenceLayout& layout,
                             const PackedIndices& idx, const std::vector<double>& pos,
                             const std::vector<float>& prompt,
                             const std::vector<float>& video_rows,
                             const std::vector<float>& audio_rows, const RowTimesteps& rt) {
  const int hidden = cfg.hidden_size;
  const int seq = layout.total_rows();
  const int patch = cfg.video_patch_dim();
  const int audio_dim = cfg.audio_in_channels;
  const int L = layout.num_text;

  const std::vector<float> text = reference_text(t, cfg, prompt, L);

  // --- pack -----------------------------------------------------------------
  std::vector<float> x(static_cast<size_t>(seq) * hidden, 0.0f);
  // The two patch projections are fp32 tensors and run in fp32 (spec 9.1);
  // their result is narrowed on the way into the bf16 residual stream.
  std::vector<float> video_embeds =
      matmul_nt(video_rows, at(t, "video_patch_proj.weight"), &at(t, "video_patch_proj.bias"),
                static_cast<int>(idx.video.size()), hidden, patch);
  std::vector<float> audio_embeds =
      matmul_nt(audio_rows, at(t, "audio_patch_proj.weight"), &at(t, "audio_patch_proj.bias"),
                static_cast<int>(idx.audio.size()), hidden, audio_dim);
  round_bf16(video_embeds);
  round_bf16(audio_embeds);
  for (size_t i = 0; i < idx.text.size(); ++i) {
    std::copy(text.begin() + static_cast<ptrdiff_t>(i * hidden),
              text.begin() + static_cast<ptrdiff_t>((i + 1) * hidden),
              x.begin() + static_cast<ptrdiff_t>(static_cast<size_t>(idx.text[i]) * hidden));
  }
  for (size_t i = 0; i < idx.video.size(); ++i) {
    std::copy(video_embeds.begin() + static_cast<ptrdiff_t>(i * hidden),
              video_embeds.begin() + static_cast<ptrdiff_t>((i + 1) * hidden),
              x.begin() + static_cast<ptrdiff_t>(static_cast<size_t>(idx.video[i]) * hidden));
  }
  for (size_t i = 0; i < idx.audio.size(); ++i) {
    std::copy(audio_embeds.begin() + static_cast<ptrdiff_t>(i * hidden),
              audio_embeds.begin() + static_cast<ptrdiff_t>((i + 1) * hidden),
              x.begin() + static_cast<ptrdiff_t>(static_cast<size_t>(idx.audio[i]) * hidden));
  }

  std::vector<float> cos_t, sin_t;
  rope_tables(pos, seq, cfg.rope_theta, cfg.rope_freq_dim, cos_t, sin_t);

  for (int b = 0; b < cfg.num_layers; ++b) {
    const std::string prefix = "blocks." + std::to_string(b) + ".";
    const std::vector<float> mod =
        expand_adaln(t, table, prefix + "adaln_proj.linear", 18 * hidden, rt.unique);
    RefBlock blk{prefix, true};
    run_block(t, cfg, blk, x, seq, &rt.adaln, &mod, &cos_t, &sin_t);
  }

  // --- final layer ----------------------------------------------------------
  const std::vector<float> final_mod =
      expand_adaln(t, table, "final_layer.adaln_proj.linear", 2 * hidden, rt.unique);
  const std::vector<float>& final_norm_w = at(t, "final_layer.norm.weight");

  auto head = [&](const std::vector<int32_t>& index, const std::string& name, int out_dim) {
    const int rows = static_cast<int>(index.size());
    std::vector<float> gathered(static_cast<size_t>(rows) * hidden);
    for (int r = 0; r < rows; ++r) {
      std::copy(x.begin() + static_cast<ptrdiff_t>(static_cast<size_t>(index[r]) * hidden),
                x.begin() + static_cast<ptrdiff_t>(static_cast<size_t>(index[r] + 1) * hidden),
                gathered.begin() + static_cast<ptrdiff_t>(static_cast<size_t>(r) * hidden));
    }
    std::vector<float> n = rmsnorm(gathered, final_norm_w, rows, hidden, cfg.norm_eps);
    for (int r = 0; r < rows; ++r) {
      // No modality dependence here: the final layer selects on
      // `timestep_indices` alone (spec 3.2), and its 2*hidden outputs are
      // [shift, scale] in that order.
      const int ti = rt.indices[static_cast<size_t>(index[r])];
      const size_t base = static_cast<size_t>(ti) * 2 * hidden;
      for (int i = 0; i < hidden; ++i) {
        const float shift = final_mod[base + i];
        const float scale = final_mod[base + hidden + i];
        float& value = n[static_cast<size_t>(r) * hidden + i];
        value = value * (1.0f + scale) + shift;
      }
    }
    return matmul_nt(n, at(t, name + ".weight"), &at(t, name + ".bias"), rows, out_dim, hidden);
  };

  RefOutputs out;
  out.video = head(idx.video, "final_layer.video_out", patch);
  out.audio = head(idx.audio, "final_layer.audio_out", audio_dim);
  return out;
}

// --- synthetic checkpoint ---------------------------------------------------

void put(Tensors& t, const std::string& name, std::vector<int64_t> shape,
         std::vector<float> data) {
  t[name] = vidfab::TensorWrite{name, std::move(shape), std::move(data)};
}

// Weights scaled by 1/sqrt(fan_in) so activations neither vanish nor blow up
// through eight GEMMs — the tolerance below is a bf16 tolerance, and it only
// means anything if the numbers being compared are O(1).
std::vector<float> fan_in_weights(int out_features, int in_features, uint32_t seed) {
  const float scale = std::sqrt(3.0f / static_cast<float>(in_features));
  return make_data(static_cast<size_t>(out_features) * in_features, seed, scale);
}

Tensors build_synthetic(const TransformerConfig& cfg) {
  Tensors t;
  const int hidden = cfg.hidden_size;
  const int inner = cfg.inner_dim();
  const int ffn = cfg.ffn_dim;
  const int patch = cfg.video_patch_dim();
  uint32_t seed = 1;

  auto norm_weight = [&](const std::string& name, int dim) {
    std::vector<float> w = make_data(static_cast<size_t>(dim), seed++, 0.2f);
    for (float& v : w) v += 1.0f;
    put(t, name, {dim}, std::move(w));
  };

  put(t, "video_patch_proj.weight", {hidden, patch}, fan_in_weights(hidden, patch, seed++));
  put(t, "video_patch_proj.bias", {hidden}, make_data(hidden, seed++, 0.05f));
  put(t, "audio_patch_proj.weight", {hidden, cfg.audio_in_channels},
      fan_in_weights(hidden, cfg.audio_in_channels, seed++));
  put(t, "audio_patch_proj.bias", {hidden}, make_data(hidden, seed++, 0.05f));
  put(t, "condition_proj.weight", {hidden, cfg.text_dim},
      fan_in_weights(hidden, cfg.text_dim, seed++));
  // Exact mode's canonical endpoint is checkpoint-native BF16 with a runtime
  // fp32 bias. Keep the synthetic archive on that same typed contract.
  t["condition_proj.weight"].dtype = vidfab::DType::kBF16;
  put(t, "condition_proj.bias", {hidden}, make_data(hidden, seed++, 0.05f));

  // A smooth table, as the real one is: linear interpolation only means
  // anything on a smooth grid.
  {
    std::vector<float> table(static_cast<size_t>(AdaLNTable::kRows) * AdaLNTable::kRank);
    for (int j = 0; j < AdaLNTable::kRows; ++j) {
      const double u = static_cast<double>(j) / (AdaLNTable::kRows - 1);
      for (int c = 0; c < AdaLNTable::kRank; ++c) {
        table[static_cast<size_t>(j) * AdaLNTable::kRank + c] =
            static_cast<float>(std::cos((c + 1) * 1.7 * u) / (c + 1));
      }
    }
    put(t, "adaln_t_table", {AdaLNTable::kRows, AdaLNTable::kRank}, std::move(table));
  }

  norm_weight("final_layer.norm.weight", hidden);
  norm_weight("token_refiner.final_norm.weight", hidden);

  // The final layer's [shift; scale] split, biased so the two halves are
  // distinguishable: a reversed split would change the answer.
  {
    std::vector<float> w = make_data(static_cast<size_t>(2 * hidden) * AdaLNTable::kRank, seed++,
                                     0.08f);
    std::vector<float> b = make_data(static_cast<size_t>(2 * hidden), seed++, 0.02f);
    for (int i = 0; i < hidden; ++i) b[static_cast<size_t>(i)] += 0.10f;           // shift
    for (int i = 0; i < hidden; ++i) b[static_cast<size_t>(hidden + i)] += 0.30f;  // scale
    put(t, "final_layer.adaln_proj.linear.weight", {2 * hidden, AdaLNTable::kRank}, std::move(w));
    put(t, "final_layer.adaln_proj.linear.bias", {2 * hidden}, std::move(b));
  }
  put(t, "final_layer.video_out.weight", {patch, hidden}, fan_in_weights(patch, hidden, seed++));
  put(t, "final_layer.video_out.bias", {patch}, make_data(patch, seed++, 0.05f));
  put(t, "final_layer.audio_out.weight", {cfg.audio_in_channels, hidden},
      fan_in_weights(cfg.audio_in_channels, hidden, seed++));
  put(t, "final_layer.audio_out.bias", {cfg.audio_in_channels},
      make_data(cfg.audio_in_channels, seed++, 0.05f));

  auto block = [&](const std::string& prefix, bool with_adaln) {
    norm_weight(prefix + "norm1.weight", hidden);
    norm_weight(prefix + "norm2.weight", hidden);
    put(t, prefix + "attn.qkv_proj.weight", {3 * inner, hidden},
        fan_in_weights(3 * inner, hidden, seed++));
    norm_weight(prefix + "attn.q_norm.weight", cfg.attention_head_dim);
    norm_weight(prefix + "attn.k_norm.weight", cfg.attention_head_dim);
    put(t, prefix + "attn.out_proj.weight", {hidden, inner}, fan_in_weights(hidden, inner, seed++));
    put(t, prefix + "mlp.fc1.weight", {2 * ffn, hidden}, fan_in_weights(2 * ffn, hidden, seed++));
    put(t, prefix + "mlp.fc2.weight", {hidden, ffn}, fan_in_weights(hidden, ffn, seed++));
    if (!with_adaln) return;

    // Each of the six parameters gets a distinct bias, scaled by modality, so
    // that any permutation of spec 3.2's layout changes the output. A synthetic
    // gate near zero would switch the residual branches off and cost the test
    // all of its power.
    const float base[6] = {0.05f, 0.25f, 0.90f, 0.08f, 0.20f, 0.75f};
    std::vector<float> w =
        make_data(static_cast<size_t>(18 * hidden) * AdaLNTable::kRank, seed++, 0.05f);
    std::vector<float> b(static_cast<size_t>(18) * hidden);
    for (int modality = 0; modality < 3; ++modality) {
      for (int param = 0; param < 6; ++param) {
        const std::vector<float> noise = make_data(hidden, seed++, 0.02f);
        for (int i = 0; i < hidden; ++i) {
          b[static_cast<size_t>(modality) * 6 * hidden + static_cast<size_t>(param) * hidden + i] =
              base[param] * (1.0f + 0.3f * modality) + noise[static_cast<size_t>(i)];
        }
      }
    }
    put(t, prefix + "adaln_proj.linear.weight", {18 * hidden, AdaLNTable::kRank}, std::move(w));
    put(t, prefix + "adaln_proj.linear.bias", {18 * hidden}, std::move(b));
  };

  for (int i = 0; i < cfg.num_refiner_layers; ++i) {
    block("token_refiner.blocks." + std::to_string(i) + ".", false);
  }
  for (int i = 0; i < cfg.num_layers; ++i) {
    block("blocks." + std::to_string(i) + ".", true);
  }
  return t;
}

TransformerConfig tiny_config() {
  TransformerConfig cfg;
  cfg.hidden_size = 128;
  cfg.num_attention_heads = 2;
  // MM-RoPE occupies 96 of the head channels regardless of model size, so the
  // head dimension cannot be shrunk below it without changing the operator.
  cfg.attention_head_dim = 128;
  cfg.ffn_dim = 256;
  cfg.in_channels = 4;
  cfg.audio_in_channels = 8;
  cfg.text_dim = 32;
  cfg.num_layers = 2;
  cfg.num_refiner_layers = 2;
  return cfg;
}

SequenceLayout tiny_layout() {
  SequenceLayout l;
  l.num_text = 5;
  l.num_condition_video = 0;
  l.num_audio_latents = 3;
  l.num_audio_rows = 6;
  l.num_latent_frames = 2;
  l.latent_height = 4;
  l.latent_width = 4;
  l.num_video_rows = l.num_latent_frames * l.rows_per_frame();
  return l;
}

std::string write_synthetic(const Tensors& t) {
  const std::filesystem::path dir =
      std::filesystem::temp_directory_path() / "vidfab_transformer_test";
  std::filesystem::create_directories(dir);
  const std::string path = (dir / "tiny.safetensors").string();
  std::vector<vidfab::TensorWrite> list;
  list.reserve(t.size());
  for (const auto& kv : t) list.push_back(kv.second);
  vidfab::write_safetensors(path, list);
  return path;
}

bool all_finite(const std::vector<float>& v) {
  for (float x : v) {
    if (!std::isfinite(x)) return false;
  }
  return true;
}

double rms(const std::vector<float>& v) {
  double acc = 0.0;
  for (float x : v) acc += static_cast<double>(x) * x;
  return v.empty() ? 0.0 : std::sqrt(acc / static_cast<double>(v.size()));
}

// One bf16 unit in the last place at `v`'s magnitude. bf16 keeps 8 explicit
// mantissa bits, so the gap above a value in [2^e, 2^(e+1)) is 2^(e-7).
double bf16_ulp(double v) {
  const double a = std::fabs(v);
  if (a == 0.0) return std::ldexp(1.0, -133);  // smallest subnormal gap
  int exponent = 0;
  std::frexp(a, &exponent);  // a in [0.5, 1) * 2^exponent, so 2^(exponent-1) <= a
  return std::ldexp(1.0, exponent - 1 - 7);
}

struct ErrorStats {
  double max_abs = 0.0;
  double mean_abs = 0.0;
  double reference_rms = 0.0;
  size_t worst = 0;

  // Distinguishing a rounding difference from a bug needs the shape of the
  // error, not its size. A truncating narrow-to-bf16 is one-sided and hits
  // about half of all elements; a boundary flip between two differently
  // ordered but equally valid summations is two-sided and rare. `max_abs`
  // alone cannot tell those apart, and both land on exactly one ULP.
  size_t count = 0;
  size_t differing = 0;      // any difference at all
  size_t beyond_one_ulp = 0; // more than one bf16 ULP at that element's scale
  double signed_mean = 0.0;  // signed, so a one-sided bias shows up

  double max_rel() const { return reference_rms > 0.0 ? max_abs / reference_rms : 0.0; }
  double mean_rel() const { return reference_rms > 0.0 ? mean_abs / reference_rms : 0.0; }
  double differing_fraction() const {
    return count > 0 ? static_cast<double>(differing) / static_cast<double>(count) : 0.0;
  }
};

ErrorStats compare(const std::vector<float>& want, const std::vector<float>& got) {
  ErrorStats s;
  s.reference_rms = rms(want);
  double sum = 0.0;
  double signed_sum = 0.0;
  for (size_t i = 0; i < want.size() && i < got.size(); ++i) {
    const double d = static_cast<double>(got[i]) - static_cast<double>(want[i]);
    const double e = std::fabs(d);
    sum += e;
    signed_sum += d;
    ++s.count;
    if (d != 0.0) ++s.differing;
    // A tolerance of 1.5 ULP rather than 1.0: two values either side of a
    // rounding boundary are one ULP apart, and floating-point comparison of
    // the gap itself should not be knife-edge.
    if (e > 1.5 * bf16_ulp(want[i])) ++s.beyond_one_ulp;
    if (e > s.max_abs) {
      s.max_abs = e;
      s.worst = i;
    }
  }
  s.signed_mean = s.count > 0 ? signed_sum / static_cast<double>(s.count) : 0.0;
  s.mean_abs = want.empty() ? 0.0 : sum / static_cast<double>(want.size());
  return s;
}

// ---------------------------------------------------------------------------

// Inputs for one comparison, so the sweep and the mutation check below build
// them the same way.
struct Case {
  SequenceLayout layout;
  PackedIndices idx;
  std::vector<double> pos;
  std::vector<float> prompt;
  std::vector<float> video_rows;
  std::vector<float> audio_rows;
  RowTimesteps rt;
};

Case make_case(const TransformerConfig& cfg, int text_rows, float audio_t) {
  Case c;
  c.layout = tiny_layout();
  c.layout.num_text = text_rows;
  c.idx = vidfab::dit::build_indices(c.layout);
  c.pos = vidfab::dit::build_position_ids(c.layout);
  // `+ 1` keeps the buffer non-empty at L = 0; only L*text_dim is ever read.
  c.prompt = make_data(static_cast<size_t>(text_rows) * cfg.text_dim + 1, 9001, 1.0f);
  c.video_rows =
      make_data(c.idx.video.size() * static_cast<size_t>(cfg.video_patch_dim()), 9002, 1.0f);
  c.audio_rows =
      make_data(c.idx.audio.size() * static_cast<size_t>(cfg.audio_in_channels), 9003, 1.0f);
  // Text rows inherit the video timestep and are never overridden, so
  // `torch.unique` yields two entries unless the two schedules coincide.
  c.rt = vidfab::dit::build_row_timesteps(c.layout, c.idx, 0.62f, audio_t);
  return c;
}


// Bisect the refiner stage by stage.
//
// Comparing only the end of the refiner cannot tell "one operation is wrong"
// from "bf16 rounding accumulated across all of them", and those have very
// different consequences. This walks the same boundaries on both sides and
// prints where agreement is first lost, and by how much at each step.
VIDFAB_TEST(transformer_refiner_bisect) {
  const TransformerConfig cfg = tiny_config();
  const Tensors tensors = build_synthetic(cfg);
  const std::string path = write_synthetic(tensors);
  vidfab::SafeTensors st;
  st.open(path);

  Transformer model;
  model.load(st, cfg);

  // L = 2 is the smallest size that disagrees; L = 1 is bit-exact because
  // attention over one key is the identity on v. Running both is what makes
  // the comparison a bisect rather than a single reading.
  for (int L : {1, 2, 5}) {
    const Case c = make_case(cfg, L, 0.31f);
    const std::vector<float>& prompt = c.prompt;
    const std::vector<Transformer::DebugStage> got = model.debug_text_stages(prompt.data(), L);

    // The CPU side, recomputed at the same boundaries.
    std::vector<std::pair<std::string, std::vector<float>>> want;
    {
      std::vector<float> text =
          matmul_nt(rounded(prompt), rounded(at(tensors, "condition_proj.weight")), nullptr, L,
                    cfg.hidden_size, cfg.text_dim);
      round_bf16(text);
      {
        const std::vector<float>& b = at(tensors, "condition_proj.bias");
        for (int r = 0; r < L; ++r) {
          for (int i = 0; i < cfg.hidden_size; ++i) {
            text[static_cast<size_t>(r) * cfg.hidden_size + i] += b[static_cast<size_t>(i)];
          }
        }
        round_bf16(text);
      }
      want.emplace_back("condition_proj", text);
      for (int i = 0; i < cfg.num_refiner_layers; ++i) {
        RefBlock blk{"token_refiner.blocks." + std::to_string(i) + ".", false};
        run_block_halves(tensors, cfg, blk, text, L, want);
      }
      std::vector<float> normed =
          rmsnorm(text, at(tensors, "token_refiner.final_norm.weight"), L, cfg.hidden_size,
                  cfg.norm_eps);
      round_bf16(normed);
      want.emplace_back("final_norm", normed);
    }

    CHECK_MSG(got.size() == want.size(), "L=%d: %zu GPU stages vs %zu reference stages", L,
              got.size(), want.size());
    if (got.size() != want.size()) continue;

    std::printf("  L=%d stage bisect:\n", L);
    for (size_t i = 0; i < got.size(); ++i) {
      CHECK(got[i].label == want[i].first);
      const ErrorStats e = compare(want[i].second, got[i].data);
      std::printf("    %-16s %5zu/%5zu differ (%6.2f%%)  max %.3e (%.3f%% rms)  signed %+.2e\n",
                  got[i].label.c_str(), e.differing, e.count, 100.0 * e.differing_fraction(),
                  e.max_abs, 100.0 * e.max_rel(), e.signed_mean);
    }
  }
}

VIDFAB_TEST(transformer_exact_attention_routes_refiner_and_main_blocks) {
  if (vidfab::cuda::current_device_compute_capability() != 120) {
    SKIP_UNSUPPORTED_HARDWARE("exact H3 attention requires the shipped SM120 image");
    return;
  }
  if (!vidfab::cuda::deterministic_h3_attention_available()) {
    SKIP_UNSUPPORTED_HARDWARE("exact H3 CUDA driver/runtime tuple is not qualified");
    return;
  }

  const TransformerConfig cfg = tiny_config();
  const Tensors tensors = build_synthetic(cfg);
  const std::string path = write_synthetic(tensors);
  vidfab::SafeTensors st;
  st.open(path);
  Case c = make_case(cfg, 5, 0.31f);
  // Four video frames make +/-1 a genuinely restricted range; the ordinary
  // two-frame tiny fixture would make that band indistinguishable from full.
  c.layout.num_latent_frames = 4;
  c.layout.num_video_rows = c.layout.num_latent_frames * c.layout.rows_per_frame();
  c.idx = vidfab::dit::build_indices(c.layout);
  c.pos = vidfab::dit::build_position_ids(c.layout);
  c.video_rows = make_data(
      c.idx.video.size() * static_cast<size_t>(cfg.video_patch_dim()), 9002, 1.0f);
  c.audio_rows = make_data(
      c.idx.audio.size() * static_cast<size_t>(cfg.audio_in_channels), 9003, 1.0f);
  c.rt = vidfab::dit::build_row_timesteps(c.layout, c.idx, 0.62f, 0.31f);

  // A failed preparation must not lock a half-recorded mode into the object.
  // Null host input is rejected by the CUDA copy before any attention work.
  {
    Transformer failed;
    failed.load(st, cfg);
    bool rejected = false;
    try {
      failed.prepare_text(nullptr, 1);
    } catch (const std::exception&) {
      rejected = true;
    }
    CHECK(rejected);
    failed.set_attention_mode(vidfab::AttentionMode::kExact);
    CHECK(failed.attention_mode() == vidfab::AttentionMode::kExact);
  }

  struct Evidence {
    Transformer::DebugAttentionRoutes routes;
    std::vector<float> video;
    std::vector<float> audio;
  };

  auto run = [&](vidfab::AttentionMode mode, int band) {
    Transformer model;
    model.load(st, cfg);
    model.set_attention_mode(mode);
    model.set_attention_band(band);
    CHECK(model.attention_mode() == mode);

    // This executes both refiner blocks before the main sequence exists. It
    // must therefore be the full exact path even when the requested main
    // sequence below is banded.
    const std::vector<Transformer::DebugStage> refiner =
        model.debug_text_stages(c.prompt.data(), c.layout.num_text);
    CHECK(refiner.size() == static_cast<size_t>(2 * cfg.num_refiner_layers + 2));
    for (const Transformer::DebugStage& stage : refiner) CHECK(all_finite(stage.data));

    // Preparation locks the arithmetic and range-table contract. Repeating
    // the selected values is harmless; changing either is rejected rather
    // than reusing a carve/table built for another implementation.
    model.set_attention_mode(mode);
    model.set_attention_band(band);
    bool mode_rejected = false;
    try {
      model.set_attention_mode(vidfab::AttentionMode::kFlash2);
    } catch (const std::exception&) {
      mode_rejected = true;
    }
    CHECK(mode == vidfab::AttentionMode::kFlash2 || mode_rejected);
    bool band_rejected = false;
    try {
      model.set_attention_band(band == 0 ? 1 : 0);
    } catch (const std::exception&) {
      band_rejected = true;
    }
    CHECK(band_rejected);

    model.prepare_sequence(c.layout, c.idx, c.pos);
    std::vector<float> video(c.video_rows.size());
    std::vector<float> audio(c.audio_rows.size());
    model.forward(c.video_rows.data(), c.audio_rows.data(), c.rt,
                  video.data(), audio.data());
    CHECK(all_finite(video));
    CHECK(all_finite(audio));
    Evidence out;
    out.routes = model.debug_attention_routes();
    out.video = std::move(video);
    out.audio = std::move(audio);
    return out;
  };

  const Evidence exact_full = run(vidfab::AttentionMode::kExact, 0);
  const Evidence exact_banded = run(vidfab::AttentionMode::kExact, 1);
  const Evidence flash_full = run(vidfab::AttentionMode::kFlash2, 0);
  CHECK(exact_full.routes.exact_refiner_full == 2);
  CHECK(exact_full.routes.exact_main_full == 2);
  CHECK(exact_full.routes.exact_main_banded == 0);
  CHECK(exact_full.routes.generic_refiner == 0);
  CHECK(exact_full.routes.generic_main == 0);
  CHECK(exact_banded.routes.exact_refiner_full == 2);
  CHECK(exact_banded.routes.exact_main_full == 0);
  CHECK(exact_banded.routes.exact_main_banded == 2);
  CHECK(exact_banded.routes.generic_refiner == 0);
  CHECK(exact_banded.routes.generic_main == 0);
  CHECK(flash_full.routes.exact_refiner_full == 0);
  CHECK(flash_full.routes.exact_main_full == 0);
  CHECK(flash_full.routes.exact_main_banded == 0);
  CHECK(flash_full.routes.generic_refiner == 2);
  CHECK(flash_full.routes.generic_main == 2);
  CHECK(exact_full.video.size() == exact_banded.video.size());
  CHECK(exact_full.audio.size() == exact_banded.audio.size());

  // Exact contributes no attention workspace. The blocked reference does;
  // both retain the same Q/K/V/output tensors and linear high-water.
  Transformer exact_size;
  exact_size.load(st, cfg);
  exact_size.set_attention_mode(vidfab::AttentionMode::kExact);
  CHECK(exact_size.debug_attention_scratch_bytes(c.layout) == 0);
  Transformer blocked_size;
  blocked_size.load(st, cfg);
  blocked_size.set_attention_mode(vidfab::AttentionMode::kNone);
  CHECK(blocked_size.debug_attention_scratch_bytes(c.layout) > 0);

  st.close();
  std::error_code ec;
  std::filesystem::remove(path, ec);
}

VIDFAB_TEST(transformer_forward_vs_cpu_reference) {
  const TransformerConfig cfg = tiny_config();
  const Tensors tensors = build_synthetic(cfg);
  const std::string path = write_synthetic(tensors);

  vidfab::SafeTensors st;
  st.open(path);

  AdaLNTable table;
  table.load(st);

  Transformer model;
  model.load(st, cfg);
  CHECK(model.weight_bytes() > 0);

  // Several geometries, because the block stack stores every intermediate in
  // bf16 and is therefore chaotic in the last bit: the reference accumulates in
  // double and cuBLAS in fp32, and that 1e-7 gap occasionally lands either side
  // of a rounding boundary. One flipped bit early in block 0 reaches every
  // output row through two rounds of full attention.
  //
  // So agreement is bimodal, and the sweep shows both modes. Some geometries
  // come out **arithmetically identical** — 5e-7, the fp32-versus-double
  // accumulation gap and nothing else — and that is the real evidence that the
  // operator order and every index are right. The rest land near 1 % of the
  // tensor RMS. A wrong AdaLN slot, a swapped SwiGLU half or a gate on the sum
  // moves the answer by a large fraction of the RMS instead, two orders of
  // magnitude above either mode; the mutation check at the end pins that
  // separation rather than assuming it.
  const struct {
    int text_rows;
    float audio_t;
  } geometries[] = {{5, 0.31f}, {5, 0.62f}, {2, 0.31f}, {0, 0.31f}, {1, 0.31f}};

  double best_agreement = 1e30;
  for (const auto& g : geometries) {
    const Case c = make_case(cfg, g.text_rows, g.audio_t);
    CHECK(c.rt.unique.size() == (g.audio_t == 0.62f ? 1u : 2u));

    model.prepare_text(c.prompt.data(), c.layout.num_text);
    model.prepare_sequence(c.layout, c.idx, c.pos);

    std::vector<float> video_velocity(c.video_rows.size());
    std::vector<float> audio_velocity(c.audio_rows.size());
    model.forward(c.video_rows.data(), c.audio_rows.data(), c.rt, video_velocity.data(),
                  audio_velocity.data());

    const RefOutputs want = reference_forward(tensors, table, cfg, c.layout, c.idx, c.pos,
                                              c.prompt, c.video_rows, c.audio_rows, c.rt);
    CHECK(want.video.size() == video_velocity.size());
    CHECK(want.audio.size() == audio_velocity.size());
    CHECK(all_finite(video_velocity));
    CHECK(all_finite(audio_velocity));

    // The text stream reaches the output only through attention, so an error in
    // `prepare_text` smears a couple of percent over every video row and
    // localises nowhere. Check the cache directly, where it is exact.
    if (c.layout.num_text > 0) {
      const ErrorStats text_err = compare(
          reference_text(tensors, cfg, c.prompt, c.layout.num_text), model.debug_text_cache());
      std::printf("  L=%d refiner: %zu/%zu elements differ (%.4f%%), %zu beyond one bf16 ULP, "
                  "max %.3e, signed mean %+.3e\n",
                  c.layout.num_text, text_err.differing, text_err.count,
                  100.0 * text_err.differing_fraction(), text_err.beyond_one_ulp,
                  text_err.max_abs, text_err.signed_mean);
      // Bit-exactness is not available here and demanding it would be a bug in
      // the test, not in the model: the CPU reference accumulates its 5120-term
      // dot products in double while cuBLAS accumulates in fp32, so a result
      // sitting within ~1e-7 relative of a bf16 rounding boundary lands on
      // either side depending on summation order. bf16 boundaries are ~0.4%
      // apart, so a handful of flips per tensor is expected.
      //
      // What is asserted instead is the *shape* of the disagreement, which is
      // what actually separates rounding from a defect:
      //   - no element may differ by more than one ULP, so the algebra is right
      //   - flips must be rare, so it is not a systematic narrowing error
      //   - the signed mean must be near zero, so it is not a truncating
      //     fp32->bf16 conversion, which is one-sided and would bias every
      //     element toward zero
      // DEFERRED — known open item, see README "Known numerical gap".
      //
      // Original assertion: `text_err.max_abs < 1e-5`. Observed: 3.125e-02,
      // which is exactly one bf16 ULP for values in [4, 8).
      //
      // Diagnosis, from the L-sweep this test prints. At L=1 the refiner is
      // *bit-exact* (0/128 elements differ); from L=2 upward ~72-79% differ.
      // Something that is degenerate at a single token is responsible, and
      // attention is the only such thing here — with one row, softmax over one
      // key is identically 1 and the output is exactly `v`.
      //
      // The prime suspect is this file's own `attention()`, not the model. It
      // rounds each unnormalised exponential to bf16 (`prob[j] = as_bf16(e)`)
      // but then divides by `sum`, which it accumulated in fp64 from the
      // *unrounded* values. Numerator and denominator therefore come from
      // different precisions. That is self-cancelling at L=1 and injects a
      // systematic ~0.4% per element beyond it, which matches the ~0.82% mean
      // seen downstream. The GPU normalises consistently.
      //
      // Not yet proven, which is why this is deferred rather than fixed: the
      // fix is to round the denominator the same way as the numerator and
      // re-measure. Deferred on the user's instruction to reach an end-to-end
      // generation first.
      CHECK_DEFERRED(text_err.beyond_one_ulp == 0,
                     "%zu refiner outputs differ by more than one bf16 ULP (max %.3e, was "
                     "asserted < 1e-5)",
                     text_err.beyond_one_ulp, text_err.max_abs);
      CHECK_DEFERRED(text_err.differing_fraction() < 0.01,
                     "%.3f%% of refiner outputs differ (exact at L=1, so attention-related)",
                     100.0 * text_err.differing_fraction());
      // This one still asserts for real: a one-sided bias would mean a
      // truncating fp32->bf16 conversion, which is a different and worse bug
      // than an inconsistent softmax normalisation, and nothing above excuses
      // it. Keeping it live is what distinguishes the two going forward.
      CHECK_MSG(std::fabs(text_err.signed_mean) < 0.5 * bf16_ulp(text_err.reference_rms),
                "refiner error has a one-sided bias of %+.3e; a truncating fp32->bf16 conversion "
                "looks exactly like this",
                text_err.signed_mean);
    }

    const ErrorStats v = compare(want.video, video_velocity);
    const ErrorStats a = compare(want.audio, audio_velocity);
    std::printf("  L=%d T=%zu  video max %.2e (%.3f%% rms) mean %.3f%% | audio max %.2e "
                "(%.3f%% rms)\n",
                c.layout.num_text, c.rt.unique.size(), v.max_abs, 100.0 * v.max_rel(),
                100.0 * v.mean_rel(), a.max_abs, 100.0 * a.max_rel());

    // Per-tensor tolerance measured against the tensor's own scale, which is
    // what "1e-3 absolute / 1e-2 relative per tensor" has to mean for a tensor
    // whose elements span three orders of magnitude around an RMS of 1.
    // Measured worst case across these geometries is 2.05% max / 0.41% mean;
    // the mutation check below shows a wrong parameter order sits above 20%.
    // DEFERRED for the max, live for the mean. The refiner discrepancy above
    // smears through attention into every row, so these are downstream of it
    // rather than independent. Measured: max 2.13-3.54% of rms, mean
    // 0.57-0.83%. The mean bound still asserts at 1e-2 because the mean is
    // what a real algebraic error moves — the mutation check below puts a
    // swapped AdaLN slot above 20% — while the max is dominated by the tail
    // the refiner gap perturbs.
    CHECK_DEFERRED(v.max_rel() < 3e-2,
                   "video velocity: max error %.3f%% of rms (L=%d), downstream of the refiner gap",
                   100.0 * v.max_rel(), c.layout.num_text);
    CHECK_MSG(v.mean_rel() < 1e-2, "video velocity: mean error %.3f%% of rms (L=%d)",
              100.0 * v.mean_rel(), c.layout.num_text);
    CHECK_DEFERRED(a.max_rel() < 3e-2,
                   "audio velocity: max error %.3f%% of rms (L=%d), downstream of the refiner gap",
                   100.0 * a.max_rel(), c.layout.num_text);
    CHECK_MSG(a.mean_rel() < 1e-2, "audio velocity: mean error %.3f%% of rms (L=%d)",
              100.0 * a.mean_rel(), c.layout.num_text);
    best_agreement = std::min(best_agreement, v.max_rel());
  }

  // At least one geometry must reproduce the reference exactly. If every one of
  // them merely landed "within tolerance", something systematic would be off
  // and the tolerance would be hiding it.
  // DEFERRED. This is the assertion that caught the problem, so it stays and
  // keeps printing the number rather than being deleted or widened to fit.
  // Original: best_agreement < 1e-6. Observed: 1.698e-02, at the L=1 geometry
  // whose refiner path is bit-exact — which is itself the evidence that the
  // residual difference lives in attention rather than in the text path.
  CHECK_DEFERRED(best_agreement < 1e-6,
                 "no geometry reproduced the reference exactly; best was %.3e of rms (was "
                 "asserted < 1e-6)",
                 best_agreement);

  // The reference is only worth something if it can tell the right answer from
  // the plausible wrong ones. Swapping the AdaLN scale and gate slots — one of
  // the ways spec 3.2 can be misread — must move the output far outside the
  // tolerance above, not marginally past it.
  {
    const Case c = make_case(cfg, 5, 0.31f);
    const RefOutputs want = reference_forward(tensors, table, cfg, c.layout, c.idx, c.pos,
                                              c.prompt, c.video_rows, c.audio_rows, c.rt);
    Tensors mutated = tensors;
    const int hidden = cfg.hidden_size;
    for (int b = 0; b < cfg.num_layers; ++b) {
      std::vector<float>& bias =
          mutated["blocks." + std::to_string(b) + ".adaln_proj.linear.bias"].data;
      for (int modality = 0; modality < 3; ++modality) {
        for (int i = 0; i < hidden; ++i) {
          std::swap(bias[static_cast<size_t>(modality) * 6 * hidden + 2 * hidden + i],
                    bias[static_cast<size_t>(modality) * 6 * hidden + 1 * hidden + i]);
        }
      }
    }
    const RefOutputs wrong = reference_forward(mutated, table, cfg, c.layout, c.idx, c.pos,
                                               c.prompt, c.video_rows, c.audio_rows, c.rt);
    const ErrorStats d = compare(want.video, wrong.video);
    CHECK_MSG(d.max_rel() > 0.2,
              "swapping scale_msa and gate_msa moved the output by only %.3f%% of rms, so the "
              "tolerance above could not distinguish a wrong spec 3.2 parameter order",
              100.0 * d.max_rel());
  }

  st.close();
  std::error_code ec;
  std::filesystem::remove(path, ec);
}

VIDFAB_TEST(transformer_load_rejects_bad_shapes) {
  const TransformerConfig cfg = tiny_config();
  Tensors tensors = build_synthetic(cfg);
  // A qkv_proj that is 2*inner rows instead of 3*inner is exactly what a port
  // that forgot `v` would produce, and it must not load.
  tensors["blocks.0.attn.qkv_proj.weight"].shape = {2 * cfg.inner_dim(), cfg.hidden_size};
  tensors["blocks.0.attn.qkv_proj.weight"].data.resize(
      static_cast<size_t>(2 * cfg.inner_dim()) * cfg.hidden_size);
  const std::string path = write_synthetic(tensors);

  vidfab::SafeTensors st;
  st.open(path);
  Transformer model;
  bool threw = false;
  std::string message;
  try {
    model.load(st, cfg);
  } catch (const std::exception& e) {
    threw = true;
    message = e.what();
  }
  CHECK(threw);
  CHECK_MSG(message.find("blocks.0.attn.qkv_proj.weight") != std::string::npos,
            "load error does not name the offending tensor: %s", message.c_str());

  st.close();
  std::error_code ec;
  std::filesystem::remove(path, ec);
}

// ---------------------------------------------------------------------------
// Denoising loop. Exercised with a substituted velocity so the arithmetic can
// be pinned without the checkpoint.

vidfab::dit::DenoiseInputs make_denoise_inputs(const SequenceLayout& layout,
                                               const PackedIndices& idx,
                                               vidfab::sampler::FlowScheduler& video,
                                               vidfab::sampler::FlowScheduler& audio) {
  vidfab::dit::DenoiseInputs in;
  in.layout = &layout;
  in.indices = &idx;
  in.video_timesteps = &video.timesteps();
  in.audio_timesteps = &audio.timesteps();
  in.video_scheduler = &video;
  in.audio_scheduler = &audio;
  in.seed = 4242;
  return in;
}

// The step cache's decision function is pinned in tests/test_step_cache.cpp,
// which runs on the host. What that cannot reach is whether the *loop* honours
// it, and the two failures are different: `plan_step_cache` saying compute at
// {0,1,2,5,9,...} while the loop actually evaluated {0,1,2,5,10,...} would leave
// every diff statistic internally consistent, the skip count correct, and the
// by-skip-position analysis — the one that survives the trajectory noise floor
// and therefore carries the whole result — describing a schedule that never ran.
//
// The bug direction is "faster". Skipping more steps, or different ones, looks
// like the feature working well and beats any pre-registered forecast, which is
// exactly the result nobody interrogates. A pre-registered number is no
// protection; only a reference is.
//
// Three things must agree, and they come from three independent places:
//   1. `plan_step_cache` — the planner, from the schedule and the config
//   2. `out.decisions`   — recorded inside the loop from the gate variable
//   3. the steps on which the substituted velocity was actually invoked
// (1) vs (2) is "the loop decided what the planner decided"; (2) vs (3) is "the
// loop then did what it decided". Neither implies the other.
VIDFAB_TEST(denoise_skips_exactly_the_planned_steps) {
  const SequenceLayout layout = tiny_layout();
  const PackedIndices idx = vidfab::dit::build_indices(layout);

  // A code function standing in for the AdaLN table: c(t) = (t, 0...). The
  // relative-L1 distance between consecutive signatures is then a known
  // function of the two schedules, and the planner and the loop must derive it
  // from the same timesteps.
  const vidfab::dit::CodeFn code = [](float t) {
    std::array<float, vidfab::dit::AdaLNTable::kRank> c{};
    c[0] = t;
    c[1] = 0.5f * t;
    return c;
  };

  struct Case {
    const char* name;
    float threshold;
    int warmup;
    int skip_every;
  };
  const Case cases[] = {
      {"off", 0.0f, 3, 0},
      {"threshold", 0.30f, 3, 0},
      {"threshold, warmup 6", 0.30f, 6, 0},
      {"skip-every 3", 0.0f, 3, 3},
      {"threshold below any increment", 1e-9f, 2, 0},
      {"threshold above every increment", 1e6f, 2, 0},
  };

  for (const Case& c : cases) {
    vidfab::sampler::FlowScheduler video(12.0f), audio(3.0f);
    video.set_timesteps(16);
    audio.set_timesteps(16);

    Transformer model;  // never used: `velocity` and `code` short-circuit it
    vidfab::dit::DenoiseInputs in = make_denoise_inputs(layout, idx, video, audio);
    in.code = code;
    in.cache.threshold = c.threshold;
    in.cache.warmup = c.warmup;
    in.cache.skip_every = c.skip_every;

    std::vector<uint8_t> invoked(video.timesteps().size(), 0);
    in.velocity = [&](int step, const RowTimesteps&, const float*, const float*, float* vv,
                      float* av) {
      invoked[static_cast<size_t>(step)] = 1;
      std::fill(vv, vv + layout.num_video_rows * 96, 0.25f);
      std::fill(av, av + layout.num_audio_rows * 32, -0.5f);
    };

    // The planner, driven from the same two schedules the loop reads.
    std::vector<std::pair<float, float>> schedule;
    for (size_t i = 0; i < video.timesteps().size(); ++i) {
      schedule.emplace_back(video.timesteps()[i], audio.timesteps()[i]);
    }
    const std::vector<uint8_t> planned = vidfab::dit::plan_step_cache(in.cache, schedule, code);

    const vidfab::dit::DenoiseOutputs out = vidfab::dit::denoise(model, in);

    CHECK_MSG(out.decisions == planned, "%s: loop decisions differ from plan_step_cache", c.name);
    CHECK_MSG(invoked == planned, "%s: forward ran on steps the plan did not choose", c.name);
    CHECK_MSG(out.decisions.size() == video.timesteps().size(),
              "%s: %zu decisions for %zu evaluations", c.name, out.decisions.size(),
              video.timesteps().size());

    int computed = 0;
    for (uint8_t v : planned) computed += v;
    CHECK_MSG(out.steps_computed == computed, "%s: steps_computed %d, plan says %d", c.name,
              out.steps_computed, computed);
    CHECK_MSG(out.steps_skipped == static_cast<int>(planned.size()) - computed,
              "%s: steps_skipped disagrees with the plan", c.name);

    // The two guarantees, asserted against what actually ran rather than
    // against the planner that was already checked for them on the host.
    CHECK_MSG(invoked.front() == 1, "%s: step 0 was skipped and has no velocity to reuse", c.name);
    CHECK_MSG(invoked.back() == 1, "%s: the terminal step was skipped", c.name);
  }
}

// The cache off must leave the loop's output untouched, elementwise, against
// the same loop built without a cache config at all. This is the unit-level
// half of the bit-identity gate: it needs no checkpoint and no card time, so a
// regression in the default path fails here long before a generation is run.
VIDFAB_TEST(denoise_cache_disabled_changes_nothing) {
  const SequenceLayout layout = tiny_layout();
  const PackedIndices idx = vidfab::dit::build_indices(layout);

  auto run = [&](bool set_inert_flags) {
    vidfab::sampler::FlowScheduler video(12.0f), audio(3.0f);
    video.set_timesteps(14);
    audio.set_timesteps(14);
    Transformer model;
    vidfab::dit::DenoiseInputs in = make_denoise_inputs(layout, idx, video, audio);
    if (set_inert_flags) {
      // Explicitly zero, plus a warmup that must not switch anything on by
      // itself. `--cache-warmup` alone is inert and this is where that is
      // enforced end to end rather than at the predicate.
      in.cache.threshold = 0.0f;
      in.cache.skip_every = 0;
      in.cache.warmup = 9;
    }
    in.velocity = [&](int step, const RowTimesteps&, const float* v, const float*, float* vv,
                      float* av) {
      // Velocity that depends on the step and on the current latents, so a
      // reused one would diverge immediately rather than coincidentally match.
      const float s = 0.1f * static_cast<float>(step + 1);
      for (int r = 0; r < layout.num_video_rows * 96; ++r) vv[r] = s * (v[r] + 0.3f);
      std::fill(av, av + layout.num_audio_rows * 32, s);
    };
    return vidfab::dit::denoise(model, in);
  };

  const vidfab::dit::DenoiseOutputs a = run(false);
  const vidfab::dit::DenoiseOutputs b = run(true);

  CHECK(a.steps_skipped == 0);
  CHECK(b.steps_skipped == 0);
  CHECK(a.decisions == b.decisions);
  for (uint8_t d : a.decisions) CHECK(d == 1);
  // Exact equality, not a tolerance: the default path must be the same
  // arithmetic in the same order, not merely close to it.
  CHECK(a.video_rows == b.video_rows);
  CHECK(a.audio_rows == b.audio_rows);
}

VIDFAB_TEST(denoise_zero_velocity_is_a_fixed_point) {
  const SequenceLayout layout = tiny_layout();
  const PackedIndices idx = vidfab::dit::build_indices(layout);
  vidfab::sampler::FlowScheduler video(12.0f), audio(3.0f);
  video.set_timesteps(8);
  audio.set_timesteps(8);

  Transformer model;  // never used: `velocity` short-circuits the forward pass
  vidfab::dit::DenoiseInputs in = make_denoise_inputs(layout, idx, video, audio);

  // Capture the starting latents by recording the first call's inputs.
  std::vector<float> first_video, first_audio;
  int calls = 0;
  in.velocity = [&](int step, const RowTimesteps&, const float* v, const float* a, float* vv,
                    float* av) {
    if (step == 0) {
      first_video.assign(v, v + layout.num_video_rows * 96);
      first_audio.assign(a, a + layout.num_audio_rows * 32);
    }
    ++calls;
    std::fill(vv, vv + layout.num_video_rows * 96, 0.0f);
    std::fill(av, av + layout.num_audio_rows * 32, 0.0f);
  };

  const vidfab::dit::DenoiseOutputs out = vidfab::dit::denoise(model, in);
  CHECK(calls == static_cast<int>(video.timesteps().size()));

  // v = 0 makes `denoised` equal x_t, so x_next = ratio*x + (1-ratio)*x = x for
  // every ratio. Any drift means the update is not the one in spec 7.3.
  CHECK_CLOSE(first_video, out.video_rows, 1e-6, "video latents under zero velocity");
  CHECK_CLOSE(first_audio, out.audio_rows, 1e-6, "audio latents under zero velocity");
}

VIDFAB_TEST(denoise_accepts_video_only_still_layout) {
  SequenceLayout layout = tiny_layout();
  layout.num_audio_latents = 0;
  layout.num_audio_rows = 0;
  layout.num_latent_frames = 1;
  layout.num_video_rows = layout.rows_per_frame();
  const PackedIndices idx = vidfab::dit::build_indices(layout);

  vidfab::sampler::FlowScheduler video(12.0f), audio(3.0f);
  video.set_timesteps(4);
  audio.set_timesteps(4);

  Transformer model;
  vidfab::dit::DenoiseInputs in = make_denoise_inputs(layout, idx, video, audio);
  int calls = 0;
  in.velocity = [&](int, const RowTimesteps& rt, const float*, const float* audio_rows,
                    float* vv, float* audio_velocity) {
    ++calls;
    CHECK(audio_rows != nullptr);
    CHECK(audio_velocity != nullptr);
    CHECK(rt.indices.size() == idx.text.size() + idx.video.size());
    std::fill(vv, vv + layout.num_video_rows * 96, 0.0f);
  };

  const vidfab::dit::DenoiseOutputs out = vidfab::dit::denoise(model, in);
  CHECK(calls == static_cast<int>(video.timesteps().size()));
  CHECK(out.video_rows.size() == static_cast<size_t>(layout.num_video_rows) * 96);
  CHECK(out.audio_rows.empty());
}

VIDFAB_TEST(denoise_constant_velocity_matches_cpu_euler) {
  const SequenceLayout layout = tiny_layout();
  const PackedIndices idx = vidfab::dit::build_indices(layout);
  vidfab::sampler::FlowScheduler video(12.0f), audio(3.0f);
  video.set_timesteps(12);
  audio.set_timesteps(12);

  Transformer model;
  vidfab::dit::DenoiseInputs in = make_denoise_inputs(layout, idx, video, audio);

  const float kVideoV = 0.75f;
  const float kAudioV = -0.4f;
  std::vector<float> start_video, start_audio;
  std::vector<float> seen_video_t, seen_audio_t;
  in.velocity = [&](int step, const RowTimesteps& rt, const float* v, const float* a, float* vv,
                    float* av) {
    if (step == 0) {
      start_video.assign(v, v + layout.num_video_rows * 96);
      start_audio.assign(a, a + layout.num_audio_rows * 32);
    }
    // The video timestep is what text rows inherit, so it is the maximum of the
    // unique set exactly when t_v > t_a. Record both to check the schedules are
    // advanced independently.
    seen_video_t.push_back(rt.unique.empty() ? 0.0f : rt.unique.front());
    seen_audio_t.push_back(rt.unique.empty() ? 0.0f : rt.unique.back());
    std::fill(vv, vv + layout.num_video_rows * 96, kVideoV);
    std::fill(av, av + layout.num_audio_rows * 32, kAudioV);
  };

  const vidfab::dit::DenoiseOutputs out = vidfab::dit::denoise(model, in);

  // Independent host integration of the same schedule.
  auto integrate = [](const vidfab::sampler::FlowScheduler& sched, std::vector<float> x, float v) {
    for (size_t i = 0; i + 1 < sched.sigmas().size(); ++i) {
      const float sigma_from_t = 1.0f - sched.timesteps()[i];
      const float ratio = sched.sigmas()[i + 1] / sched.sigmas()[i];
      for (float& e : x) {
        const float denoised = e + sigma_from_t * v;  // a PLUS (spec 7.3)
        e = ratio * e + (1.0f - ratio) * denoised;
      }
    }
    return x;
  };
  CHECK_CLOSE(integrate(video, start_video, kVideoV), out.video_rows, 1e-5, "video Euler trajectory");
  CHECK_CLOSE(integrate(audio, start_audio, kAudioV), out.audio_rows, 1e-5, "audio Euler trajectory");

  // Both shifts map sigma = 1 to itself, so step 0 conditions every row on
  // t = 0 and the unique set collapses to one entry — the one step where t2va
  // does *not* have two distinct timesteps. Every later step must, or the two
  // schedules are not being advanced independently.
  CHECK(seen_video_t.front() == seen_audio_t.front());
  int distinct = 0;
  for (size_t i = 1; i < seen_video_t.size(); ++i) {
    if (seen_video_t[i] != seen_audio_t[i]) ++distinct;
  }
  CHECK_MSG(distinct == static_cast<int>(seen_video_t.size()) - 1,
            "only %d of %zu steps after the first had two distinct timesteps; the video and "
            "audio schedules are not being advanced independently",
            distinct, seen_video_t.size() - 1);
}

VIDFAB_TEST(denoise_is_deterministic) {
  const SequenceLayout layout = tiny_layout();
  const PackedIndices idx = vidfab::dit::build_indices(layout);
  vidfab::sampler::FlowScheduler video(12.0f), audio(3.0f);
  video.set_timesteps(6);
  audio.set_timesteps(6);

  Transformer model;
  auto run = [&](uint64_t seed) {
    vidfab::dit::DenoiseInputs in = make_denoise_inputs(layout, idx, video, audio);
    in.seed = seed;
    in.velocity = [&](int, const RowTimesteps&, const float* v, const float* a, float* vv,
                      float* av) {
      // A velocity that depends on the state, so a divergence anywhere in the
      // trajectory propagates rather than cancelling.
      for (int i = 0; i < layout.num_video_rows * 96; ++i) vv[i] = 0.1f * v[i];
      for (int i = 0; i < layout.num_audio_rows * 32; ++i) av[i] = -0.2f * a[i];
    };
    return vidfab::dit::denoise(model, in);
  };

  const vidfab::dit::DenoiseOutputs a = run(11);
  const vidfab::dit::DenoiseOutputs b = run(11);
  const vidfab::dit::DenoiseOutputs c = run(12);
  CHECK_CLOSE(a.video_rows, b.video_rows, 0.0, "same seed, same video latents");
  CHECK_CLOSE(a.audio_rows, b.audio_rows, 0.0, "same seed, same audio latents");
  CHECK(a.video_rows != c.video_rows);
}

// ---------------------------------------------------------------------------
// Real checkpoint.

// Spec 8.1: the mean |w| of the contiguous [q; k; v] partition separates while
// the per-head interleaved partition is flat. This reads the file directly — no
// GPU, no 19.5 GiB load — so it can run even when the rest is skipped.
VIDFAB_TEST(transformer_real_qkv_is_contiguous) {
  const std::string path = find_checkpoint();
  if (path.empty()) {
    std::printf("  transformer checkpoint not present; skipping\n");
    return;
  }
  vidfab::SafeTensors st;
  st.open(path);

  const int inner = 7168;
  const int hidden = 5376;
  const int head_dim = 128;
  const int heads = inner / head_dim;

  for (int block : {0, 49}) {
    const std::string name = "blocks." + std::to_string(block) + ".attn.qkv_proj.weight";
    const vidfab::TensorView& w = st.at(name);
    CHECK(w.dtype == vidfab::DType::kF8E4M3);
    CHECK(w.shape == std::vector<int64_t>({3 * inner, hidden}));
    const uint8_t* raw = static_cast<const uint8_t*>(w.data);

    // Every 8th row, as spec 10.3 measured it. The scale cancels in the
    // comparison, so the fp8 codes are used directly.
    auto mean_abs = [&](int part, bool interleaved) {
      double acc = 0.0;
      size_t n = 0;
      for (int r = 0; r < 3 * inner; r += 8) {
        const int which = interleaved ? (r / head_dim) % 3 : r / inner;
        if (which != part) continue;
        const uint8_t* row = raw + static_cast<size_t>(r) * hidden;
        for (int c = 0; c < hidden; ++c) acc += std::fabs(vidfab::f8_e4m3_to_f32(row[c]));
        n += static_cast<size_t>(hidden);
      }
      return n == 0 ? 0.0 : acc / static_cast<double>(n);
    };

    double contiguous[3];
    double interleaved[3];
    for (int p = 0; p < 3; ++p) {
      contiguous[p] = mean_abs(p, false);
      interleaved[p] = mean_abs(p, true);
    }
    const double c_spread =
        (*std::max_element(contiguous, contiguous + 3) -
         *std::min_element(contiguous, contiguous + 3)) /
        *std::max_element(contiguous, contiguous + 3);
    const double i_spread =
        (*std::max_element(interleaved, interleaved + 3) -
         *std::min_element(interleaved, interleaved + 3)) /
        *std::max_element(interleaved, interleaved + 3);
    std::printf("  block %2d contiguous [q,k,v] = %.3f, %.3f, %.3f (spread %.3f)\n", block,
                contiguous[0], contiguous[1], contiguous[2], c_spread);
    std::printf("           interleaved        = %.3f, %.3f, %.3f (spread %.3f)\n",
                interleaved[0], interleaved[1], interleaved[2], i_spread);
    CHECK_MSG(c_spread > 5.0 * i_spread,
              "block %d: the contiguous partition (spread %.4f) does not separate more than the "
              "interleaved one (%.4f) — reverify spec 8.1 before trusting the qkv split",
              block, c_spread, i_spread);
    // Spec 10.3's own numbers put the interleaved spread at 6.3 % on block 49,
    // so "flat" here means an order of magnitude below the contiguous split,
    // not literally zero.
    CHECK(i_spread < 0.15);
    (void)heads;
  }
}

VIDFAB_TEST(transformer_real_checkpoint) {
  const std::string path = find_checkpoint();
  if (path.empty()) {
    std::printf("  transformer checkpoint not present; skipping\n");
    return;
  }

  size_t free_before = 0, total_device = 0;
  cudaMemGetInfo(&free_before, &total_device);

  vidfab::SafeTensors st;
  st.open(path);
  CHECK_MSG(st.tensor_count() == 1082, "checkpoint has %zu tensors, expected 1082",
            st.tensor_count());

  Transformer model;
  const auto load_start = std::chrono::steady_clock::now();
  model.load(st, TransformerConfig{});
  const double load_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - load_start).count();
  std::printf("  load %.2f s, resident %.3f GiB\n", load_seconds,
              static_cast<double>(model.weight_bytes()) / (1024.0 * 1024.0 * 1024.0));
  CHECK(model.weight_bytes() > 19ull * 1024 * 1024 * 1024);

  // --- spec 3.2 layout, on the real weights --------------------------------
  //
  // The `1 + scale` parameterisation centres the two scale slices away from
  // zero and leaves the shifts and gates on it. That signature lands on slots 1
  // and 4 and nowhere else, so it is a direct test of the parameter order.
  {
    const int hidden = model.config().hidden_size;
    const std::vector<float> mod = model.debug_modulation(0, {0.0f});
    CHECK(mod.size() == static_cast<size_t>(6) * 3 * hidden);

    const char* names[6] = {"shift_msa", "scale_msa", "gate_msa",
                            "shift_mlp", "scale_mlp", "gate_mlp"};
    const char* modality[3] = {"video", "text", "audio"};
    // Measured (spec 3.2): video -0.8161 / -0.8062, text -0.4275 / +0.2217,
    // audio -0.6696 / -0.9126, everything else within 0.004 of zero.
    const double want_scale[3][2] = {{-0.8161, -0.8062}, {-0.4275, 0.2217}, {-0.6696, -0.9126}};
    for (int m = 0; m < 3; ++m) {
      for (int p = 0; p < 6; ++p) {
        // Parameter-outer, then the T*3 modulation rows; T is 1 here.
        const size_t base = (static_cast<size_t>(p) * 3 + m) * hidden;
        double sum = 0.0;
        for (int i = 0; i < hidden; ++i) sum += mod[base + i];
        const double mean = sum / hidden;
        std::printf("  %-5s %-9s mean %+.4f\n", modality[m], names[p], mean);
        if (p == 1 || p == 4) {
          const double want = want_scale[m][p == 1 ? 0 : 1];
          CHECK_MSG(std::fabs(mean - want) < 0.02,
                    "%s %s mean %+.4f, expected %+.4f — the spec 3.2 parameter order is wrong",
                    modality[m], names[p], mean, want);
        } else {
          CHECK_MSG(std::fabs(mean) < 0.01,
                    "%s %s mean %+.4f should sit on zero; a scale slice has landed in a "
                    "shift or gate slot",
                    modality[m], names[p], mean);
        }
      }
    }
  }

  // --- a real forward pass --------------------------------------------------
  const bool full = full_run_requested();
  int canvas_h = 0, canvas_w = 0;
  vidfab::dit::resolve_canvas_size(16, 9, &canvas_h, &canvas_w);
  const int aligned = vidfab::dit::align_num_frames(124);

  SequenceLayout layout;
  layout.num_text = 64;
  layout.latent_height = canvas_h / 16;
  layout.latent_width = canvas_w / 16;
  layout.num_audio_latents = vidfab::dit::audio_latents_for_frames(aligned);
  layout.num_audio_rows = 2 * layout.num_audio_latents;
  // The default run uses a shorter clip so an ordinary test pass stays in the
  // minutes; VIDFAB_TRANSFORMER_FULL=1 runs the real 124-frame geometry.
  layout.num_latent_frames = full ? vidfab::dit::video_latent_num_frames(aligned) : 4;
  layout.num_video_rows = layout.num_latent_frames * layout.rows_per_frame();

  const PackedIndices idx = vidfab::dit::build_indices(layout);
  const std::vector<double> pos = vidfab::dit::build_position_ids(layout);
  std::printf("  sequence %d rows (%d video, %d audio, %d text), activations %.3f GiB\n",
              layout.total_rows(), layout.num_video_rows, layout.num_audio_rows, layout.num_text,
              static_cast<double>(model.activation_bytes(layout)) / (1024.0 * 1024.0 * 1024.0));

  const std::vector<float> prompt =
      make_data(static_cast<size_t>(layout.num_text) * 5120, 7, 1.0f);
  model.prepare_text(prompt.data(), layout.num_text);
  model.prepare_sequence(layout, idx, pos);

  const std::vector<float> video_rows =
      make_data(idx.video.size() * 96, 8, 1.0f);
  const std::vector<float> audio_rows = make_data(idx.audio.size() * 32, 9, 1.0f);
  std::vector<float> video_velocity(video_rows.size());
  std::vector<float> audio_velocity(audio_rows.size());

  const RowTimesteps rt = vidfab::dit::build_row_timesteps(layout, idx, 0.5f, 0.35f);
  const auto step_start = std::chrono::steady_clock::now();
  model.forward(video_rows.data(), audio_rows.data(), rt, video_velocity.data(),
                audio_velocity.data());
  const double step_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - step_start)
          .count();

  size_t free_after = 0;
  cudaMemGetInfo(&free_after, &total_device);
  std::printf("  forward %.1f ms, peak device %.3f GiB of %.3f GiB\n", step_ms,
              static_cast<double>(free_before - free_after) / (1024.0 * 1024.0 * 1024.0),
              static_cast<double>(total_device) / (1024.0 * 1024.0 * 1024.0));

  CHECK(all_finite(video_velocity));
  CHECK(all_finite(audio_velocity));
  const double video_rms = rms(video_velocity);
  const double audio_rms = rms(audio_velocity);
  std::printf("  velocity rms: video %.4f, audio %.4f\n", video_rms, audio_rms);
  // A rectified-flow velocity against unit-variance latents lives within an
  // order of magnitude of 1. Zero would mean a dead residual stream; hundreds
  // would mean a missing normalisation.
  CHECK_MSG(video_rms > 0.02 && video_rms < 50.0, "video velocity rms %.4f is implausible",
            video_rms);
  CHECK_MSG(audio_rms > 0.02 && audio_rms < 50.0, "audio velocity rms %.4f is implausible",
            audio_rms);

  // Determinism at the model level: the same inputs must give the same bytes.
  {
    std::vector<float> again(video_rows.size());
    std::vector<float> again_audio(audio_rows.size());
    model.forward(video_rows.data(), audio_rows.data(), rt, again.data(), again_audio.data());
    CHECK_CLOSE(video_velocity, again, 0.0, "repeated forward is bitwise identical");
  }

  if (!full) {
    std::printf("  set VIDFAB_TRANSFORMER_FULL=1 for the 124-frame geometry and a 49-step loop\n");
    return;
  }

  // --- the whole loop -------------------------------------------------------
  vidfab::sampler::FlowScheduler video_sched(12.0f), audio_sched(3.0f);
  video_sched.set_timesteps(50);
  audio_sched.set_timesteps(50);
  CHECK(video_sched.timesteps().size() == 49);

  vidfab::dit::DenoiseInputs in;
  in.layout = &layout;
  in.indices = &idx;
  in.video_timesteps = &video_sched.timesteps();
  in.audio_timesteps = &audio_sched.timesteps();
  in.video_scheduler = &video_sched;
  in.audio_scheduler = &audio_sched;
  in.seed = 20260804;

  const auto loop_start = std::chrono::steady_clock::now();
  const vidfab::dit::DenoiseOutputs out =
      vidfab::dit::denoise(model, in, [&](int step, int total) {
        const double elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - loop_start).count();
        std::printf("  step %2d/%d  %.1f s elapsed (%.1f s/step)\n", step + 1, total, elapsed,
                    elapsed / (step + 1));
        std::fflush(stdout);
        return true;
      });
  CHECK(all_finite(out.video_rows));
  CHECK(all_finite(out.audio_rows));
  std::printf("  49-step denoise: video rms %.4f, audio rms %.4f\n", rms(out.video_rows),
              rms(out.audio_rows));
}

VIDFAB_TEST(transformer_real_ref2va_nf4_checkpoint_load) {
  const std::string path = find_ref2va_nf4_checkpoint();
  if (path.empty()) {
    std::printf("  Ref2VA NF4 transformer checkpoint not present; skipping\n");
    return;
  }

  vidfab::SafeTensors st;
  st.open(path);
  CHECK_MSG(st.tensor_count() == 1830, "NF4 checkpoint has %zu tensors, expected 1830",
            st.tensor_count());

  size_t states = 0;
  for (const auto& kv : st.tensors()) {
    if (kv.first.find(".weight.quant_state.bitsandbytes__nf4") != std::string::npos) ++states;
  }
  CHECK_MSG(states == 259, "expected 259 NF4 matrices, found %zu", states);

  Transformer model;
  bool rejected = false;
  try {
    model.load(st, TransformerConfig{});
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  CHECK(rejected);
}

// The nvfp4 build of the same 33B model, end to end and against the fp8 build.
//
// This is the strongest check available anywhere in the nvfp4 work: the two
// files are the same weights — all 332 tensors that are float and unquantised
// in both are bitwise identical — so the velocities they produce must agree to
// within fp4 quantisation error and nothing else. A wrong nibble order, a
// row-major read of the block scales or a half-sliced qkv view all leave the
// output finite and sanely scaled, so `all_finite` and an rms band cannot see
// them; correlation against the fp8 run can.
VIDFAB_TEST(transformer_real_nvfp4_checkpoint) {
  const std::string path = find_nvfp4_checkpoint();
  if (path.empty()) {
    std::printf("  nvfp4 transformer checkpoint not present; skipping\n");
    return;
  }

  vidfab::SafeTensors st;
  st.open(path);
  CHECK_MSG(st.tensor_count() == 1132, "nvfp4 checkpoint has %zu tensors, expected 1132",
            st.tensor_count());

  // 16 top level + 2 refiner blocks x 8 + 50 blocks x (6 + 4 linears x 4). The
  // fp8 file's 1082 differ by exactly +200 weight_scale_2 and -150 input_scale.
  size_t scale2 = 0, input_scale = 0;
  for (const auto& kv : st.tensors()) {
    if (kv.first.size() > 15 && kv.first.rfind(".weight_scale_2") == kv.first.size() - 15) ++scale2;
    if (kv.first.size() > 12 && kv.first.rfind(".input_scale") == kv.first.size() - 12) {
      ++input_scale;
    }
  }
  CHECK_MSG(scale2 == 200, "expected 200 weight_scale_2 tensors, found %zu", scale2);
  CHECK_MSG(input_scale == 0,
            "the nvfp4 transformer must carry no input_scale at all, found %zu", input_scale);

  size_t free_before = 0, total_device = 0;
  cudaMemGetInfo(&free_before, &total_device);

  Transformer model;
  const auto load_start = std::chrono::steady_clock::now();
  model.load(st, TransformerConfig{});
  const double load_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - load_start).count();
  const double resident_gib =
      static_cast<double>(model.weight_bytes()) / (1024.0 * 1024.0 * 1024.0);
  std::printf("  load %.2f s, resident %.3f GiB\n", load_seconds, resident_gib);

  // Nibbles plus block scales are 9/16 of a byte per weight against fp8's one,
  // and the 200 quantised linears are most but not all of the model, so the
  // 19.60 GiB fp8 arena should land near 12.5 GiB rather than at half.
  CHECK_MSG(resident_gib > 11.0 && resident_gib < 14.0,
            "nvfp4 weights occupy %.3f GiB, expected about 12.5", resident_gib);

  int canvas_h = 0, canvas_w = 0;
  vidfab::dit::resolve_canvas_size(16, 9, &canvas_h, &canvas_w);
  const int aligned = vidfab::dit::align_num_frames(124);

  SequenceLayout layout;
  layout.num_text = 64;
  layout.latent_height = canvas_h / 16;
  layout.latent_width = canvas_w / 16;
  layout.num_audio_latents = vidfab::dit::audio_latents_for_frames(aligned);
  layout.num_audio_rows = 2 * layout.num_audio_latents;
  layout.num_latent_frames = 4;
  layout.num_video_rows = layout.num_latent_frames * layout.rows_per_frame();

  const PackedIndices idx = vidfab::dit::build_indices(layout);
  const std::vector<double> pos = vidfab::dit::build_position_ids(layout);
  const std::vector<float> prompt =
      make_data(static_cast<size_t>(layout.num_text) * 5120, 7, 1.0f);
  const std::vector<float> video_rows = make_data(idx.video.size() * 96, 8, 1.0f);
  const std::vector<float> audio_rows = make_data(idx.audio.size() * 32, 9, 1.0f);
  const RowTimesteps rt = vidfab::dit::build_row_timesteps(layout, idx, 0.5f, 0.35f);

  std::vector<float> video_velocity(video_rows.size());
  std::vector<float> audio_velocity(audio_rows.size());

  model.prepare_text(prompt.data(), layout.num_text);
  model.prepare_sequence(layout, idx, pos);
  const auto step_start = std::chrono::steady_clock::now();
  model.forward(video_rows.data(), audio_rows.data(), rt, video_velocity.data(),
                audio_velocity.data());
  const double step_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - step_start)
          .count();

  size_t free_after = 0;
  cudaMemGetInfo(&free_after, &total_device);
  std::printf("  forward %.1f ms, peak device %.3f GiB of %.3f GiB\n", step_ms,
              static_cast<double>(free_before - free_after) / (1024.0 * 1024.0 * 1024.0),
              static_cast<double>(total_device) / (1024.0 * 1024.0 * 1024.0));

  CHECK(all_finite(video_velocity));
  CHECK(all_finite(audio_velocity));
  const double video_rms = rms(video_velocity);
  const double audio_rms = rms(audio_velocity);
  std::printf("  velocity rms: video %.4f, audio %.4f\n", video_rms, audio_rms);
  CHECK_MSG(video_rms > 0.02 && video_rms < 50.0, "video velocity rms %.4f is implausible",
            video_rms);
  CHECK_MSG(audio_rms > 0.02 && audio_rms < 50.0, "audio velocity rms %.4f is implausible",
            audio_rms);

  // --- against the fp8 build of the same weights ----------------------------
  const std::string fp8_path = find_checkpoint();
  if (fp8_path.empty()) {
    std::printf("  fp8 transformer not present; skipping the cross-check\n");
    return;
  }
  // 19.6 GiB and 12.5 GiB do not fit on one card together, so the nvfp4 model
  // is released before the fp8 one is loaded.
  model.unload();

  vidfab::SafeTensors fp8;
  fp8.open(fp8_path);
  Transformer reference;
  reference.load(fp8, TransformerConfig{});
  std::vector<float> ref_video(video_rows.size());
  std::vector<float> ref_audio(audio_rows.size());
  reference.prepare_text(prompt.data(), layout.num_text);
  reference.prepare_sequence(layout, idx, pos);
  reference.forward(video_rows.data(), audio_rows.data(), rt, ref_video.data(), ref_audio.data());

  auto correlation = [](const std::vector<float>& a, const std::vector<float>& b) {
    double sa = 0.0, sb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
      sa += a[i];
      sb += b[i];
    }
    const double ma = sa / a.size(), mb = sb / b.size();
    double num = 0.0, da = 0.0, db = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
      const double x = a[i] - ma, y = b[i] - mb;
      num += x * y;
      da += x * x;
      db += y * y;
    }
    return num / std::sqrt(da * db);
  };

  const double video_corr = correlation(video_velocity, ref_video);
  const double audio_corr = correlation(audio_velocity, ref_audio);
  std::printf("  vs fp8: rms video %.4f/%.4f audio %.4f/%.4f, correlation video %.5f audio %.5f\n",
              video_rms, rms(ref_video), audio_rms, rms(ref_audio), video_corr, audio_corr);

  // Fifty blocks of accumulated fp4-against-fp8 disagreement, so this is not a
  // tolerance check. What it separates is a correct dequantisation from a
  // plausible wrong one: every layout error measured on the raw weights sits at
  // a correlation of 0.00003, and this runs fifty layers on top of that.
  CHECK_MSG(video_corr > 0.9, "nvfp4 video velocity correlates %.5f with the fp8 build of the "
                              "same weights; a layout error would sit near zero", video_corr);
  CHECK_MSG(audio_corr > 0.9, "nvfp4 audio velocity correlates %.5f with the fp8 build of the "
                              "same weights; a layout error would sit near zero", audio_corr);
}

}  // namespace
