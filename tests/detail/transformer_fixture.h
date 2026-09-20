#pragma once

// Private shared fixtures for the transformer suites.
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
// on SLOPFAB_TRANSFORMER_FULL=1 so an ordinary test run stays minutes rather than
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

#include "../harness.h"
#include "slopfab/dit/adaln.h"
#include "slopfab/dit/denoise.h"
#include "slopfab/dit/packing.h"
#include "slopfab/dit/transformer.h"
#include "slopfab/cuda/deterministic_attention.cuh"
#include "slopfab/cuda/device.h"
#include "slopfab/dtype.h"
#include "slopfab/safetensors.h"
#include "slopfab/safetensors_write.h"
#include "slopfab/sampler/scheduler.h"


namespace {
namespace {

using slopfab::dit::AdaLNTable;
using slopfab::dit::PackedIndices;
using slopfab::dit::RowTimesteps;
using slopfab::dit::SequenceLayout;
using slopfab::dit::Transformer;
using slopfab::dit::TransformerConfig;
using slopfab::test::make_data;

using Tensors = std::map<std::string, slopfab::TensorWrite>;

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
  const char* v = std::getenv("SLOPFAB_TRANSFORMER_FULL");
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
float as_bf16(float v) { return slopfab::bf16_to_f32(slopfab::f32_to_bf16(v)); }

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
  t[name] = slopfab::TensorWrite{name, std::move(shape), std::move(data)};
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
  t["condition_proj.weight"].dtype = slopfab::DType::kBF16;
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

std::string write_synthetic(const Tensors& t,
    const std::map<std::string, std::string>& metadata = {}) {
  const std::filesystem::path dir =
      std::filesystem::temp_directory_path() / "slopfab_transformer_test";
  std::filesystem::create_directories(dir);
  const std::string path = (dir / "tiny.safetensors").string();
  std::vector<slopfab::TensorWrite> list;
  list.reserve(t.size());
  for (const auto& kv : t) list.push_back(kv.second);
  slopfab::write_safetensors(path, list, metadata);
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
  c.idx = slopfab::dit::build_indices(c.layout);
  c.pos = slopfab::dit::build_position_ids(c.layout);
  // `+ 1` keeps the buffer non-empty at L = 0; only L*text_dim is ever read.
  c.prompt = make_data(static_cast<size_t>(text_rows) * cfg.text_dim + 1, 9001, 1.0f);
  c.video_rows =
      make_data(c.idx.video.size() * static_cast<size_t>(cfg.video_patch_dim()), 9002, 1.0f);
  c.audio_rows =
      make_data(c.idx.audio.size() * static_cast<size_t>(cfg.audio_in_channels), 9003, 1.0f);
  // Text rows inherit the video timestep and are never overridden, so
  // `torch.unique` yields two entries unless the two schedules coincide.
  c.rt = slopfab::dit::build_row_timesteps(c.layout, c.idx, 0.62f, audio_t);
  return c;
}


// Bisect the refiner stage by stage.
//
// Comparing only the end of the refiner cannot tell "one operation is wrong"
// from "bf16 rounding accumulated across all of them", and those have very
// different consequences. This walks the same boundaries on both sides and
// prints where agreement is first lost, and by how much at each step.

// ---------------------------------------------------------------------------
// Denoising loop. Exercised with a substituted velocity so the arithmetic can
// be pinned without the checkpoint.

slopfab::dit::DenoiseInputs make_denoise_inputs(const SequenceLayout& layout,
                                               const PackedIndices& idx,
                                               slopfab::sampler::FlowScheduler& video,
                                               slopfab::sampler::FlowScheduler& audio) {
  slopfab::dit::DenoiseInputs in;
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

// The cache off must leave the loop's output untouched, elementwise, against
// the same loop built without a cache config at all. This is the unit-level
// half of the bit-identity gate: it needs no checkpoint and no card time, so a
// regression in the default path fails here long before a generation is run.

// ---------------------------------------------------------------------------
// Real checkpoint.

// Spec 8.1: the mean |w| of the contiguous [q; k; v] partition separates while
// the per-head interleaved partition is flat. This reads the file directly — no
// GPU, no 19.5 GiB load — so it can run even when the rest is skipped.

// The nvfp4 build of the same 33B model, end to end and against the fp8 build.
//
// This is the strongest check available anywhere in the nvfp4 work: the two
// files are the same weights — all 332 tensors that are float and unquantised
// in both are bitwise identical — so the velocities they produce must agree to
// within fp4 quantisation error and nothing else. A wrong nibble order, a
// row-major read of the block scales or a half-sliced qkv view all leave the
// output finite and sanely scaled, so `all_finite` and an rms band cannot see
// them; correlation against the fp8 run can.


}  // namespace

}  // namespace
