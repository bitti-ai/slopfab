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
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "harness.h"
#include "vidfab/dit/adaln.h"
#include "vidfab/dit/denoise.h"
#include "vidfab/dit/packing.h"
#include "vidfab/dit/transformer.h"
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

std::string find_checkpoint() {
  for (const char* prefix : {"", "../", "../../", "../../../"}) {
    const std::string p =
        std::string(prefix) + "weights/transformer/fl2va_pruned_fp8_scaled.safetensors";
    if (std::filesystem::exists(p)) return p;
  }
  return {};
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

  // --- text stream: context_embedder, then the token refiner ----------------
  //
  // `context_embedder` runs in the block dtype, so the fp32 conditioning
  // embedding is narrowed first.
  std::vector<float> text =
      matmul_nt(rounded(prompt), rounded(at(t, "condition_proj.weight")),
                &at(t, "condition_proj.bias"), L, hidden, cfg.text_dim);
  round_bf16(text);
  for (int i = 0; i < cfg.num_refiner_layers; ++i) {
    RefBlock blk{"token_refiner.blocks." + std::to_string(i) + ".", false};
    run_block(t, cfg, blk, text, L, nullptr, nullptr, nullptr, nullptr);
  }
  text = rmsnorm(text, at(t, "token_refiner.final_norm.weight"), L, hidden, cfg.norm_eps);
  round_bf16(text);

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

// ---------------------------------------------------------------------------

VIDFAB_TEST(transformer_forward_vs_cpu_reference) {
  const TransformerConfig cfg = tiny_config();
  const SequenceLayout layout = tiny_layout();
  const Tensors tensors = build_synthetic(cfg);
  const std::string path = write_synthetic(tensors);

  vidfab::SafeTensors st;
  st.open(path);

  AdaLNTable table;
  table.load(st);

  Transformer model;
  model.load(st, cfg);
  CHECK(model.weight_bytes() > 0);

  const PackedIndices idx = vidfab::dit::build_indices(layout);
  const std::vector<double> pos = vidfab::dit::build_position_ids(layout);

  const std::vector<float> prompt =
      make_data(static_cast<size_t>(layout.num_text) * cfg.text_dim, 9001, 1.0f);
  const std::vector<float> video_rows = make_data(
      idx.video.size() * static_cast<size_t>(cfg.video_patch_dim()), 9002, 1.0f);
  const std::vector<float> audio_rows =
      make_data(idx.audio.size() * static_cast<size_t>(cfg.audio_in_channels), 9003, 1.0f);

  // Two distinct timesteps, which is the t2va norm: video and text share t_v,
  // audio gets t_a, and torch.unique sorts them ascending.
  const RowTimesteps rt = vidfab::dit::build_row_timesteps(layout, idx, 0.62f, 0.31f);
  CHECK(rt.unique.size() == 2);
  CHECK(rt.unique[0] < rt.unique[1]);

  model.prepare_text(prompt.data(), layout.num_text);
  model.prepare_sequence(layout, idx, pos);

  std::vector<float> video_velocity(video_rows.size());
  std::vector<float> audio_velocity(audio_rows.size());
  model.forward(video_rows.data(), audio_rows.data(), rt, video_velocity.data(),
                audio_velocity.data());

  const RefOutputs want =
      reference_forward(tensors, table, cfg, layout, idx, pos, prompt, video_rows, audio_rows, rt);

  CHECK(want.video.size() == video_velocity.size());
  CHECK(want.audio.size() == audio_velocity.size());
  CHECK(all_finite(video_velocity));
  CHECK(all_finite(audio_velocity));
  {
    double maxe = 0, sume = 0;
    for (size_t i = 0; i < want.video.size(); ++i) {
      const double e = std::fabs(static_cast<double>(want.video[i]) - video_velocity[i]);
      maxe = std::max(maxe, e);
      sume += e;
    }
    std::printf("  DIAG video rms=%.4f max_err=%.4e mean_err=%.4e (%.3f%% of rms)\n",
                rms(want.video), maxe, sume / want.video.size(), 100.0 * maxe / rms(want.video));
  }
  // The block stack runs bf16; the reference runs fp64-accumulated fp32. Per
  // spec, correctness is 1e-3 absolute or 1e-2 relative elementwise.
  CHECK_CLOSE_REL(want.video, video_velocity, 1e-3, 1e-2, "video velocity vs CPU reference");
  CHECK_CLOSE_REL(want.audio, audio_velocity, 1e-3, 1e-2, "audio velocity vs CPU reference");

  // The reference is only worth something if it can tell the right answer from
  // the plausible wrong ones. Swapping the AdaLN scale and gate slots — one of
  // the ways spec 3.2 can be misread — must move the output well outside
  // tolerance.
  {
    Tensors mutated = tensors;
    const int hidden = cfg.hidden_size;
    for (int b = 0; b < cfg.num_layers; ++b) {
      std::vector<float>& bias =
          mutated["blocks." + std::to_string(b) + ".adaln_proj.linear.bias"].data;
      for (int modality = 0; modality < 3; ++modality) {
        for (int i = 0; i < hidden; ++i) {
          const size_t g = static_cast<size_t>(modality) * 6 * hidden + 2 * hidden + i;
          const size_t s = static_cast<size_t>(modality) * 6 * hidden + 1 * hidden + i;
          std::swap(bias[g], bias[s]);
        }
      }
    }
    const RefOutputs wrong = reference_forward(mutated, table, cfg, layout, idx, pos, prompt,
                                               video_rows, audio_rows, rt);
    double worst = 0.0;
    for (size_t i = 0; i < wrong.video.size(); ++i) {
      worst = std::max(worst, std::fabs(static_cast<double>(wrong.video[i]) - want.video[i]));
    }
    CHECK_MSG(worst > 1e-2,
              "swapping scale_msa and gate_msa changed the output by only %.3e, so the test "
              "cannot detect a wrong spec 3.2 parameter order",
              worst);
  }

  st.close();
  std::error_code ec;
  std::filesystem::remove(path, ec);
}

VIDFAB_TEST(zzz_ablation_diagnostic) {
  const TransformerConfig cfg = tiny_config();
  const SequenceLayout layout = tiny_layout();
  const PackedIndices idx = vidfab::dit::build_indices(layout);
  const std::vector<double> pos = vidfab::dit::build_position_ids(layout);
  const std::vector<float> prompt =
      make_data(static_cast<size_t>(layout.num_text) * cfg.text_dim, 9001, 1.0f);
  const std::vector<float> video_rows = make_data(
      idx.video.size() * static_cast<size_t>(cfg.video_patch_dim()), 9002, 1.0f);
  const std::vector<float> audio_rows =
      make_data(idx.audio.size() * static_cast<size_t>(cfg.audio_in_channels), 9003, 1.0f);
  const RowTimesteps rt = vidfab::dit::build_row_timesteps(layout, idx, 0.62f, 0.31f);

  auto trial = [&](const char* name, bool msa, bool mlp, int text_rows = -1,
                   bool zero_pos = false, bool kill_refiner_attn = false,
                   bool kill_refiner_ffn = false) {
    SequenceLayout lay = layout;
    if (text_rows >= 0) lay.num_text = text_rows;
    const PackedIndices idx2 = vidfab::dit::build_indices(lay);
    std::vector<double> pos2 = vidfab::dit::build_position_ids(lay);
    if (zero_pos) std::fill(pos2.begin(), pos2.end(), 0.0);
    const std::vector<float> prompt2 =
        make_data(static_cast<size_t>(lay.num_text) * cfg.text_dim + 1, 9001, 1.0f);
    const RowTimesteps rt2 = vidfab::dit::build_row_timesteps(lay, idx2, 0.62f, 0.31f);
    Tensors tensors = build_synthetic(cfg);
    const int hidden = cfg.hidden_size;
    for (int b = 0; b < cfg.num_layers; ++b) {
      std::vector<float>& bias =
          tensors["blocks." + std::to_string(b) + ".adaln_proj.linear.bias"].data;
      std::vector<float>& w =
          tensors["blocks." + std::to_string(b) + ".adaln_proj.linear.weight"].data;
      for (int m = 0; m < 3; ++m) {
        for (int i = 0; i < hidden; ++i) {
          if (!msa) {
            const size_t p = static_cast<size_t>(m) * 6 * hidden + 2 * hidden + i;
            bias[p] = 0.0f;
            for (int r = 0; r < 8; ++r) w[p * 8 + r] = 0.0f;
          }
          if (!mlp) {
            const size_t p = static_cast<size_t>(m) * 6 * hidden + 5 * hidden + i;
            bias[p] = 0.0f;
            for (int r = 0; r < 8; ++r) w[p * 8 + r] = 0.0f;
          }
        }
      }
    }
    const std::string path = write_synthetic(tensors);
    vidfab::SafeTensors st;
    st.open(path);
    AdaLNTable table;
    table.load(st);
    Transformer model;
    model.load(st, cfg);
    model.prepare_text(prompt2.data(), lay.num_text);
    model.prepare_sequence(lay, idx2, pos2);
    std::vector<float> vv(video_rows.size()), av(audio_rows.size());
    model.forward(video_rows.data(), audio_rows.data(), rt2, vv.data(), av.data());
    const RefOutputs want = reference_forward(tensors, table, cfg, lay, idx2, pos2, prompt2,
                                              video_rows, audio_rows, rt2);
    double maxe = 0;
    for (size_t i = 0; i < vv.size(); ++i) {
      maxe = std::max(maxe, std::fabs(static_cast<double>(vv[i]) - want.video[i]));
    }
    double maxa = 0;
    for (size_t i = 0; i < av.size(); ++i) {
      maxa = std::max(maxa, std::fabs(static_cast<double>(av[i]) - want.audio[i]));
    }
    std::printf("  ABLATE %-14s msa=%d mlp=%d  video rms %.4f max %.3e | audio max %.3e\n", name,
                msa ? 1 : 0, mlp ? 1 : 0, rms(want.video), maxe, maxa);
    st.close();
    std::error_code ec;
    std::filesystem::remove(path, ec);
  };

  trial("io-only", false, false);
  trial("attention", true, false);
  trial("attn-notext", true, false, 0);
  trial("attn-zeropos", true, false, -1, true);
  trial("attn-notext-zp", true, false, 0, true);
  trial("ffn", false, true);
  trial("full", true, true);
  CHECK(true);
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
                                               const vidfab::sampler::FlowScheduler& video,
                                               const vidfab::sampler::FlowScheduler& audio) {
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

  // The two shifted grids differ, so the two timesteps in the unique set must
  // differ at every step: one scheduler driving both would collapse them.
  int distinct = 0;
  for (size_t i = 0; i < seen_video_t.size(); ++i) {
    if (seen_video_t[i] != seen_audio_t[i]) ++distinct;
  }
  CHECK_MSG(distinct == static_cast<int>(seen_video_t.size()),
            "only %d of %zu steps had two distinct timesteps; the video and audio schedules are "
            "not being advanced independently",
            distinct, seen_video_t.size());
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
    CHECK(i_spread < 0.05);
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

}  // namespace
