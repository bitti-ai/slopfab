#include "runtime.cuh"
#include "slopfab/int8_weight.h"
#include "slopfab/tensor_convert.h"
#include <cmath>
#include <stdexcept>

namespace slopfab::moss {
namespace {
__device__ float f(BFloat x) {
  return __bfloat162float(x);
}

__global__ void causal_conv(const BFloat* x, const BFloat* w, BFloat* y, int rows) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * 8192)
    return;
  const int r = i / 8192, c = i % 8192;
  float value = 0;
  for (int k = 0; k < 4; ++k)
    if (r + k >= 3)
      value += f(x[(r + k - 3) * 8192 + c]) * f(w[c * 4 + k]);
  // The causal-conv implementation applies SiLU before casting the output.
  y[i] = __float2bfloat16(value / (1.f + expf(-value)));
}

__global__ void recurrent_delta(const BFloat* qkv, const BFloat* a, const BFloat* b,
                                const float* alog, const BFloat* bias, BFloat* out, int rows) {
  const int head = blockIdx.x, channel = threadIdx.x;
  __shared__ float q[128], k[128], reduction[128];
  float state[128] = {};
  for (int r = 0; r < rows; ++r) {
    const int kh = head / 2;
    q[channel] = f(qkv[r * 8192 + kh * 128 + channel]);
    k[channel] = f(qkv[r * 8192 + 2048 + kh * 128 + channel]);
    reduction[channel] = q[channel] * q[channel];
    __syncthreads();
    for (int d = 64; d; d /= 2) {
      if (channel < d)
        reduction[channel] += reduction[channel + d];
      __syncthreads();
    }
    const float qi = rsqrtf(reduction[0] + 1e-6f) / sqrtf(128.f);
    __syncthreads();
    reduction[channel] = k[channel] * k[channel];
    __syncthreads();
    for (int d = 64; d; d /= 2) {
      if (channel < d)
        reduction[channel] += reduction[channel + d];
      __syncthreads();
    }
    const float ki = rsqrtf(reduction[0] + 1e-6f);
    __syncthreads();
    q[channel] *= qi;
    k[channel] *= ki;
    __syncthreads();
    const float av = f(a[r * 32 + head]) + f(bias[head]);
    const float softplus = av > 20.f ? av : log1pf(expf(av));
    const float decay = expf(-expf(alog[head]) * softplus);
    const float beta = f(__float2bfloat16(1.f / (1.f + expf(-f(b[r * 32 + head])))));
    float memory = 0;
    for (int j = 0; j < 128; ++j) {
      state[j] *= decay;
      memory += state[j] * k[j];
    }
    const float delta = (f(qkv[r * 8192 + 4096 + head * 128 + channel]) - memory) * beta;
    float output = 0;
    for (int j = 0; j < 128; ++j) {
      state[j] += k[j] * delta;
      output += state[j] * q[j];
    }
    out[r * 4096 + head * 128 + channel] = __float2bfloat16(output);
    __syncthreads();
  }
}

Tensor embedding(Runtime& rt, const std::vector<int32_t>& ids) {
  const std::string name = "model.language_model.embed_tokens.weight";
  const auto& w = rt.checkpoint.at(name);
  if (w.shape != std::vector<int64_t>{248320, 4096})
    throw std::invalid_argument("MOSS requires the Qwen3.5-9B vocabulary and hidden width");
  std::vector<float> values(ids.size() * 4096);
  Int8WeightState quant;
  if (w.dtype == DType::kI8)
    quant = read_int8_weight(rt.checkpoint, name, "Qwen3.5 embedding");
  for (size_t r = 0; r < ids.size(); ++r) {
    if (ids[r] < 0 || ids[r] >= 248320)
      throw std::invalid_argument("Qwen3.5 token outside vocabulary");
    if (w.dtype != DType::kI8) {
      TensorView row = w;
      row.shape = {4096};
      row.nbytes = 4096 * dtype_size(w.dtype);
      row.data = static_cast<const uint8_t*>(w.data) + size_t(ids[r]) * row.nbytes;
      const auto decoded = to_f32(row);
      std::copy(decoded.begin(), decoded.end(), values.begin() + r * 4096);
      continue;
    }
    const int group = quant.rotation_group;
    for (int c = 0; c < 4096; c += group) {
      float* v = values.data() + r * 4096 + c;
      for (int i = 0; i < group; ++i)
        v[i] = quant.codes[size_t(ids[r]) * 4096 + c + i];
      for (int stride = 1; stride < group; stride *= 4)
        for (int base = 0; base < group; base += 4 * stride)
          for (int i = 0; i < stride; ++i) {
            const int j = base + i;
            const float a = v[j], b = v[j + stride], cc = v[j + 2 * stride], d = v[j + 3 * stride];
            v[j] = a + b + cc - d;
            v[j + stride] = a + b - cc + d;
            v[j + 2 * stride] = a - b + cc + d;
            v[j + 3 * stride] = -a + b + cc + d;
          }
      for (int i = 0; i < group; ++i)
        v[i] *= quant.scales[ids[r]] / std::sqrt(float(group));
    }
  }
  return from_host(values, int(ids.size()), 4096);
}
}

Tensor delta_attention(Runtime& rt, const Tensor& x, const std::string& p) {
  auto mixed = rt.project(x, p + "in_proj_qkv");
  Tensor convolved(x.rows, 8192);
  causal_conv<<<(x.rows * 8192 + 255) / 256, 256>>>(mixed.ptr(), rt.weight(p + "conv1d.weight"),
                                                    convolved.ptr(), x.rows);
  auto a = rt.project(x, p + "in_proj_a"), b = rt.project(x, p + "in_proj_b");
  Tensor out(x.rows, 4096);
  recurrent_delta<<<32, 128>>>(convolved.ptr(), a.ptr(), b.ptr(), rt.float_weight(p + "A_log"),
                               rt.weight(p + "dt_bias"), out.ptr(), x.rows);
  SLOPFAB_CUDA_CHECK(cudaGetLastError());
  auto z = rt.project(x, p + "in_proj_z");
  auto normalized = rt.norm(out, p + "norm.weight", 128, false, nullptr, 0, &z);
  return rt.project(normalized, p + "out_proj");
}

Tensor encode_text(Runtime& rt, const std::vector<int32_t>& ids) {
  if (ids.empty() || ids.size() > 4096)
    throw std::invalid_argument("MOSS prompt must contain 1..4096 tokens");
  auto x = embedding(rt, ids);
  auto rope = text_rope(x.rows);
  for (int layer = 0; layer < 32; ++layer) {
    const std::string p = "model.language_model.layers." + std::to_string(layer) + ".";
    auto y = rt.norm(x, p + "input_layernorm.weight", 0, true);
    Tensor update;
    if (layer % 4 != 3)
      update = delta_attention(rt, y, p + "linear_attn.");
    else {
      const std::string a = p + "self_attn.";
      auto qgate = rt.project(y, a + "q_proj");
      auto q = slice(qgate, 0, 4096, 16), gate = slice(qgate, 256, 4096, 16);
      q = rt.norm(q, a + "q_norm.weight", 256, true);
      auto k = rt.norm(rt.project(y, a + "k_proj"), a + "k_norm.weight", 256, true);
      auto v = rt.project(y, a + "v_proj");
      rotary(q, 16, rope, 64);
      rotary(k, 4, rope, 64);
      auto attended = rt.attend(q, k, v, 16, 4, true);
      update = rt.project(activate(attended, &gate, false, true), a + "o_proj");
    }
    residual(x, update);
    y = rt.norm(x, p + "post_attention_layernorm.weight", 0, true);
    auto gate = rt.project(y, p + "mlp.gate_proj");
    auto up = rt.project(y, p + "mlp.up_proj");
    update = rt.project(activate(up, &gate), p + "mlp.down_proj");
    residual(x, update);
  }
  return rt.norm(x, "model.language_model.norm.weight", 0, true);
}
} // namespace slopfab::moss
