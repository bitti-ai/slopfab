#include "runtime.cuh"
#include "slopfab/cuda/gemm.cuh"
#include "slopfab/int8_weight.h"
#include "slopfab/tensor_convert.h"
#include "slopfab/text/encoder.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace slopfab::moss {
namespace {
__device__ float f(BFloat x) {
  return __bfloat162float(x);
}

__device__ float bf(float x) {
  return f(__float2bfloat16(x));
}

__device__ float silu(float x) {
  return x / (1.f + expf(-x));
}

__global__ void fused_bias(const float* x, const BFloat* bias, BFloat* y, int n, int cols) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = __float2bfloat16(x[i] + f(bias[i % cols]));
}

__device__ float sum_block(float x) {
  __shared__ float sums[32];
  for (int d = 16; d; d /= 2)
    x += __shfl_down_sync(0xffffffff, x, d);
  if ((threadIdx.x & 31) == 0)
    sums[threadIdx.x / 32] = x;
  __syncthreads();
  x = threadIdx.x < blockDim.x / 32 ? sums[threadIdx.x] : 0;
  if (threadIdx.x < 32)
    for (int d = 16; d; d /= 2)
      x += __shfl_down_sync(0xffffffff, x, d);
  if (threadIdx.x == 0)
    sums[0] = x;
  __syncthreads();
  return sums[0];
}

__global__ void norm_kernel(const BFloat* x, const BFloat* w, BFloat* y, int width, int cols,
                            bool offset, const BFloat* mod, int shift, const BFloat* gate) {
  const int row = blockIdx.x;
  float sum = 0;
  for (int i = threadIdx.x; i < width; i += blockDim.x) {
    float a = f(x[size_t(row) * width + i]);
    sum += a * a;
  }
  const float inv = rsqrtf(sum_block(sum) / width + 1e-6f);
  for (int i = threadIdx.x; i < width; i += blockDim.x) {
    const size_t index = size_t(row) * width + i;
    float a = f(x[index]) * inv;
    // Qwen3.5 offset RMSNorm multiplies in float before the final cast.
    a = offset ? bf(a * (1.f + f(w[i]))) : bf(bf(a) * f(w[i]));
    if (mod) {
      const int c = index % cols;
      a = bf(bf(a * bf(1.f + f(mod[(shift + 1) * cols + c]))) + f(mod[shift * cols + c]));
    }
    if (gate)
      a *= silu(f(gate[index]));
    y[index] = __float2bfloat16(a);
  }
}

__global__ void slice_kernel(const BFloat* x, BFloat* y, int rows, int source_cols, int offset,
                             int cols, int groups) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * cols)
    return;
  const int c = i % cols, group_width = cols / groups;
  y[i] = x[size_t(i / cols) * source_cols + (c / group_width) * (source_cols / groups) + offset +
           c % group_width];
}

__global__ void residual_kernel(BFloat* x, const BFloat* y, int n, int cols, const BFloat* mod,
                                int gate) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    x[i] =
        __float2bfloat16(f(x[i]) + (mod ? bf(f(y[i]) * f(mod[gate * cols + i % cols])) : f(y[i])));
}

__global__ void act_kernel(const BFloat* x, const BFloat* gate, BFloat* y, int n, bool situ,
                           bool sigmoid_gate) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  float a = f(x[i]);
  if (gate) {
    const float g = f(gate[i]);
    if (situ) {
      // Preserve the BF16 operations used by the PyTorch reference.
      float gg = bf(4.f * bf(tanhf(bf(g / 4.f))));
      gg = bf(gg * bf(1.f / (1.f + expf(-g))));
      a = bf(25.f * bf(tanhf(bf(a / 25.f)))) * gg;
    } else
      a *= bf(sigmoid_gate ? 1.f / (1.f + expf(-g)) : silu(g));
  } else
    a = silu(a);
  y[i] = __float2bfloat16(a);
}

__global__ void rope_kernel(BFloat* x, const BFloat* rope, int rows, int heads, int dim,
                            int rotary_dim) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int half = rotary_dim / 2;
  if (i >= rows * heads * half)
    return;
  const int j = i % half, h = (i / half) % heads, r = i / (half * heads);
  const int index = (r * heads + h) * dim + j;
  const float a = f(x[index]), b = f(x[index + half]);
  const float s = f(rope[r * rotary_dim + j]), c = f(rope[r * rotary_dim + half + j]);
  x[index] = __float2bfloat16(bf(a * c) - bf(b * s));
  x[index + half] = __float2bfloat16(bf(b * c) + bf(a * s));
}
}

Runtime::Runtime(const std::string& path) {
  checkpoint.open(path);
  SLOPFAB_CUBLAS_CHECK(cuda::cublas_create(&blas));
  linear.init(blas, nullptr);
}

Runtime::~Runtime() {
  if (blas)
    cuda::cublas_destroy(blas);
}

const void* Runtime::upload(const std::string& key, const void* data, size_t bytes) {
  auto it = storage_.find(key);
  if (it != storage_.end())
    return it->second.get();
  cuda::DeviceBuffer<uint8_t> buffer(bytes);
  SLOPFAB_CUDA_CHECK(cudaMemcpy(buffer.get(), data, bytes, cudaMemcpyHostToDevice));
  return storage_.emplace(key, std::move(buffer)).first->second.get();
}

const BFloat* Runtime::weight(const std::string& name) {
  auto it = storage_.find(name);
  if (it != storage_.end())
    return reinterpret_cast<const BFloat*>(it->second.get());
  const auto values = to_f32(checkpoint.at(name));
  std::vector<BFloat> bits(values.size());
  for (size_t i = 0; i < bits.size(); ++i)
    bits[i] = __float2bfloat16(values[i]);
  return static_cast<const BFloat*>(upload(name, bits.data(), bits.size() * 2));
}

const float* Runtime::float_weight(const std::string& name) {
  const auto key = name + ":f32";
  auto it = storage_.find(key);
  if (it != storage_.end())
    return reinterpret_cast<const float*>(it->second.get());
  const auto values = to_f32(checkpoint.at(name));
  return static_cast<const float*>(upload(key, values.data(), values.size() * 4));
}

Tensor Runtime::project(const Tensor& x, const std::string& name) {
  auto it = linears_.find(name);
  if (it == linears_.end()) {
    const auto& tensor = checkpoint.at(name + ".weight");
    if (tensor.shape.size() != 2 || tensor.shape[1] != x.cols)
      throw std::invalid_argument("linear shape mismatch: " + name);
    cuda::QuantWeight w;
    w.in_features = int(tensor.shape[1]);
    w.out_features = int(tensor.shape[0]);
    if (tensor.dtype == DType::kI8) {
      const auto state = read_int8_weight(checkpoint, name + ".weight", "CUDA model runtime");
      w.format = cuda::QuantFormat::kI8;
      w.data = upload(name + ":int8", tensor.data, tensor.nbytes);
      w.weight_scale = float_weight(name + ".weight_scale");
      w.per_channel_scale = true;
      w.convrot = state.rotation_group > 1;
      w.convrot_group = state.rotation_group;
    } else if (tensor.dtype == DType::kBF16 || tensor.dtype == DType::kF32 ||
               tensor.dtype == DType::kF16) {
      w.format = cuda::QuantFormat::kBF16;
      w.data = weight(name + ".weight");
    } else
      throw std::invalid_argument("unsupported model weight dtype: " + name);
    if (checkpoint.find(name + ".bias")) {
      if (checkpoint.at(name + ".bias").shape != std::vector<int64_t>{w.out_features})
        throw std::invalid_argument("linear bias shape mismatch: " + name);
      w.bias = weight(name + ".bias");
      w.bias_format = cuda::QuantFormat::kBF16;
    }
    it = linears_.emplace(name, w).first;
  }
  const auto& w = it->second;
  if (x.cols != w.in_features)
    throw std::invalid_argument("linear input mismatch: " + name);
  Tensor y(x.rows, w.out_features);
  scratch.reserve(cuda::linear_workspace_bytes(w, x.rows, cuda::ComputeType::kBF16));
  cuda::Workspace::Scope scope(scratch);
  if (w.has_bias()) {
    // torch.nn.Linear adds bias to the FP32 accumulator before BF16 rounding.
    // H3's existing runner deliberately uses its own post-rounding bias contract.
    const auto* dense = linear.prepare(w, scratch);
    const BFloat* input = x.ptr();
    if (w.convrot) {
      auto* rotated = scratch.alloc_n<BFloat>(size_t(x.rows) * x.cols);
      cuda::launch_convrot(input, rotated, x.rows, x.cols, w.convrot_group, nullptr);
      input = rotated;
    }
    cuda::DeviceBuffer<float> accumulator(size_t(x.rows) * w.out_features);
    const float alpha = 1.f, beta = 0.f;
    SLOPFAB_CUBLAS_CHECK(cuda::cublas_gemm_ex(
        blas, CUBLAS_OP_T, CUBLAS_OP_N, w.out_features, x.rows, w.in_features, &alpha, dense,
        CUDA_R_16BF, w.in_features, input, CUDA_R_16BF, w.in_features, &beta, accumulator.get(),
        CUDA_R_32F, w.out_features, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    const int n = x.rows * w.out_features;
    fused_bias<<<(n + 255) / 256, 256>>>(accumulator.get(), static_cast<const BFloat*>(w.bias),
                                         y.ptr(), n, w.out_features);
  } else
    linear.forward(w, x.ptr(), x.rows, y.ptr(), scratch);
  return y;
}

Tensor Runtime::norm(const Tensor& x, const std::string& name, int width, bool offset,
                     const Tensor* mod, int shift, const Tensor* gate) {
  if (!width)
    width = x.cols;
  if (width <= 0 || x.cols % width || checkpoint.at(name).numel() != width ||
      (mod && (shift < 0 || size_t(shift + 2) * x.cols > mod->data.size())) ||
      (gate && (gate->rows != x.rows || gate->cols != x.cols)))
    throw std::invalid_argument("normalization shape mismatch: " + name);
  Tensor y(x.rows, x.cols);
  norm_kernel<<<x.rows * x.cols / width, 256>>>(x.ptr(), weight(name), y.ptr(), width, x.cols,
                                                offset, mod ? mod->ptr() : nullptr, shift,
                                                gate ? gate->ptr() : nullptr);
  SLOPFAB_CUDA_CHECK(cudaGetLastError());
  return y;
}

Tensor Runtime::attend(const Tensor& q, const Tensor& k, const Tensor& v, int heads, int kv_heads,
                       bool causal) {
  if (q.rows != k.rows || k.rows != v.rows || k.cols != v.cols || q.cols % heads ||
      k.cols % kv_heads || q.cols / heads != k.cols / kv_heads)
    throw std::invalid_argument("attention tensor shape mismatch");
  if (causal) {
    text::CausalAttentionConfig config;
    config.seq_len = q.rows;
    config.num_heads = heads;
    config.num_kv_heads = kv_heads;
    config.head_dim = q.cols / heads;
    scratch.reserve(text::causal_attention_workspace_bytes(config));
    Tensor out(q.rows, q.cols);
    cuda::Workspace::Scope scope(scratch);
    text::causal_attention_forward(blas, nullptr, q.ptr(), k.ptr(), v.ptr(), out.ptr(), config,
                                   scratch);
    return out;
  }
  AttentionDescriptor desc;
  desc.query_tokens = q.rows;
  desc.key_value_tokens = k.rows;
  desc.query_heads = heads;
  desc.key_value_heads = kv_heads;
  desc.head_dim = q.cols / heads;
  desc.mask = causal ? AttentionMask::kCausal : AttentionMask::kFull;
  cuda::AttentionExecutionPolicy policy;
  const auto plan = cuda::AttentionPlan::compile(desc, policy);
  scratch.reserve(plan.workspace_bytes());
  Tensor out(q.rows, q.cols);
  cuda::Workspace::Scope scope(scratch);
  plan.forward(blas, nullptr, q.ptr(), k.ptr(), v.ptr(), out.ptr(), scratch);
  return out;
}

Tensor from_host(const std::vector<float>& values, int rows, int cols) {
  if (values.size() != size_t(rows) * cols)
    throw std::invalid_argument("tensor upload shape");
  Tensor result(rows, cols);
  std::vector<BFloat> bits(values.size());
  for (size_t i = 0; i < bits.size(); ++i)
    bits[i] = __float2bfloat16(values[i]);
  SLOPFAB_CUDA_CHECK(
      cudaMemcpy(result.ptr(), bits.data(), bits.size() * 2, cudaMemcpyHostToDevice));
  return result;
}

std::vector<float> to_host(const Tensor& x) {
  std::vector<BFloat> bits(size_t(x.rows) * x.cols);
  SLOPFAB_CUDA_CHECK(cudaMemcpy(bits.data(), x.ptr(), bits.size() * 2, cudaMemcpyDeviceToHost));
  std::vector<float> values(bits.size());
  for (size_t i = 0; i < bits.size(); ++i)
    values[i] = __bfloat162float(bits[i]);
  return values;
}

Tensor slice(const Tensor& x, int offset, int cols, int groups) {
  if (groups <= 0 || cols <= 0 || x.cols % groups || cols % groups || offset < 0 ||
      offset + cols / groups > x.cols / groups)
    throw std::invalid_argument("slice shape mismatch");
  Tensor out(x.rows, cols);
  slice_kernel<<<(x.rows * cols + 255) / 256, 256>>>(x.ptr(), out.ptr(), x.rows, x.cols, offset,
                                                     cols, groups);
  return out;
}

Tensor concatenate(const Tensor& a, const Tensor& b) {
  if (a.cols != b.cols)
    throw std::invalid_argument("concatenate width mismatch");
  Tensor out(a.rows + b.rows, a.cols);
  SLOPFAB_CUDA_CHECK(cudaMemcpy(out.ptr(), a.ptr(), a.data.nbytes(), cudaMemcpyDeviceToDevice));
  SLOPFAB_CUDA_CHECK(cudaMemcpy(out.ptr() + size_t(a.rows) * a.cols, b.ptr(), b.data.nbytes(),
                                cudaMemcpyDeviceToDevice));
  return out;
}

void residual(Tensor& x, const Tensor& update, const Tensor* mod, int gate) {
  if (x.rows != update.rows || x.cols != update.cols ||
      (mod && (gate < 0 || size_t(gate + 1) * x.cols > mod->data.size())))
    throw std::invalid_argument("residual shape mismatch");
  const int n = x.rows * x.cols;
  residual_kernel<<<(n + 255) / 256, 256>>>(x.ptr(), update.ptr(), n, x.cols,
                                            mod ? mod->ptr() : nullptr, gate);
}

Tensor activate(const Tensor& x, const Tensor* gate, bool situ, bool sigmoid_gate) {
  if (gate && (x.rows != gate->rows || x.cols != gate->cols))
    throw std::invalid_argument("activation gate shape mismatch");
  Tensor y(x.rows, x.cols);
  const int n = x.rows * x.cols;
  act_kernel<<<(n + 255) / 256, 256>>>(x.ptr(), gate ? gate->ptr() : nullptr, y.ptr(), n, situ,
                                       sigmoid_gate);
  return y;
}

void rotary(Tensor& x, int heads, const Tensor& rope, int dim) {
  const int n = x.rows * heads * dim / 2;
  rope_kernel<<<(n + 255) / 256, 256>>>(x.ptr(), rope.ptr(), x.rows, heads, x.cols / heads, dim);
}

Tensor image_rope(int height, int width, int length, bool text_only) {
  const int pixels = text_only ? 0 : height * width, rows = pixels + length;
  std::vector<float> table(size_t(rows) * 128);
  for (int r = 0; r < rows; ++r) {
    const float pos[] = {0.f, float(r < pixels ? length : r - pixels),
                         r < pixels ? r / width - (height - 1) * .5f : 0.f,
                         r < pixels ? r % width - (width - 1) * .5f : 0.f};
    for (int axis = 0; axis < 4; ++axis)
      for (int i = 0; i < 16; ++i) {
        float phase = pos[axis] / std::pow(10000.f, i / 16.f);
        table[r * 128 + axis * 16 + i] = std::sin(phase);
        table[r * 128 + 64 + axis * 16 + i] = std::cos(phase);
      }
  }
  return from_host(table, rows, 128);
}

Tensor text_rope(int length) {
  std::vector<float> table(size_t(length) * 64);
  for (int r = 0; r < length; ++r)
    for (int i = 0; i < 32; ++i) {
      float phase = r / std::pow(10000000.f, i / 32.f);
      table[r * 64 + i] = std::sin(phase);
      table[r * 64 + 32 + i] = std::cos(phase);
    }
  return from_host(table, length, 64);
}
} // namespace slopfab::moss
