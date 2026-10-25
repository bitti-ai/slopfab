#define SLOPFAB_MUSIC_VULKAN 1
#include "runtime.cuh"
#include "slopfab/int8_weight.h"
#include "slopfab/tensor_convert.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace slopfab::music_gpu {
namespace {
thread_local Runtime* current = nullptr;

vk::Device make_device(vk::Instance& instance) {
  auto devices = instance.enumerate_devices();
  if (devices.empty())
    throw std::runtime_error("Vulkan music: no compute device");
  // Prefer the device with the largest local heap.
  auto selected = devices.begin();
  for (auto it = devices.begin(); it != devices.end(); ++it)
    if (it->info().device_local_bytes() > selected->info().device_local_bytes())
      selected = it;
  vk::DeviceOptions options;
  options.portable_arithmetic = true;
  options.enable_storage_buffer_16bit = true;
  options.enable_timeline_semaphore = true;
  options.enable_shader_float16 = selected->info().shader_float16;
  options.enable_shader_int64 = selected->info().shader_int64;
  options.enable_cooperative_matrix = selected->info().cooperative_matrix;
  return selected->create_device(options);
}

vk::TensorContextOptions context_options() {
  vk::TensorContextOptions o;
  o.max_batch_operators = 128;
  o.pipeline_sets =
      vk::TensorPipelineSet::kCore | vk::TensorPipelineSet::kAudio | vk::TensorPipelineSet::kMusic;
  return o;
}

uint16_t bits(float v) {
  uint32_t u;
  std::memcpy(&u, &v, 4);
  return uint16_t((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}
}

float round_bf(float v) {
  uint32_t u = uint32_t(bits(v)) << 16;
  std::memcpy(&v, &u, 4);
  return v;
}

Engine::Engine()
    : instance(vk::Instance::create()), device(make_device(instance)),
      context(device, context_options()) {
}

Engine::~Engine() {
  try {
    flush();
    device.wait_idle();
  } catch (...) {
  }
}

vk::TensorBatch& Engine::batch() {
  // Complete each operation before submitting its consumers. Batched music
  // graphs showed intermittent corruption during repeated seeded generation;
  // keep the conservative submission path until batching is validated.
  if (pending && pending.remaining_operator_capacity() < 128)
    flush();
  if (!pending)
    pending = context.begin_batch();
  return pending;
}

void Engine::flush() {
  if (pending) {
    if (pending.remaining_operator_capacity() == 128) {
      pending = {};
      return;
    }
    auto done = pending.submit();
    pending = {};
    done.wait();
  }
}

vk::DeviceTensor Engine::allocate(int rows, int cols, ScalarType type) {
  if (rows <= 0 || cols <= 0)
    throw std::invalid_argument("Vulkan music: invalid tensor shape");
  uint64_t shape[] = {uint64_t(rows), uint64_t(cols)};
  return pending ? pending.allocate(TensorLayout::contiguous(shape, 2), type)
                 : context.allocate(TensorLayout::contiguous(shape, 2), type);
}

vk::DeviceTensor Engine::upload(const std::vector<float>& values, int rows, int cols, bool fp32,
                                bool vector) {
  if (values.size() != size_t(rows) * cols)
    throw std::invalid_argument("Vulkan music: upload shape");
  flush();
  uint64_t count = values.size();
  auto result = vector ? context.allocate(TensorLayout::contiguous(&count, 1),
                                          fp32 ? ScalarType::kFloat32 : ScalarType::kBFloat16)
                       : allocate(rows, cols, fp32 ? ScalarType::kFloat32 : ScalarType::kBFloat16);
  if (fp32)
    context.upload_transient(result, values.data(), values.size());
  else {
    std::vector<uint16_t> v(values.size());
    for (size_t i = 0; i < v.size(); ++i)
      v[i] = bits(values[i]);
    context.upload_transient_bytes(result, v.data(), v.size() * 2);
  }
  return result;
}

void Engine::op(vk::DeviceTensor& x, vk::DeviceTensor& w, vk::DeviceTensor& b, vk::DeviceTensor& y,
                std::initializer_list<uint32_t> params, uint32_t groups) {
  uint32_t p[16] = {};
  std::copy(params.begin(), params.end(), p);
  batch().music_operation(x, w, b, y, p, groups);
}

Tensor::Tensor(int r, int c) : rows(r), cols(c) {
  if (!current)
    throw std::logic_error("Vulkan music: tensor outside runtime scope");
  engine = current->engine;
  data = engine->allocate(r, c);
}

Runtime::Runtime(const std::string& path) {
  checkpoint.open(path);
  previous_ = current;
  engine = current ? current->engine : std::make_shared<Engine>();
  current = this;
}

Runtime::~Runtime() {
  try {
    engine->flush();
  } catch (...) {
  }
  current = previous_;
}

vk::DeviceTensor& Runtime::weight(const std::string& name) {
  auto it = weights_.find(name);
  if (it != weights_.end())
    return it->second;
  auto v = to_f32(checkpoint.at(name));
  return weights_.emplace(name, engine->upload(v, 1, int(v.size()))).first->second;
}

vk::DeviceTensor& Runtime::float_weight(const std::string& name) {
  auto key = name + ":f32";
  auto it = weights_.find(key);
  if (it != weights_.end())
    return it->second;
  auto v = to_f32(checkpoint.at(name));
  return weights_.emplace(key, engine->upload(v, 1, int(v.size()), true)).first->second;
}

Tensor Runtime::project(const Tensor& x, const std::string& name, bool fused_bias) {
  const auto& view = checkpoint.at(name + ".weight");
  if (view.shape.size() < 2 || view.numel() / view.shape[0] != x.cols)
    throw std::invalid_argument("Vulkan music linear shape: " + name);
  int out = int(view.shape[0]);
  auto it = matrices_.find(name);
  if (it == matrices_.end()) {
    Matrix m;
    if (view.dtype != DType::kI8 && view.dtype != DType::kBF16 && view.dtype != DType::kF16 &&
        view.dtype != DType::kF32)
      throw std::invalid_argument("Vulkan music unsupported weight dtype: " + name);
    Int8WeightState quant;
    if (view.dtype == DType::kI8) {
      quant = read_int8_weight(checkpoint, name + ".weight", "Vulkan music");
      m.group = quant.rotation_group;
    }
    auto bias_view = checkpoint.find(name + ".bias");
    std::vector<float> bias;
    if (bias_view) {
      bias = to_f32(*bias_view);
      if (bias.size() != size_t(out))
        throw std::invalid_argument("Vulkan music linear bias");
    }
    // Tile large vocabulary matrices below the portable storage-buffer limit.
    int tile = int(std::min<uint64_t>(64ull << 20, engine->device.info().max_storage_buffer_bytes) /
                   (2 * x.cols));
    tile = std::max(1, (tile / 16) * 16);
    for (int first = 0; first < out; first += tile) {
      int count = std::min(tile, out - first);
      std::vector<float> values(size_t(count) * x.cols);
      if (view.dtype == DType::kI8) {
        for (int r = 0; r < count; ++r)
          for (int c = 0; c < x.cols; ++c)
            values[size_t(r) * x.cols + c] =
                float(quant.codes[size_t(first + r) * x.cols + c]) * quant.scales[first + r];
      } else {
        auto v = view;
        v.shape = {count, x.cols};
        v.nbytes = values.size() * dtype_size(v.dtype);
        v.data =
            static_cast<const uint8_t*>(view.data) + size_t(first) * x.cols * dtype_size(v.dtype);
        values = to_f32(v);
      }
      m.tiles.push_back(engine->upload(values, count, x.cols));
      if (bias_view)
        m.biases.push_back(
            engine->upload(std::vector<float>(bias.begin() + first, bias.begin() + first + count),
                           1, count, false, true));
    }
    it = matrices_.emplace(name, std::move(m)).first;
  }
  auto& m = it->second;
  Tensor rotated;
  auto* input = &x.data;
  if (m.group > 1) {
    if (m.group > 256 || x.cols % m.group)
      throw std::invalid_argument("Vulkan music ConvRot group");
    rotated = Tensor(x.rows, x.cols);
    uint32_t groups = x.rows * x.cols / m.group;
    engine->op(x.data, x.data, x.data, rotated.data, {10, groups, uint32_t(m.group)}, groups);
    input = &rotated.data;
  }
  // Full tiles select cooperative GEMM where available. Preserve its
  // accumulation order for autoregressive models; reduction-based matvecs
  // can amplify small rounding differences through YuE's later layers.
  const auto& info = engine->device.info();
  bool cooperative = info.cooperative_matrix_enabled && info.shader_bfloat16_type &&
                     info.shader_bfloat16_cooperative_matrix &&
                     info.cooperative_matrix_bf16_f32_16x16x16 && info.subgroup_size == 32;
  int gemm_rows = cooperative ? ((x.rows + 63) / 64) * 64 : x.rows;
  Tensor padded;
  if (gemm_rows != x.rows) {
    padded = Tensor(gemm_rows, x.cols);
    music_vulkan::gpu_zero(padded);
    engine->batch().copy_rows(*input, padded.data, 0, 0, x.rows);
    input = &padded.data;
  }
  Tensor output(x.rows, out);
  int first = 0;
  for (size_t j = 0; j < m.tiles.size(); ++j) {
    auto& w = m.tiles[j];
    int count = int(w.layout().extent[0]);
    Tensor piece(gemm_rows, count);
    auto* bias = m.biases.empty() ? nullptr : &m.biases[j];
    DenseGemmPlanDesc d;
    d.max_rows = gemm_rows;
    d.out_features = count;
    d.in_features = x.cols;
    d.bias = bias ? (fused_bias ? DenseGemmBias::kBFloat16Fused : DenseGemmBias::kBFloat16)
                  : DenseGemmBias::kNone;
    auto plan = vk::DenseGemmPlan::create(engine->context, d);
    plan.record(engine->batch(), *input, w, piece.data, gemm_rows, 0, 0, bias);
    piece.rows = x.rows;
    music_vulkan::gpu_copy_columns(piece, output, first);
    first += count;
  }
  if (!cache_dense_weights)
    matrices_.erase(it);
  return output;
}

Tensor Runtime::norm(const Tensor& x, const std::string& name, int width) {
  if (!width)
    width = x.cols;
  if (x.cols % width || checkpoint.at(name).numel() != width)
    throw std::invalid_argument("Vulkan music norm shape");
  Tensor out(x.rows, x.cols);
  auto& w = weight(name);
  uint32_t rows = x.rows * x.cols / width;
  engine->op(x.data, w, w, out.data, {7, rows, uint32_t(width)}, rows);
  return out;
}

Tensor Runtime::attend(const Tensor& q, const Tensor& k, const Tensor& v, int heads, int kv_heads,
                       bool causal, bool fp32) {
  if (heads < 1 || kv_heads < 1 || heads % kv_heads || q.cols % heads || k.cols % kv_heads ||
      q.cols / heads != k.cols / kv_heads || k.rows != v.rows || k.cols != v.cols ||
      (causal && q.rows != k.rows))
    throw std::invalid_argument("Vulkan music attention shape");
  Tensor out(q.rows, q.cols);
  int dim = q.cols / heads;
  int tile = std::min<int>(
      32, int(engine->device.info().max_storage_buffer_bytes / (uint64_t(heads) * k.rows * 4)));
  if (tile < 1)
    throw std::invalid_argument("Vulkan music attention workspace exceeds device limit");
  for (int first = 0; first < q.rows; first += tile) {
    int count = std::min(tile, q.rows - first);
    uint32_t n = heads * count * k.rows;
    auto scores = engine->allocate(heads * count, k.rows, ScalarType::kFloat32);
    auto probs = engine->allocate(heads * count, k.rows, ScalarType::kFloat32);
    engine->op(q.data, k.data, k.data, scores,
               {11, n, uint32_t(dim), uint32_t(k.rows), uint32_t(count), uint32_t(first),
                uint32_t(heads), uint32_t(kv_heads), uint32_t(causal)},
               (n + 255) / 256);
    engine->op(scores, scores, scores, probs,
               {12, uint32_t(heads * count), uint32_t(k.rows), uint32_t(fp32)}, heads * count);
    engine->op(probs, v.data, v.data, out.data,
               {13, uint32_t(heads * count * dim), uint32_t(dim), uint32_t(k.rows), uint32_t(count),
                uint32_t(first), uint32_t(heads), uint32_t(kv_heads)},
               (heads * count * dim + 255) / 256);
  }
  return out;
}

void Runtime::release_prefix(const std::string& prefix) {
  engine->flush();
  for (auto it = matrices_.begin(); it != matrices_.end();)
    if (it->first.rfind(prefix, 0) == 0)
      it = matrices_.erase(it);
    else
      ++it;
  for (auto it = weights_.begin(); it != weights_.end();)
    if (it->first.rfind(prefix, 0) == 0)
      it = weights_.erase(it);
    else
      ++it;
}

Tensor from_host(const std::vector<float>& v, int r, int c) {
  Tensor x(r, c);
  x.data = x.engine->upload(v, r, c);
  return x;
}

std::vector<float> to_host(const Tensor& x) {
  x.engine->flush();
  std::vector<uint16_t> bits(x.data.layout().elements());
  x.engine->context.download_bytes(x.data, bits.data(), bits.size() * 2);
  std::vector<float> values(size_t(x.rows) * x.cols);
  for (size_t i = 0; i < values.size(); ++i) {
    uint32_t u = uint32_t(bits[i]) << 16;
    std::memcpy(&values[i], &u, 4);
  }
  return values;
}

Tensor slice(const Tensor& x, int offset, int cols, int groups) {
  if (groups < 1 || cols < 1 || cols % groups || x.cols % groups || offset < 0 ||
      offset + cols / groups > x.cols / groups)
    throw std::invalid_argument("Vulkan music slice shape");
  Tensor y(x.rows, cols);
  uint32_t n = x.rows * cols;
  x.engine->op(x.data, x.data, x.data, y.data,
               {0, n, uint32_t(cols), uint32_t(x.cols), uint32_t(offset), uint32_t(groups)},
               (n + 255) / 256);
  return y;
}

Tensor concatenate(const Tensor& a, const Tensor& b) {
  if (a.cols != b.cols)
    throw std::invalid_argument("Vulkan music concatenate shape");
  Tensor y(a.rows + b.rows, a.cols);
  music_vulkan::gpu_copy_rows(a, y, 0, 0, a.rows);
  music_vulkan::gpu_copy_rows(b, y, 0, a.rows, b.rows);
  return y;
}

void residual(Tensor& x, const Tensor& v) {
  if (x.rows != v.rows || x.cols != v.cols)
    throw std::invalid_argument("Vulkan music residual shape");
  uint32_t n = x.rows * x.cols;
  x.engine->op(x.data, v.data, v.data, x.data, {2, n}, (n + 255) / 256);
}

Tensor activate(const Tensor& x, const Tensor* gate) {
  if (gate && (gate->rows != x.rows || gate->cols != x.cols))
    throw std::invalid_argument("Vulkan music activation shape");
  Tensor y(x.rows, x.cols);
  uint32_t n = x.rows * x.cols;
  auto& w = gate ? gate->data : x.data;
  x.engine->op(x.data, w, w, y.data, {3, n, 0, uint32_t(gate != nullptr)}, (n + 255) / 256);
  return y;
}

void rotary(Tensor& x, int heads, const Tensor& pos, int dim) {
  if (heads < 1 || x.cols % heads || dim % 2 || dim > x.cols / heads || pos.rows != x.rows ||
      pos.cols != dim)
    throw std::invalid_argument("Vulkan music rotary shape");
  uint32_t n = x.rows * heads * dim / 2;
  x.engine->op(x.data, pos.data, pos.data, x.data,
               {4, n, uint32_t(x.cols / heads), uint32_t(heads), uint32_t(dim)}, (n + 255) / 256);
}
}

namespace slopfab::music_vulkan {
float gpu_bf(float v) {
  return music_gpu::round_bf(v);
}

void gpu_copy_rows(const Tensor& x, Tensor& y, int first, int dest, int count) {
  if (first < 0 || dest < 0 || count < 1 || first + count > x.rows ||
      uint64_t(dest + count) > y.data.layout().extent[0] || x.cols != y.cols)
    throw std::invalid_argument("Vulkan music row copy");
  x.engine->batch().copy_rows(x.data, y.data, first, dest, count);
}

void gpu_copy_columns(const Tensor& x, Tensor& y, int offset) {
  if (x.rows != y.rows || offset < 0 || offset + x.cols > y.cols)
    throw std::invalid_argument("Vulkan music column copy");
  uint32_t n = x.rows * x.cols;
  x.engine->op(x.data, x.data, x.data, y.data,
               {1, n, uint32_t(x.cols), uint32_t(y.cols), uint32_t(offset)}, (n + 255) / 256);
}

void gpu_zero(Tensor& x) {
  uint32_t n = x.rows * x.cols;
  x.engine->op(x.data, x.data, x.data, x.data, {6, n}, (n + 255) / 256);
}

void gpu_condition(Runtime& rt, Tensor& x, const Tensor& t, const std::string& name) {
  auto& pe = rt.weight(name);
  uint32_t n = x.rows * x.cols;
  if (t.rows != 1 || t.cols != x.cols || pe.layout().elements() < n)
    throw std::invalid_argument("Vulkan music positional condition shape");
  x.engine->op(x.data, t.data, pe, x.data, {5, n, uint32_t(x.cols)}, (n + 255) / 256);
}

Tensor gpu_yue_norm(Runtime& rt, const Tensor& x, const std::string& name, int width) {
  int rows = x.rows * x.cols / width, outputs = 1;
  while (outputs * 2 <= rows && outputs < 512)
    outputs *= 2;
  uint32_t threads = std::min(width > 128 ? width / 4 : width, 512 / std::min(outputs, 16));
  Tensor y(x.rows, x.cols);
  auto& w = rt.weight(name);
  x.engine->op(x.data, w, w, y.data, {8, uint32_t(rows), uint32_t(width), threads}, rows);
  return y;
}

Tensor gpu_ln(Runtime& rt, const Tensor& x, const std::string& name) {
  Tensor y(x.rows, x.cols);
  auto& w = rt.weight(name + ".gamma");
  auto& b = rt.weight(name + ".beta");
  if (w.layout().elements() != uint64_t(x.cols) || b.layout().elements() != uint64_t(x.cols))
    throw std::invalid_argument("Vulkan music layer norm shape");
  x.engine->op(x.data, w, b, y.data, {9, uint32_t(x.rows), uint32_t(x.cols)}, x.rows);
  return y;
}

Tensor gpu_causal_attention(Runtime& rt, const Tensor& q, const Tensor& k, const Tensor& v) {
  return rt.attend(q, k, v, 16, 8, true);
}
}
