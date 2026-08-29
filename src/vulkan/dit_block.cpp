#include "vidfab/vulkan/dit_block.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vidfab/attention.h"
#include "vidfab/dtype.h"
#include "vidfab/nf4.h"
#include "vidfab/tensor_convert.h"
#include "vidfab/vulkan/linear.h"

namespace vidfab::vulkan {
namespace {

TensorLayout matrix(uint64_t a, uint64_t b) {
  const uint64_t shape[] = {a, b}; return TensorLayout::contiguous(shape, 2);
}
TensorLayout three(uint64_t a, uint64_t b, uint64_t c) {
  const uint64_t shape[] = {a, b, c}; return TensorLayout::contiguous(shape, 3);
}
TensorLayout vector(uint64_t n) { return TensorLayout::contiguous(&n, 1); }
uint64_t bytes(const DeviceTensor& t) {
  return t ? t.layout().bytes(t.type()) : 0;
}
uint64_t checked_product(uint64_t a, uint64_t b, const char* what) {
  if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a)
    throw std::overflow_error(std::string("Vulkan H3 block: ") + what + " overflow");
  return a * b;
}
void validate_config(const H3BlockConfig& c) {
  if (c.sequence == 0 || c.hidden == 0 || c.heads == 0 ||
      c.head_dim != 128 || c.heads * c.head_dim == 0 ||
      c.heads * c.head_dim > std::numeric_limits<uint32_t>::max() ||
      c.ffn == 0 || c.timesteps == 0 || c.modalities == 0 ||
      c.adaln_rank == 0 || !std::isnormal(c.epsilon) || c.epsilon <= 0.0f) {
    throw std::invalid_argument("Vulkan H3 block: invalid configuration");
  }
}
void require_shape(const TensorView& v, const std::vector<int64_t>& shape,
                   const std::string& name) {
  if (v.shape != shape)
    throw std::runtime_error("Vulkan H3 block: '" + name + "' has wrong shape");
}
std::vector<uint16_t> bf16_vector(const SafeTensors& st,
                                  const std::string& name, uint32_t count) {
  const TensorView& v = st.at(name); require_shape(v, {count}, name);
  std::vector<float> wide = to_f32(v); std::vector<uint16_t> result(count);
  for (uint32_t i = 0; i < count; ++i) result[i] = f32_to_bf16(wide[i]);
  return result;
}
float scalar(const TensorView& v, const std::string& name) {
  std::vector<float> values = to_f32(v);
  if (values.size() != 1 || !std::isfinite(values[0]))
    throw std::runtime_error("Vulkan H3 block: invalid scalar '" + name + "'");
  return values[0];
}

struct Projection {
  LinearWeight weight;
  DeviceTensor dense;
  uint32_t out = 0, in = 0;
  uint64_t persistent_bytes() const noexcept {
    return weight.resident_bytes() + bytes(dense);
  }
};

Projection load_projection(TensorContext& context, const SafeTensors& st,
                           const std::string& name, uint32_t out, uint32_t in,
                           uint32_t source_out = 0, uint32_t row_offset = 0) {
  if (source_out == 0) source_out = out;
  if (row_offset > source_out || out > source_out - row_offset)
    throw std::invalid_argument("Vulkan H3 block: invalid projection slice");
  const TensorView& w = st.at(name + ".weight");
  LinearWeightUpload u; u.out_features = out; u.in_features = in;
  std::vector<float> scale_storage, map_storage, nested_map_storage,
      nested_absmax_storage;
  std::vector<uint16_t> pre_scale_storage;
  const uint64_t elements = checked_product(out, in, "projection elements");
  const uint64_t source_elements = checked_product(source_out, in, "projection source");
  const uint64_t element_offset = checked_product(row_offset, in, "projection offset");

  const TensorView* block_scale = st.find(name + ".weight_scale");
  const bool nvfp4 = w.dtype == DType::kU8 && block_scale &&
      block_scale->dtype == DType::kF8E4M3 && in % 64 == 0;
  if (is_nf4_weight(st, name)) {
    const NF4State state = read_nf4_state(st, name, "Vulkan H3 block");
    if (state.shape != std::vector<int64_t>{source_out, in} ||
        element_offset % 16384 != 0 || elements % 16384 != 0)
      throw std::runtime_error("Vulkan H3 block: NF4 projection slice is not nested-block aligned");
    const TensorView& absmax = st.at(name + ".weight.absmax");
    const TensorView& qmap = st.at(name + ".weight.quant_map");
    const TensorView& nested_map = st.at(name + ".weight.nested_quant_map");
    const TensorView& nested_absmax = st.at(name + ".weight.nested_absmax");
    map_storage = to_f32(qmap); nested_map_storage = to_f32(nested_map);
    nested_absmax_storage = to_f32(nested_absmax);
    u.format = LinearWeightFormat::kNF4;
    u.data = static_cast<const uint8_t*>(w.data) + element_offset / 2;
    u.data_bytes = elements / 2;
    u.nf4_absmax = static_cast<const uint8_t*>(absmax.data) + element_offset / 64;
    u.nf4_absmax_count = elements / 64;
    u.nf4_quant_map = map_storage.data(); u.nf4_quant_map_count = map_storage.size();
    u.nf4_nested_quant_map = nested_map_storage.data();
    u.nf4_nested_quant_map_count = nested_map_storage.size();
    const uint64_t nested_begin = element_offset / 16384;
    const uint64_t nested_count = elements / 16384;
    if (nested_begin + nested_count > nested_absmax_storage.size())
      throw std::runtime_error("Vulkan H3 block: NF4 nested scale slice escapes tensor");
    u.nf4_nested_absmax = nested_absmax_storage.data() + nested_begin;
    u.nf4_nested_absmax_count = nested_count;
    u.nf4_nested_offset = state.nested_offset;
  } else if (nvfp4) {
    require_shape(w, {source_out, static_cast<int64_t>(in / 2)}, name + ".weight");
    require_shape(*block_scale,
                  {source_out, static_cast<int64_t>(in / 16)},
                  name + ".weight_scale");
    if (row_offset % 128 != 0 || out % 128 != 0)
      throw std::runtime_error("Vulkan H3 block: NVFP4 row slice is not 128-row aligned");
    const TensorView* global = st.find(name + ".weight_scale_2");
    if (!global) throw std::runtime_error("Vulkan H3 block: NVFP4 weight lacks scale_2");
    u.format = LinearWeightFormat::kNVFloat4;
    u.data = static_cast<const uint8_t*>(w.data) + element_offset / 2;
    u.data_bytes = elements / 2;
    u.block_scale = static_cast<const uint8_t*>(block_scale->data) + element_offset / 16;
    u.block_scale_count = elements / 16;
    u.global_scale = scalar(*global, name + ".weight_scale_2");
  } else {
    require_shape(w, {source_out, in}, name + ".weight");
    uint64_t element_bytes = 0;
    switch (w.dtype) {
      case DType::kF32: u.format = LinearWeightFormat::kFloat32; element_bytes = 4; break;
      case DType::kF16: u.format = LinearWeightFormat::kFloat16; element_bytes = 2; break;
      case DType::kBF16: u.format = LinearWeightFormat::kBFloat16; element_bytes = 2; break;
      case DType::kF8E4M3: u.format = LinearWeightFormat::kFloat8E4M3; element_bytes = 1; break;
      case DType::kI8: u.format = LinearWeightFormat::kInt8; element_bytes = 1; break;
      default: throw std::runtime_error("Vulkan H3 block: unsupported dtype for '" + name + "'");
    }
    if (w.nbytes != source_elements * element_bytes)
      throw std::runtime_error("Vulkan H3 block: projection byte count mismatch");
    u.data = static_cast<const uint8_t*>(w.data) + element_offset * element_bytes;
    u.data_bytes = elements * element_bytes;
    if (u.format == LinearWeightFormat::kFloat8E4M3) {
      const TensorView* sv = st.find(name + ".weight_scale");
      if (!sv) throw std::runtime_error("Vulkan H3 block: FP8 weight lacks scalar scale");
      scale_storage = {scalar(*sv, name + ".weight_scale")};
      u.weight_scale = scale_storage.data(); u.weight_scale_count = 1;
      if (const TensorView* input_scale = st.find(name + ".input_scale")) {
        u.fp8_input_scale = scalar(*input_scale, name + ".input_scale");
        u.has_fp8_input_scale = true;
      }
    } else if (u.format == LinearWeightFormat::kInt8) {
      const TensorView& sv = st.at(name + ".weight_scale");
      std::vector<float> all = to_f32(sv);
      if (all.size() != source_out)
        throw std::runtime_error("Vulkan H3 block: INT8 scale shape mismatch");
      scale_storage.assign(all.begin() + row_offset, all.begin() + row_offset + out);
      u.weight_scale = scale_storage.data(); u.weight_scale_count = out;
    }
  }
  if (const TensorView* pre = st.find(name + ".pre_quant_scale")) {
    require_shape(*pre, {in}, name + ".pre_quant_scale");
    std::vector<float> wide = to_f32(*pre); pre_scale_storage.resize(in);
    for (uint32_t i = 0; i < in; ++i) pre_scale_storage[i] = f32_to_bf16(wide[i]);
    u.pre_quant_scale_bf16 = pre_scale_storage.data(); u.pre_quant_scale_count = in;
  }
  Projection result; result.out = out; result.in = in;
  result.weight = LinearWeight::upload(context, u);
  if (result.weight.format() != LinearWeightFormat::kNVFloat4) {
    result.dense = context.allocate(matrix(out, in), ScalarType::kBFloat16);
    TensorBatch batch = context.begin_batch();
    result.weight.materialize_bf16(batch, result.dense); batch.submit().wait();
  }
  return result;
}

}  // namespace

struct ExactH3BlockScratch::Impl {
  TensorContext* context = nullptr; H3BlockConfig config;
  uint32_t modulation_rows = 0;
  DeviceTensor modulation, normed, q, k, v, attention, branch, fused, activation;
  DeviceTensor hidden_a, hidden_b, inner_a, inner_b, ffn_a, ffn_b;
  StreamedNVFP4WeightCache cache;
  DenseGemmPlan q_plan, k_plan, v_plan, out_plan, fc1_plan, fc2_plan;
  H3AttentionPlan attention_plan;
  explicit Impl(TensorContext& owner, const H3BlockConfig& c)
      : context(&owner), config(c) {
    const uint32_t inner = c.heads * c.head_dim;
    modulation_rows = c.timesteps * c.modalities;
    const uint64_t logical_table = checked_product(modulation_rows, c.hidden, "modulation");
    const uint64_t align_elements = std::max<uint64_t>(
        1, (owner.storage_binding_alignment() + 3) / 4);
    const uint64_t table_stride = ((logical_table + align_elements - 1) / align_elements) * align_elements;
    modulation = owner.allocate(vector(checked_product(6, table_stride, "modulation arena")));
    normed = owner.allocate(matrix(c.sequence, c.hidden), ScalarType::kBFloat16);
    q = owner.allocate(three(c.sequence, c.heads, c.head_dim), ScalarType::kBFloat16);
    k = owner.allocate(three(c.sequence, c.heads, c.head_dim), ScalarType::kBFloat16);
    v = owner.allocate(three(c.sequence, c.heads, c.head_dim), ScalarType::kBFloat16);
    attention = owner.allocate(matrix(c.sequence, inner), ScalarType::kBFloat16);
    branch = owner.allocate(matrix(c.sequence, c.hidden), ScalarType::kBFloat16);
    fused = owner.allocate(matrix(c.sequence, 2ull * c.ffn), ScalarType::kBFloat16);
    activation = owner.allocate(matrix(c.sequence, c.ffn), ScalarType::kBFloat16);
    hidden_a = owner.allocate(matrix(c.sequence, c.hidden), ScalarType::kBFloat16);
    hidden_b = owner.allocate(matrix(c.sequence, c.hidden), ScalarType::kBFloat16);
    inner_a = owner.allocate(matrix(c.sequence, inner), ScalarType::kBFloat16);
    inner_b = owner.allocate(matrix(c.sequence, inner), ScalarType::kBFloat16);
    ffn_a = owner.allocate(matrix(c.sequence, c.ffn), ScalarType::kBFloat16);
    ffn_b = owner.allocate(matrix(c.sequence, c.ffn), ScalarType::kBFloat16);
    const uint64_t largest = std::max({checked_product(inner, c.hidden, "q"),
        checked_product(c.hidden, inner, "out"),
        checked_product(2ull * c.ffn, c.hidden, "fc1"),
        checked_product(c.hidden, c.ffn, "fc2")});
    cache = StreamedNVFP4WeightCache::create(owner, largest);
    auto plan = [&](uint32_t out, uint32_t in) {
      return DenseGemmPlan::create(owner, {c.sequence, out, in,
          DenseGemmMode::kBFloat16, DenseGemmBias::kNone, false});
    };
    q_plan = plan(inner, c.hidden); k_plan = plan(inner, c.hidden);
    v_plan = plan(inner, c.hidden); out_plan = plan(c.hidden, inner);
    fc1_plan = plan(2 * c.ffn, c.hidden); fc2_plan = plan(c.hidden, c.ffn);
    attention_plan = H3AttentionPlan::create(owner, {c.sequence, c.heads,
        c.head_dim, exact_attention_scale(c.head_dim)});
  }
  uint64_t reserved() const noexcept {
    return bytes(modulation) + bytes(normed) + bytes(q) + bytes(k) + bytes(v) +
        bytes(attention) + bytes(branch) + bytes(fused) + bytes(activation) +
        bytes(hidden_a) + bytes(hidden_b) + bytes(inner_a) + bytes(inner_b) +
        bytes(ffn_a) + bytes(ffn_b) + cache.dense_bytes();
  }
};

struct ExactH3BlockStage::Impl {
  struct Weights {
    DeviceTensor norm1, norm2, q_norm, k_norm, adaln_w, adaln_b;
    Projection q, k, v, out, fc1, fc2;
    uint64_t bytes() const noexcept {
      return ::vidfab::vulkan::bytes(norm1) + ::vidfab::vulkan::bytes(norm2) +
          ::vidfab::vulkan::bytes(q_norm) + ::vidfab::vulkan::bytes(k_norm) +
          ::vidfab::vulkan::bytes(adaln_w) + ::vidfab::vulkan::bytes(adaln_b) +
          q.persistent_bytes() + k.persistent_bytes() + v.persistent_bytes() +
          out.persistent_bytes() + fc1.persistent_bytes() + fc2.persistent_bytes();
    }
  };
  TensorContext* context = nullptr; H3BlockConfig config;
  std::unique_ptr<Weights> weights;
  Impl(TensorContext& owner, const H3BlockConfig& c) : context(&owner), config(c) {}
};

ExactH3BlockScratch::ExactH3BlockScratch() = default;
ExactH3BlockScratch::~ExactH3BlockScratch() = default;
ExactH3BlockScratch::ExactH3BlockScratch(std::shared_ptr<Impl> p) : impl_(std::move(p)) {}
ExactH3BlockScratch::ExactH3BlockScratch(ExactH3BlockScratch&&) noexcept = default;
ExactH3BlockScratch& ExactH3BlockScratch::operator=(ExactH3BlockScratch&&) noexcept = default;
ExactH3BlockScratch ExactH3BlockScratch::create(TensorContext& context,
                                                const H3BlockConfig& config) {
  validate_config(config); context.require_exact_fp32_vae_normalization();
  context.require_exact_vae_pointwise(); context.require_exact_h3_attention();
  return ExactH3BlockScratch(std::make_shared<Impl>(context, config));
}
uint64_t ExactH3BlockScratch::reserved_bytes() const noexcept {
  return impl_ ? impl_->reserved() : 0;
}

ExactH3BlockStage::ExactH3BlockStage() = default;
ExactH3BlockStage::~ExactH3BlockStage() = default;
ExactH3BlockStage::ExactH3BlockStage(std::shared_ptr<Impl> p) : impl_(std::move(p)) {}
ExactH3BlockStage::ExactH3BlockStage(ExactH3BlockStage&&) noexcept = default;
ExactH3BlockStage& ExactH3BlockStage::operator=(ExactH3BlockStage&&) noexcept = default;
ExactH3BlockStage ExactH3BlockStage::create(TensorContext& context,
                                            const H3BlockConfig& config) {
  validate_config(config); context.require_exact_fp32_vae_normalization();
  context.require_exact_vae_pointwise(); context.require_exact_h3_attention();
  return ExactH3BlockStage(std::make_shared<Impl>(context, config));
}

void ExactH3BlockStage::load(const SafeTensors& st, uint32_t layer) {
  if (!impl_) throw std::logic_error("Vulkan H3 block: empty stage");
  Impl& s = *impl_; const H3BlockConfig& c = s.config;
  const uint32_t inner = c.heads * c.head_dim;
  const std::string p = "blocks." + std::to_string(layer) + ".";
  auto next = std::make_unique<Impl::Weights>();
  auto upload_bf = [&](const std::string& name, uint32_t n) {
    std::vector<uint16_t> host = bf16_vector(st, name, n);
    DeviceTensor result = s.context->allocate(vector(n), ScalarType::kBFloat16);
    s.context->upload_bytes(result, host.data(), host.size() * 2); return result;
  };
  next->norm1 = upload_bf(p + "norm1.weight", c.hidden);
  next->norm2 = upload_bf(p + "norm2.weight", c.hidden);
  next->q_norm = upload_bf(p + "attn.q_norm.weight", c.head_dim);
  next->k_norm = upload_bf(p + "attn.k_norm.weight", c.head_dim);
  const std::string qkv = p + "attn.qkv_proj";
  next->q = load_projection(*s.context, st, qkv, inner, c.hidden, 3 * inner, 0);
  next->k = load_projection(*s.context, st, qkv, inner, c.hidden, 3 * inner, inner);
  next->v = load_projection(*s.context, st, qkv, inner, c.hidden, 3 * inner, 2 * inner);
  next->out = load_projection(*s.context, st, p + "attn.out_proj", c.hidden, inner);
  next->fc1 = load_projection(*s.context, st, p + "mlp.fc1", 2 * c.ffn, c.hidden);
  next->fc2 = load_projection(*s.context, st, p + "mlp.fc2", c.hidden, c.ffn);
  const uint64_t adaln_out = checked_product(
      checked_product(c.modalities, 6, "AdaLN"), c.hidden, "AdaLN");
  const TensorView& aw = st.at(p + "adaln_proj.linear.weight");
  const TensorView& ab = st.at(p + "adaln_proj.linear.bias");
  require_shape(aw, {static_cast<int64_t>(adaln_out), c.adaln_rank}, aw.name);
  require_shape(ab, {static_cast<int64_t>(adaln_out)}, ab.name);
  std::vector<float> wide_w = to_f32(aw), wide_b = to_f32(ab);
  next->adaln_w = s.context->allocate(matrix(adaln_out, c.adaln_rank));
  next->adaln_b = s.context->allocate(vector(adaln_out));
  s.context->upload(next->adaln_w, wide_w.data(), wide_w.size());
  s.context->upload(next->adaln_b, wide_b.data(), wide_b.size());
  s.weights = std::move(next);
}

namespace {
DeviceTensor& transformed_input(TensorBatch& batch, Projection& p,
                                DeviceTensor& input, DeviceTensor& a,
                                DeviceTensor& b) {
  DeviceTensor* selected = &input;
  if (p.weight.has_pre_quant_scale()) {
    p.weight.apply_pre_quant_scale(batch, *selected, a); selected = &a;
  }
  if (p.weight.applies_convrot()) {
    p.weight.apply_convrot(batch, *selected, selected == &a ? b : a);
    selected = selected == &a ? &b : &a;
  }
  return *selected;
}
void projection(TensorBatch& batch, Projection& p, const DenseGemmPlan& plan,
                StreamedNVFP4WeightCache& cache, DeviceTensor& input,
                DeviceTensor& output, DeviceTensor& transform_a,
                DeviceTensor& transform_b, uint32_t rows) {
  DeviceTensor& source = transformed_input(batch, p, input, transform_a, transform_b);
  const uint32_t tiled_rows = rows / 64 * 64;
  auto record = [&](auto& weight) {
    if (tiled_rows != 0)
      plan.record(batch, source, weight, output, tiled_rows);
    if (tiled_rows != rows)
      plan.record(batch, source, weight, output, rows - tiled_rows,
                  tiled_rows, tiled_rows);
  };
  if (p.weight.format() == LinearWeightFormat::kNVFloat4) {
    PreparedNVFP4WeightView prepared = cache.prepare(batch, p.weight, plan);
    record(prepared);
  } else {
    record(p.dense);
  }
}
}

void ExactH3BlockStage::record(TensorBatch& batch, DeviceTensor& tokens,
    DeviceTensor& selectors, DeviceTensor& code, DeviceTensor& cosine,
    DeviceTensor& sine, ExactH3BlockScratch& scratch,
    const H3AttentionRanges* ranges) const {
  if (!impl_ || !impl_->weights) throw std::logic_error("Vulkan H3 block: not loaded");
  if (!scratch.impl_ || scratch.impl_->context != impl_->context)
    throw std::invalid_argument("Vulkan H3 block: incompatible scratch");
  const H3BlockConfig& c = impl_->config; auto& s = *scratch.impl_;
  if (s.config.sequence != c.sequence || s.config.hidden != c.hidden ||
      s.config.heads != c.heads || s.config.head_dim != c.head_dim ||
      s.config.ffn != c.ffn || s.config.timesteps != c.timesteps ||
      s.config.modalities != c.modalities || s.config.adaln_rank != c.adaln_rank)
    throw std::invalid_argument("Vulkan H3 block: scratch configuration mismatch");
  const auto tv = tokens.view(), av = selectors.view(), cv = code.view();
  if (tv.type != ScalarType::kBFloat16 || tv.layout.rank != 2 ||
      tv.layout.extent[0] != c.sequence || tv.layout.extent[1] != c.hidden ||
      av.type != ScalarType::kInt32 || av.layout.rank != 1 ||
      av.layout.extent[0] != c.sequence || cv.type != ScalarType::kFloat32 ||
      cv.layout.rank != 2 || cv.layout.extent[0] != c.timesteps ||
      cv.layout.extent[1] != c.adaln_rank)
    throw std::invalid_argument("Vulkan H3 block: invalid activation tensors");
  constexpr uint32_t kOperators = 40;
  if (batch.remaining_operator_capacity() < kOperators)
    throw std::logic_error("Vulkan H3 block: insufficient batch capacity");
  auto& w = *impl_->weights;
  batch.dit_expand_adaln(w.adaln_w, w.adaln_b, code, s.modulation,
                         c.modalities, 6, c.hidden);
  batch.rms_norm_modulate_bf16_table(tokens, w.norm1, s.modulation,
                                     s.modulation_rows, 1, 0, selectors, s.normed, c.epsilon);
  projection(batch, w.q, s.q_plan, s.cache, s.normed, s.q,
             s.hidden_a, s.hidden_b, c.sequence);
  projection(batch, w.k, s.k_plan, s.cache, s.normed, s.k,
             s.hidden_a, s.hidden_b, c.sequence);
  projection(batch, w.v, s.v_plan, s.cache, s.normed, s.v,
             s.hidden_a, s.hidden_b, c.sequence);
  batch.rms_norm_heads_bf16(s.q, w.q_norm, s.q, c.heads, c.head_dim, c.epsilon);
  batch.rms_norm_heads_bf16(s.k, w.k_norm, s.k, c.heads, c.head_dim, c.epsilon);
  batch.rope_h3_bf16(s.q, cosine, sine);
  batch.rope_h3_bf16(s.k, cosine, sine);
  s.attention_plan.record(batch, s.q, s.k, s.v, s.attention, ranges);
  projection(batch, w.out, s.out_plan, s.cache, s.attention, s.branch,
             s.inner_a, s.inner_b, c.sequence);
  batch.dit_add_gated_bf16_table(tokens, s.branch, s.modulation,
                                 s.modulation_rows, 2, selectors);
  batch.rms_norm_modulate_bf16_table(tokens, w.norm2, s.modulation,
                                     s.modulation_rows, 4, 3, selectors, s.normed, c.epsilon);
  projection(batch, w.fc1, s.fc1_plan, s.cache, s.normed, s.fused,
             s.hidden_a, s.hidden_b, c.sequence);
  batch.dit_swiglu_bf16(s.fused, s.activation);
  projection(batch, w.fc2, s.fc2_plan, s.cache, s.activation, s.branch,
             s.ffn_a, s.ffn_b, c.sequence);
  batch.dit_add_gated_bf16_table(tokens, s.branch, s.modulation,
                                 s.modulation_rows, 5, selectors);
}

void ExactH3BlockStage::unload() noexcept { if (impl_) impl_->weights.reset(); }
bool ExactH3BlockStage::loaded() const noexcept { return impl_ && impl_->weights; }
const H3BlockConfig& ExactH3BlockStage::config() const noexcept { return impl_->config; }
uint64_t ExactH3BlockStage::persistent_bytes() const noexcept {
  return impl_ && impl_->weights ? impl_->weights->bytes() : 0;
}
uint64_t ExactH3BlockStage::peak_device_bytes(
    const ExactH3BlockScratch& scratch) const noexcept {
  return persistent_bytes() + scratch.reserved_bytes();
}

}  // namespace vidfab::vulkan
