#include "slopfab/vulkan/sol_h3_vae.h"
#include "slopfab/vulkan/tensor.h"
#include "slopfab/vulkan/gemm.h"
#include "slopfab/vae/sol_h3.h"
#include "slopfab/tensor_convert.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <map>
#include <stdexcept>

namespace slopfab::vulkan {
namespace {
constexpr uint32_t D = 2048, Inner = 8192, Chunk = 512;
TensorLayout matrix(uint64_t rows, uint64_t columns) {
  const uint64_t dims[] = {rows, columns};
  return TensorLayout::contiguous(dims, 2);
}
TensorLayout vector(uint64_t count) { return TensorLayout::contiguous(&count, 1); }
TensorLayout attention_shape(uint64_t rows) {
  const uint64_t dims[] = {rows, 32, 64};
  return TensorLayout::contiguous(dims, 3);
}
uint16_t bf_bits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, 4);
  if ((bits & 0x7fffffffu) > 0x7f800000u) return 0x7fc0;
  return uint16_t((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}
struct Linear { DeviceTensor w, bias; };
struct Block { DeviceTensor n1, n2, s1, s2; Linear qkv, out, w1, w2; };
struct Window {
  DeviceTensor indices;
  uint32_t patches = 0, keys = 0;
};
struct AttentionSlot {
  DeviceTensor q, k, v, output;
  PreparedAttentionInputs prepared;
  BlockedAttentionPlan plan;
  AttentionSlot(TensorContext& context, uint32_t rows)
      : q(context.allocate(attention_shape(rows), ScalarType::kBFloat16)),
        k(context.allocate(attention_shape(rows), ScalarType::kBFloat16)),
        v(context.allocate(attention_shape(rows), ScalarType::kBFloat16)),
        output(context.allocate(attention_shape(rows), ScalarType::kBFloat16)),
        prepared(PreparedAttentionInputs::create(context, {rows, 32, 64, .125f})),
        plan(BlockedAttentionPlan::create(context, {rows, 32, 64, .125f})) {}
};
TensorContextOptions options() {
  TensorContextOptions result;
  result.pipeline_sets = TensorPipelineSet::kCore | TensorPipelineSet::kSolVae |
                         TensorPipelineSet::kBlockedAttention;
  result.max_batch_operators = 128;
  return result;
}
} // namespace

struct SolH3Decoder::Impl {
  TensorContext context;
  TensorBatch pending;
  std::vector<Block> blocks;
  Linear pq, embed, head;
  DeviceTensor reg, norm_w, norm_b;
  size_t bytes = 0;
  bool loaded = false;
  explicit Impl(const Device& device) : context(device, options()) {
    context.require_blocked_attention();
  }
  void flush() {
    if (pending) {
      auto done = pending.submit();
      pending = {};
      done.wait();
    }
  }
  TensorBatch& batch(uint32_t capacity = 16) {
    if (pending && pending.remaining_operator_capacity() < capacity) flush();
    if (!pending) pending = context.begin_batch();
    return pending;
  }
  DeviceTensor load_tensor(const SafeTensors& st, const std::string& name,
                           const TensorLayout& layout, bool bf16 = false) {
    auto result = context.allocate(layout, bf16 ? ScalarType::kBFloat16 : ScalarType::kFloat32);
    const auto values = to_f32(st.at(name));
    if (bf16) {
      std::vector<uint16_t> words(values.size());
      std::transform(values.begin(), values.end(), words.begin(), bf_bits);
      context.upload_transient_bytes(result, words.data(), words.size() * 2);
    } else {
      context.upload_transient(result, values.data(), values.size());
    }
    bytes += values.size() * (bf16 ? 2 : 4);
    return result;
  }
  Linear load_linear(const SafeTensors& st, const std::string& prefix, uint32_t out,
                      uint32_t in, bool bf16) {
    return {load_tensor(st, prefix + ".weight", matrix(out, in), bf16),
            load_tensor(st, prefix + ".bias", vector(out), bf16)};
  }
  // Unused bindings alias input. The recording seam merges aliases before barriers.
  void op(TensorBatch& batch, std::initializer_list<uint32_t> params, uint32_t groups,
           DeviceTensor& input, DeviceTensor& weight, DeviceTensor& bias, DeviceTensor& output,
           DeviceTensor* extra0 = nullptr, DeviceTensor* extra1 = nullptr,
           DeviceTensor* extra2 = nullptr, DeviceTensor* extra3 = nullptr,
           uint32_t writes = 8) {
    uint32_t p[16] = {};
    std::copy(params.begin(), params.end(), p);
    DeviceTensor* bindings[] = {&input, &weight, &bias, &output,
                               extra0 ? extra0 : &input, extra1 ? extra1 : &input,
                               extra2 ? extra2 : &input, extra3 ? extra3 : &input};
    batch.sol_vae_operation(bindings, writes, p, groups);
  }
};

SolH3Decoder::SolH3Decoder(const Device& device) : impl_(std::make_unique<Impl>(device)) {}
SolH3Decoder::~SolH3Decoder() = default;
SolH3Decoder::SolH3Decoder(SolH3Decoder&&) noexcept = default;
SolH3Decoder& SolH3Decoder::operator=(SolH3Decoder&&) noexcept = default;
size_t SolH3Decoder::weight_bytes() const { return impl_->bytes; }

void SolH3Decoder::load(const SafeTensors& checkpoint) {
  vae::validate_sol_h3_vae(checkpoint);
  auto& d = *impl_;
  d.flush(); d.loaded = false; d.blocks.clear(); d.bytes = 0;
  checkpoint.prefetch();
  d.pq = d.load_linear(checkpoint, "post_quant_conv", 24, 24, true);
  d.embed = d.load_linear(checkpoint, "x_embedder", D, 24, false);
  d.head = d.load_linear(checkpoint, "proj_out", 3072, D, false);
  d.reg = d.load_tensor(checkpoint, "register_tokens", matrix(4, D));
  d.norm_w = d.load_tensor(checkpoint, "norm_out.weight", vector(D));
  d.norm_b = d.load_tensor(checkpoint, "norm_out.bias", vector(D));
  for (int i = 0; i < 24; ++i) {
    const auto p = "blocks." + std::to_string(i) + ".";
    Block b;
    b.n1 = d.load_tensor(checkpoint, p + "norm1.weight", vector(D));
    b.n2 = d.load_tensor(checkpoint, p + "norm2.weight", vector(D));
    b.s1 = d.load_tensor(checkpoint, p + "scale1", vector(D));
    b.s2 = d.load_tensor(checkpoint, p + "scale2", vector(D));
    b.qkv = d.load_linear(checkpoint, p + "attn.to_qkv", 3 * D, D, true);
    b.out = d.load_linear(checkpoint, p + "attn.to_out", D, D, true);
    b.w1 = d.load_linear(checkpoint, p + "ff.w1", 2 * Inner, D, true);
    b.w2 = d.load_linear(checkpoint, p + "ff.w2", D, Inner, true);
    d.blocks.push_back(std::move(b));
  }
  d.loaded = true;
}

vae::DecodedVideo SolH3Decoder::decode(const float* z, int T, int H, int W,
    const std::vector<float>& mean, const std::vector<float>& stddev, const vae::DecodeSchedule&) {
  auto& d = *impl_;
  if (!d.loaded || !z || mean.size() != 24 || stddev.size() != 24)
    throw std::invalid_argument("Vulkan Sol-H3-VAE: load weights and provide 24-channel latents/statistics");
  auto geom = vae::build_sol_h3_geometry(T, H, W);
  const uint32_t N = geom.patches, M = geom.rows, nb = (T + 4) / 5;
  auto& ctx = d.context;
  struct Discard { Impl& impl; ~Discard() { impl.pending = {}; } } discard{d};
  auto alloc = [&](uint64_t rows, uint64_t cols, bool bf16 = false) {
    return ctx.allocate(matrix(rows, cols), bf16 ? ScalarType::kBFloat16 : ScalarType::kFloat32);
  };
  auto make_plan = [&](uint32_t out, uint32_t in, bool bf16) {
    return DenseGemmPlan::create(ctx, {Chunk, out, in,
      bf16 ? DenseGemmMode::kBFloat16 : DenseGemmMode::kFloat32,
      bf16 ? DenseGemmBias::kBFloat16Fused : DenseGemmBias::kFloat32});
  };
  auto pq_plan = make_plan(24, 24, true), embed_plan = make_plan(D, 24, false);
  auto qkv_plan = make_plan(3 * D, D, true), out_plan = make_plan(D, D, true);
  auto w1_plan = make_plan(2 * Inner, D, true), w2_plan = make_plan(D, Inner, true);
  auto head_plan = make_plan(3072, D, false);
  auto x = alloc(M, D), h = alloc(Chunk, D, true), branch = alloc(Chunk, D, true);
  auto ffn = alloc(Chunk, 2 * Inner, true), activation = alloc(Chunk, Inner, true);
  auto qkv = alloc(Chunk, 3 * D, true);
  {
    auto input = alloc(24, N), raw = alloc(Chunk, 24, true), pq = alloc(Chunk, 24, true);
    auto pq32 = alloc(Chunk, 24), embedded = alloc(Chunk, D);
    auto dm = ctx.allocate(vector(24)), ds = ctx.allocate(vector(24));
    ctx.upload_transient(input, z, size_t(24) * N);
    ctx.upload(dm, mean.data(), 24); ctx.upload(ds, stddev.data(), 24);
    for (uint32_t first = 0; first < N; first += Chunk) {
      const uint32_t count = std::min(Chunk, N - first);
      auto& batch = d.batch();
      d.op(batch, {0, count, first, N, Chunk}, Chunk * 12 / 256, input, dm, ds, raw);
      pq_plan.record(batch, raw, d.pq.w, pq, Chunk, 0, 0, &d.pq.bias);
      batch.convert(pq, pq32);
      embed_plan.record(batch, pq32, d.embed.w, embedded, Chunk, 0, 0, &d.embed.bias);
      d.op(batch, {1, count, first}, (count * D + 255) / 256, embedded, embedded, embedded, x);
    }
    d.op(d.batch(), {2, nb * 5, N}, (nb * 5 * D + 255) / 256, d.reg, d.reg, d.reg, x);
    d.flush();
  }
  {
    auto q = alloc(M, D, true), k = alloc(M, D, true), v = alloc(M, D, true), out = alloc(M, D, true);
    DeviceTensor co[2], si[2];
    std::vector<Window> windows[2], globals;
    std::map<uint32_t, std::unique_ptr<AttentionSlot>> slots;
    auto upload_windows = [&](const std::vector<vae::SolH3Window>& source, std::vector<Window>& dest) {
      for (const auto& w : source) {
        Window win;
        win.patches = w.patches; win.keys = uint32_t(w.rows.size());
        win.indices = ctx.allocate(vector(win.keys), ScalarType::kInt32);
        ctx.upload_bytes(win.indices, w.rows.data(), w.rows.size() * 4);
        if (!slots.count(win.keys)) slots[win.keys] = std::make_unique<AttentionSlot>(ctx, win.keys);
        dest.push_back(std::move(win));
      }
    };
    for (int parity = 0; parity < 2; ++parity) {
      co[parity] = alloc(M, 48); si[parity] = alloc(M, 48);
      ctx.upload(co[parity], geom.rope[parity].cosine.data(), geom.rope[parity].cosine.size());
      ctx.upload(si[parity], geom.rope[parity].sine.data(), geom.rope[parity].sine.size());
      upload_windows(geom.windows[parity], windows[parity]);
    }
    upload_windows(geom.blocks, globals);
    auto attention = [&](Window& win, bool global) {
      auto& slot = *slots.at(win.keys);
      auto& batch = d.batch();
      const uint32_t groups = (win.keys * 1024 + 255) / 256;
      d.op(batch, {7, win.keys, 0, M}, groups, q, win.indices, q, slot.q);
      d.op(batch, {7, win.keys, 0, M}, groups, k, win.indices, k, slot.k);
      d.op(batch, {7, win.keys, 0, M}, groups, v, win.indices, v, slot.v);
      auto prepared = slot.prepared.prepare(batch, slot.q, slot.k, slot.v);
      const uint32_t offset = global ? win.patches : 0, count = global ? 5 : win.patches;
      slot.plan.record(batch, prepared, slot.output, offset, count, 0);
      d.op(batch, {8, count, offset, 0, M}, (count * 1024 + 255) / 256,
           slot.output, win.indices, slot.output, out);
    };
    for (uint32_t layer = 0; layer < 24; ++layer) {
      if (progress && !progress(layer, 24)) throw std::runtime_error("Sol-H3-VAE decode cancelled");
      auto& b = d.blocks[layer];
      for (uint32_t first = 0; first < M; first += Chunk) {
        const uint32_t count = std::min(Chunk, M - first);
        auto& batch = d.batch();
        d.op(batch, {3, count, first, D}, Chunk, x, b.n1, x, h);
        qkv_plan.record(batch, h, b.qkv.w, qkv, Chunk, 0, 0, &b.qkv.bias);
        d.op(batch, {4, count, first, N}, count * 32, qkv, co[layer % 2], si[layer % 2], q,
             &k, &v, nullptr, nullptr, 8 | 16 | 32);
      }
      for (auto& win : windows[layer % 2]) attention(win, false);
      for (auto& win : globals) attention(win, true);
      for (uint32_t first = 0; first < M; first += Chunk) {
        const uint32_t count = std::min(Chunk, M - first);
        auto& batch = d.batch();
        out_plan.record(batch, out, b.out.w, branch, count, first, 0, &b.out.bias);
        d.op(batch, {5, count, first}, (count * D + 255) / 256, branch, b.s1, branch, x);
        d.op(batch, {3, count, first, D}, Chunk, x, b.n2, x, h);
        w1_plan.record(batch, h, b.w1.w, ffn, Chunk, 0, 0, &b.w1.bias);
        d.op(batch, {6, Chunk}, Chunk * 4096 / 256, ffn, ffn, ffn, activation);
        w2_plan.record(batch, activation, b.w2.w, branch, Chunk, 0, 0, &b.w2.bias);
        d.op(batch, {5, count, first}, (count * D + 255) / 256, branch, b.s2, branch, x);
      }
      d.flush();
    }
  } // Q/K/V and attention scratch are released before pixel allocation.
  vae::DecodedVideo result;
  result.frames = vae::sol_h3_frames(T); result.height = H * 16; result.width = W * 16;
  const size_t plane = size_t(result.frames) * result.height * result.width;
  result.data.resize(plane * 3);
  auto red = alloc(result.frames, uint64_t(result.height) * result.width);
  auto green = alloc(result.frames, uint64_t(result.height) * result.width);
  auto blue = alloc(result.frames, uint64_t(result.height) * result.width);
  auto normed = alloc(Chunk, D), projected = alloc(Chunk, 3072);
  for (uint32_t first = 0; first < N; first += Chunk) {
    const uint32_t count = std::min(Chunk, N - first);
    auto& batch = d.batch();
    d.op(batch, {9, count, first, D}, Chunk, x, d.norm_w, d.norm_b, normed);
    head_plan.record(batch, normed, d.head.w, projected, Chunk, 0, 0, &d.head.bias);
    d.op(batch, {10, count, first, uint32_t(H), uint32_t(W), uint32_t(result.frames)},
         (count * 3072 + 255) / 256, projected, projected, projected, projected,
         nullptr, &red, &green, &blue, 32 | 64 | 128);
  }
  d.flush();
  ctx.download(red, result.data.data(), plane);
  ctx.download(green, result.data.data() + plane, plane);
  ctx.download(blue, result.data.data() + plane * 2, plane);
  if (progress && !progress(24, 24)) throw std::runtime_error("Sol-H3-VAE decode cancelled");
  return result;
}
} // namespace slopfab::vulkan
