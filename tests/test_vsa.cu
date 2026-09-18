#include "harness.h"
#include "slopfab/cuda/vsa_attention.cuh"
#include "slopfab/dit/vsa.h"
#include "slopfab/dit/transformer.h"
#include "slopfab/dtype.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <numeric>

namespace {
float bf(float x) { return slopfab::bf16_to_f32(slopfab::f32_to_bf16(x)); }

void verify_vsa(int dim) {
  using namespace slopfab;
  using namespace slopfab::cuda;
  dit::SequenceLayout layout;
  layout.num_text = 65; layout.num_audio_rows = 3;
  layout.num_latent_frames = 5; layout.latent_height = 10; layout.latent_width = 14;
  layout.num_video_rows = 5 * 5 * 7;
  const auto tiles = dit::build_vsa_tiles(layout);
  const int seq = layout.total_rows(), heads = 2, n = int(tiles.sizes.size()), inner = heads * dim;
  const size_t count = size_t(seq) * inner;
  std::vector<__nv_bfloat16> q(count), k(count), v(count), gate(count);
  auto fill = [&](auto& x, unsigned seed, float scale) {
    auto data = test::make_data(x.size(), seed, scale);
    for (size_t i = 0; i < x.size(); ++i) x[i] = __float2bfloat16(data[i]);
  };
  fill(q, 71, 1.0f); fill(k, 83, 1.0f); fill(v, 97, 1.0f); fill(gate, 113, 0.8f);
  std::vector<double> qp(size_t(n) * inner), kp(qp.size()), vp(qp.size());
  auto pool = [&](const auto& src, auto& dest) {
    for (int t = 0; t < n; ++t)
      for (int j = 0; j < inner; ++j) {
        double sum = 0;
        for (int r = 0; r < tiles.sizes[t]; ++r)
          sum += __bfloat162float(src[size_t(tiles.rows[t * 64 + r]) * inner + j]);
        dest[size_t(t) * inner + j] = sum / tiles.sizes[t];
      }
  };
  pool(q, qp); pool(k, kp); pool(v, vp);
  std::vector<uint8_t> masks(size_t(heads) * n * n);
  std::vector<float> compressed(size_t(n) * inner);
  for (int head = 0; head < heads; ++head)
    for (int t = 0; t < n; ++t) {
      std::vector<double> scores(n);
      for (int key = 0; key < n; ++key)
        for (int d = 0; d < dim; ++d)
          scores[key] += qp[size_t(t) * inner + head * dim + d] * kp[size_t(key) * inner + head * dim + d] / std::sqrt(double(dim));
      std::vector<int> order(n - tiles.prefix_tiles);
      std::iota(order.begin(), order.end(), tiles.prefix_tiles);
      std::sort(order.begin(), order.end(), [&](int a, int b) { return scores[a] > scores[b]; });
      auto* mask = masks.data() + (size_t(head) * n + t) * n;
      for (int key = 0; key < n; ++key) mask[key] = t < tiles.prefix_tiles || key < tiles.prefix_tiles;
      for (int i = 0; i < (int(order.size()) + 4) / 5; ++i) mask[order[i]] = 1;
      const double maximum = *std::max_element(scores.begin(), scores.end());
      double total = 0;
      for (auto& score : scores) { score = std::exp(score - maximum); total += score; }
      for (int d = 0; d < dim; ++d) {
        double value = 0;
        for (int key = 0; key < n; ++key) value += scores[key] / total * vp[size_t(key) * inner + head * dim + d];
        compressed[size_t(t) * inner + head * dim + d] = bf(float(value));
      }
    }
  std::vector<float> expected(count), expected_gate(count);
  for (int row = 0; row < seq; ++row)
    for (int head = 0; head < heads; ++head) {
      const auto* mask = masks.data() + (size_t(head) * n + tiles.row_tiles[row]) * n;
      std::vector<double> scores(seq, -std::numeric_limits<double>::infinity());
      for (int key = 0; key < seq; ++key) {
        if (!mask[tiles.row_tiles[key]]) continue;
        double score = 0;
        for (int d = 0; d < dim; ++d)
          score += double(__bfloat162float(q[size_t(row) * inner + head * dim + d])) * __bfloat162float(k[size_t(key) * inner + head * dim + d]);
        scores[key] = score / std::sqrt(double(dim));
      }
      const double maximum = *std::max_element(scores.begin(), scores.end());
      double sum = 0;
      for (auto& score : scores) { score = std::exp(score - maximum); sum += score; }
      for (int d = 0; d < dim; ++d) {
        double value = 0;
        for (int key = 0; key < seq; ++key) value += scores[key] / sum * __bfloat162float(v[size_t(key) * inner + head * dim + d]);
        const size_t i = size_t(row) * inner + head * dim + d;
        expected[i] = bf(float(value));
        expected_gate[i] = bf(expected[i] + bf(__bfloat162float(gate[i]) * compressed[size_t(tiles.row_tiles[row]) * inner + head * dim + d]));
      }
    }
  Stream stream;
  DeviceBuffer<int32_t> rows(tiles.rows.size()), sizes(tiles.sizes.size()), row_tiles(tiles.row_tiles.size());
  rows.copy_from_host(tiles.rows.data(), tiles.rows.size(), stream.get());
  sizes.copy_from_host(tiles.sizes.data(), tiles.sizes.size(), stream.get());
  row_tiles.copy_from_host(tiles.row_tiles.data(), tiles.row_tiles.size(), stream.get());
  DeviceBuffer<__nv_bfloat16> dq(count), dk(count), dv(count), dg(count), output(count), coarse(size_t(n) * inner);
  dq.copy_from_host(q.data(), count, stream.get()); dk.copy_from_host(k.data(), count, stream.get());
  dv.copy_from_host(v.data(), count, stream.get()); dg.copy_from_host(gate.data(), count, stream.get());
  VsaConfig config{n, tiles.prefix_tiles, heads, dim, rows.get(), sizes.get(), row_tiles.get()};
  Workspace ws; ws.reserve(vsa_attention_workspace_bytes(n, heads, dim));
  vsa_attention_forward(stream.get(), dq.get(), dk.get(), dv.get(), output.get(), coarse.get(), config, ws);
  std::vector<__nv_bfloat16> got(count), got_coarse(size_t(n) * inner);
  output.copy_to_host(got.data(), count, stream.get()); coarse.copy_to_host(got_coarse.data(), got_coarse.size(), stream.get());
  SLOPFAB_CUDA_CHECK(cudaStreamSynchronize(stream.get()));
  double max_error = 0, coarse_error = 0;
  for (size_t i = 0; i < count; ++i) max_error = std::max(max_error, double(std::fabs(expected[i] - __bfloat162float(got[i]))));
  for (size_t i = 0; i < compressed.size(); ++i) coarse_error = std::max(coarse_error, double(std::fabs(compressed[i] - __bfloat162float(got_coarse[i]))));
  CHECK_MSG(max_error < 0.004, "VSA sparse D=%d max error %g", dim, max_error);
  CHECK_MSG(coarse_error < 0.001, "VSA pooled D=%d max error %g", dim, coarse_error);
  // Offset/chunk boundaries need not coincide with tile boundaries.
  for (int start = 0; start < seq; start += 37)
    vsa_add_compression(stream.get(), output.get() + size_t(start) * inner,
        dg.get() + size_t(start) * inner, coarse.get(), start, std::min(37, seq - start), config);
  output.copy_to_host(got.data(), count, stream.get());
  SLOPFAB_CUDA_CHECK(cudaStreamSynchronize(stream.get()));
  max_error = 0;
  for (size_t i = 0; i < count; ++i) max_error = std::max(max_error, double(std::fabs(expected_gate[i] - __bfloat162float(got[i]))));
  CHECK_MSG(max_error < 0.004, "VSA gated D=%d max error %g", dim, max_error);
}
}  // namespace

SLOPFAB_TEST(vsa_attention_cpu_reference) {
  verify_vsa(64);
  verify_vsa(128);
}

SLOPFAB_TEST(vsa_real_checkpoint_forward) {
  const char* path = std::getenv("SLOPFAB_VSA_CHECKPOINT");
  if (!path) { SKIP_MISSING_FIXTURE("set SLOPFAB_VSA_CHECKPOINT for the FastH3 V2 integration test"); return; }
  using namespace slopfab;
  dit::SequenceLayout layout;
  layout.num_text = 8; layout.num_audio_latents = 7; layout.num_audio_rows = 14;
  layout.num_latent_frames = 5; layout.latent_height = 10; layout.latent_width = 14;
  layout.num_video_rows = 175;
  SafeTensors checkpoint; checkpoint.open(path);
  auto prompt = test::make_data(8 * 5120, 31, 0.1f);
  auto video = test::make_data(175 * 96, 41, 0.5f);
  auto audio = test::make_data(14 * 32, 51, 0.5f);
  std::vector<float> resident_video, resident_audio;
  const auto timesteps = dit::build_row_timesteps(layout, dit::build_indices(layout), 0.2f, 0.3f);
  // Three blocks exercise reuse of the streamer's two transfer slots.
  for (int offload : {0, 3}) {
    dit::Transformer model;
    dit::TransformerLoadOptions options; options.layout = &layout; options.offload_blocks = offload;
    model.load(checkpoint, {}, nullptr, options);
    CHECK(model.num_blocks() == 50);
    CHECK(model.offloaded_blocks() == offload);
    model.prepare_text(prompt.data(), 8);
    model.prepare_sequence(layout, dit::build_indices(layout), dit::build_position_ids(layout));
    std::vector<float> ov(video.size()), oa(audio.size());
    model.forward(video.data(), audio.data(), timesteps, ov.data(), oa.data());
    CHECK(std::all_of(ov.begin(), ov.end(), [](float x) { return std::isfinite(x); }));
    CHECK(std::all_of(oa.begin(), oa.end(), [](float x) { return std::isfinite(x); }));
    CHECK(std::any_of(ov.begin(), ov.end(), [](float x) { return x != 0; }));
    if (offload == 0) { resident_video = ov; resident_audio = oa; }
    else { CHECK(ov == resident_video); CHECK(oa == resident_audio); }
  }
}
