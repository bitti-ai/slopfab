#pragma once
#include "../harness.h"
#include "slopfab/dit/veda.h"
#include "slopfab/safetensors_write.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <numeric>

namespace slopfab::test {
struct VedaFixture {
  std::string path;
  std::shared_ptr<dit::VedaRequest> request;
  std::vector<uint16_t> q, k, v;

  explicit VedaFixture(int dim, const std::string& budget, const std::string& backend,
                       bool globals = true) {
    path = (std::filesystem::temp_directory_path() / ("slopfab_veda_" + backend + ".safetensors"))
               .string();
    constexpr int heads = 2;
    std::map<std::string, std::string> metadata{
        {"format", "miowtion-veda-predictor-v1"},
        {"num_layers", "2"},
        {"num_heads", "2"},
        {"head_dim", std::to_string(dim)},
        {"keep_ratio", "0.1"},
        {"plans",
         R"({"test":{"geometry":"test","grid":[5,5,9],"shapes":["4x4x8","2x8x8"],"head_shape":[[0,1],[1,0]]}})"}};
    std::vector<TensorWrite> weights;
    for (int layer = 0; layer < 2; ++layer)
      for (int part = 0; part < 2; ++part) {
        const auto name = "layers." + std::to_string(layer) + (part == 0 ? ".proj_q" : ".proj_k");
        weights.push_back(
            {name,
             {heads, 3 * dim, dim},
             make_data(size_t(heads) * 3 * dim * dim, 701 + layer * 11 + part, .04f)});
      }
    write_safetensors(path, weights, metadata);
    dit::SequenceLayout layout;
    layout.num_text = globals ? 3 : 0;
    layout.num_condition_video = globals ? 65 : 0;
    layout.num_audio_rows = globals ? 67 : 0;
    layout.num_latent_frames = 5;
    layout.latent_height = 10;
    layout.latent_width = 18;
    layout.num_video_rows = 225;
    dit::VedaOptions options;
    options.predictor_path = path;
    options.generated_sparsity = budget;
    request = dit::prepare_veda(options, layout);
    const size_t count = size_t(layout.total_rows()) * heads * dim;
    auto fill = [&](auto& out, unsigned seed) {
      const auto values = make_data(count, seed, 1.0f);
      out.resize(count);
      for (size_t i = 0; i < count; ++i)
        out[i] = f32_to_bf16(values[i]);
    };
    fill(q, 71);
    fill(k, 83);
    fill(v, 97);
  }

  ~VedaFixture() {
    request.reset();
    std::error_code ec;
    std::filesystem::remove(path, ec);
  }

  std::vector<float> reference(int layer) const {
    const int dim = request->bundle.head_dim, heads = request->bundle.heads,
              seq = request->layout.total_rows();
    const auto pq = request->bundle.projection(layer, true),
               pk = request->bundle.projection(layer, false);
    std::vector<float> out(q.size());
    for (int head = 0; head < heads; ++head) {
      const int shape = request->plan.head_shape[layer][head];
      const auto& t = request->tilings[shape];
      const int n = t.video_tiles, tiles = int(t.sizes.size());
      std::vector<int> row_tile(seq);
      for (int tile = 0; tile < tiles; ++tile)
        for (int i = 0; i < t.sizes[tile]; ++i)
          row_tile[t.rows[tile * 128 + i]] = tile;
      auto predict = [&](const auto& input, const auto& projection) {
        std::vector<double> predicted(size_t(n) * dim);
        for (int tile = 0; tile < n; ++tile) {
          std::vector<double> features(3 * dim);
          for (int d = 0; d < dim; ++d) {
            double sum = 0, high = -std::numeric_limits<double>::infinity(), low = -high;
            for (int r = 0; r < t.sizes[tile]; ++r) {
              const double value =
                  bf16_to_f32(input[(size_t(t.rows[tile * 128 + r]) * heads + head) * dim + d]);
              sum += value;
              high = std::max(high, value);
              low = std::min(low, value);
            }
            features[d] = sum / t.sizes[tile];
            features[dim + d] = high;
            features[2 * dim + d] = low;
          }
          for (int d = 0; d < dim; ++d) {
            double sum = features[d];
            for (int f = 0; f < 3 * dim; ++f)
              sum += features[f] * bf16_to_f32(projection[(size_t(head) * 3 * dim + f) * dim + d]);
            predicted[size_t(tile) * dim + d] = sum;
          }
        }
        return predicted;
      };
      const auto qhat = predict(q, pq), khat = predict(k, pk);
      std::vector<uint8_t> mask(size_t(tiles) * tiles);
      for (int row = 0; row < tiles; ++row) {
        for (int col = 0; col < tiles; ++col)
          mask[size_t(row) * tiles + col] = row >= n || col >= n;
        if (row >= n)
          continue;
        std::vector<double> scores(n);
        for (int col = 0; col < n; ++col)
          for (int d = 0; d < dim; ++d)
            scores[col] += qhat[size_t(row) * dim + d] * khat[size_t(col) * dim + d];
        scores[row] = std::numeric_limits<double>::infinity();
        std::vector<int> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) {
          return scores[a] > scores[b] || (scores[a] == scores[b] && a < b);
        });
        for (int i = 0; i < request->keeps[shape][row]; ++i)
          mask[size_t(row) * tiles + order[i]] = 1;
      }
      for (int row = 0; row < seq; ++row) {
        std::vector<double> scores(seq, -std::numeric_limits<double>::infinity());
        for (int col = 0; col < seq; ++col) {
          if (!mask[size_t(row_tile[row]) * tiles + row_tile[col]])
            continue;
          double dot = 0;
          for (int d = 0; d < dim; ++d)
            dot += double(bf16_to_f32(q[(size_t(row) * heads + head) * dim + d])) *
                   bf16_to_f32(k[(size_t(col) * heads + head) * dim + d]);
          scores[col] = dot / std::sqrt(double(dim));
        }
        const double maximum = *std::max_element(scores.begin(), scores.end());
        double sum = 0;
        for (auto& x : scores) {
          x = std::exp(x - maximum);
          sum += x;
        }
        for (int d = 0; d < dim; ++d) {
          double value = 0;
          for (int col = 0; col < seq; ++col)
            value += scores[col] / sum * bf16_to_f32(v[(size_t(col) * heads + head) * dim + d]);
          out[(size_t(row) * heads + head) * dim + d] = bf16_to_f32(f32_to_bf16(float(value)));
        }
      }
    }
    return out;
  }
};
} // namespace slopfab::test
