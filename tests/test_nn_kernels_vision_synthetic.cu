#include "detail/nn_kernels_fixture.h"

SLOPFAB_TEST_CATEGORY(qwen_vision_layernorm_and_gelu, "synthetic") {
  const int rows = 2, dim = 7;
  const auto x = bf16_round(make_data(rows * dim, 8101, 2.0f));
  const auto w = bf16_round(make_data(dim, 8102, 0.3f));
  const auto b = bf16_round(make_data(dim, 8103, 0.2f));
  std::vector<float> want(rows * dim);
  for (int r = 0; r < rows; ++r) {
    float mean = 0, var = 0;
    for (int j = 0; j < dim; ++j) mean += x[r * dim + j];
    mean /= dim;
    for (int j = 0; j < dim; ++j) { float d = x[r * dim + j] - mean; var += d * d; }
    const float inv = 1.0f / std::sqrt(var / dim + 1e-6f);
    for (int j = 0; j < dim; ++j)
      want[r * dim + j] = (x[r * dim + j] - mean) * inv * w[j] + b[j];
  }
  BfBuf dx(x), dw(w), db(b), out(rows * dim);
  slopfab::cuda::launch_layernorm_affine(dx.p(), dw.p(), db.p(), out.p(), rows, dim, 1e-6f, nullptr);
  CHECK_CLOSE(want, out.host(), 2e-2, "vision layernorm");

  const auto gx = bf16_round(std::vector<float>{-3, -1, 0, 0.5f, 2});
  std::vector<float> gw(gx.size());
  for (size_t i = 0; i < gx.size(); ++i) {
    const float v = gx[i];
    gw[i] = .5f * v * (1 + std::tanh(0.7978845608028654f * (v + .044715f * v * v * v)));
  }
  BfBuf dg(gx);
  slopfab::cuda::launch_gelu_tanh(dg.p(), gx.size(), nullptr);
  CHECK_CLOSE(gw, dg.host(), 1e-2, "vision gelu tanh");
}



SLOPFAB_TEST_CATEGORY(qwen_vision_merge_and_deepstack_scatter, "synthetic") {
  const int groups = 2, dim = 3;
  const auto x = bf16_round(make_data(groups * 4 * dim, 8110, 1.0f));
  BfBuf dx(x), merged(x.size());
  slopfab::cuda::launch_merge_four_rows(dx.p(), merged.p(), groups, dim, nullptr);
  CHECK_CLOSE(x, merged.host(), 0, "merge four contiguous rows");

  const std::vector<int32_t> rows = {1, 4};
  const auto add = bf16_round(std::vector<float>{1, 2, 3, -1, -.5f, .25f});
  auto base = bf16_round(make_data(6 * dim, 8111, .2f));
  auto want = base;
  for (int r = 0; r < 2; ++r)
    for (int d = 0; d < dim; ++d) want[rows[r] * dim + d] += add[r * dim + d];
  BfBuf dadd(add), dbase(base); auto didx = to_device_i32(rows);
  slopfab::cuda::launch_scatter_add_rows(dadd.p(), didx.get(), dbase.p(), 2, dim, nullptr);
  CHECK_CLOSE(want, dbase.host(), 2e-2, "deepstack additive scatter");
}
