#include "detail/nn_kernels_fixture.h"
#include "detail/nn_attention_fixture.h"

SLOPFAB_TEST_CATEGORY(attention_sol_pipeline_real_scale_finite, "synthetic") {
  REQUIRE_SM120_TEST("Sol TMA pipeline");
  CublasScope cb;
  // More than 64 physical blocks exercises multiple compacted approximate
  // groups and a ragged tail with activation ranges observed in H3 captures.
  const int seq = 4097, dim = 128;
  const auto q = bf16_round(make_data(size_t(seq) * dim, 941u, 14.0f));
  const auto k = bf16_round(make_data(size_t(seq) * dim, 942u, 12.0f));
  const auto v = bf16_round(make_data(size_t(seq) * dim, 943u, 72.0f));
  BfBuf dq(q), dk(k), dv(v), dout(size_t(seq) * dim);
  slopfab::cuda::AttentionConfig cfg;
  cfg.seq_len = seq;
  cfg.num_heads = 1;
  cfg.head_dim = dim;
  cfg.exact_prefix = 419;
  cfg.sol_pipeline = true;
  cfg.sol_beta = 1.0f;
  Workspace ws;
  ws.reserve(slopfab::cuda::attention_workspace_bytes(cfg, slopfab::cuda::AttentionBackend::kSol));
  slopfab::cuda::attention_forward(cb.h, nullptr, dq.p(), dk.p(), dv.p(), dout.p(), cfg,
                                   slopfab::cuda::AttentionBackend::kSol, ws);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  const auto got = dout.host();
  size_t bad = 0;
  for (float x : got)
    bad += !std::isfinite(x);
  CHECK_MSG(bad == 0, "Sol real-scale grouped pipeline produced %zu non-finite values", bad);
}
