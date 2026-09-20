#include "detail/nn_kernels_fixture.h"
#include "detail/nn_attention_fixture.h"

SLOPFAB_TEST_CATEGORY(attention_blocked, "synthetic") {
  CublasScope cb;
  const int seq = 512;
  const int heads = 4;
  const int head_dim = 128;
  const int width = heads * head_dim;

  const std::vector<float> q = bf16_round(make_data(size_t(seq) * width, 161u, 0.3f));
  const std::vector<float> k = bf16_round(make_data(size_t(seq) * width, 162u, 0.3f));
  const std::vector<float> v = bf16_round(make_data(size_t(seq) * width, 163u, 1.0f));

  BfBuf dq(q), dk(k), dv(v);

  slopfab::cuda::AttentionConfig cfg;
  cfg.seq_len = seq;
  cfg.num_heads = heads;
  cfg.head_dim = head_dim;
  CHECK_NEAR(cfg.effective_scale(), 1.0 / std::sqrt(128.0), 1e-7);  // computed in fp32

  const std::vector<float> want =
      cpu_attention(q, k, v, seq, heads, heads, head_dim, cfg.effective_scale());

  // The result must not depend on query_block. A broken online-softmax rescale
  // is silent under a single block size and only shows up here.
  std::vector<std::vector<float>> results;
  const int blocks[] = {64, 128, 512};
  for (int bq : blocks) {
    cfg.query_block = bq;
    BfBuf dout(size_t(seq) * width);
    Workspace ws;
    ws.reserve(slopfab::cuda::attention_workspace_bytes(cfg, slopfab::cuda::AttentionBackend::kBlocked));
    slopfab::cuda::attention_forward(cb.h, nullptr, dq.p(), dk.p(), dv.p(), dout.p(), cfg,
                                    slopfab::cuda::AttentionBackend::kBlocked, ws);
    SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
    results.push_back(dout.host());
    CHECK_CLOSE_REL(want, results.back(), 1e-3, 1e-2,
                    ("attention vs dense CPU, query_block " + std::to_string(bq)).c_str());
  }
  CHECK_CLOSE_REL(results[0], results[1], 1e-3, 1e-2, "attention query_block 64 == 128");
  CHECK_CLOSE_REL(results[0], results[2], 1e-3, 1e-2, "attention query_block 64 == 512");

  // The fused backend must agree with the dense CPU reference and with the
  // blocked backend it replaces. It takes no workspace, so pass an empty one --
  // if it ever starts allocating, this fails rather than silently reading
  // whatever the caller happened to leave reserved.
  {
    cfg.query_block = 1024;
    BfBuf dfused(size_t(seq) * width);
    Workspace ws;
    CHECK(slopfab::cuda::attention_workspace_bytes(cfg, slopfab::cuda::AttentionBackend::kFused) == 0);
    slopfab::cuda::attention_forward(cb.h, nullptr, dq.p(), dk.p(), dv.p(), dfused.p(), cfg,
                                    slopfab::cuda::AttentionBackend::kFused, ws);
    SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
    const std::vector<float> got = dfused.host();
    CHECK_CLOSE_REL(want, got, 1e-3, 1e-2, "fused attention vs dense CPU");
    CHECK_CLOSE_REL(results[0], got, 1e-3, 1e-2, "fused attention == blocked attention");
  }

  // head_dim outside the instantiated set must be refused, not silently wrong.
  {
    slopfab::cuda::AttentionConfig odd = cfg;
    odd.head_dim = 96;
    bool threw = false;
    try {
      Workspace ws;
      BfBuf dodd(size_t(seq) * heads * 96);
      slopfab::cuda::attention_forward(cb.h, nullptr, dq.p(), dk.p(), dv.p(), dodd.p(), odd,
                                      slopfab::cuda::AttentionBackend::kFused, ws);
    } catch (const std::exception&) {
      threw = true;
    }
    CHECK(threw);
  }
}

SLOPFAB_TEST_CATEGORY(attention_fused_ragged_tail, "synthetic") {
  CublasScope cb;
  const int heads = 3;
  const int head_dim = 128;
  const int width = heads * head_dim;

  for (int seq : {17, 61, 127, 199}) {
    const std::vector<float> q = bf16_round(make_data(size_t(seq) * width, 401u + seq, 0.3f));
    const std::vector<float> k = bf16_round(make_data(size_t(seq) * width, 402u + seq, 0.3f));
    const std::vector<float> v = bf16_round(make_data(size_t(seq) * width, 403u + seq, 1.0f));
    BfBuf dq(q), dk(k), dv(v), dout(size_t(seq) * width);

    slopfab::cuda::AttentionConfig cfg;
    cfg.seq_len = seq;
    cfg.num_heads = heads;
    cfg.head_dim = head_dim;

    const std::vector<float> want =
        cpu_attention(q, k, v, seq, heads, heads, head_dim, cfg.effective_scale());

    Workspace ws;
    slopfab::cuda::attention_forward(cb.h, nullptr, dq.p(), dk.p(), dv.p(), dout.p(), cfg,
                                    slopfab::cuda::AttentionBackend::kFused, ws);
    SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
    CHECK_CLOSE_REL(want, dout.host(), 1e-3, 1e-2,
                    ("fused attention, ragged seq " + std::to_string(seq)).c_str());
  }
}

SLOPFAB_TEST_CATEGORY(attention_compact_queries_match_full_buffers, "synthetic") {
  CublasScope cb;
  for (int dim : {64, 128}) for (int seq : {1, 127, 128, 259, 2051}) {
    const int width = 2 * dim;
    const auto q = bf16_round(make_data(size_t(seq) * width, 9701, 0.3f));
    const auto k = bf16_round(make_data(size_t(seq) * width, 9702, 0.3f));
    const auto v = bf16_round(make_data(size_t(seq) * width, 9703, 1.0f));
    BfBuf dq(q), dk(k), dv(v), full(size_t(seq) * width);
    slopfab::cuda::AttentionConfig cfg;
    cfg.seq_len = seq; cfg.num_heads = 2; cfg.head_dim = dim;
    for (bool banded : {false, true}) {
      if (banded && seq < 259) continue;
      slopfab::cuda::DeviceBuffer<int32_t> bands;
      if (banded) {
        std::vector<int32_t> ranges;
        for (int tile = 0; tile < (seq + 127) / 128; ++tile)
          ranges.insert(ranges.end(), {0, 64, 128 + (tile % 2) * 64, (seq + 63) / 64 * 64});
        bands.allocate(ranges.size()); bands.copy_from_host(ranges.data(), ranges.size());
      }
      cfg.band_ranges = bands.get();
      Workspace ws;
      slopfab::cuda::attention_forward(cb.h, nullptr, dq.p(), dk.p(), dv.p(), full.p(),
                                       cfg, slopfab::cuda::AttentionBackend::kFused, ws);
      const auto expected = full.host();
      for (int chunk : {128, 256}) {
        std::vector<float> actual;
        for (int start = 0; start < seq; start += chunk) {
          const int rows = std::min(chunk, seq - start);
          const size_t count = static_cast<size_t>(rows) * width;
          BfBuf compact_q(std::vector<float>(q.begin() + static_cast<size_t>(start) * width,
                                            q.begin() + static_cast<size_t>(start + rows) * width));
          BfBuf compact_out(std::vector<float>(count + 16, 42.0f));
          slopfab::cuda::attention_forward_query_chunk(nullptr, compact_q.p(), dk.p(), dv.p(),
                                                       compact_out.p(), cfg, start, rows);
          const auto got = compact_out.host();
          CHECK(std::all_of(got.begin() + count, got.end(), [](float x) { return x == 42.0f; }));
          actual.insert(actual.end(), got.begin(), got.begin() + count);
        }
        CHECK_CLOSE(expected, actual, 0.0, "compact Q/output preserves Flash2 bits and band offsets");
      }
    }
  }
}

SLOPFAB_TEST_CATEGORY(attention_sol, "synthetic") {
  REQUIRE_SM120_TEST("Sol attention");
  CublasScope cb;
  const int seq = 263;  // four full blocks, local routes, and a ragged tail
  const int heads = 2;
  const int dim = 128;
  const int width = heads * dim;
  const std::vector<float> q = bf16_round(make_data(size_t(seq) * width, 911u, 0.3f));
  std::vector<float> k = bf16_round(make_data(size_t(seq) * width, 912u, 0.3f));
  const std::vector<float> v = bf16_round(make_data(size_t(seq) * width, 913u, 1.0f));

  slopfab::cuda::AttentionConfig cfg;
  cfg.seq_len = seq;
  cfg.num_heads = heads;
  cfg.head_dim = dim;
  cfg.sol_pipeline = true;
  CHECK_NEAR(cfg.sol_beta, 1.0f, 0.0);
  const size_t bytes =
      slopfab::cuda::attention_workspace_bytes(cfg, slopfab::cuda::AttentionBackend::kSol);
  CHECK(bytes > 0);
  CHECK(bytes < size_t(seq) * heads * dim * sizeof(float));

  auto run = [&](const std::vector<float>& keys, float beta) {
    BfBuf dq(q), dk(keys), dv(v), dout(size_t(seq) * width);
    cfg.sol_beta = beta;
    Workspace ws;
    ws.reserve(bytes);
    slopfab::cuda::attention_forward(cb.h, nullptr, dq.p(), dk.p(), dv.p(), dout.p(), cfg,
                                    slopfab::cuda::AttentionBackend::kSol, ws);
    SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
    return dout.host();
  };

  // A cutoff below every finite proxy selects every block, reducing Sol-Attn
  // to exact attention. This pins the common online-softmax state and tail.
  const std::vector<float> want =
      cpu_attention(q, k, v, seq, heads, heads, dim, cfg.effective_scale());
  CHECK_CLOSE_REL(want, run(k, -1.0e6f), 1e-3, 1e-2,
                  "Sol-Attn all-selected equals dense attention");

  // If K is constant within each physical block, the zeroth-order correction
  // is mathematically exact even when every routable block is rejected. This
  // independently catches mean-vs-sum and ragged-tail multiplicity mistakes.
  for (int row = 0; row < seq; ++row) {
    const int source = (row / 64) * 64;
    for (int x = 0; x < width; ++x) k[size_t(row) * width + x] = k[size_t(source) * width + x];
  }
  const std::vector<float> constant_want =
      cpu_attention(q, k, v, seq, heads, heads, dim, cfg.effective_scale());
  CHECK_CLOSE_REL(constant_want, run(k, 1.0e6f), 1e-3, 1e-2,
                  "Sol-Attn rejected constant-K blocks equal dense attention");

  // Forced-exact prefix includes the crossing physical block, so selecting an
  // arbitrary multimodal boundary cannot approximate any prefix key.
  cfg.exact_prefix = 70;
  CHECK_CLOSE_REL(constant_want, run(k, 1.0e6f), 1e-3, 1e-2,
                  "Sol-Attn forced-exact prefix");

  // Nonconstant mixed-route oracle: six blocks ensure prefix, local and
  // threshold-selected routes coexist with corrected rejected routes.
  {
    const int mixed_seq = 321;
    std::vector<float> mq = bf16_round(make_data(size_t(mixed_seq) * dim, 921u, 0.8f));
    std::vector<float> mk = bf16_round(make_data(size_t(mixed_seq) * dim, 922u, 0.8f));
    std::vector<float> mv = bf16_round(make_data(size_t(mixed_seq) * dim, 923u, 1.0f));
    int selected = 0, rejected = 0;
    const std::vector<float> oracle = cpu_sol_attention(
        mq, mk, mv, mixed_seq, 70, 1.0f / std::sqrt(float(dim)), 1.0f, &selected, &rejected);
    CHECK(selected > 0 && rejected > 0);
    BfBuf dqm(mq), dkm(mk), dvm(mv), dom(size_t(mixed_seq) * dim);
    slopfab::cuda::AttentionConfig mixed;
    mixed.seq_len = mixed_seq;
    mixed.num_heads = 1;
    mixed.head_dim = dim;
    mixed.exact_prefix = 70;
    mixed.sol_pipeline = true;
    Workspace mixed_ws;
    mixed_ws.reserve(slopfab::cuda::attention_workspace_bytes(
        mixed, slopfab::cuda::AttentionBackend::kSol));
    slopfab::cuda::attention_forward(cb.h, nullptr, dqm.p(), dkm.p(), dvm.p(), dom.p(), mixed,
                                    slopfab::cuda::AttentionBackend::kSol, mixed_ws);
    SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
    CHECK_CLOSE_REL(oracle, dom.host(), 2e-3, 2e-2, "Sol-Attn mixed-route CPU oracle");
  }
}

SLOPFAB_TEST_CATEGORY(attention_sol_pipeline_exact, "synthetic") {
  REQUIRE_SM120_TEST("Sol TMA pipeline");
  CublasScope cb;
  const int seq = 128, heads = 1, dim = 128;
  const auto q = bf16_round(make_data(size_t(seq) * dim, 921u, 0.3f));
  const auto k = bf16_round(make_data(size_t(seq) * dim, 922u, 0.3f));
  const auto v = bf16_round(make_data(size_t(seq) * dim, 923u, 1.0f));
  const auto want = cpu_attention(q, k, v, seq, heads, heads, dim,
                                  1.0f / std::sqrt(float(dim)));
  BfBuf dq(q), dk(k), dv(v), dout(size_t(seq) * dim);
  slopfab::cuda::AttentionConfig cfg;
  cfg.seq_len = seq; cfg.num_heads = heads; cfg.head_dim = dim;
  cfg.exact_prefix = seq; cfg.sol_pipeline = true;
  Workspace ws;
  ws.reserve(slopfab::cuda::attention_workspace_bytes(
      cfg, slopfab::cuda::AttentionBackend::kSol));
  slopfab::cuda::attention_forward(cb.h, nullptr, dq.p(), dk.p(), dv.p(), dout.p(), cfg,
                                  slopfab::cuda::AttentionBackend::kSol, ws);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK_CLOSE_REL(want, dout.host(), 2e-3, 2e-2,
                  "Sol SM120 pipeline exact oracle");
}

SLOPFAB_TEST_CATEGORY(attention_sol_pipeline_mixed, "synthetic") {
  REQUIRE_SM120_TEST("Sol TMA pipeline");
  CublasScope cb;
  const int seq=384, dim=128;
  const auto q=bf16_round(make_data(size_t(seq)*dim,931u,0.8f));
  const auto k=bf16_round(make_data(size_t(seq)*dim,932u,0.8f));
  const auto v=bf16_round(make_data(size_t(seq)*dim,933u,1.0f));
  int selected=0,rejected=0;
  const auto want=cpu_sol_attention(q,k,v,seq,70,1.0f/std::sqrt(float(dim)),
                                    1.0f,&selected,&rejected);
  CHECK(selected>0 && rejected>0);
  BfBuf dq(q),dk(k),dv(v),dout(size_t(seq)*dim);
  slopfab::cuda::AttentionConfig cfg;
  cfg.seq_len=seq; cfg.num_heads=1; cfg.head_dim=dim;
  cfg.exact_prefix=70; cfg.sol_pipeline=true;
  Workspace ws; ws.reserve(slopfab::cuda::attention_workspace_bytes(
      cfg,slopfab::cuda::AttentionBackend::kSol));
  slopfab::cuda::attention_forward(cb.h,nullptr,dq.p(),dk.p(),dv.p(),dout.p(),cfg,
                                  slopfab::cuda::AttentionBackend::kSol,ws);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK_CLOSE_REL(want,dout.host(),2e-3,2e-2,"Sol pipeline mixed-route CPU oracle");
}

SLOPFAB_TEST_CATEGORY(attention_sol_pipeline_large_pooled_v, "synthetic") {
  REQUIRE_SM120_TEST("Sol TMA pipeline");
  CublasScope cb;
  // A 64-row pooled V sum is ~131k here: well beyond BF16's finite range,
  // while the corrected attention result remains a perfectly finite ~2k.
  // This reproduces the end-to-end failure that motivated keeping the
  // approximate V contraction in FP32.
  const int seq=321,dim=128;
  const auto q=bf16_round(make_data(size_t(seq)*dim,951u,0.4f));
  auto k=bf16_round(make_data(size_t(seq)*dim,952u,0.4f));
  std::vector<float> v(size_t(seq)*dim);
  for(int row=0;row<seq;++row) {
    const int source=(row/64)*64;
    for(int x=0;x<dim;++x) {
      k[size_t(row)*dim+x]=k[size_t(source)*dim+x];
      v[size_t(row)*dim+x]=2048.0f+float((x%7)-3)*8.0f;
    }
  }
  const auto want=cpu_attention(q,k,v,seq,1,1,dim,1.0f/std::sqrt(float(dim)));
  BfBuf dq(q),dk(k),dv(v),dout(size_t(seq)*dim);
  slopfab::cuda::AttentionConfig cfg;
  cfg.seq_len=seq;cfg.num_heads=1;cfg.head_dim=dim;
  cfg.exact_prefix=0;cfg.sol_pipeline=true;cfg.sol_beta=1.0e6f;
  Workspace ws;ws.reserve(slopfab::cuda::attention_workspace_bytes(
      cfg,slopfab::cuda::AttentionBackend::kSol));
  slopfab::cuda::attention_forward(cb.h,nullptr,dq.p(),dk.p(),dv.p(),dout.p(),cfg,
                                  slopfab::cuda::AttentionBackend::kSol,ws);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  const auto got=dout.host();
  size_t bad=0;
  for(float x:got) bad+=!std::isfinite(x);
  CHECK_MSG(bad==0,"Sol large pooled-V pipeline produced %zu non-finite values",bad);
  CHECK_CLOSE_REL(want,got,2e-3,2e-2,"Sol pipeline large pooled-V FP32 correction");
}

SLOPFAB_TEST_CATEGORY(attention_sol_rejects_invalid_error_weights, "synthetic") {
  REQUIRE_SM120_TEST("Sol attention");
  slopfab::cuda::AttentionConfig cfg;
  cfg.seq_len=64;cfg.num_heads=1;cfg.head_dim=128;
  slopfab::cuda::Workspace ws;
  auto rejected=[&]() {
    try {
      slopfab::cuda::sol_attention_forward(nullptr,nullptr,nullptr,nullptr,nullptr,cfg,ws);
      return false;
    } catch(const std::runtime_error&) { return true; }
  };
  cfg.sol_error_k=-1.0f;CHECK(rejected());
  cfg.sol_error_k=std::numeric_limits<float>::infinity();CHECK(rejected());
  cfg.sol_error_k=0.0f;cfg.sol_error_v=-1.0f;CHECK(rejected());
  cfg.sol_error_v=std::numeric_limits<float>::quiet_NaN();CHECK(rejected());
}

SLOPFAB_TEST_CATEGORY(attention_sol_zero_error_weight_ignores_infinite_residual, "synthetic") {
  REQUIRE_SM120_TEST("Sol attention");
  CublasScope cb;
  const int seq=321,dim=128;
  std::vector<float> q(size_t(seq)*dim,0.0f),k(size_t(seq)*dim),v;
  v=bf16_round(make_data(size_t(seq)*dim,961u,1.0f));
  // Squaring this finite BF16 value overflows FP32 residual preprocessing,
  // while alternating signs keep the centroid and zero-Q proxy finite.
  for(int row=0;row<seq;++row)for(int d=0;d<dim;++d)
    k[size_t(row)*dim+d]=((row+d)&1)?-3.0e38f:3.0e38f;
  k=bf16_round(k);
  BfBuf dq(q),dk(k),dv(v),dout(size_t(seq)*dim);
  slopfab::cuda::AttentionConfig cfg;
  cfg.seq_len=seq;cfg.num_heads=1;cfg.head_dim=dim;cfg.sol_pipeline=true;
  cfg.sol_beta=1.0e6f;cfg.sol_error_k=0.0f;cfg.sol_error_v=0.0f;
  Workspace ws;ws.reserve(slopfab::cuda::attention_workspace_bytes(
      cfg,slopfab::cuda::AttentionBackend::kSol));
  slopfab::cuda::attention_forward(cb.h,nullptr,dq.p(),dk.p(),dv.p(),dout.p(),cfg,
                                  slopfab::cuda::AttentionBackend::kSol,ws);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  size_t bad=0;for(float x:dout.host())bad+=!std::isfinite(x);
  CHECK_MSG(bad==0,"zero Sol error weights consumed infinite residual: %zu nonfinite",bad);
}

SLOPFAB_TEST_CATEGORY(attention_sage2_architecture_dispatch, "synthetic") {
  using Variant = slopfab::cuda::Sage2KernelVariant;
  CHECK(slopfab::cuda::sage2_ampere_dynamic_smem_bytes(64) == 20 * 1024);
  CHECK(slopfab::cuda::sage2_ampere_dynamic_smem_bytes(128) == 40 * 1024);
  CHECK(slopfab::cuda::sage2_variant_for_compute_capability(79) == Variant::kUnsupported);
  CHECK(slopfab::cuda::sage2_variant_for_compute_capability(80) == Variant::kUnsupported);
  CHECK(slopfab::cuda::sage2_variant_for_compute_capability(85) == Variant::kUnsupported);
  CHECK(slopfab::cuda::sage2_variant_for_compute_capability(86) == Variant::kAmpereFp16);
  CHECK(slopfab::cuda::sage2_variant_for_compute_capability(88) == Variant::kAmpereFp16);
  CHECK(slopfab::cuda::sage2_variant_for_compute_capability(89) == Variant::kUnsupported);
  CHECK(slopfab::cuda::sage2_variant_for_compute_capability(90) == Variant::kUnsupported);
  CHECK(slopfab::cuda::sage2_variant_for_compute_capability(120) == Variant::kBlackwellFp8);
  CHECK(slopfab::cuda::sage2_variant_for_compute_capability(121) == Variant::kUnsupported);
}

SLOPFAB_TEST_CATEGORY(attention_sage2, "synthetic") {
  CublasScope cb;
  const int seq = 199;
  const int heads = 2;
  const int head_dim = 128;
  const int width = heads * head_dim;
  const std::vector<float> q = bf16_round(make_data(size_t(seq) * width, 871u, 0.3f));
  const std::vector<float> k = bf16_round(make_data(size_t(seq) * width, 872u, 0.3f));
  const std::vector<float> v = bf16_round(make_data(size_t(seq) * width, 873u, 1.0f));
  const std::vector<float> want =
      cpu_attention(q, k, v, seq, heads, heads, head_dim, 1.0f / std::sqrt(128.0f));
  BfBuf dq(q), dk(k), dv(v), dout(size_t(seq) * width);
  slopfab::cuda::AttentionConfig cfg;
  cfg.seq_len = seq;
  cfg.num_heads = heads;
  cfg.head_dim = head_dim;
  Workspace ws;
  const size_t bytes =
      slopfab::cuda::attention_workspace_bytes(cfg, slopfab::cuda::AttentionBackend::kSage2);
  CHECK(bytes > 0);
  ws.reserve(bytes);
  slopfab::cuda::attention_forward(cb.h, nullptr, dq.p(), dk.p(), dv.p(), dout.p(), cfg,
                                  slopfab::cuda::AttentionBackend::kSage2, ws);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  const std::vector<float> got = dout.host();
  CHECK_CLOSE_REL(want, got, 2.5e-2, 1e-1, "sage2 attention vs dense CPU");

  // Quantization is deterministic, including the ragged Q/K/V padding.
  BfBuf again(size_t(seq) * width);
  ws.clear();
  slopfab::cuda::attention_forward(cb.h, nullptr, dq.p(), dk.p(), dv.p(), again.p(), cfg,
                                  slopfab::cuda::AttentionBackend::kSage2, ws);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(dout.bits() == again.bits());

  cfg.band_ranges = reinterpret_cast<const int32_t*>(uintptr_t{16});
  bool threw = false;
  try {
    ws.clear();
    slopfab::cuda::attention_forward(cb.h, nullptr, dq.p(), dk.p(), dv.p(), again.p(), cfg,
                                    slopfab::cuda::AttentionBackend::kSage2, ws);
  } catch (const std::exception&) {
    threw = true;
  }
  CHECK(threw);
}

SLOPFAB_TEST_CATEGORY(attention_sage2_head_dims_bit_stable, "synthetic") {
  const int heads = 8, kv_heads = 2;
  const bool check_blackwell_hashes = test_is_sm120();
  if (!check_blackwell_hashes) {
    SKIP_UNSUPPORTED_HARDWARE(
        "Blackwell Sage hashes are SM120-specific; dense-reference and determinism still run");
  }
  struct Shape { int seq; int head_dim; uint64_t digest; };
  const Shape shapes[] = {
      {64, 64, 0x3dece381009ce78cull},  {199, 128, 0xc755489173f5999cull},
      {130, 64, 0x51488ef18c4440adull}, {256, 128, 0xd01e0a63a85a68a9ull},
      {65, 64, 0x00fca10d10e3d6b9ull},  {321, 128, 0x0a67856f531f85c4ull},
  };

  std::vector<uint16_t> first[6];
  for (int round = 0; round < 2; ++round) {
    for (int si = 0; si < 6; ++si) {
      const int seq = shapes[si].seq, head_dim = shapes[si].head_dim;
      const size_t qn = size_t(seq) * heads * head_dim;
      const size_t kn = size_t(seq) * kv_heads * head_dim;
      const std::vector<float> q = bf16_round(make_data(qn, 4101u + si, 0.3f));
      const std::vector<float> k = bf16_round(make_data(kn, 4201u + si, 0.3f));
      const std::vector<float> v = bf16_round(make_data(kn, 4301u + si, 1.0f));
      BfBuf dq(q), dk(k), dv(v), dout(qn);
      slopfab::cuda::AttentionConfig cfg;
      cfg.seq_len = seq;
      cfg.num_heads = heads;
      cfg.head_dim = head_dim;
      Workspace ws;
      ws.reserve(slopfab::cuda::sage2_workspace_bytes(cfg, kv_heads));
      slopfab::cuda::sage2_attention_forward(nullptr, dq.p(), dk.p(), dv.p(), dout.p(), cfg,
                                            kv_heads, ws);
      SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
      const std::vector<uint16_t> bits = dout.bits();

      if (round == 0) {
        const std::vector<float> want =
            cpu_attention(q, k, v, seq, heads, kv_heads, head_dim,
                          1.0f / std::sqrt(float(head_dim)));
        CHECK_CLOSE_REL(want, dout.host(), 3.0e-2, 1.2e-1, "sage2 GQA vs dense CPU");
        first[si] = bits;
        // FNV-1a over the raw bf16 payload, against the frozen table above.
        uint64_t h = 1469598103934665603ull;
        for (uint16_t b : bits) {
          h = (h ^ (b & 0xffu)) * 1099511628211ull;
          h = (h ^ (b >> 8)) * 1099511628211ull;
        }
        if (check_blackwell_hashes) {
          CHECK_MSG(h == shapes[si].digest,
                    "sage2 output moved at seq=%d D=%d kv=%d: digest %016llx, expected %016llx",
                    seq, head_dim, kv_heads, static_cast<unsigned long long>(h),
                    static_cast<unsigned long long>(shapes[si].digest));
        }
      } else {
        size_t bad = 0;
        for (size_t i = 0; i < bits.size(); ++i) bad += bits[i] != first[si][i];
        CHECK_MSG(bad == 0,
                  "sage2 seq=%d D=%d drifted across interleaved head dims: %zu of %zu bf16 differ",
                  seq, head_dim, bad, bits.size());
      }
    }
  }
}

SLOPFAB_TEST_CATEGORY(attention_fused_head_dim_64, "synthetic") {
  CublasScope cb;
  const int heads = 5;
  const int head_dim = 64;
  const int width = heads * head_dim;

  for (int seq : {64, 129, 200}) {
    const std::vector<float> q = bf16_round(make_data(size_t(seq) * width, 601u + seq, 0.3f));
    const std::vector<float> k = bf16_round(make_data(size_t(seq) * width, 602u + seq, 0.3f));
    const std::vector<float> v = bf16_round(make_data(size_t(seq) * width, 603u + seq, 1.0f));
    BfBuf dq(q), dk(k), dv(v), dout(size_t(seq) * width);

    slopfab::cuda::AttentionConfig cfg;
    cfg.seq_len = seq;
    cfg.num_heads = heads;
    cfg.head_dim = head_dim;
    CHECK(slopfab::cuda::attention_preferred_backend(cfg) == slopfab::cuda::AttentionBackend::kFused);

    const std::vector<float> want =
        cpu_attention(q, k, v, seq, heads, heads, head_dim, cfg.effective_scale());

    Workspace ws;
    slopfab::cuda::attention_forward(cb.h, nullptr, dq.p(), dk.p(), dv.p(), dout.p(), cfg,
                                    slopfab::cuda::AttentionBackend::kFused, ws);
    SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
    CHECK_CLOSE_REL(want, dout.host(), 1e-3, 1e-2,
                    ("fused attention, head_dim 64, seq " + std::to_string(seq)).c_str());
  }

  // Grouped-query at 64 wide as well: `kvld` is num_kv_heads*64 there, which is
  // the stride the staging walks, and a kv head offset of `kv_head*64` is the
  // one place a 128-wide assumption would survive every test above.
  {
    const int seq = 96;
    const int kv_heads = 1;
    const std::vector<float> q = bf16_round(make_data(size_t(seq) * heads * head_dim, 611u, 0.3f));
    const std::vector<float> k =
        bf16_round(make_data(size_t(seq) * kv_heads * head_dim, 612u, 0.3f));
    const std::vector<float> v =
        bf16_round(make_data(size_t(seq) * kv_heads * head_dim, 613u, 1.0f));
    BfBuf dq(q), dk(k), dv(v), dout(size_t(seq) * heads * head_dim);

    slopfab::cuda::AttentionConfig cfg;
    cfg.seq_len = seq;
    cfg.num_heads = heads;
    cfg.head_dim = head_dim;

    const std::vector<float> want =
        cpu_attention(q, k, v, seq, heads, kv_heads, head_dim, cfg.effective_scale());
    Workspace ws;
    slopfab::cuda::attention_forward_gqa(cb.h, nullptr, dq.p(), dk.p(), dv.p(), dout.p(), cfg,
                                        kv_heads, slopfab::cuda::AttentionBackend::kFused, ws);
    SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
    CHECK_CLOSE_REL(want, dout.host(), 1e-3, 1e-2, "fused gqa attention, head_dim 64");
  }
}

SLOPFAB_TEST_CATEGORY(attention_fused_banded, "synthetic") {
  CublasScope cb;
  using slopfab::dit::SequenceLayout;

  SequenceLayout layout;
  layout.num_text = 5;
  layout.num_audio_rows = 8;
  layout.num_latent_frames = 12;
  layout.latent_height = 12;
  layout.latent_width = 14;  // R = 6*7 = 42, coprime-ish with the 64-row block
  layout.num_video_rows = layout.num_latent_frames * layout.rows_per_frame();

  const int seq = layout.total_rows();
  const int heads = 2;
  const int head_dim = 128;
  const int width = heads * head_dim;
  const int tile = slopfab::cuda::attention_fused_query_tile();
  const int align = slopfab::cuda::attention_fused_key_align();
  CHECK(layout.rows_per_frame() == 42);
  CHECK(layout.rows_per_frame() % align != 0);  // the rounding must be exercised

  const std::vector<float> q = bf16_round(make_data(size_t(seq) * width, 701u, 0.3f));
  const std::vector<float> k = bf16_round(make_data(size_t(seq) * width, 702u, 0.3f));
  const std::vector<float> v = bf16_round(make_data(size_t(seq) * width, 703u, 1.0f));
  BfBuf dq(q), dk(k), dv(v);

  slopfab::cuda::AttentionConfig cfg;
  cfg.seq_len = seq;
  cfg.num_heads = heads;
  cfg.head_dim = head_dim;

  for (int band : {2, 3, 5, 64}) {
    const slopfab::dit::BandedKeyRanges r =
        slopfab::dit::build_banded_key_ranges(layout, band, tile, align);
    DeviceBuffer<int32_t> dranges(r.ranges.size());
    dranges.copy_from_host(r.ranges.data(), r.ranges.size());

    BfBuf dout(size_t(seq) * width);
    slopfab::cuda::AttentionConfig banded = cfg;
    banded.band_ranges = dranges.get();
    Workspace ws;
    slopfab::cuda::attention_forward(cb.h, nullptr, dq.p(), dk.p(), dv.p(), dout.p(), banded,
                                    slopfab::cuda::AttentionBackend::kFused, ws);
    SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());

    const std::vector<float> want = cpu_attention_banded(
        q, k, v, seq, heads, heads, head_dim, cfg.effective_scale(), r.ranges, tile);
    CHECK_CLOSE_REL(want, dout.host(), 1e-3, 1e-2,
                    ("banded fused attention, band +/-" + std::to_string(band)).c_str());
  }

  // A band wide enough to span the sequence must reproduce full attention
  // exactly -- same bytes, not merely the same tolerance. If it does not, the
  // banded path differs from the unbanded one for reasons that have nothing to
  // do with banding.
  {
    const slopfab::dit::BandedKeyRanges wide =
        slopfab::dit::build_banded_key_ranges(layout, 64, tile, align);
    DeviceBuffer<int32_t> dranges(wide.ranges.size());
    dranges.copy_from_host(wide.ranges.data(), wide.ranges.size());

    BfBuf dfull(size_t(seq) * width), dwide(size_t(seq) * width);
    Workspace ws;
    slopfab::cuda::attention_forward(cb.h, nullptr, dq.p(), dk.p(), dv.p(), dfull.p(), cfg,
                                    slopfab::cuda::AttentionBackend::kFused, ws);
    slopfab::cuda::AttentionConfig banded = cfg;
    banded.band_ranges = dranges.get();
    slopfab::cuda::attention_forward(cb.h, nullptr, dq.p(), dk.p(), dv.p(), dwide.p(), banded,
                                    slopfab::cuda::AttentionBackend::kFused, ws);
    SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
    CHECK_MSG(dfull.bits() == dwide.bits(),
              "a band covering the sequence is not bit-identical to full attention");
  }

  // A narrow band must actually change the answer. Without this the test above
  // would pass just as well against a kernel that ignored `band_ranges`
  // entirely -- which is the failure the blocked backend throws to avoid.
  {
    const slopfab::dit::BandedKeyRanges narrow =
        slopfab::dit::build_banded_key_ranges(layout, 1, tile, align);
    DeviceBuffer<int32_t> dranges(narrow.ranges.size());
    dranges.copy_from_host(narrow.ranges.data(), narrow.ranges.size());
    BfBuf dfull(size_t(seq) * width), dnarrow(size_t(seq) * width);
    Workspace ws;
    slopfab::cuda::attention_forward(cb.h, nullptr, dq.p(), dk.p(), dv.p(), dfull.p(), cfg,
                                    slopfab::cuda::AttentionBackend::kFused, ws);
    slopfab::cuda::AttentionConfig banded = cfg;
    banded.band_ranges = dranges.get();
    slopfab::cuda::attention_forward(cb.h, nullptr, dq.p(), dk.p(), dv.p(), dnarrow.p(), banded,
                                    slopfab::cuda::AttentionBackend::kFused, ws);
    SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
    CHECK_MSG(dfull.bits() != dnarrow.bits(), "a +/-1 frame band did not change the output");
  }

  // The blocked backend has no banding and must refuse rather than quietly
  // returning full attention under a banded caller's name.
  {
    DeviceBuffer<int32_t> dranges(4);
    const std::vector<int32_t> z = {0, seq, 0, 0};
    dranges.copy_from_host(z.data(), z.size());
    slopfab::cuda::AttentionConfig banded = cfg;
    banded.band_ranges = dranges.get();
    BfBuf dout(size_t(seq) * width);
    Workspace ws;
    ws.reserve(slopfab::cuda::attention_workspace_bytes(banded,
                                                       slopfab::cuda::AttentionBackend::kBlocked));
    bool threw = false;
    try {
      slopfab::cuda::attention_forward(cb.h, nullptr, dq.p(), dk.p(), dv.p(), dout.p(), banded,
                                      slopfab::cuda::AttentionBackend::kBlocked, ws);
    } catch (const std::exception&) {
      threw = true;
    }
    CHECK(threw);
  }
}

SLOPFAB_TEST_CATEGORY(attention_fused_probability_precision, "synthetic") {
  CublasScope cb;
  const int seq = 1024;
  const int heads = 4;
  const int head_dim = 128;
  const int width = heads * head_dim;

  // Same amplitudes as the blocked path's test, so the two ratios are directly
  // comparable: scores at a few units, the order post-q_norm/k_norm H3 produces.
  const std::vector<float> q = bf16_round(make_data(size_t(seq) * width, 181u, 0.5f));
  const std::vector<float> k = bf16_round(make_data(size_t(seq) * width, 182u, 0.5f));
  const std::vector<float> v = bf16_round(make_data(size_t(seq) * width, 183u, 1.0f));
  BfBuf dq(q), dk(k), dv(v), dout(size_t(seq) * width);

  slopfab::cuda::AttentionConfig cfg;
  cfg.seq_len = seq;
  cfg.num_heads = heads;
  cfg.head_dim = head_dim;
  CHECK(slopfab::cuda::attention_preferred_backend(cfg) == slopfab::cuda::AttentionBackend::kFused);

  const std::vector<float> want =
      cpu_attention(q, k, v, seq, heads, heads, head_dim, cfg.effective_scale());

  double norm_sq = 0.0;
  for (double x : want) norm_sq += x * x;
  const double want_rms = std::sqrt(norm_sq / double(want.size()));

  // The floor: what rounding the exact answer to the output's own format costs.
  // Error below this is invisible by construction, whatever produced it.
  const std::vector<float> rounded = bf16_round(want);
  double floor_sq = 0.0;
  for (size_t i = 0; i < want.size(); ++i) {
    const double d = double(rounded[i]) - want[i];
    floor_sq += d * d;
  }
  const double output_floor_rel = std::sqrt(floor_sq / double(want.size())) / want_rms;

  Workspace ws;
  slopfab::cuda::attention_forward(cb.h, nullptr, dq.p(), dk.p(), dv.p(), dout.p(), cfg,
                                  slopfab::cuda::AttentionBackend::kFused, ws);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  const std::vector<float> got = dout.host();

  double err_sq = 0.0;
  double worst_abs = 0.0;
  for (size_t i = 0; i < got.size(); ++i) {
    const double d = double(got[i]) - want[i];
    err_sq += d * d;
    worst_abs = std::max(worst_abs, std::fabs(d));
  }
  const double rms_rel = std::sqrt(err_sq / double(got.size())) / want_rms;

  std::printf("  fused P precision: rms_rel %.3g (bf16 output floor %.3g, ratio %.2f), "
              "worst abs %.3g\n",
              rms_rel, output_floor_rel, rms_rel / output_floor_rel, worst_abs);
  CHECK_MSG(rms_rel < 1.10 * output_floor_rel,
            "fused probabilities: rms_rel %.4g is %.2fx the %.4g bf16 output floor, over the 1.10x "
            "bar; bf16 probabilities model at 1.21-1.27x",
            rms_rel, rms_rel / output_floor_rel, output_floor_rel);
}

SLOPFAB_TEST_CATEGORY(attention_fused_gqa, "synthetic") {
  CublasScope cb;
  const int seq = 96;
  const int heads = 6;
  const int kv_heads = 2;
  const int head_dim = 128;

  const std::vector<float> q = bf16_round(make_data(size_t(seq) * heads * head_dim, 511u, 0.3f));
  const std::vector<float> k = bf16_round(make_data(size_t(seq) * kv_heads * head_dim, 512u, 0.3f));
  const std::vector<float> v = bf16_round(make_data(size_t(seq) * kv_heads * head_dim, 513u, 1.0f));
  BfBuf dq(q), dk(k), dv(v), dout(size_t(seq) * heads * head_dim);

  slopfab::cuda::AttentionConfig cfg;
  cfg.seq_len = seq;
  cfg.num_heads = heads;
  cfg.head_dim = head_dim;

  const std::vector<float> want =
      cpu_attention(q, k, v, seq, heads, kv_heads, head_dim, cfg.effective_scale());

  Workspace ws;
  slopfab::cuda::attention_forward_gqa(cb.h, nullptr, dq.p(), dk.p(), dv.p(), dout.p(), cfg,
                                      kv_heads, slopfab::cuda::AttentionBackend::kFused, ws);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK_CLOSE_REL(want, dout.host(), 1e-3, 1e-2, "fused gqa attention vs dense CPU");
}

SLOPFAB_TEST_CATEGORY(attention_fp16_score_tile, "synthetic") {
  CublasScope cb;
  const int seq = 1024;
  const int heads = 4;
  const int head_dim = 128;
  const int width = heads * head_dim;

  // Amplitude 0.5 over 128 channels puts scores at a few units — the same order
  // as post-q_norm/k_norm H3, and far from the 65504 where fp16 would saturate.
  const std::vector<float> q = bf16_round(make_data(size_t(seq) * width, 181u, 0.5f));
  const std::vector<float> k = bf16_round(make_data(size_t(seq) * width, 182u, 0.5f));
  const std::vector<float> v = bf16_round(make_data(size_t(seq) * width, 183u, 1.0f));
  BfBuf dq(q), dk(k), dv(v);

  slopfab::cuda::AttentionConfig cfg;
  cfg.seq_len = seq;
  cfg.num_heads = heads;
  cfg.head_dim = head_dim;

  const std::vector<float> want =
      cpu_attention(q, k, v, seq, heads, heads, head_dim, cfg.effective_scale());

  double norm_sq = 0.0;
  for (float f : want) norm_sq += double(f) * f;
  const double want_rms = std::sqrt(norm_sq / double(want.size()));

  // `out` is bf16, so *any* correct implementation is at least this far from the
  // fp64 reference. Measuring the kernel against a fixed constant would mostly
  // measure that floor — one bf16 ulp is 0.4 % relative — so the bound below is
  // expressed as a multiple of it instead.
  const std::vector<float> rounded = bf16_round(want);
  double floor_sq = 0.0;
  for (size_t i = 0; i < want.size(); ++i) {
    const double d = double(rounded[i]) - want[i];
    floor_sq += d * d;
  }
  const double output_floor_rel = std::sqrt(floor_sq / double(want.size())) / want_rms;

  // Key blocks that do and do not divide the sequence, plus one that leaves a
  // short tail, crossed with two query blocks.
  const int key_blocks[] = {0, 128, 256, 384, 1024};
  const int query_blocks[] = {256, 1024};
  std::vector<std::vector<float>> results;
  double worst_rms_rel = 0.0;
  double worst_abs = 0.0;

  for (int kb : key_blocks) {
    for (int qb : query_blocks) {
      cfg.key_block = kb;
      cfg.query_block = qb;
      BfBuf dout(size_t(seq) * width);
      Workspace ws;
      ws.reserve(
          slopfab::cuda::attention_workspace_bytes(cfg, slopfab::cuda::AttentionBackend::kBlocked));
      slopfab::cuda::attention_forward(cb.h, nullptr, dq.p(), dk.p(), dv.p(), dout.p(), cfg,
                                      slopfab::cuda::AttentionBackend::kBlocked, ws);
      SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
      const std::vector<float> got = dout.host();

      double err_sq = 0.0;
      for (size_t i = 0; i < got.size(); ++i) {
        const double d = double(got[i]) - want[i];
        err_sq += d * d;
        worst_abs = std::max(worst_abs, std::fabs(d));
      }
      worst_rms_rel = std::max(worst_rms_rel, std::sqrt(err_sq / double(got.size())) / want_rms);

      CHECK_CLOSE_REL(want, got, 1e-3, 1e-2,
                      ("attention key_block " + std::to_string(kb) + " query_block " +
                       std::to_string(qb))
                          .c_str());
      results.push_back(got);
    }
  }

  // Every split must agree to the same tolerance. A correction factor applied to
  // the wrong operand, or skipped when the running max happens not to move,
  // shows up here and nowhere else in this file.
  for (size_t i = 1; i < results.size(); ++i) {
    CHECK_CLOSE_REL(results[0], results[i], 1e-3, 1e-2,
                    "attention result independent of key/query block");
  }

  // Recorded, not merely bounded. Measured: rms_rel 1.67e-3 against a 1.66e-3
  // bf16 output floor, a ratio of 1.01 — the score path contributes
  // sqrt(1.67^2 - 1.66^2) = 1.8e-4, well under the quantisation of the format it
  // is written to. At this output precision an fp16 score tile is
  // indistinguishable from exact arithmetic.
  //
  // Be clear about what the assertion is worth: at ratio 1.01 it has no power to
  // separate fp16 probabilities from bf16 ones, because neither is visible
  // through a bf16 output. It catches gross regressions — an fp8 tile, a lost
  // fp32 accumulator, a correction factor applied to the wrong operand — and the
  // key/query block sweep above is what actually pins the online softmax.
  std::printf("  fp16 score tile: worst rms_rel %.3g (bf16 output floor %.3g, ratio %.2f), "
              "worst abs %.3g, output rms %.3g\n",
              worst_rms_rel, output_floor_rel, worst_rms_rel / output_floor_rel, worst_abs,
              want_rms);
  CHECK_MSG(worst_rms_rel < 1.3 * output_floor_rel,
            "fp16 score tile rms_rel %.4g exceeds 1.3x the %.4g bf16 output floor", worst_rms_rel,
            output_floor_rel);
}

SLOPFAB_TEST_CATEGORY(attention_gqa, "synthetic") {
  CublasScope cb;
  const int seq = 256;
  const int heads = 8;
  const int kv_heads = 2;
  const int head_dim = 64;

  const std::vector<float> q = bf16_round(make_data(size_t(seq) * heads * head_dim, 171u, 0.3f));
  const std::vector<float> k = bf16_round(make_data(size_t(seq) * kv_heads * head_dim, 172u, 0.3f));
  const std::vector<float> v = bf16_round(make_data(size_t(seq) * kv_heads * head_dim, 173u, 1.0f));

  BfBuf dq(q), dk(k), dv(v);
  BfBuf dout(size_t(seq) * heads * head_dim);

  slopfab::cuda::AttentionConfig cfg;
  cfg.seq_len = seq;
  cfg.num_heads = heads;
  cfg.head_dim = head_dim;
  cfg.query_block = 64;

  Workspace ws;
  ws.reserve(slopfab::cuda::attention_workspace_bytes(cfg, slopfab::cuda::AttentionBackend::kBlocked));
  slopfab::cuda::attention_forward_gqa(cb.h, nullptr, dq.p(), dk.p(), dv.p(), dout.p(), cfg,
                                      kv_heads, slopfab::cuda::AttentionBackend::kBlocked, ws);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());

  const std::vector<float> want =
      cpu_attention(q, k, v, seq, heads, kv_heads, head_dim, cfg.effective_scale());
  CHECK_CLOSE_REL(want, dout.host(), 1e-3, 1e-2, "gqa attention, 8 query heads over 2 kv heads");

  // A wrong query-head -> kv-head mapping (say h % kv_heads instead of
  // h / group) would still be finite, so pin that it is not what happened.
  std::vector<float> wrong(size_t(seq) * heads * head_dim);
  {
    std::vector<float> k_perm(k.size());
    std::vector<float> v_perm(k.size());
    for (int t = 0; t < seq; ++t) {
      for (int kv = 0; kv < kv_heads; ++kv) {
        const int other = (kv + 1) % kv_heads;
        for (int d = 0; d < head_dim; ++d) {
          k_perm[size_t(t) * kv_heads * head_dim + kv * head_dim + d] =
              k[size_t(t) * kv_heads * head_dim + other * head_dim + d];
          v_perm[size_t(t) * kv_heads * head_dim + kv * head_dim + d] =
              v[size_t(t) * kv_heads * head_dim + other * head_dim + d];
        }
      }
    }
    wrong = cpu_attention(q, k_perm, v_perm, seq, heads, kv_heads, head_dim, cfg.effective_scale());
  }
  CHECK_MSG(max_abs_diff(wrong, dout.host()) > 0.01,
            "gqa must map query head h to kv head h/(H/Hkv) (max diff %.4g)",
            max_abs_diff(wrong, dout.host()));
}
