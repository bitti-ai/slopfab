#include "detail/nn_kernels_fixture.h"
#include "detail/nn_weight_fixture.h"
#include "detail/nn_timing_fixture.h"

SLOPFAB_TEST_CATEGORY(production_shape_timings, "benchmark") {
  CublasScope cb;
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  SLOPFAB_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
  std::printf("  device free %.2f GiB of %.2f GiB\n", double(free_bytes) / (1 << 30),
              double(total_bytes) / (1 << 30));

  // The default 124-frame request packs 37710 rows. If the card cannot hold the
  // buffers alongside everything else, fall back and say so rather than skip.
  int seq = 37710;
  const int heads = 56;
  const int head_dim = 128;
  const int width = heads * head_dim;
  const int model_dim = 5376;

  // q,k,v,out at [seq, 7168] bf16 plus the score tile and accumulator.
  while (seq > 1024 && size_t(seq) * width * 2ull * 4ull + (400ull << 20) > free_bytes * 3 / 4) {
    seq /= 2;
  }
  if (seq != 37710) {
    std::printf("  NOTE: attention measured at seq %d, not 37710, to fit in free memory\n", seq);
  }

  Timer timer;

  {
    BfBuf q(size_t(seq) * width), k(size_t(seq) * width), v(size_t(seq) * width),
        out(size_t(seq) * width);
    // Random, not memset: see fill_bf16_kernel. Constant q and k also make every
    // score identical, so the online softmax never rescales and its running
    // maximum never moves — the kernel would be measured on the one input that
    // exercises none of its data-dependent work.
    fill_random(q, 0x5EEDu);
    fill_random(k, 0xA17Eu);
    fill_random(v, 0xC0FFu);

    slopfab::cuda::AttentionConfig cfg;
    cfg.seq_len = seq;
    cfg.num_heads = heads;
    cfg.head_dim = head_dim;
    cfg.query_block = 1024;
    // The production shape must dispatch to the fused kernel on its own, not
    // only when a benchmark names it. A silent fall back to the blocked path
    // would still produce correct output and a believable number -- it is only
    // visible if something asserts which backend the pipeline would pick.
    CHECK(slopfab::cuda::attention_preferred_backend(cfg) ==
          slopfab::cuda::AttentionBackend::kFused);

    const size_t ws_bytes =
        slopfab::cuda::attention_workspace_bytes(cfg, slopfab::cuda::AttentionBackend::kBlocked);
    Workspace ws;
    ws.reserve(ws_bytes);
    std::printf("  attention workspace %.2f GiB (score tile, accumulator, fp16 k/v/q)\n",
                double(ws_bytes) / (1 << 30));
    // Something else on this box touches the GPU intermittently: roughly one run
    // in six comes back at half throughput. Take the best of three passes rather
    // than believing a single number.
    float ms = 1e30f;
    for (int pass = 0; pass < 3; ++pass) {
      ms = std::min(ms, timer.measure(
                            [&] {
                              slopfab::cuda::attention_forward(
                                  cb.h, nullptr, q.p(), k.p(), v.p(), out.p(), cfg,
                                  slopfab::cuda::AttentionBackend::kBlocked, ws);
                            },
                            1, 3));
    }
    const double flops = 4.0 * double(seq) * seq * head_dim * heads;
    std::printf("  attention   seq=%-6d heads=56 head_dim=128  %8.2f ms  (%.1f TFLOP/s)\n", seq, ms,
                flops / (ms * 1e-3) / 1e12);
    CHECK(ms > 0.0f);

    // Same buffers, same timer, same best-of-three: the only honest way to
    // report what removing the HBM round trip actually bought.
    Workspace none;
    float fms = 1e30f;
    for (int pass = 0; pass < 3; ++pass) {
      fms = std::min(fms, timer.measure(
                              [&] {
                                slopfab::cuda::attention_forward(
                                    cb.h, nullptr, q.p(), k.p(), v.p(), out.p(), cfg,
                                    slopfab::cuda::AttentionBackend::kFused, none);
                              },
                              1, 3));
    }
    std::printf("  attention   fused (no workspace)                %8.2f ms  (%.1f TFLOP/s)\n", fms,
                flops / (fms * 1e-3) / 1e12);
    std::printf("  fused speedup %.2fx over blocked, workspace %.2f GiB -> 0\n", ms / fms,
                double(ws_bytes) / (1 << 30));
    CHECK(fms > 0.0f);

    // Independent acceptance measurement: conversion/smoothing and the
    // quantized attention kernel are one timed operation.
    const std::vector<uint16_t> flash_bits = out.bits();
    const size_t sage_ws_bytes =
        slopfab::cuda::attention_workspace_bytes(cfg, slopfab::cuda::AttentionBackend::kSage2);
    Workspace sage_ws;
    sage_ws.reserve(sage_ws_bytes);
    float sms = 1e30f;
    for (int pass = 0; pass < 3; ++pass) {
      sms = std::min(sms, timer.measure(
                              [&] {
                                slopfab::cuda::attention_forward(
                                    cb.h, nullptr, q.p(), k.p(), v.p(), out.p(), cfg,
                                    slopfab::cuda::AttentionBackend::kSage2, sage_ws);
                              },
                              1, 3));
    }
    const std::vector<uint16_t> sage_bits = out.bits();
    slopfab::cuda::attention_forward(cb.h, nullptr, q.p(), k.p(), v.p(), out.p(), cfg,
                                     slopfab::cuda::AttentionBackend::kSage2, sage_ws);
    SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
    const std::vector<uint16_t> sage_repeat = out.bits();
    double err2 = 0.0, ref2 = 0.0, dot = 0.0, got2 = 0.0, max_abs = 0.0;
    size_t mismatches = 0;
    for (size_t i = 0; i < flash_bits.size(); ++i) {
      const double a = slopfab::bf16_to_f32(flash_bits[i]);
      const double b = slopfab::bf16_to_f32(sage_bits[i]);
      const double e = b - a;
      err2 += e * e;
      ref2 += a * a;
      dot += a * b;
      got2 += b * b;
      max_abs = std::max(max_abs, std::abs(e));
      mismatches += sage_bits[i] != sage_repeat[i];
    }
    std::printf("  attention   sage2 (all conversions)             %8.2f ms  (%.2fx flash2)\n", sms,
                fms / sms);
    std::printf(
        "  sage2 workspace %.3f GiB, rel_L2 %.6f corr %.6f max_abs %.6g repeat_mismatch %zu\n",
        double(sage_ws_bytes) / (1 << 30), std::sqrt(err2 / ref2), dot / std::sqrt(ref2 * got2),
        max_abs, mismatches);
    CHECK(sms > 0.0f);
  }

  {
    // q_norm/k_norm: 128-wide rows, the narrow-row path. Bytes are read+write of
    // the activation; the 128-element weight stays in L2.
    const int qk_heads = 56;
    const int qk_dim = 128;
    const size_t n = size_t(seq) * qk_heads * qk_dim;
    const size_t wn = size_t(qk_dim);
    BfBuf x(n), w(wn);
    SLOPFAB_CUDA_CHECK(cudaMemset(x.raw.get(), 0x3C, x.raw.nbytes()));
    SLOPFAB_CUDA_CHECK(cudaMemset(w.raw.get(), 0x3F, w.raw.nbytes()));
    float ms = 1e30f;
    for (int pass = 0; pass < 3; ++pass) {
      ms = std::min(ms, timer.measure(
                            [&] {
                              slopfab::cuda::launch_head_rmsnorm(x.p(), w.p(), seq, qk_heads,
                                                                 qk_dim, 1e-5f, nullptr);
                            },
                            3, 20));
    }
    const double bytes = 2.0 * double(n) * 2.0;
    std::printf("  head_rmsnorm rows=%-6d heads=56 dim=128     %8.3f ms  (%.0f GB/s)\n", seq, ms,
                bytes / (ms * 1e-3) / 1e9);
    CHECK(ms > 0.0f);
  }

  // The four transformer GEMM shapes, at the same row count.
  struct Shape {
    const char* name;
    int out_features;
    int in_features;
  };

  const Shape shapes[] = {
      {"qkv_proj", 21504, 5376},
      {"out_proj", 5376, 7168},
      {"mlp.fc1", 28672, 5376},
      {"mlp.fc2", 5376, 14336},
  };

  slopfab::cuda::LinearRunner runner;
  runner.init(cb.h, nullptr);
  for (const Shape& sh : shapes) {
    const size_t xn = size_t(seq) * sh.in_features;
    const size_t yn = size_t(seq) * sh.out_features;
    const size_t wn = size_t(sh.out_features) * sh.in_features;
    SLOPFAB_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    if ((xn * 2 + yn * 2 + wn * 3) > free_bytes * 3 / 4) {
      std::printf("  linear %-9s SKIPPED, needs more than the free memory\n", sh.name);
      continue;
    }
    DeviceBuffer<uint8_t> raw(wn);
    fill_random_f8(raw, 0x1234u);
    const std::vector<float> sv(1, 8.1264e-3f);
    DeviceBuffer<float> dscale = to_device(sv);
    BfBuf dx(xn), dy(yn);
    fill_random(dx, 0x9ABCu);

    slopfab::cuda::QuantWeight qw;
    qw.format = slopfab::cuda::QuantFormat::kF8E4M3;
    qw.data = raw.get();
    qw.out_features = sh.out_features;
    qw.in_features = sh.in_features;
    qw.weight_scale = dscale.get();

    Workspace ws;
    ws.reserve(slopfab::cuda::linear_workspace_bytes(qw, seq, slopfab::cuda::ComputeType::kBF16) +
               256);
    float ms = 1e30f;
    for (int pass = 0; pass < 2; ++pass) {
      ms = std::min(ms, timer.measure(
                            [&] {
                              runner.forward(qw, dx.p(), seq, dy.p(), ws);
                            },
                            2, 10));
    }
    const double flops = 2.0 * double(seq) * sh.out_features * sh.in_features;
    std::printf("  linear %-9s [%5d,%5d] rows=%-6d  %8.3f ms  (%.1f TFLOP/s)\n", sh.name,
                sh.out_features, sh.in_features, seq, ms, flops / (ms * 1e-3) / 1e12);
    CHECK(ms > 0.0f);
  }

  {
    const size_t norm_n = size_t(seq) * model_dim;
    const size_t weight_n = size_t(model_dim);
    BfBuf x(norm_n), out(norm_n), w(weight_n);
    SLOPFAB_CUDA_CHECK(cudaMemset(x.raw.get(), 0x3C, x.raw.nbytes()));
    SLOPFAB_CUDA_CHECK(cudaMemset(w.raw.get(), 0x3F, w.raw.nbytes()));
    float ms_norm = 1e30f;
    for (int pass = 0; pass < 3; ++pass) {
      ms_norm = std::min(ms_norm, timer.measure(
                                      [&] {
                                        slopfab::cuda::launch_rmsnorm(x.p(), w.p(), out.p(), seq,
                                                                      model_dim, 1e-5f, nullptr);
                                      },
                                      3, 20));
    }
    const double bytes = 3.0 * double(seq) * model_dim * 2.0;
    std::printf("  rmsnorm     rows=%-6d dim=5376              %8.3f ms  (%.0f GB/s)\n", seq,
                ms_norm, bytes / (ms_norm * 1e-3) / 1e9);

    const int mod_rows = 3;
    DeviceBuffer<float> scale(size_t(mod_rows) * model_dim);
    DeviceBuffer<float> shift(size_t(mod_rows) * model_dim);
    scale.zero();
    shift.zero();
    std::vector<int32_t> a(seq);
    for (int r = 0; r < seq; ++r)
      a[r] = r % mod_rows;
    DeviceBuffer<int32_t> da = to_device_i32(a);
    float ms_mod = 1e30f;
    for (int pass = 0; pass < 3; ++pass) {
      ms_mod = std::min(ms_mod, timer.measure(
                                    [&] {
                                      slopfab::cuda::launch_rmsnorm_modulate(
                                          x.p(), w.p(), scale.get(), shift.get(), da.get(), out.p(),
                                          seq, model_dim, 1e-5f, nullptr);
                                    },
                                    3, 20));
    }
    std::printf("  rmsnorm_mod rows=%-6d dim=5376              %8.3f ms\n", seq, ms_mod);
    CHECK(ms_norm > 0.0f && ms_mod > 0.0f);
  }
}
