#include "detail/nn_kernels_fixture.h"
#include "detail/nn_weight_fixture.h"
#include "detail/nn_timing_fixture.h"

SLOPFAB_TEST_CATEGORY(nvfp4_dequant_timings, "benchmark") {
  Timer timer;

  const struct {
    const char* name;
    int out_features;
    int in_features;
  } shapes[] = {
      {"blocks.N.attn.qkv_proj", 21504, 5376},
      {"blocks.N.attn.out_proj", 5376, 7168},
      {"blocks.N.mlp.fc1", 28672, 5376},
      {"blocks.N.mlp.fc2", 5376, 14336},
  };

  for (const auto& s : shapes) {
    const size_t n = size_t(s.out_features) * s.in_features;
    DeviceBuffer<uint8_t> dw(n / 2);
    DeviceBuffer<uint8_t> dsc(n / size_t(slopfab::cuda::kNVFP4BlockSize));
    BfBuf ddst(n);
    // 0x38 is e4m3 1.0; the codes themselves do not affect timing.
    SLOPFAB_CUDA_CHECK(cudaMemset(dw.get(), 0x52, dw.nbytes()));
    SLOPFAB_CUDA_CHECK(cudaMemset(dsc.get(), 0x38, dsc.nbytes()));

    float ms = 1e30f;
    for (int pass = 0; pass < 3; ++pass) {
      ms = std::min(ms, timer.measure(
                            [&] {
                              slopfab::cuda::launch_dequant_nvfp4(dw.get(), dsc.get(), 1.0f,
                                                                  ddst.p(), s.out_features,
                                                                  s.in_features, nullptr);
                            },
                            3, 20));
    }
    const double bytes = double(n) * 2.0 + double(n) / 2.0 + double(n) / 16.0;
    std::printf("  %-24s %6d x %5d  %7.3f ms  %7.1f GB/s\n", s.name, s.out_features, s.in_features,
                ms, bytes / (double(ms) * 1e-3) / 1e9);
    CHECK(ms > 0.0f);
  }
}

SLOPFAB_TEST_CATEGORY(nvfp4_activation_cost, "benchmark") {
  REQUIRE_SM120_TEST("native NVFP4 GEMM");
  const int rows = 128, out = 256;
  const std::vector<float> wd_full = make_gaussian(size_t(out) * 5376, 2468u, 0.05f);

  std::printf("  nvfp4 activation cost vs a bf16-activation reference:\n");
  std::printf("      %-26s %6s %8s %8s %8s %9s\n", "distribution", "K", "rms_rel", "median", "p99",
              "corr");

  struct Case {
    const char* name;
    int seed;
    int hot_stride; // 0 = none
    bool uniform;
  };

  const Case cases[] = {{"gaussian", 1234, 0, false},
                        {"gaussian + hot channels", 5678, 61, false},
                        {"uniform", 4321, 0, true}};

  for (const Case& c : cases) {
    for (int in : {128, 512, 2048, 5376}) {
      std::vector<float> x = c.uniform ? make_data(size_t(rows) * in, uint32_t(c.seed), 1.0f)
                                       : make_gaussian(size_t(rows) * in, uint32_t(c.seed), 1.0f);
      // Post-norm transformer activations are not clean Gaussians: a handful of
      // channels run an order of magnitude hot and persist across rows. That is
      // the case block scaling exists for, and no uniform generator produces it.
      if (c.hot_stride) {
        for (int ch = 0; ch < in; ch += c.hot_stride) {
          for (int r = 0; r < rows; ++r)
            x[size_t(r) * in + ch] *= 20.0f;
        }
      }
      x = bf16_round(x);

      const std::vector<float> wd(wd_full.begin(), wd_full.begin() + size_t(out) * in);
      const NvfpPacked w = pack_nvfp4(wd, out, in, 1.0f, true, true);
      const std::vector<float> got = run_native_nvfp4(x, w, rows, out, in, 1.0f);

      // Same weight both sides, so the only difference is the activation.
      const std::vector<float> ref = cpu_matmul_nt(x, w.dense, rows, out, in);

      std::vector<double> rel;
      rel.reserve(got.size());
      double num = 0.0, den = 0.0, hi_err = 0.0, lo_err = 0.0;
      int hi_n = 0, lo_n = 0;
      double amax = 0.0;
      for (float v : ref)
        amax = std::max(amax, std::fabs(double(v)));
      for (size_t i = 0; i < got.size(); ++i) {
        const double a = std::fabs(double(ref[i]));
        const double d = std::fabs(double(got[i]) - ref[i]);
        num += d * d;
        den += double(ref[i]) * ref[i];
        if (a > 1e-6)
          rel.push_back(d / a);
        // Error against output magnitude: quantisation noise is roughly flat in
        // absolute terms, so it shows up as a huge *relative* error wherever
        // the output cancelled towards zero and a small one where it did not.
        if (a > 0.5 * amax) {
          hi_err += d;
          ++hi_n;
        } else if (a < 0.05 * amax) {
          lo_err += d;
          ++lo_n;
        }
      }
      std::sort(rel.begin(), rel.end());
      const double med = rel.empty() ? 0.0 : rel[rel.size() / 2];
      const double p99 = rel.empty() ? 0.0 : rel[rel.size() * 99 / 100];
      std::printf("      %-26s %6d %8.4f %8.4f %8.4f %9.6f\n", c.name, in,
                  std::sqrt(num / std::max(den, 1e-30)), med, p99, correlation(ref, got));
      if (in == 5376) {
        std::printf("          mean |err| on the largest half of outputs %.3e, "
                    "on the smallest twentieth %.3e\n",
                    hi_n ? hi_err / hi_n : 0.0, lo_n ? lo_err / lo_n : 0.0);
      }
      CHECK(std::sqrt(num / std::max(den, 1e-30)) < 1.0); // still the same matrix
    }
  }
}

SLOPFAB_TEST_CATEGORY(nvfp4_activation_cost_real_weights, "benchmark") {
  REQUIRE_SM120_TEST("native NVFP4 GEMM");
  std::string path;
  for (const char* prefix : {"", "../", "../../", "../../../"}) {
    const std::string p =
        std::string(prefix) + "weights/transformer/MiniMax_H3_FL2VA_pruned_nvfp4.safetensors";
    if (std::filesystem::exists(p)) {
      path = p;
      break;
    }
  }
  if (path.empty()) {
    SKIP_MISSING_FIXTURE("  nvfp4 real-weight error: checkpoint absent, skipped\n");
    return;
  }

  slopfab::SafeTensors st;
  st.open(path);
  const int rows = 128;
  const char* names[] = {"blocks.0.attn.qkv_proj", "blocks.0.mlp.fc2"};
  for (const char* name : names) {
    const slopfab::TensorView& wv = st.at(std::string(name) + ".weight");
    const slopfab::TensorView& sv = st.at(std::string(name) + ".weight_scale");
    const slopfab::TensorView& gv = st.at(std::string(name) + ".weight_scale_2");
    const int out = int(wv.shape[0]);
    const int in = int(wv.shape[1]) * 2;
    float global = 1.0f;
    std::memcpy(&global, gv.data, sizeof(float));

    // One 128-row slab, which is a whole number of scale tiles and so slices
    // cleanly out of both arrays.
    const int slab = 128;
    const int kb = in / 16;
    std::vector<uint8_t> wpacked(static_cast<const uint8_t*>(wv.data),
                                 static_cast<const uint8_t*>(wv.data) + size_t(slab) * in / 2);
    std::vector<uint8_t> wscale(static_cast<const uint8_t*>(sv.data),
                                static_cast<const uint8_t*>(sv.data) + size_t(slab) * kb);

    // What those bytes mean, straight from the file.
    std::vector<float> dense(size_t(slab) * in);
    for (int o = 0; o < slab; ++o) {
      for (int i = 0; i < in; ++i) {
        const float s = slopfab::f8_e4m3_to_f32(wscale[nvfp4_scale_slot(o, i / 16, kb)]) * global;
        const uint8_t byte = wpacked[(size_t(o) * in + i) / 2];
        dense[size_t(o) * in + i] =
            slopfab::f4_e2m1_to_f32(i % 2 == 0 ? (byte >> 4) : (byte & 0x0F)) * s;
      }
    }

    // Post-RMSNorm activations: unit RMS per row is what the norm produces, and
    // the hot channels are what makes the case hard.
    std::vector<float> x = make_gaussian(size_t(rows) * in, 777u, 1.0f);
    for (int ch = 0; ch < in; ch += 61) {
      for (int r = 0; r < rows; ++r)
        x[size_t(r) * in + ch] *= 20.0f;
    }
    x = bf16_round(x);

    NvfpPacked w;
    w.data = std::move(wpacked);
    w.scale = std::move(wscale);
    w.dense = dense;
    const std::vector<float> got = run_native_nvfp4(x, w, rows, slab, in, global);
    const std::vector<float> ref = cpu_matmul_nt(x, dense, rows, slab, in);
    const std::vector<float> fp4_ref =
        cpu_matmul_nt(host_quantise_act(x, rows, in), dense, rows, slab, in);

    std::printf("  %-24s out=%-6d in=%-6d  vs bf16 act: rms_rel %.4f corr %.6f | "
                "vs fp4 act: rms_rel %.6f\n",
                name, out, in, rms_rel(ref, got), correlation(ref, got), rms_rel(fp4_ref, got));
    // The operand path on the shipped bytes, with the format's own cost taken
    // out of both sides. This one is an assertion.
    CHECK_CLOSE_REL(fp4_ref, got, 1e-3, 1e-2, "native nvfp4 GEMM on a shipped weight");
  }
}

SLOPFAB_TEST_CATEGORY(nvfp4_gemm_production_timings, "benchmark") {
  REQUIRE_SM120_TEST("native NVFP4 GEMM timings");
  CublasScope cb;
  cudaDeviceProp prop{};
  SLOPFAB_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("  %s: %d SMs, %d KB shared per SM\n", prop.name, prop.multiProcessorCount,
              int(prop.sharedMemPerMultiprocessor >> 10));

  struct Shape {
    const char* name;
    int out, in;
  };

  const Shape shapes[] = {{"qkv_proj", 21504, 5376},
                          {"attn.out_proj", 5376, 7168},
                          {"mlp.fc1", 28672, 5376},
                          {"mlp.fc2", 5376, 14336}};

  Timer timer;
  for (const Shape& s : shapes) {
    // The bytes never leave the device and their values do not affect timing.
    DeviceBuffer<uint8_t> dw(size_t(s.out) * s.in / 2), dws(size_t(s.out) * s.in / 16);
    SLOPFAB_CUDA_CHECK(cudaMemset(dw.get(), 0x25, dw.nbytes()));
    SLOPFAB_CUDA_CHECK(cudaMemset(dws.get(), 0x38, dws.nbytes()));

    for (int rows : {512, 2048, 8192}) {
      BfBuf dx(size_t(rows) * s.in), dy(size_t(rows) * s.out);
      SLOPFAB_CUDA_CHECK(cudaMemset(dx.raw.get(), 0x3C, dx.raw.nbytes()));

      Workspace ws;
      ws.reserve(slopfab::cuda::nvfp4_gemm_workspace_bytes(rows, s.in) + (1u << 20));
      Workspace dqws;
      dqws.reserve(size_t(s.out) * s.in * sizeof(__nv_bfloat16) + 4096);

      const double flops = 2.0 * rows * s.out * s.in;
      // What the native path must move: the weight, its scales, the activation
      // in and the result out. The reference path moves the dequantised weight
      // twice on top of all of it.
      const double bytes = double(s.out) * s.in / 2 + double(s.out) * s.in / 16 +
                           2.0 * rows * s.in + 2.0 * rows * s.out;

      float ms = 1e30f;
      for (int pass = 0; pass < 3; ++pass) {
        ms = std::min(ms, timer.measure(
                              [&] {
                                slopfab::cuda::nvfp4_gemm_forward(dx.p(), dw.get(), dws.get(), 1.0f,
                                                                  dy.p(), rows, s.out, s.in, ws,
                                                                  nullptr);
                              },
                              2, 5));
      }

      // The reference path, same shape, same process: the production
      // dequantiser followed by a cuBLAS bf16 GEMM.
      float dq_ms = 1e30f;
      for (int pass = 0; pass < 3; ++pass) {
        dq_ms =
            std::min(dq_ms, timer.measure(
                                [&] {
                                  Workspace::Scope scope(dqws);
                                  __nv_bfloat16* wb =
                                      dqws.alloc_n<__nv_bfloat16>(size_t(s.out) * s.in);
                                  slopfab::cuda::launch_dequant_nvfp4(dw.get(), dws.get(), 1.0f, wb,
                                                                      s.out, s.in, nullptr);
                                  const float alpha = 1.0f, beta = 0.0f;
                                  SLOPFAB_CUBLAS_CHECK(slopfab::cuda::cublas_gemm_ex(
                                      cb.h, CUBLAS_OP_T, CUBLAS_OP_N, s.out, rows, s.in, &alpha, wb,
                                      CUDA_R_16BF, s.in, dx.p(), CUDA_R_16BF, s.in, &beta, dy.p(),
                                      CUDA_R_16BF, s.out, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
                                },
                                2, 5));
      }

      std::printf("  %-14s out=%-6d in=%-6d rows=%-5d  native %8.3f ms %6.1f TFLOP/s %6.0f GB/s"
                  "  |  dequant+cuBLAS %8.3f ms %6.1f TFLOP/s  |  x%.2f\n",
                  s.name, s.out, s.in, rows, ms, flops / (ms * 1e-3) / 1e12,
                  bytes / (ms * 1e-3) / 1e9, dq_ms, flops / (dq_ms * 1e-3) / 1e12, dq_ms / ms);
      CHECK(ms > 0.0f && dq_ms > 0.0f);
    }
  }
}
