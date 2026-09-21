#include "detail/nn_kernels_fixture.h"
#include "detail/nn_weight_fixture.h"

SLOPFAB_TEST_CATEGORY(nn_dequant_nvfp4, "synthetic") {
  // 256 rows spans two row-tiles and 128 columns gives eight scale blocks, so
  // both halves of the tile index vary; a one-tile case would pass under a
  // layout that ignored the tiling entirely.
  const int out_features = 256;
  const int in_features = 128;
  const float global = 1.3580322e-3f; // the qkv_proj block 0 value, not a power of two
  const Nvfp4Weight w = make_nvfp4(out_features, in_features, global, 20260804u);

  // All sixteen E2M1 codes have to appear or the pattern coverage claim below
  // is empty.
  bool seen[16] = {false};
  for (uint8_t c : w.codes)
    seen[c] = true;
  for (int c = 0; c < 16; ++c)
    CHECK(seen[c]);

  DeviceBuffer<uint8_t> dw(w.packed.size());
  dw.copy_from_host(w.packed.data(), w.packed.size());
  DeviceBuffer<uint8_t> dsc(w.stored.size());
  dsc.copy_from_host(w.stored.data(), w.stored.size());
  BfBuf ddst(w.codes.size());

  slopfab::cuda::launch_dequant_nvfp4(dw.get(), dsc.get(), global, ddst.p(), out_features,
                                      in_features, nullptr);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  const std::vector<float> got = ddst.host();

  // E2M1 has one mantissa bit and e4m3 three, so the product carries at most
  // five significant bits and bf16 holds eight: with the same operation order
  // the two sides agree bit for bit.
  CHECK_CLOSE(nvfp4_reference(w), got, 0.0, "dequant nvfp4");

  // Each wrong form, and the distance from it. These are the checks that give
  // the one above any power.
  const std::vector<float> swapped = nvfp4_reference(w, 16, /*swap_nibbles=*/true);
  CHECK_MSG(max_abs_diff(swapped, got) > 1e-4,
            "dequant nvfp4 must take the even element from the HIGH nibble (max diff %.4g)",
            max_abs_diff(swapped, got));

  const std::vector<float> stride32 = nvfp4_reference(w, 32);
  CHECK_MSG(max_abs_diff(stride32, got) > 1e-4,
            "dequant nvfp4 must scale in blocks of 16, not 32 (max diff %.4g)",
            max_abs_diff(stride32, got));

  // Block scales read row-major instead of through the tile map.
  {
    Nvfp4Weight flat = w;
    flat.stored = w.scales; // as if the file were plain [out, in/16]
    DeviceBuffer<uint8_t> dflat(flat.stored.size());
    dflat.copy_from_host(flat.stored.data(), flat.stored.size());
    BfBuf dout(w.codes.size());
    slopfab::cuda::launch_dequant_nvfp4(dw.get(), dflat.get(), global, dout.p(), out_features,
                                        in_features, nullptr);
    SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
    CHECK_MSG(max_abs_diff(dout.host(), got) > 1e-4,
              "dequant nvfp4 must unswizzle the 128x4 block-scale tiling (max diff %.4g)",
              max_abs_diff(dout.host(), got));
  }

  // A shape the tile map cannot address must be refused rather than silently
  // mis-offset: 200 does not divide by 128, 96 does not divide by 64.
  for (auto bad : {std::pair<int, int>(200, 128), std::pair<int, int>(256, 96)}) {
    bool threw = false;
    try {
      slopfab::cuda::launch_dequant_nvfp4(dw.get(), dsc.get(), global, ddst.p(), bad.first,
                                          bad.second, nullptr);
    } catch (const std::exception&) {
      threw = true;
    }
    CHECK_MSG(threw, "launch_dequant_nvfp4 must refuse %dx%d", bad.first, bad.second);
  }
}

SLOPFAB_TEST_CATEGORY(nn_dequant_nvfp4_tile_shapes, "synthetic") {
  const float global = 1.3580322e-3f;

  struct Shape {
    int out;
    int in;
  };

  const Shape shapes[] = {
      {128, 64},  // exactly one tile, the degenerate case
      {128, 512}, // one row-tile, eight k-tiles
      {384, 128}, // three row-tiles: not a power of two
      {256, 320}, // five k-tiles, likewise
      {512, 64},  // one k-tile, four row-tiles
  };
  for (int i = 0; i < 5; ++i) {
    const int out_features = shapes[i].out, in_features = shapes[i].in;
    const Nvfp4Weight w = make_nvfp4(out_features, in_features, global, 90210u + 7u * i);
    DeviceBuffer<uint8_t> dw(w.packed.size());
    dw.copy_from_host(w.packed.data(), w.packed.size());
    DeviceBuffer<uint8_t> dsc(w.stored.size());
    dsc.copy_from_host(w.stored.data(), w.stored.size());
    BfBuf ddst(w.codes.size());
    slopfab::cuda::launch_dequant_nvfp4(dw.get(), dsc.get(), global, ddst.p(), out_features,
                                        in_features, nullptr);
    SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
    const std::vector<float> got = ddst.host();
    const std::vector<float> want = nvfp4_reference(w);
    CHECK_MSG(max_abs_diff(want, got) == 0.0,
              "dequant nvfp4 %dx%d (tiles %dx%d) is not exact: max diff %.6g", out_features,
              in_features, out_features / 128, in_features / 64, max_abs_diff(want, got));

    // A scale that lands on the wrong tile still produces plausible output, so
    // check that this shape can tell the difference at all: perturb one stored
    // scale byte and require the result to move.
    if (w.stored.size() > 1) {
      std::vector<uint8_t> poked = w.stored;
      const size_t at = poked.size() / 3;
      poked[at] = uint8_t(poked[at] ^ 0x08u); // one exponent step
      DeviceBuffer<uint8_t> dpoke(poked.size());
      dpoke.copy_from_host(poked.data(), poked.size());
      BfBuf dalt(w.codes.size());
      slopfab::cuda::launch_dequant_nvfp4(dw.get(), dpoke.get(), global, dalt.p(), out_features,
                                          in_features, nullptr);
      SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
      CHECK_MSG(max_abs_diff(dalt.host(), got) > 0.0,
                "dequant nvfp4 %dx%d ignored a perturbed block scale at byte %zu", out_features,
                in_features, at);
    }
  }
}

SLOPFAB_TEST_CATEGORY(linear_nvfp4, "synthetic") {
  CublasScope cb;
  const int rows = 29;
  const int in_features = 128;
  const int out_features = 256;
  const float global = 2.899169921875e-3f;

  const Nvfp4Weight w = make_nvfp4(out_features, in_features, global, 4242u);
  const std::vector<float> wdq = nvfp4_reference(w);
  const std::vector<float> x = bf16_round(make_data(size_t(rows) * in_features, 4243u, 0.1f));

  DeviceBuffer<uint8_t> dw(w.packed.size());
  dw.copy_from_host(w.packed.data(), w.packed.size());
  DeviceBuffer<uint8_t> dsc(w.stored.size());
  dsc.copy_from_host(w.stored.data(), w.stored.size());
  BfBuf dx(x), dy(size_t(rows) * out_features);

  slopfab::cuda::QuantWeight qw;
  qw.format = slopfab::cuda::QuantFormat::kNVFP4;
  qw.data = dw.get();
  qw.out_features = out_features;
  qw.in_features = in_features;
  qw.block_scale = dsc.get();
  qw.global_scale = global;
  CHECK(qw.stored_bytes() == size_t(out_features) * in_features / 2);

  slopfab::cuda::LinearRunner runner;
  runner.init(cb.h, nullptr);
  Workspace ws;
  ws.reserve(slopfab::cuda::linear_workspace_bytes(qw, rows, slopfab::cuda::ComputeType::kBF16) +
             256);
  runner.forward(qw, dx.p(), rows, dy.p(), ws);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());

  const std::vector<float> want = cpu_matmul_nt(x, wdq, rows, out_features, in_features);
  CHECK_CLOSE_REL(want, dy.host(), 1e-3, 1e-2, "linear nvfp4 dequantise-then-GEMM");

  // A GEMM against the nibble-swapped weight is well scaled and completely
  // wrong, which is the shape of the failure this format invites.
  const std::vector<float> wrong = cpu_matmul_nt(x, nvfp4_reference(w, 16, /*swap_nibbles=*/true),
                                                 rows, out_features, in_features);
  CHECK_MSG(max_abs_diff(wrong, dy.host()) > 1e-3,
            "linear nvfp4 must not match the nibble-swapped weight (max diff %.4g)",
            max_abs_diff(wrong, dy.host()));

  // A weight the checkpoint did not flag full_precision takes the native
  // tensor-core path when `set_native` is on. **That path does not compute the
  // same thing as this one, and the difference is arithmetic rather than
  // implementation**: it quantises the activation to nvfp4 too, because neither
  // shipped checkpoint carries an input_scale. E2M1 has one mantissa bit, so
  // the composed result sits about 9% rms from a bf16-activation reference, at
  // every K -- signal and error both grow as sqrt(K), so a dot product cannot
  // average it away. Asserting agreement with the dequantised path here would
  // be asserting that 4-bit activations are free.
  //
  // So the bound is pointed at what it can describe. The native path is held to
  // the same 1e-3 / 1e-2 against a reference that quantises the activation the
  // same way, which is a real assertion -- it passes at ~2e-3 and fails on any
  // operand, stride or scale error -- and the distance from the bf16 path is
  // reported rather than asserted. `nvfp4_activation_cost` measures that
  // distance properly, across three distributions and four values of K;
  // `nvfp4_gemm_exact_fp4_activations` is the control that separates the two.
  const std::vector<float> got = dy.host();
  if (test_is_sm120()) {
    runner.set_native(true);
    runner.forward(qw, dx.p(), rows, dy.p(), ws);
    SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
    const std::vector<float> native = dy.host();
    runner.set_native(false);
    CHECK_CLOSE_REL(cpu_matmul_nt(host_quantise_act(x, rows, in_features), wdq, rows, out_features,
                                  in_features),
                    native, 1e-3, 1e-2, "linear nvfp4 native vs an fp4-activation reference");
    std::printf("  linear nvfp4 native vs dequantised: rms_rel %.4f (4-bit activations)\n",
                rms_rel(got, native));
    // Still the same matrix, and still the same one the dequantised path
    // computes: a layout error would take the correlation to ~0, not to 0.99.
    CHECK(correlation(got, native) > 0.99);
  } else {
    SKIP_UNSUPPORTED_HARDWARE("native half of linear_nvfp4 requires the shipped SM120 image");
  }

  // With an AWQ activation scale the runner must scale the activation, not the
  // weight — the two differ because the GEMM is not symmetric in them.
  {
    const std::vector<float> pqs = bf16_round(make_data(in_features, 4244u, 0.5f));
    BfBuf dpqs(pqs);
    slopfab::cuda::QuantWeight aw = qw;
    aw.pre_quant_scale = dpqs.p();

    std::vector<float> xs(x.size());
    for (int r = 0; r < rows; ++r) {
      for (int i = 0; i < in_features; ++i) {
        xs[size_t(r) * in_features + i] =
            slopfab::bf16_to_f32(slopfab::f32_to_bf16(x[size_t(r) * in_features + i] * pqs[i]));
      }
    }
    Workspace ws2;
    ws2.reserve(slopfab::cuda::linear_workspace_bytes(aw, rows, slopfab::cuda::ComputeType::kBF16) +
                256);
    runner.forward(aw, dx.p(), rows, dy.p(), ws2);
    SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
    CHECK_CLOSE_REL(cpu_matmul_nt(xs, wdq, rows, out_features, in_features), dy.host(), 1e-3, 1e-2,
                    "linear nvfp4 with pre_quant_scale");
    CHECK_MSG(max_abs_diff(got, dy.host()) > 1e-3,
              "pre_quant_scale must actually reach the activation (max diff %.4g)",
              max_abs_diff(got, dy.host()));
  }
}

SLOPFAB_TEST_CATEGORY(nvfp4_mma_operand_layout, "synthetic") {
  REQUIRE_SM120_TEST("NVFP4 MMA operand-layout test");
  uint8_t A[16][64], B[64][8];
  for (int r = 0; r < 16; ++r)
    for (int k = 0; k < 64; ++k)
      A[r][k] = static_cast<uint8_t>((r * 7 + k * 3) % 15);
  for (int k = 0; k < 64; ++k)
    for (int c = 0; c < 8; ++c)
      B[k][c] = static_cast<uint8_t>((k * 5 + c * 11) % 15);

  // Each register packs eight consecutive-k nibbles; the register pair splits
  // rows at 8 and the pair-of-pairs splits k at 32.
  std::vector<uint32_t> ha(32 * 4, 0), hb(32 * 2, 0), hs(32, 0x38383838u);
  for (int lane = 0; lane < 32; ++lane) {
    const int gid = lane >> 2, tig = lane & 3;
    for (int i = 0; i < 8; ++i) {
      ha[lane * 4 + 0] |= uint32_t(A[gid][tig * 8 + i] & 15) << (4 * i);
      ha[lane * 4 + 1] |= uint32_t(A[gid + 8][tig * 8 + i] & 15) << (4 * i);
      ha[lane * 4 + 2] |= uint32_t(A[gid][32 + tig * 8 + i] & 15) << (4 * i);
      ha[lane * 4 + 3] |= uint32_t(A[gid + 8][32 + tig * 8 + i] & 15) << (4 * i);
      hb[lane * 2 + 0] |= uint32_t(B[tig * 8 + i][gid] & 15) << (4 * i);
      hb[lane * 2 + 1] |= uint32_t(B[32 + tig * 8 + i][gid] & 15) << (4 * i);
    }
  }

  DeviceBuffer<uint32_t> da(ha.size()), db(hb.size()), ds(hs.size());
  da.copy_from_host(ha.data(), ha.size());
  db.copy_from_host(hb.data(), hb.size());
  ds.copy_from_host(hs.data(), hs.size());
  DeviceBuffer<float> dout(128);
  nvfp4_mma_kernel<<<1, 32>>>(da.get(), db.get(), ds.get(), dout.get());
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  const std::vector<float> got = to_host(dout);

  std::vector<float> want(128, 0.0f);
  for (int r = 0; r < 16; ++r)
    for (int c = 0; c < 8; ++c)
      for (int k = 0; k < 64; ++k)
        want[r * 8 + c] += e2m1_ref(A[r][k]) * e2m1_ref(B[k][c]);
  // fp4 products of these magnitudes accumulate exactly in fp32, so this is an
  // equality, not a tolerance.
  CHECK_CLOSE_REL(want, got, 0.0, 0.0, "nvfp4 m16n8k64 A/B/accumulator layout");

  // Block scales: A and B all 1.0, so every row sums to 4 blocks x 16 = 64.
  // Doubling one (lane, byte) must lift exactly its own row by exactly 16.
  const std::vector<uint32_t> ones(32 * 4, 0x22222222u), onesb(32 * 2, 0x22222222u);
  DeviceBuffer<uint32_t> ua(ones.size()), ub(onesb.size()), dsx(32);
  ua.copy_from_host(ones.data(), ones.size());
  ub.copy_from_host(onesb.data(), onesb.size());
  for (int r : {0, 3, 7, 8, 11, 15}) {
    for (int blk : {0, 3}) {
      std::vector<uint32_t> s(32, 0x38383838u);
      const int lane = scale_lane_for_row(r);
      s[lane] = (s[lane] & ~(0xFFu << (8 * blk))) | (uint32_t(0x40) << (8 * blk)); // e4m3 2.0
      dsx.copy_from_host(s.data(), s.size());
      nvfp4_mma_kernel<<<1, 32>>>(ua.get(), ub.get(), dsx.get(), dout.get());
      SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
      const std::vector<float> g = to_host(dout);
      for (int rr = 0; rr < 16; ++rr) {
        const float expect = rr == r ? 80.0f : 64.0f; // 64 + 16 on the scaled row
        CHECK_NEAR(g[rr * 8], expect, 1e-3);
      }
    }
  }
}

SLOPFAB_TEST_CATEGORY(nvfp4_mma_b_scale_operand_layout, "synthetic") {
  REQUIRE_SM120_TEST("NVFP4 MMA block-scale test");
  const std::vector<uint32_t> ones(32 * 4, 0x22222222u), onesb(32 * 2, 0x22222222u);
  DeviceBuffer<uint32_t> da(ones.size()), db(onesb.size()), dsa(32), dsb(32);
  da.copy_from_host(ones.data(), ones.size());
  db.copy_from_host(onesb.data(), onesb.size());
  const std::vector<uint32_t> unit(32, 0x38383838u); // four e4m3 1.0 scales
  dsa.copy_from_host(unit.data(), unit.size());
  DeviceBuffer<float> dout(128);

  // Every element and every scale 1.0, so each output is 4 blocks x 16 = 64.
  dsb.copy_from_host(unit.data(), unit.size());
  nvfp4_mma_bscale_kernel<<<1, 32>>>(da.get(), db.get(), dsa.get(), dsb.get(), dout.get());
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK_CLOSE(std::vector<float>(128, 64.0f), to_host(dout), 0.0, "nvfp4 B unit scales");

  int live_lanes = 0;
  for (int lane = 0; lane < 32; ++lane) {
    for (int blk : {0, 2, 3}) {
      std::vector<uint32_t> s(32, 0x38383838u);
      s[lane] = (s[lane] & ~(0xFFu << (8 * blk))) | (uint32_t(0x40) << (8 * blk)); // e4m3 2.0
      dsb.copy_from_host(s.data(), s.size());
      nvfp4_mma_bscale_kernel<<<1, 32>>>(da.get(), db.get(), dsa.get(), dsb.get(), dout.get());
      SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
      const std::vector<float> g = to_host(dout);
      const int live_col = (lane % 4 == 0) ? lane / 4 : -1;
      bool ok = true;
      for (int r = 0; r < 16; ++r) {
        for (int c = 0; c < 8; ++c) {
          ok = ok && std::fabs(g[r * 8 + c] - (c == live_col ? 80.0f : 64.0f)) < 1e-3;
        }
      }
      if (live_col >= 0 && blk == 0)
        ++live_lanes;
      CHECK_MSG(ok, "B scale lane %d byte %d: expected %s", lane, blk,
                live_col >= 0 ? "its own column lifted by one block" : "no effect at all");
    }
  }
  CHECK_MSG(live_lanes == 8, "exactly eight B scale lanes should be live, saw %d", live_lanes);
}

SLOPFAB_TEST_CATEGORY(nvfp4_rounding_reference, "synthetic") {
  std::vector<float> vals = {0.0f, 0.25f, 0.75f, 1.25f, 1.75f, 2.5f,
                             3.5f, 5.0f,  6.0f,  1e4f,  -1e4f, 0.0f};
  const std::vector<float> r = make_data(512, 909u, 8.0f);
  vals.insert(vals.end(), r.begin(), r.end());

  DeviceBuffer<float> din(vals.size());
  din.copy_from_host(vals.data(), vals.size());
  const int pairs = int(vals.size() / 2);
  DeviceBuffer<uint8_t> d4(pairs), d8(pairs);
  nvfp4_cvt_probe_kernel<<<(pairs + 127) / 128, 128>>>(din.get(), d4.get(), d8.get(), pairs);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<uint8_t> h4(pairs), h8(pairs);
  d4.copy_to_host(h4.data(), h4.size());
  d8.copy_to_host(h8.data(), h8.size());

  int bad4 = 0, bad8 = 0;
  for (int i = 0; i < pairs; ++i) {
    // The instruction packs the even index in the LOW nibble. That the
    // checkpoint does the opposite is a separate statement about a separate
    // layer, and conflating the two is exactly the trap this file exists for.
    if ((h4[i] & 0x0F) != host_e2m1(vals[i * 2]))
      ++bad4;
    if ((h4[i] >> 4) != host_e2m1(vals[i * 2 + 1]))
      ++bad4;
    if (vals[i * 2] >= 0.0f && h8[i] != host_e4m3(vals[i * 2]))
      ++bad8;
  }
  CHECK_MSG(bad4 == 0, "host e2m1 encoder disagrees with cvt.rn.satfinite on %d of %d", bad4,
            pairs * 2);
  CHECK_MSG(bad8 == 0, "host e4m3 encoder disagrees with cvt.rn.satfinite on %d of %d", bad8,
            pairs);
}

SLOPFAB_TEST_CATEGORY(nvfp4_activation_quantisation, "synthetic") {
  const int rows = 7;
  const int dim = 64;
  std::vector<float> x = make_gaussian(size_t(rows) * dim, 4242u, 0.7f);
  // Three blocks with a story: all zeros, far under e4m3's smallest scale, and
  // far over its largest. None may produce a NaN and none may wrap.
  for (int i = 0; i < 16; ++i)
    x[0 * dim + i] = 0.0f;
  for (int i = 0; i < 16; ++i)
    x[1 * dim + 16 + i] = 1e-6f * float(i + 1);
  for (int i = 0; i < 16; ++i)
    x[2 * dim + 32 + i] = 5000.0f;
  const std::vector<float> xr = bf16_round(x);

  BfBuf dx(x);
  DeviceBuffer<uint8_t> dq(size_t(rows) * dim / 2), ds(size_t(rows) * dim / 16);
  slopfab::cuda::launch_quantize_nvfp4_activations(dx.p(), dq.get(), ds.get(), rows, dim, nullptr);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<uint8_t> hq(dq.size()), hs(ds.size());
  dq.copy_to_host(hq.data(), hq.size());
  ds.copy_to_host(hs.data(), hs.size());

  // The activation buffer is written by this project, for this instruction, so
  // it owes the checkpoint's conventions nothing: low nibble is the even
  // element and the scales are plain row-major.
  int bad_scale = 0, bad_nibble = 0;
  for (int r = 0; r < rows; ++r) {
    for (int b = 0; b < dim / 16; ++b) {
      float amax = 0.0f;
      for (int i = 0; i < 16; ++i) {
        amax = std::max(amax, std::fabs(xr[size_t(r) * dim + b * 16 + i]));
      }
      const uint8_t s8 = host_e4m3(amax * (1.0f / 6.0f));
      if (hs[size_t(r) * (dim / 16) + b] != s8)
        ++bad_scale;
      const float sd = slopfab::f8_e4m3_to_f32(s8);
      const float inv = sd > 0.0f ? 1.0f / sd : 0.0f;
      for (int i = 0; i < 16; ++i) {
        const size_t flat = size_t(r) * dim + b * 16 + i;
        const uint8_t got = (i % 2 == 0) ? (hq[flat / 2] & 0x0F) : (hq[flat / 2] >> 4);
        if (host_e2m1(xr[flat] * inv) != got)
          ++bad_nibble;
      }
    }
  }
  CHECK_MSG(bad_scale == 0, "%d of %d block scales are not amax/6 rounded to e4m3", bad_scale,
            int(hs.size()));
  CHECK_MSG(bad_nibble == 0, "%d of %d nibbles differ from the host rule", bad_nibble,
            int(xr.size()));

  // Zero block: scale zero, nibbles zero, and no NaN out of the reciprocal.
  CHECK(hs[0] == 0);
  bool zeros = true;
  for (int i = 0; i < 8; ++i)
    zeros = zeros && hq[i] == 0;
  CHECK(zeros);

  // Underflow flushes the whole block rather than clipping it onto the grid.
  // e4m3's smallest positive is 2^-9, so a block whose largest element is under
  // 6 * 2^-10 has no representable scale and becomes zero. Documented, not
  // accidental: it is why the header argues a per-tensor activation scale buys
  // nothing for post-norm activations, whose blocks are nowhere near this.
  CHECK(hs[dim / 16 + 1] == 0);

  // Overflow saturates. 5000 wants a scale of 833, e4m3 stops at 448, and the
  // elements then clip at 6 — finite and too small, never wrapped, never NaN.
  CHECK(slopfab::f8_e4m3_to_f32(hs[2 * (dim / 16) + 2]) == 448.0f);
  bool clipped = true;
  for (int i = 0; i < 8; ++i) {
    const uint8_t byte = hq[(size_t(2) * dim + 32) / 2 + i];
    clipped = clipped && (byte & 0x0F) == 7 && (byte >> 4) == 7; // +6 both halves
  }
  CHECK(clipped);
}

SLOPFAB_TEST_CATEGORY(nvfp4_gemm_matches_cpu_reference, "synthetic") {
  REQUIRE_SM120_TEST("native NVFP4 GEMM");

  struct Shape {
    int rows, out, in;
  };

  // Ragged row counts on purpose: 1 leaves 127 rows of a tile as padding, 200
  // leaves a 72-row tail, 129 leaves a one-row second tile. The contraction
  // covers half a staging tile (64), one (128), one and a half (192) and two
  // and a half (320).
  const Shape shapes[] = {
      {1, 128, 64}, {17, 128, 128}, {128, 256, 192}, {200, 128, 320}, {129, 256, 64}};

  for (const Shape& s : shapes) {
    const std::vector<float> x = bf16_round(make_gaussian(size_t(s.rows) * s.in, 71u + s.in, 0.8f));
    const std::vector<float> wd = make_gaussian(size_t(s.out) * s.in, 33u + s.out, 0.05f);
    const NvfpPacked w = pack_nvfp4(wd, s.out, s.in, 0.7f, true, true);
    // The reference sees the same activation the kernel does. This check is
    // about the GEMM; what 4-bit activations cost is a different question,
    // measured in `nvfp4_activation_cost`.
    CHECK_CLOSE_REL(cpu_matmul_nt(host_quantise_act(x, s.rows, s.in), w.dense, s.rows, s.out, s.in),
                    run_native_nvfp4(x, w, s.rows, s.out, s.in, 0.7f), 1e-3, 1e-2,
                    "native nvfp4 GEMM vs CPU reference");
  }
}

SLOPFAB_TEST_CATEGORY(nvfp4_gemm_disk_layout, "synthetic") {
  REQUIRE_SM120_TEST("native NVFP4 GEMM");
  const int rows = 64, out = 128, in = 192;
  const std::vector<float> x = bf16_round(make_gaussian(size_t(rows) * in, 515u, 0.8f));
  const std::vector<float> wd = make_gaussian(size_t(out) * in, 616u, 0.05f);
  const NvfpPacked right = pack_nvfp4(wd, out, in, 1.0f, true, true);
  const std::vector<float> want =
      cpu_matmul_nt(host_quantise_act(x, rows, in), right.dense, rows, out, in);

  CHECK_CLOSE_REL(want, run_native_nvfp4(x, right, rows, out, in, 1.0f), 1e-3, 1e-2,
                  "checkpoint layout: high nibble even, block scales swizzled");

  const struct {
    bool high_even, swizzled;
    const char* what;
  } wrong[] = {
      {false, true, "low nibble even"},
      {true, false, "block scales row-major rather than 128x4 tiled"},
      {false, false, "both conventions inverted"},
  };

  for (const auto& c : wrong) {
    const NvfpPacked bad = pack_nvfp4(wd, out, in, 1.0f, c.high_even, c.swizzled);
    const double rel = rms_rel(want, run_native_nvfp4(x, bad, rows, out, in, 1.0f));
    CHECK_MSG(rel > 0.2, "%s must disagree with the checkpoint layout, rms_rel %.4f", c.what, rel);
  }

  // A block stride error inside an otherwise correct swizzle: every row's block
  // scales rotated by one. Every byte is still present and still in the right
  // tile, so nothing about the size or the value histogram gives it away.
  NvfpPacked rot = right;
  const int kb = in / 16;
  for (int m = 0; m < out; ++m) {
    for (int b = 0; b < kb; ++b) {
      rot.scale[nvfp4_scale_slot(m, b, kb)] = right.scale[nvfp4_scale_slot(m, (b + 1) % kb, kb)];
    }
  }
  const double rel = rms_rel(want, run_native_nvfp4(x, rot, rows, out, in, 1.0f));
  CHECK_MSG(rel > 0.2, "block scales rotated by one block must disagree, rms_rel %.4f", rel);

  // The shapes the 128x4 tiling cannot address are refused, not guessed at.
  CHECK(slopfab::cuda::nvfp4_gemm_supported(out, in));
  CHECK(!slopfab::cuda::nvfp4_gemm_supported(out, in + 16)); // in % 64 != 0
  CHECK(!slopfab::cuda::nvfp4_gemm_supported(out + 64, in)); // out % 128 != 0
}

SLOPFAB_TEST_CATEGORY(nvfp4_gemm_global_scale, "synthetic") {
  REQUIRE_SM120_TEST("native NVFP4 GEMM");
  const int rows = 32, out = 128, in = 128;
  const std::vector<float> x = bf16_round(make_gaussian(size_t(rows) * in, 808u, 0.8f));
  const std::vector<float> wd = make_gaussian(size_t(out) * in, 909u, 0.05f);
  const NvfpPacked w = pack_nvfp4(wd, out, in, 1.0f, true, true);

  // The same stored bytes twice, with only the scalar changed, so anything the
  // quantiser does cancels and what is left is the epilogue's multiply.
  const std::vector<float> at_one = run_native_nvfp4(x, w, rows, out, in, 1.0f);
  const float g = 3.25f;
  const std::vector<float> at_g = run_native_nvfp4(x, w, rows, out, in, g);

  std::vector<float> lifted(at_one.size()), twice(at_one.size());
  for (size_t i = 0; i < at_one.size(); ++i) {
    lifted[i] = at_one[i] * g;
    twice[i] = at_one[i] * g * g;
  }
  // Not equality, and the reason is worth stating because the tempting
  // argument is wrong: g has three significant bits, but `at_one` is already
  // rounded to bf16, so scaling it multiplies a half-ULP error by 3.25 before
  // `at_g`'s own rounding is added. Two bf16 ULP apart is correct behaviour.
  // The bar still has all the power it needs: getting the count wrong moves
  // the answer by a factor of 3.25.
  CHECK_CLOSE_REL(lifted, at_g, 1e-3, 1e-2, "global scale applied exactly once");
  CHECK(rms_rel(at_one, at_g) > 0.5); // not zero times
  CHECK(rms_rel(twice, at_g) > 0.5);  // not twice

  // And the scalar the checkpoint actually carries reaches the same answer
  // whether it is folded into the stored scales or passed alongside them.
  const NvfpPacked folded = pack_nvfp4(wd, out, in, g, true, true);
  CHECK_CLOSE_REL(cpu_matmul_nt(host_quantise_act(x, rows, in), folded.dense, rows, out, in),
                  run_native_nvfp4(x, folded, rows, out, in, g), 1e-3, 1e-2,
                  "global scale against a weight packed for it");
}

SLOPFAB_TEST_CATEGORY(nvfp4_gemm_exact_fp4_activations, "synthetic") {
  REQUIRE_SM120_TEST("native NVFP4 GEMM");
  const int rows = 128, out = 256, in = 512;
  const float grid[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

  std::vector<float> x(size_t(rows) * in);
  const std::vector<float> u = make_data(x.size(), 31337u, 0.5f);
  for (int r = 0; r < rows; ++r) {
    for (int b = 0; b < in / 16; ++b) {
      // Powers of two from 2^-4 to 2^3, all exactly e4m3.
      const float s = std::ldexp(1.0f, ((r * 7 + b * 3) % 8) - 4);
      for (int i = 0; i < 16; ++i) {
        const size_t flat = size_t(r) * in + b * 16 + i;
        const int code = int((u[flat] + 0.5f) * 8.0f) & 7;
        const float sign = (int((u[flat] + 0.5f) * 64.0f) & 1) ? -1.0f : 1.0f;
        x[flat] = sign * grid[code] * s;
      }
      x[size_t(r) * in + b * 16] = 6.0f * s; // pin amax so the scale round-trips
    }
  }
  // Every value is a small multiple of a power of two, so bf16 holds it exactly
  // and the buffer the kernel reads is the buffer built here.
  CHECK(bf16_round(x) == x);
  CHECK(host_quantise_act(x, rows, in) == x);

  const std::vector<float> wd = make_gaussian(size_t(out) * in, 2468u, 0.05f);
  const NvfpPacked w = pack_nvfp4(wd, out, in, 1.0f, true, true);
  const std::vector<float> got = run_native_nvfp4(x, w, rows, out, in, 1.0f);
  const std::vector<float> want = cpu_matmul_nt(x, w.dense, rows, out, in);

  std::printf("  nvfp4 exact-fp4 activations: rms_rel %.6f  corr %.6f\n", rms_rel(want, got),
              correlation(want, got));
  CHECK_CLOSE_REL(want, got, 1e-3, 1e-2, "native nvfp4 GEMM on fp4-exact activations");
}
