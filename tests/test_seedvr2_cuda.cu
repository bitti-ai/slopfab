#include "harness.h"
#include "../src/seedvr2/runtime.cuh"
#include "slopfab/tensor_convert.h"
#include <cstdlib>
#include <filesystem>
using namespace slopfab;
using namespace slopfab::seedvr2;

SLOPFAB_TEST(seedvr2_torch_operator_parity) {
  const char* fixture = std::getenv("SLOPFAB_SEEDVR2_FIXTURE");
  if (!fixture || !std::filesystem::exists(fixture)) {
    SKIP_MISSING_FIXTURE("Run tools/seedvr2_goldens.py and set SLOPFAB_SEEDVR2_FIXTURE");
    return;
  }
  if (cuda::device_count() == 0) {
    SKIP_UNSUPPORTED_HARDWARE("CUDA required");
    return;
  }
  SafeTensors f;
  f.open(fixture);
  Runtime r;
  r.clear(f);
  auto input = [&](const std::string& name) {
    const auto& v = f.at(name);
    CHECK(v.shape.size() == 4);
    return r.upload(to_f32(v), int(v.shape[0]), int(v.shape[1]), int(v.shape[2]), int(v.shape[3]));
  };
  auto check = [&](const std::string& name, const Tensor& x, float tolerance) {
    auto expected = to_f32(f.at(name + ".expected"));
    auto actual = r.download(x);
    CHECK_CLOSE(expected, actual, tolerance, name.c_str());
  };
  auto x = input("input");
  check("conv", r.conv(x, "conv"), 0.004f);
  check("down", r.conv(x, "down", true, 2), 0.004f);
  check("spatial", r.conv(x, "spatial", true, 1), 0.004f);
  check("norm", r.groupnorm(x, "norm"), 0.02f);
  auto q = input("rms.input");
  check("linear", r.linear(q, "linear"), 0.0001f);
  check("rms", r.rms(q, "rms.weight"), 0.02f);
  auto emb = input("emb");
  r.modulate(q, emb, "ada", 1, false);
  check("mod", q, 0.0001f);
  q = input("rms.input");
  r.modulate(q, emb, "ada", 1, true);
  check("gate", q, 0.0001f);
  check("up", r.upsample(input("up.input"), 2), 0);
}
