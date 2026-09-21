// Unit tests. Hand-rolled rather than gtest to keep the dependency count at
// zero; the harness is a few dozen lines and reports the same information.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "harness.h"
#include "slopfab/dtype.h"
#include "slopfab/json.h"
#include "slopfab/safetensors.h"
#include "slopfab/sampler/scheduler.h"
#include "slopfab/tensor_convert.h"
#include "slopfab/w4a8.h"

namespace {

using ::slopfab::test::throws;

// --- json -------------------------------------------------------------------

void test_json() {
  TEST("json");
  using namespace slopfab;

  const json::Value v = json::parse(R"({
    "name": "hello",
    "n": -12.5e2,
    "flag": true,
    "nothing": null,
    "list": [1, 2, 3],
    "nested": {"inner": {"deep": "yes"}}
  })");

  CHECK(v.is_object());
  CHECK(v.find("name")->as_string() == "hello");
  CHECK_NEAR(v.find("n")->as_number(), -1250.0, 1e-9);
  CHECK(v.find("flag")->as_bool() == true);
  CHECK(v.find("nothing")->is_null());
  CHECK(v.find("list")->as_array().size() == 3);
  CHECK(v.find("list")->as_array()[2].as_int() == 3);
  CHECK(v.find("nested")->find("inner")->find("deep")->as_string() == "yes");
  CHECK(v.find("absent") == nullptr);

  // Escapes. Written as \u escapes rather than literal characters so the test
  // cannot be broken by a tool re-encoding the source file.
  //   Ä      -> 2-byte UTF-8 (C3 84)
  //   😀 -> surrogate pair for U+1F600, 4-byte UTF-8 (F0 9F 98 80)
  const json::Value esc = json::parse(R"({"s": "a\"b\\c\ndÄ😀"})");
  const std::string& s = esc.find("s")->as_string();
  CHECK(s.rfind("a\"b\\c\nd", 0) == 0);
  CHECK(s.substr(7) == "\xC3\x84\xF0\x9F\x98\x80");
  CHECK(s.size() == 13);

  // A lone high surrogate is replaced rather than aborting the parse.
  const json::Value lone = json::parse(R"({"s": "\uD83D!"})");
  CHECK(lone.find("s")->as_string() == "\xEF\xBF\xBD!");

  CHECK(json::parse("{}").as_object().empty());
  CHECK(json::parse("[]").as_array().empty());

  CHECK(throws([] {
    json::parse("{");
  }));
  CHECK(throws([] {
    json::parse("{\"a\": }");
  }));
  CHECK(throws([] {
    json::parse("{\"a\": 1} trailing");
  }));
  CHECK(throws([] {
    json::parse("\"unterminated");
  }));
  CHECK(throws([] {
    json::parse("{\"a\": tru}");
  }));
}

// --- dtype ------------------------------------------------------------------

void test_dtype() {
  TEST("dtype");
  using namespace slopfab;

  CHECK(dtype_from_string("BF16") == DType::kBF16);
  CHECK(dtype_from_string("F8_E4M3") == DType::kF8E4M3);
  CHECK(dtype_from_string("NOPE") == DType::kUnknown);
  CHECK(dtype_size(DType::kF32) == 4);
  CHECK(dtype_size(DType::kBF16) == 2);
  CHECK(dtype_size(DType::kF8E4M3) == 1);

  // bfloat16
  CHECK_NEAR(bf16_to_f32(0x3F80), 1.0f, 0.0);
  CHECK_NEAR(bf16_to_f32(0xBF80), -1.0f, 0.0);
  CHECK_NEAR(bf16_to_f32(0x4000), 2.0f, 0.0);
  CHECK_NEAR(bf16_to_f32(0x0000), 0.0f, 0.0);
  CHECK(f32_to_bf16(1.0f) == 0x3F80);
  CHECK(f32_to_bf16(-2.0f) == 0xC000);
  // Round trip through bf16 keeps ~3 decimal digits.
  CHECK_NEAR(bf16_to_f32(f32_to_bf16(3.14159f)), 3.14159f, 0.02);

  // float16, including a subnormal
  CHECK_NEAR(f16_to_f32(0x3C00), 1.0f, 0.0);
  CHECK_NEAR(f16_to_f32(0xC000), -2.0f, 0.0);
  CHECK_NEAR(f16_to_f32(0x0000), 0.0f, 0.0);
  CHECK_NEAR(f16_to_f32(0x0001), 5.9604645e-8f, 1e-12);
  CHECK(std::isinf(f16_to_f32(0x7C00)));
  CHECK(std::isnan(f16_to_f32(0x7E00)));

  // float8 e4m3: bias 7, max normal 448, 0x7F is NaN
  CHECK_NEAR(f8_e4m3_to_f32(0x00), 0.0f, 0.0);
  CHECK_NEAR(f8_e4m3_to_f32(0x38), 1.0f, 0.0);
  CHECK_NEAR(f8_e4m3_to_f32(0x3C), 1.5f, 0.0);
  CHECK_NEAR(f8_e4m3_to_f32(0xB8), -1.0f, 0.0);
  CHECK_NEAR(f8_e4m3_to_f32(0x7E), 448.0f, 0.0);
  CHECK(std::isnan(f8_e4m3_to_f32(0x7F)));
  // Smallest positive subnormal is 2^-9.
  CHECK_NEAR(f8_e4m3_to_f32(0x01), 0.001953125f, 1e-9);

  // float4 e2m1: the full representable set
  const float expect[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  for (uint8_t i = 0; i < 8; ++i) {
    CHECK_NEAR(f4_e2m1_to_f32(i), expect[i], 0.0);
    CHECK_NEAR(f4_e2m1_to_f32(static_cast<uint8_t>(i | 0x08)), -expect[i], 0.0);
  }
  // Packing order: low nibble is the even-indexed element. 0x21 holds nibble 1
  // (0.5) in the low half and nibble 2 (1.0) in the high half.
  CHECK_NEAR(f4_lo(0x21), 0.5f, 0.0);
  CHECK_NEAR(f4_hi(0x21), 1.0f, 0.0);
  CHECK_NEAR(f4_lo(0xF7), 6.0f, 0.0);  // nibble 7
  CHECK_NEAR(f4_hi(0xF7), -6.0f, 0.0); // nibble 15 = sign | 7
}

// --- safetensors ------------------------------------------------------------

// Builds a minimal but valid safetensors file so the loader is exercised
// against known bytes rather than only against real checkpoints.
std::string write_temp_safetensors(const std::string& header, const std::vector<uint8_t>& data,
                                   const std::string& filename) {
  const std::filesystem::path path = std::filesystem::temp_directory_path() / filename;
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  const uint64_t len = header.size();
  out.write(reinterpret_cast<const char*>(&len), sizeof(len));
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  if (!data.empty()) {
    out.write(reinterpret_cast<const char*>(data.data()),
              static_cast<std::streamsize>(data.size()));
  }
  out.close();
  return path.string();
}

void test_safetensors() {
  TEST("safetensors");
  using namespace slopfab;

  // Two tensors: a 2x3 f32 and a scalar f32, plus a metadata block.
  std::vector<uint8_t> data(6 * 4 + 4);
  const float values[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::memcpy(data.data(), values, sizeof(values));
  const float scalar = 0.25f;
  std::memcpy(data.data() + 24, &scalar, sizeof(scalar));

  const std::string header = R"({"__metadata__":{"format":"pt","note":"unit test"},)"
                             R"("w":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},)"
                             R"("s":{"dtype":"F32","shape":[],"data_offsets":[24,28]}})";

  const std::string path = write_temp_safetensors(header, data, "slopfab_ok.safetensors");

  SafeTensors st;
  st.open(path);
  CHECK(st.is_open());
  CHECK(st.tensor_count() == 2);
  CHECK(st.metadata().at("format") == "pt");
  CHECK(st.metadata().at("note") == "unit test");

  const TensorView& w = st.at("w");
  CHECK(w.dtype == DType::kF32);
  CHECK(w.shape.size() == 2 && w.shape[0] == 2 && w.shape[1] == 3);
  CHECK(w.numel() == 6);
  CHECK(w.nbytes == 24);
  const auto* wp = static_cast<const float*>(w.data);
  CHECK_NEAR(wp[0], 1.0f, 0.0);
  CHECK_NEAR(wp[5], 6.0f, 0.0);

  const TensorView& s = st.at("s");
  CHECK(s.is_scalar());
  CHECK(s.numel() == 1);
  CHECK_NEAR(*static_cast<const float*>(s.data), 0.25f, 0.0);

  CHECK(st.find("absent") == nullptr);
  bool threw = false;
  try {
    st.at("absent");
  } catch (const std::exception&) {
    threw = true;
  }
  CHECK(threw);

  st.close();
  CHECK(!st.is_open());
  std::filesystem::remove(path);

  // A shape that disagrees with the declared byte range must be rejected: it
  // would otherwise cause a silent misread of every following tensor.
  const std::string bad_header = R"({"w":{"dtype":"F32","shape":[2,3],"data_offsets":[0,16]}})";
  const std::string bad_path =
      write_temp_safetensors(bad_header, std::vector<uint8_t>(16, 0), "slopfab_bad.safetensors");
  bool rejected = false;
  try {
    SafeTensors bad;
    bad.open(bad_path);
  } catch (const std::exception&) {
    rejected = true;
  }
  CHECK(rejected);
  std::filesystem::remove(bad_path);

  // A range extending past EOF must be rejected too.
  const std::string over_header = R"({"w":{"dtype":"F32","shape":[100],"data_offsets":[0,400]}})";
  const std::string over_path =
      write_temp_safetensors(over_header, std::vector<uint8_t>(8, 0), "slopfab_over.safetensors");
  bool over_rejected = false;
  try {
    SafeTensors over;
    over.open(over_path);
  } catch (const std::exception&) {
    over_rejected = true;
  }
  CHECK(over_rejected);
  std::filesystem::remove(over_path);

  // A missing file is an error, not a silent empty archive.
  bool missing_rejected = false;
  try {
    SafeTensors missing;
    missing.open("this_file_does_not_exist.safetensors");
  } catch (const std::exception&) {
    missing_rejected = true;
  }
  CHECK(missing_rejected);
}

void test_w4a8_state() {
  TEST("w4a8_state");
  using namespace slopfab;

  const std::string payload =
      R"({"format":"asym_w4a8_int8","group_size":16,"convrot_groupsize":256})";
  const std::string header =
      std::string(R"({"layer.weight":{"dtype":"I8","shape":[2,128],"data_offsets":[0,256]},)") +
      R"("layer.comfy_quant":{"dtype":"U8","shape":[)" + std::to_string(payload.size()) +
      R"(],"data_offsets":[256,)" + std::to_string(256 + payload.size()) + "]}}";
  std::vector<uint8_t> data(256 + payload.size());
  std::memcpy(data.data() + 256, payload.data(), payload.size());
  const std::string path = write_temp_safetensors(header, data, "slopfab_w4a8_state.safetensors");
  SafeTensors st;
  st.open(path);
  CHECK(is_w4a8_weight(st, "layer.weight"));
  const W4A8State state = read_w4a8_state(st, "layer.weight", "test");
  CHECK(state.group_size == 16);
  CHECK(state.convrot_group_size == 256);
  CHECK(!is_w4a8_weight(st, "absent.weight"));
  st.close();
  std::error_code ec;
  std::filesystem::remove(path, ec);
}

// --- compare ----------------------------------------------------------------

void test_compare() {
  TEST("compare");
  using namespace slopfab;

  // Identical inputs produce exactly zero error.
  const std::vector<float> a = {1.0f, 2.0f, -3.0f, 0.0f};
  CompareStats same = compare(a, a);
  CHECK(same.shape_match);
  CHECK(same.count == 4);
  CHECK_NEAR(same.max_abs_err, 0.0, 0.0);
  CHECK_NEAR(same.rms_err, 0.0, 0.0);
  CHECK(same.passes(0.0, 0.0));

  // A single perturbed element drives max_abs_err and argmax.
  std::vector<float> b = a;
  b[2] = -3.5f;
  CompareStats diff = compare(a, b);
  CHECK_NEAR(diff.max_abs_err, 0.5, 1e-9);
  CHECK(diff.argmax_abs == 2);
  CHECK_NEAR(diff.lhs_at_argmax, -3.0, 1e-9);
  CHECK_NEAR(diff.rhs_at_argmax, -3.5, 1e-9);
  // Relative error is 0.5/3.
  CHECK_NEAR(diff.max_rel_err, 0.5 / 3.0, 1e-9);
  CHECK(!diff.passes(1e-3, 1e-3));
  CHECK(diff.passes(1.0, 1e-3)); // passes on absolute
  CHECK(diff.passes(1e-9, 0.2)); // passes on relative

  // Size mismatch never passes.
  CompareStats mismatch = compare(a, std::vector<float>{1.0f});
  CHECK(!mismatch.shape_match);
  CHECK(!mismatch.passes(1e9, 1e9));

  // A near-zero reference must not manufacture a huge relative error.
  CompareStats tiny = compare(std::vector<float>{1e-9f}, std::vector<float>{2e-9f});
  CHECK_NEAR(tiny.max_rel_err, 0.0, 0.0);
  CHECK(tiny.passes(1e-6, 0.0));

  // Matching NaNs agree; a NaN against a number does not.
  const float nan_v = std::nanf("");
  CHECK(compare({nan_v}, {nan_v}).nan_mismatches == 0);
  CompareStats nan_bad = compare({nan_v}, {1.0f});
  CHECK(nan_bad.nan_mismatches == 1);
  CHECK(!nan_bad.passes(1e9, 1e9));

  // to_f32 widens each stored dtype correctly.
  std::vector<uint8_t> raw(4);
  const uint16_t bf16_one = 0x3F80;
  std::memcpy(raw.data(), &bf16_one, 2);
  const uint16_t bf16_two = 0x4000;
  std::memcpy(raw.data() + 2, &bf16_two, 2);

  TensorView view;
  view.name = "t";
  view.dtype = DType::kBF16;
  view.shape = {2};
  view.data = raw.data();
  view.nbytes = 4;
  const std::vector<float> widened = to_f32(view);
  CHECK(widened.size() == 2);
  CHECK_NEAR(widened[0], 1.0f, 0.0);
  CHECK_NEAR(widened[1], 2.0f, 0.0);
}

// --- scheduler --------------------------------------------------------------

void test_scheduler() {
  TEST("scheduler");
  using slopfab::sampler::FlowScheduler;

  FlowScheduler s(12.0f);
  s.set_timesteps(50);

  const std::vector<float>& sig = s.sigmas();
  CHECK(!sig.empty());
  // The grid starts at 1 and ends at exactly 0; the shift maps both to
  // themselves.
  CHECK_NEAR(sig.front(), 1.0f, 1e-6);
  CHECK_NEAR(sig.back(), 0.0f, 0.0);
  // Strictly decreasing after duplicate collapsing.
  bool decreasing = true;
  for (size_t i = 0; i + 1 < sig.size(); ++i) {
    if (!(sig[i + 1] < sig[i]))
      decreasing = false;
  }
  CHECK(decreasing);
  // One model evaluation per sigma except the terminal zero.
  CHECK(s.timesteps().size() == sig.size() - 1);
  // t = 1 - sigma, and t increases towards clean.
  CHECK_NEAR(s.timesteps().front(), 0.0f, 1e-6);
  CHECK(s.timesteps().back() > s.timesteps().front());

  // The shift pushes sigma above the unshifted grid everywhere in between:
  // 12*b/(1+11b) > b for 0 < b < 1.
  FlowScheduler unshifted(1.0f);
  unshifted.set_timesteps(50);
  CHECK(unshifted.sigmas().size() == 50);
  bool shift_raises = true;
  for (size_t i = 1; i + 1 < unshifted.sigmas().size() && i + 1 < sig.size(); ++i) {
    if (!(sig[i] > unshifted.sigmas()[i]))
      shift_raises = false;
  }
  CHECK(shift_raises);
  // shift = 1 is the identity map.
  CHECK_NEAR(unshifted.sigmas()[10], 1.0f - 10.0f / 49.0f, 1e-6);

  // Audio runs the same scheduler at a different shift.
  FlowScheduler audio(3.0f);
  audio.set_timesteps(50);
  CHECK(audio.sigmas().size() >= 2);
  CHECK_NEAR(audio.sigmas().back(), 0.0f, 0.0);

  // A single Euler step. With v = 0 the update is a pure blend towards x_t,
  // so x stays put; the ratio only rescales the x0 contribution.
  const std::vector<float> x = {1.0f, -2.0f, 0.5f};
  const std::vector<float> zero_v = {0.0f, 0.0f, 0.0f};
  std::vector<float> out(3);
  s.step(0, x.data(), zero_v.data(), 3, out.data());
  // denoised = x + sigma*0 = x, so out = r*x + (1-r)*x = x.
  CHECK_NEAR(out[0], 1.0f, 1e-6);
  CHECK_NEAR(out[1], -2.0f, 1e-6);

  // With a non-zero velocity, verify against the formula directly. `reset()`
  // before each probe because `step` is order-sensitive now — it carries a
  // velocity history under kAb2 — and these are deliberate random accesses
  // rather than a trajectory.
  const std::vector<float> v = {0.25f, 1.0f, -0.5f};
  s.reset();
  s.step(3, x.data(), v.data(), 3, out.data());
  {
    const float t = s.timesteps()[3];
    const float sfrom = 1.0f - t;
    const float ratio = s.sigmas()[4] / s.sigmas()[3];
    for (int i = 0; i < 3; ++i) {
      const float denoised = x[static_cast<size_t>(i)] + sfrom * v[static_cast<size_t>(i)];
      const float want = ratio * x[static_cast<size_t>(i)] + (1.0f - ratio) * denoised;
      CHECK_NEAR(out[static_cast<size_t>(i)], want, 1e-6);
    }
  }

  // step() may alias its input.
  std::vector<float> inplace = x;
  s.reset();
  s.step(3, inplace.data(), v.data(), 3, inplace.data());
  CHECK_NEAR(inplace[0], out[0], 1e-6);

  // The last step lands exactly on x0, because sigma_next is 0 so ratio is 0.
  const int last = static_cast<int>(s.num_steps()) - 1;
  s.reset();
  s.step(last, x.data(), v.data(), 3, out.data());
  {
    const float sfrom = 1.0f - s.timesteps()[static_cast<size_t>(last)];
    CHECK_NEAR(out[0], x[0] + sfrom * v[0], 1e-6);
  }

  // scale_noise: t = 1 returns the clean sample, t = 0 returns pure noise.
  const std::vector<float> noise = {10.0f, 20.0f, 30.0f};
  FlowScheduler::scale_noise(x.data(), noise.data(), 1.0f, 3, out.data());
  CHECK_NEAR(out[0], 1.0f, 1e-6);
  FlowScheduler::scale_noise(x.data(), noise.data(), 0.0f, 3, out.data());
  CHECK_NEAR(out[0], 10.0f, 1e-6);
  FlowScheduler::scale_noise(x.data(), noise.data(), 0.25f, 3, out.data());
  CHECK_NEAR(out[0], 0.25f * 1.0f + 0.75f * 10.0f, 1e-6);

  CHECK(throws([] {
    FlowScheduler bad(12.0f);
    bad.set_timesteps(1);
  }));
  CHECK(throws([] {
    FlowScheduler bad(0.0f);
  }));
}

const bool registered = ::slopfab::test::register_test("json", &test_json) &&
                        ::slopfab::test::register_test("dtype", &test_dtype) &&
                        ::slopfab::test::register_test("safetensors", &test_safetensors) &&
                        ::slopfab::test::register_test("w4a8_state", &test_w4a8_state) &&
                        ::slopfab::test::register_test("compare", &test_compare) &&
                        ::slopfab::test::register_test("scheduler", &test_scheduler);

} // namespace

int main() {
  return ::slopfab::test::run_all();
}
