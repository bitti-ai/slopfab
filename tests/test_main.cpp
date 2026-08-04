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

#include "vidfab/dtype.h"
#include "vidfab/json.h"
#include "vidfab/safetensors.h"

namespace {

int g_checks = 0;
int g_failures = 0;
const char* g_current_test = "";

void check(bool ok, const char* expr, int line) {
  ++g_checks;
  if (!ok) {
    ++g_failures;
    std::fprintf(stderr, "  FAIL %s:%d  %s\n", g_current_test, line, expr);
  }
}

void check_near(double a, double b, double tol, const char* expr, int line) {
  ++g_checks;
  if (!(std::fabs(a - b) <= tol)) {
    ++g_failures;
    std::fprintf(stderr, "  FAIL %s:%d  %s  (%g vs %g)\n", g_current_test, line, expr, a, b);
  }
}

#define CHECK(expr) check((expr), #expr, __LINE__)
#define CHECK_NEAR(a, b, tol) check_near((a), (b), (tol), #a " ~= " #b, __LINE__)

#define TEST(name)                     \
  g_current_test = name;               \
  std::printf("test %s\n", name);

bool throws(void (*fn)()) {
  try {
    fn();
  } catch (const std::exception&) {
    return true;
  }
  return false;
}

// --- json -------------------------------------------------------------------

void test_json() {
  TEST("json");
  using namespace vidfab;

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

  // Escapes, including a surrogate pair (U+1F600).
  const json::Value esc = json::parse(R"({"s": "a\"b\\c\ndAe😀"})");
  const std::string& s = esc.find("s")->as_string();
  CHECK(s.find("a\"b\\c\nd") == 0);
  CHECK(s.find("Ae") != std::string::npos);
  CHECK(s.size() > 10);  // 4-byte UTF-8 emoji appended

  CHECK(json::parse("{}").as_object().empty());
  CHECK(json::parse("[]").as_array().empty());

  CHECK(throws([] { json::parse("{"); }));
  CHECK(throws([] { json::parse("{\"a\": }"); }));
  CHECK(throws([] { json::parse("{\"a\": 1} trailing"); }));
  CHECK(throws([] { json::parse("\"unterminated"); }));
  CHECK(throws([] { json::parse("{\"a\": tru}"); }));
}

// --- dtype ------------------------------------------------------------------

void test_dtype() {
  TEST("dtype");
  using namespace vidfab;

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
  CHECK_NEAR(f4_lo(0xF7), 6.0f, 0.0);   // nibble 7
  CHECK_NEAR(f4_hi(0xF7), -6.0f, 0.0);  // nibble 15 = sign | 7
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
  using namespace vidfab;

  // Two tensors: a 2x3 f32 and a scalar f32, plus a metadata block.
  std::vector<uint8_t> data(6 * 4 + 4);
  const float values[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::memcpy(data.data(), values, sizeof(values));
  const float scalar = 0.25f;
  std::memcpy(data.data() + 24, &scalar, sizeof(scalar));

  const std::string header =
      R"({"__metadata__":{"format":"pt","note":"unit test"},)"
      R"("w":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},)"
      R"("s":{"dtype":"F32","shape":[],"data_offsets":[24,28]}})";

  const std::string path = write_temp_safetensors(header, data, "vidfab_ok.safetensors");

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
  const std::string bad_header =
      R"({"w":{"dtype":"F32","shape":[2,3],"data_offsets":[0,16]}})";
  const std::string bad_path =
      write_temp_safetensors(bad_header, std::vector<uint8_t>(16, 0), "vidfab_bad.safetensors");
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
  const std::string over_header =
      R"({"w":{"dtype":"F32","shape":[100],"data_offsets":[0,400]}})";
  const std::string over_path =
      write_temp_safetensors(over_header, std::vector<uint8_t>(8, 0), "vidfab_over.safetensors");
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

}  // namespace

int main() {
  test_json();
  test_dtype();
  test_safetensors();

  std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
