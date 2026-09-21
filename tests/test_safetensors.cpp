// Tests for the mapped safetensors reader and the widening that reads out of
// it: the ranged readahead hint used by the Qwen vision tower load, and
// `to_f32`'s output buffer contract.
//
// A prefetch hint is invisible when it works and invisible when it does not,
// so the two things worth asserting are that the *extent* it is given is the
// right one and that taking the hint never changes a byte of what is read.
// Both are checked here against a file the test writes itself, because the
// real checkpoints live under `ref/` and are not committed.

#include <cstdint>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

#include "harness.h"
#include "slopfab/safetensors.h"
#include "slopfab/sha256.h"
#include "slopfab/tensor_convert.h"
#include "slopfab/text/layer_capture.h"

namespace {

using slopfab::SafeTensors;
using slopfab::TensorView;

struct Spec {
  std::string name;
  size_t elements;
  const char* dtype = "F32";
};

// The value tensor `k` holds at index `i`. An integer at most 256 scaled by a
// power of two, so it is exact in bf16 (8 significand bits) as well as in fp16
// and fp32 — the widening tests below can then assert equality rather than a
// tolerance — while still differing enough between tensors and positions that
// a range read back from the wrong place is obviously wrong.
float element(size_t k, size_t i) {
  return static_cast<float>(1 + i % 200) * static_cast<float>(1u << k);
}

// Writes a minimal but genuinely valid safetensors file: the 8-byte header
// length, a JSON header, then the tensor data in the declared order.
std::string write_file(const std::vector<Spec>& specs, const std::string& stem) {
  auto width = [](const Spec& s) -> size_t {
    return std::strcmp(s.dtype, "F32") == 0 ? 4 : 2;
  };
  std::string header = "{";
  size_t offset = 0;
  for (size_t k = 0; k < specs.size(); ++k) {
    const size_t bytes = specs[k].elements * width(specs[k]);
    if (k != 0)
      header += ",";
    header += "\"" + specs[k].name + "\":{\"dtype\":\"" + specs[k].dtype + "\",\"shape\":[" +
              std::to_string(specs[k].elements) + "],\"data_offsets\":[" + std::to_string(offset) +
              "," + std::to_string(offset + bytes) + "]}";
    offset += bytes;
  }
  header += "}";

  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / (stem + ".safetensors");
  std::ofstream out(path, std::ios::binary);
  uint64_t header_len = header.size();
  out.write(reinterpret_cast<const char*>(&header_len), 8);
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  for (size_t k = 0; k < specs.size(); ++k) {
    for (size_t i = 0; i < specs[k].elements; ++i) {
      const float v = element(k, i);
      if (width(specs[k]) == 4) {
        out.write(reinterpret_cast<const char*>(&v), 4);
      } else {
        const uint16_t h = std::strcmp(specs[k].dtype, "BF16") == 0 ? slopfab::f32_to_bf16(v)
                                                                    : slopfab::f32_to_f16(v);
        out.write(reinterpret_cast<const char*>(&h), 2);
      }
    }
  }
  out.close();
  return path.string();
}

// Byte offset of `name`'s data from the start of the mapping.
size_t offset_of(const SafeTensors& st, const char* name) {
  const auto* base = static_cast<const uint8_t*>(st.mapping_base());
  return static_cast<size_t>(static_cast<const uint8_t*>(st.at(name).data) - base);
}

uint64_t fnv64(const void* data, size_t bytes, uint64_t hash = 1469598103934665603ull) {
  const auto* cursor = static_cast<const uint8_t*>(data);
  for (size_t i = 0; i < bytes; ++i) {
    hash ^= cursor[i];
    hash *= 1099511628211ull;
  }
  return hash;
}

SLOPFAB_TEST(qwen_layer_capture_is_bounded_and_self_verifying) {
  const char abc[] = "abc";
  constexpr slopfab::Sha256Digest abc_sha{0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea,
                                          0x41, 0x41, 0x40, 0xde, 0x5d, 0xae, 0x22, 0x23,
                                          0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c,
                                          0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00, 0x15, 0xad};
  CHECK(slopfab::sha256_bytes(abc, 3) == abc_sha);

  slopfab::text::QwenLayerCapture capture;
  capture.header.sequence = 1;
  capture.header.hidden = 5120;
  capture.header.query_heads = 64;
  capture.header.kv_heads = 8;
  capture.header.head_dim = 128;
  capture.header.intermediate = 25600;
  capture.header.checkpoint_sha256.fill(0x11);
  capture.header.tokenizer_sha256.fill(0x22);
  capture.header.boundary_fnv64.fill(0x123456789abcdef0ull);
  capture.token_ids = {17};
  capture.input_bf16.resize(5120);
  for (size_t i = 0; i < capture.input_bf16.size(); ++i)
    capture.input_bf16[i] = static_cast<uint16_t>(i * 37u);
  capture.cosine.resize(128, 1.0f);
  capture.sine.resize(128, 0.0f);
  capture.header.input_fnv64 = fnv64(capture.input_bf16.data(), capture.input_bf16.size() * 2);
  capture.header.rope_fnv64 = fnv64(capture.cosine.data(), capture.cosine.size() * sizeof(float));
  capture.header.rope_fnv64 =
      fnv64(capture.sine.data(), capture.sine.size() * sizeof(float), capture.header.rope_fnv64);
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() /
      ("slopfab_qwen_capture_validation_" +
       std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count()) +
       ".vfqw");
  auto write_valid = [&] {
    slopfab::text::write_qwen_layer_capture(path.string(), capture);
  };
  auto rejects = [&] {
    try {
      (void)slopfab::text::read_qwen_layer_capture(path.string());
    } catch (const std::exception&) {
      return true;
    }
    return false;
  };
  write_valid();
  CHECK(slopfab::text::read_qwen_layer_capture(path.string()).input_bf16 == capture.input_bf16);
  slopfab::Sha256Digest memory_digest{};
  {
    std::ifstream bytes_in(path, std::ios::binary);
    const std::vector<uint8_t> bytes{std::istreambuf_iterator<char>(bytes_in),
                                     std::istreambuf_iterator<char>()};
    memory_digest = slopfab::sha256_bytes(bytes.data(), bytes.size());
  }
  CHECK(slopfab::sha256_file(path.string()) == memory_digest);

  // Truncation and trailing bytes are rejected from exact file-size preflight.
  const uintmax_t valid_size = std::filesystem::file_size(path);
  std::filesystem::resize_file(path, valid_size - 1);
  CHECK(rejects());
  write_valid();
  {
    std::ofstream out(path, std::ios::binary | std::ios::app);
    out.put('\0');
  }
  CHECK(rejects());

  // Header corruption and a production-cap overflow fail before allocation.
  write_valid();
  {
    std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
    file.put('X');
  }
  CHECK(rejects());
  write_valid();
  {
    std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
    const uint32_t oversized = 8193;
    file.seekp(12); // magic[8], version[4]
    file.write(reinterpret_cast<const char*>(&oversized), sizeof(oversized));
  }
  std::filesystem::resize_file(path, 90ull << 20); // sparse on supported filesystems
  CHECK(rejects());
  write_valid();
  {
    std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
    const uint32_t overflowing = std::numeric_limits<uint32_t>::max();
    file.seekp(12);
    file.write(reinterpret_cast<const char*>(&overflowing), sizeof(overflowing));
  }
  CHECK(rejects());

  // A changed input word and an out-of-range token both survive shape parsing
  // but fail the payload digest/domain checks.
  write_valid();
  {
    std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
    file.seekg(sizeof(slopfab::text::QwenLayerCaptureHeader) + sizeof(int32_t));
    char byte = 0;
    file.read(&byte, 1);
    byte ^= 1;
    file.seekp(sizeof(slopfab::text::QwenLayerCaptureHeader) + sizeof(int32_t));
    file.write(&byte, 1);
  }
  CHECK(rejects());
  write_valid();
  {
    std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
    const int32_t invalid_token = 151936;
    file.seekp(sizeof(slopfab::text::QwenLayerCaptureHeader));
    file.write(reinterpret_cast<const char*>(&invalid_token), sizeof(invalid_token));
  }
  CHECK(rejects());
  std::filesystem::remove(path);
}

SLOPFAB_TEST(safetensors_comfy_diffusion_namespace) {
  const std::string prefix = "model.diffusion_model.";
  const auto path = write_file({{prefix + "adaln_t_table", 8}, {prefix + "blocks.0.weight", 4}},
                               "slopfab_comfy_namespace");
  SafeTensors st;
  st.open(path);
  CHECK(st.tensor_count() == 2);
  CHECK(st.find("blocks.0.weight") == st.find(prefix + "blocks.0.weight"));
  CHECK(st.at("blocks.0.weight").name == prefix + "blocks.0.weight");
  CHECK(st.tensors().count(prefix + "blocks.0.weight") == 1);
  CHECK(st.find("absent") == nullptr);
  CHECK(static_cast<const float*>(st.at("blocks.0.weight").data)[0] == element(1, 0));
  SafeTensors moved(std::move(st));
  CHECK(moved.find("adaln_t_table") != nullptr);
  st = std::move(moved);
  CHECK(st.find("adaln_t_table") != nullptr);
  st.close();
  CHECK(st.find("adaln_t_table") == nullptr);
  std::filesystem::remove(path);

  // A mixed namespace is not an aliasable model. Exact names still work.
  const auto mixed =
      write_file({{prefix + "a", 1}, {"a", 1}, {prefix + "b", 1}}, "slopfab_comfy_mixed");
  st.open(mixed);
  CHECK(st.at("a").name == "a");
  CHECK(st.find("b") == nullptr);
  CHECK(st.find(prefix + "b") != nullptr);
  st.close();
  std::filesystem::remove(mixed);
}

SLOPFAB_TEST(safetensors_prefix_extent_bounds_the_matching_tensors) {
  // Deliberately not in name order on disk, and with a decoy whose name shares
  // the prefix's leading characters but not the prefix itself. The extent must
  // come from the three `visual.` tensors and nothing else.
  const std::vector<Spec> specs = {
      {"visual.blocks.1.weight", 8}, {"other.weight", 16}, {"visual.blocks.0.weight", 4},
      {"visualise.weight", 32}, // decoy: "visual" is a prefix, "visual." is not
      {"visual.merger.weight", 2},   {"zzz.tail", 64},
  };
  const std::string path = write_file(specs, "slopfab_prefix_extent");

  SafeTensors st;
  st.open(path);
  CHECK(st.tensor_count() == specs.size());

  const void* begin = nullptr;
  size_t bytes = 0;
  st.prefix_extent("visual.", &begin, &bytes);
  CHECK(begin != nullptr);

  // Lowest and highest of the three matching tensors by file offset, which is
  // `visual.blocks.1` (written first) through the end of `visual.merger`.
  const size_t lo = offset_of(st, "visual.blocks.1.weight");
  const size_t hi = offset_of(st, "visual.merger.weight") + st.at("visual.merger.weight").nbytes;
  const auto* base = static_cast<const uint8_t*>(st.mapping_base());
  CHECK_MSG(static_cast<const uint8_t*>(begin) == base + lo, "extent starts at %zu, expected %zu",
            static_cast<size_t>(static_cast<const uint8_t*>(begin) - base), lo);
  CHECK_MSG(bytes == hi - lo, "extent spans %zu bytes, expected %zu", bytes, hi - lo);

  // The decoy sits between them on disk, so the extent legitimately covers it.
  // That is the documented contract — a bounding extent, not a filter — and the
  // point of asserting it is that a future "tighten this" change has to notice.
  CHECK(offset_of(st, "visualise.weight") > lo);

  // A prefix nothing matches yields an empty answer rather than the whole file.
  const void* none_begin = reinterpret_cast<const void*>(~uintptr_t{0});
  size_t none_bytes = 12345;
  st.prefix_extent("absent.", &none_begin, &none_bytes);
  CHECK(none_begin == nullptr);
  CHECK(none_bytes == 0);

  // An empty prefix matches everything, which is the whole-file case.
  const void* all_begin = nullptr;
  size_t all_bytes = 0;
  st.prefix_extent("", &all_begin, &all_bytes);
  CHECK(all_begin == base + offset_of(st, "visual.blocks.1.weight"));
  CHECK(all_bytes == st.file_size() - offset_of(st, "visual.blocks.1.weight"));

  st.close();
  std::filesystem::remove(path);
}

SLOPFAB_TEST(safetensors_prefetch_range_is_advisory_and_bounded) {
  const std::vector<Spec> specs = {{"a.weight", 1024}, {"b.weight", 2048}, {"c.weight", 512}};
  const std::string path = write_file(specs, "slopfab_prefetch_range");

  SafeTensors st;
  st.open(path);
  const auto* base = static_cast<const uint8_t*>(st.mapping_base());

  // Degenerate requests are refused rather than guessed at. A hint that walks
  // off the end of the mapping would be a hint about someone else's memory.
  CHECK(st.prefetch_range(nullptr, 16) == false);
  CHECK(st.prefetch_range(base, 0) == false);
  CHECK(st.prefetch_range(base - 4096, 16) == false);
  CHECK(st.prefetch_range(base + st.file_size() + 4096, 16) == false);

  // A request that starts inside and runs past the end is clamped, not
  // refused: the caller asked about bytes that exist plus bytes that do not.
  st.prefetch_range(base + st.file_size() - 8, ~size_t{0} / 2);

  const void* begin = nullptr;
  size_t bytes = 0;
  st.prefix_extent("b.", &begin, &bytes);
  st.prefetch_range(begin, bytes);
  st.prefetch();

  // The whole point: after every hint above, every byte still reads back
  // exactly as written. `prefetch_range` returning false is a slow path, never
  // a wrong one.
  for (size_t k = 0; k < specs.size(); ++k) {
    const TensorView& v = st.at(specs[k].name);
    const auto* data = static_cast<const float*>(v.data);
    bool ok = v.nbytes == specs[k].elements * 4;
    for (size_t i = 0; ok && i < specs[k].elements; ++i) {
      ok = data[i] == element(k, i);
    }
    CHECK_MSG(ok, "%s did not read back intact after prefetching", specs[k].name.c_str());
  }

  st.close();

  // Closed reader: nothing to hint about, and no dereference of a null base.
  CHECK(st.prefetch() == false);
  CHECK(st.prefetch_range(base, 16) == false);
  const void* begin_closed = reinterpret_cast<const void*>(~uintptr_t{0});
  size_t bytes_closed = 7;
  st.prefix_extent("a.", &begin_closed, &bytes_closed);
  CHECK(begin_closed == nullptr);
  CHECK(bytes_closed == 0);

  std::filesystem::remove(path);
}

// `to_f32` used to zero its output buffer before overwriting every element of
// it, which cost a 155 MB memset per AdaLN projection on the real transformer
// checkpoint. It now resizes instead. That is only equivalent if the buffer
// ends up the same size with the same contents no matter what it held before,
// which is exactly what a reused buffer makes easy to get wrong: a shorter
// tensor followed by a longer one, or a longer one followed by a shorter one,
// must not leave any element of the previous tensor visible.
SLOPFAB_TEST(to_f32_result_does_not_depend_on_the_reused_buffer) {
  const std::vector<Spec> specs = {
      {"long.f32", 4096, "F32"}, {"short.f16", 7, "F16"},   {"mid.bf16", 300, "BF16"},
      {"empty.f32", 0, "F32"},   {"tail.f16", 5000, "F16"},
  };
  const std::string path = write_file(specs, "slopfab_to_f32_reuse");

  SafeTensors st;
  st.open(path);

  // The answer each tensor produces into a buffer that has never been used.
  std::vector<std::vector<float>> fresh(specs.size());
  for (size_t k = 0; k < specs.size(); ++k) {
    fresh[k] = slopfab::to_f32(st.at(specs[k].name));
    CHECK_MSG(fresh[k].size() == specs[k].elements, "%s widened to %zu elements, expected %zu",
              specs[k].name.c_str(), fresh[k].size(), specs[k].elements);
  }

  // The same answers out of one buffer walked in every order, so every
  // grow-then-shrink and shrink-then-grow transition is covered.
  std::vector<float> reused;
  size_t diffs = 0;
  std::string first_bad;
  for (size_t start = 0; start < specs.size(); ++start) {
    for (size_t step = 0; step < specs.size(); ++step) {
      const size_t k = (start + step) % specs.size();
      slopfab::to_f32(st.at(specs[k].name), reused);
      if (reused != fresh[k]) {
        if (diffs == 0)
          first_bad = specs[k].name;
        ++diffs;
      }
    }
  }
  CHECK_MSG(diffs == 0, "%zu reused-buffer conversions differ, first '%s'", diffs,
            first_bad.c_str());

  // And the values themselves are what was written. fp16 and bf16 hold these
  // exactly, so this is an equality, not a tolerance.
  for (size_t k = 0; k < specs.size(); ++k) {
    bool ok = true;
    size_t bad = 0;
    for (size_t i = 0; ok && i < specs[k].elements; ++i) {
      ok = fresh[k][i] == element(k, i);
      if (!ok)
        bad = i;
    }
    CHECK_MSG(ok, "%s widened to the wrong values: [%zu] is %.9g, expected %.9g",
              specs[k].name.c_str(), bad, ok ? 0.0 : static_cast<double>(fresh[k][bad]),
              static_cast<double>(element(k, bad)));
  }

  st.close();
  std::filesystem::remove(path);
}

} // namespace
