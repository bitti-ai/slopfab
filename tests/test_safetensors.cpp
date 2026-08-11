// Reader tests for the mapped safetensors file, focused on the ranged
// readahead hint used by the Qwen vision tower load.
//
// A prefetch hint is invisible when it works and invisible when it does not,
// so the two things worth asserting are that the *extent* it is given is the
// right one and that taking the hint never changes a byte of what is read.
// Both are checked here against a file the test writes itself, because the
// real checkpoints live under `ref/` and are not committed.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "harness.h"
#include "vidfab/safetensors.h"

namespace {

using vidfab::SafeTensors;
using vidfab::TensorView;

struct Spec {
  std::string name;
  size_t elements;  // F32
};

// Writes a minimal but genuinely valid safetensors file: the 8-byte header
// length, a JSON header, then the tensor data in the declared order. Every
// element of tensor `k` is written as the float `k * 1000 + i`, so a range that
// is read back wrong is obvious rather than plausible.
std::string write_file(const std::vector<Spec>& specs, const std::string& stem) {
  std::string header = "{";
  size_t offset = 0;
  for (size_t k = 0; k < specs.size(); ++k) {
    const size_t bytes = specs[k].elements * 4;
    if (k != 0) header += ",";
    header += "\"" + specs[k].name + "\":{\"dtype\":\"F32\",\"shape\":[" +
              std::to_string(specs[k].elements) + "],\"data_offsets\":[" +
              std::to_string(offset) + "," + std::to_string(offset + bytes) + "]}";
    offset += bytes;
  }
  header += "}";

  const std::filesystem::path path = std::filesystem::temp_directory_path() / (stem + ".safetensors");
  std::ofstream out(path, std::ios::binary);
  uint64_t header_len = header.size();
  out.write(reinterpret_cast<const char*>(&header_len), 8);
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  for (size_t k = 0; k < specs.size(); ++k) {
    for (size_t i = 0; i < specs[k].elements; ++i) {
      const float v = static_cast<float>(k) * 1000.0f + static_cast<float>(i);
      out.write(reinterpret_cast<const char*>(&v), 4);
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

VIDFAB_TEST(safetensors_prefix_extent_bounds_the_matching_tensors) {
  // Deliberately not in name order on disk, and with a decoy whose name shares
  // the prefix's leading characters but not the prefix itself. The extent must
  // come from the three `visual.` tensors and nothing else.
  const std::vector<Spec> specs = {
      {"visual.blocks.1.weight", 8},
      {"other.weight", 16},
      {"visual.blocks.0.weight", 4},
      {"visualise.weight", 32},  // decoy: "visual" is a prefix, "visual." is not
      {"visual.merger.weight", 2},
      {"zzz.tail", 64},
  };
  const std::string path = write_file(specs, "vidfab_prefix_extent");

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
  CHECK_MSG(static_cast<const uint8_t*>(begin) == base + lo,
            "extent starts at %zu, expected %zu",
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

VIDFAB_TEST(safetensors_prefetch_range_is_advisory_and_bounded) {
  const std::vector<Spec> specs = {{"a.weight", 1024}, {"b.weight", 2048}, {"c.weight", 512}};
  const std::string path = write_file(specs, "vidfab_prefetch_range");

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
      ok = data[i] == static_cast<float>(k) * 1000.0f + static_cast<float>(i);
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

}  // namespace
