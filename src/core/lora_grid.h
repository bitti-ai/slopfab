#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "slopfab/safetensors.h"
#include "slopfab/sha256.h"

namespace slopfab::detail {

inline constexpr const char* kLoraGridTensor = "slopfab.silu_t_emb_grid";

// Owns the small grid, not the adapter. The original identity is checked before
// replacement; all adapter mappings must be closed before embed() is called.
struct LoraGrid {
  TensorView tensor;
  std::vector<uint8_t> bytes;
  std::string adapter_path;
  std::string identity;
  uintmax_t original_size = 0;
  std::filesystem::file_time_type original_time;
  Sha256Digest original_header{};
  bool needs_embedding = false;

  void load(const SafeTensors& adapter, int width, bool allow_download);
  void embed() const;
};

} // namespace slopfab::detail
