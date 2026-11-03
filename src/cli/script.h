#pragma once
#include "slopfab/json.h"
#include <filesystem>
#include <string>
#include <vector>

namespace slopfab::cli {
int script_command(const std::vector<std::string>& args);
int run_script(const std::filesystem::path& path, const std::vector<std::string>& args,
               const std::string* source = nullptr, size_t memory_mb = 256,
               uint64_t timeout_ms = 0);
}
