#pragma once
#include "slopfab/json.h"
#include <filesystem>
#include <string>
#include <vector>

namespace slopfab::cli::updater {
inline constexpr uint64_t kMaxPackageBytes = 512ull * 1024 * 1024;
inline constexpr const char* kRepository = "bitti-ai/slopfab";

struct Release {
  std::string version, url, digest;
  uint64_t bytes = 0;
};

const std::vector<std::string>& package_files();
Release parse_release(const json::Value& value);
bool newer_version(const std::string& candidate, const std::string& installed);
std::string file_hash(const std::filesystem::path& path);
void verify_package(const std::filesystem::path& archive, const std::string& digest,
                    uint64_t bytes);
// Reject links (including Windows junctions), and non-directory parents.
void check_path(const std::filesystem::path& root, const std::filesystem::path& relative,
                bool directory = false);
void extract_package(const std::filesystem::path& archive,
                     const std::filesystem::path& fresh_directory);
// All paths reside on one filesystem. Restore replaced files if installation fails.
void install_package(const std::filesystem::path& root, const std::filesystem::path& payload,
                     const std::filesystem::path& fresh_backup);
}
