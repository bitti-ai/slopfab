#pragma once
#include <array>
#include <cstdint>
#include <filesystem>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/stat.h>
#endif

namespace slopfab::detail {
// File identity and change time prevent reuse after replacement or after an
// in-place edit that preserves size and restores the modification timestamp.
inline std::array<uint64_t, 7> lcs_file_stamp(const std::filesystem::path& path) {
#ifdef _WIN32
  HANDLE file = CreateFileW(path.c_str(), FILE_READ_ATTRIBUTES,
      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
      OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (file == INVALID_HANDLE_VALUE)
    throw std::runtime_error("Cannot inspect LCS VAE " + path.string());
  BY_HANDLE_FILE_INFORMATION info{};
  FILE_BASIC_INFO basic{};
  const bool ok = GetFileInformationByHandle(file, &info) &&
      GetFileInformationByHandleEx(file, FileBasicInfo, &basic, sizeof(basic));
  CloseHandle(file);
  if (!ok)
    throw std::runtime_error("Cannot read LCS VAE file identity " + path.string());
  return {(uint64_t(info.nFileSizeHigh) << 32) | info.nFileSizeLow,
      info.dwVolumeSerialNumber, info.nFileIndexHigh, info.nFileIndexLow,
      uint64_t(basic.LastWriteTime.QuadPart), uint64_t(basic.ChangeTime.QuadPart),
      uint64_t(basic.CreationTime.QuadPart)};
#else
  struct stat info{};
  if (stat(path.c_str(), &info) != 0)
    throw std::runtime_error("Cannot inspect LCS VAE " + path.string());
#ifdef __APPLE__
  const auto modified = info.st_mtimespec, changed = info.st_ctimespec;
#else
  const auto modified = info.st_mtim, changed = info.st_ctim;
#endif
  return {uint64_t(info.st_size), uint64_t(info.st_dev), uint64_t(info.st_ino),
      uint64_t(modified.tv_sec), uint64_t(modified.tv_nsec),
      uint64_t(changed.tv_sec), uint64_t(changed.tv_nsec)};
#endif
}

class LcsFingerprintCache {
  struct Entry {
    std::array<uint64_t, 7> stamp;
    std::string digest;
  };
  std::mutex mutex_;
  std::map<std::filesystem::path, Entry> entries_;
public:
  template <class Hash> std::string get(const std::string& path, Hash hash) {
    const auto canonical = std::filesystem::canonical(path);
    // One reader hashes a miss; concurrent users reuse the completed digest.
    std::lock_guard<std::mutex> lock(mutex_);
    const auto stamp = lcs_file_stamp(canonical);
    auto found = entries_.find(canonical);
    if (found != entries_.end() && found->second.stamp == stamp)
      return found->second.digest;
    entries_.erase(canonical);
    const auto digest = hash(canonical.string());
    if (lcs_file_stamp(canonical) != stamp)
      throw std::runtime_error("VAE changed while checking LCS; retry generation");
    if (entries_.size() >= 32)
      entries_.clear();
    entries_.emplace(canonical, Entry{stamp, digest});
    return digest;
  }
};
} // namespace slopfab::detail
