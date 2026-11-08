#include "update_package.h"
#include "slopfab/sha256.h"
#include "miniz_tinfl.h"
#include "monocypher-ed25519.h"
#include "update_key.h"
#include <algorithm>
#include <array>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace slopfab::cli::updater {
namespace fs = std::filesystem;

namespace {
[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error("update: " + message);
}

const json::Value& required(const json::Value& value, const char* key) {
  const auto* item = value.find(key);
  if (!item)
    fail(std::string("release is missing ") + key);
  return *item;
}

std::array<uint32_t, 3> version(const std::string& text) {
  std::array<uint32_t, 3> parts{};
  size_t pos = 0;
  for (size_t i = 0; i < parts.size(); ++i) {
    const size_t start = pos;
    uint64_t n = 0;
    while (pos < text.size() && text[pos] >= '0' && text[pos] <= '9') {
      n = n * 10 + unsigned(text[pos++] - '0');
      if (n > UINT32_MAX)
        fail("invalid release version");
    }
    if (pos == start || (pos - start > 1 && text[start] == '0'))
      fail("expected a MAJOR.MINOR.PATCH release version");
    parts[i] = uint32_t(n);
    if (i < 2 && (pos >= text.size() || text[pos++] != '.'))
      fail("expected a MAJOR.MINOR.PATCH release version");
  }
  if (pos != text.size())
    fail("expected a stable MAJOR.MINOR.PATCH release version");
  return parts;
}

bool digest_valid(const std::string& s) {
  return s.size() == 64 && s.find_first_not_of("0123456789abcdef") == std::string::npos;
}

uint16_t u16(const std::vector<uint8_t>& b, size_t p) {
  if (p > b.size() || b.size() - p < 2)
    fail("truncated ZIP");
  return uint16_t(b[p] | (uint16_t(b[p + 1]) << 8));
}

uint32_t u32(const std::vector<uint8_t>& b, size_t p) {
  return uint32_t(u16(b, p)) | (uint32_t(u16(b, p + 2)) << 16);
}

uint32_t crc32(const std::vector<uint8_t>& bytes) {
  static const auto table = [] {
    std::array<uint32_t, 256> result{};
    for (uint32_t i = 0; i < 256; ++i) {
      uint32_t c = i;
      for (int j = 0; j < 8; ++j)
        c = (c >> 1) ^ ((c & 1) ? 0xedb88320u : 0);
      result[i] = c;
    }
    return result;
  }();
  uint32_t c = ~0u;
  for (auto b : bytes)
    c = table[(c ^ b) & 255] ^ (c >> 8);
  return ~c;
}

bool link(const fs::path& path) {
#ifdef _WIN32
  const DWORD attr = GetFileAttributesW(path.c_str());
  if (attr != INVALID_FILE_ATTRIBUTES && (attr & FILE_ATTRIBUTE_REPARSE_POINT))
    return true;
#endif
  return fs::is_symlink(fs::symlink_status(path));
}
}

const std::vector<std::string>& package_files() {
  // Keep in sync with tools/package_bundle.py. Executables are installed last.
  static const std::vector<std::string> names{"mods/download/MOD.md",
                                              "mods/download/main.js",
                                              "mods/download/catalog.json",
                                              "mods/licenses/MOD.md",
                                              "mods/licenses/main.js",
                                              "mods/licenses/licenses.json",
#ifdef _WIN32
                                              "slopfab",
                                              "slopfab.exe"
#else
                                              "slopfab.exe",
                                              "slopfab"
#endif
  };
  return names;
}

Release parse_release(const json::Value& value) {
  if (required(value, "draft").as_bool() || required(value, "prerelease").as_bool())
    fail("latest release must be published and stable");
  const auto tag = required(value, "tag_name").as_string();
  Release result;
  result.version = !tag.empty() && tag.front() == 'v' ? tag.substr(1) : tag;
  version(result.version);
  const std::string name = "slopfab-" + result.version + "-windows-linux-x64.zip";
  size_t matches = 0;
  for (const auto& asset : required(value, "assets").as_array()) {
    if (required(asset, "name").as_string() != name)
      continue;
    ++matches;
    result.url = required(asset, "browser_download_url").as_string();
    if (result.url !=
        "https://github.com/" + std::string(kRepository) + "/releases/download/" + tag + "/" + name)
      fail("unexpected GitHub package URL");
    const auto* hash = asset.find("digest");
    if (!hash || !hash->is_string() || hash->as_string().substr(0, 7) != "sha256:")
      fail("GitHub release asset has no SHA256 digest; re-upload the package");
    result.digest = hash->as_string().substr(7);
    if (!digest_valid(result.digest))
      fail("invalid GitHub package digest");
    const double bytes = required(asset, "size").as_number();
    if (!(bytes > 0 && bytes <= double(kMaxPackageBytes)) || bytes != double(uint64_t(bytes)))
      fail("invalid or oversized release package");
    result.bytes = uint64_t(bytes);
  }
  if (matches != 1)
    fail("latest release must contain exactly one " + name);
  matches = 0;
  for (const auto& asset : required(value, "assets").as_array()) {
    if (required(asset, "name").as_string() != name + ".sig")
      continue;
    ++matches;
    result.signature_url = required(asset, "browser_download_url").as_string();
    if (result.signature_url != result.url + ".sig" || required(asset, "size").as_number() != 64)
      fail("invalid release signature asset");
  }
  if (matches != 1)
    fail("latest release must contain exactly one " + name + ".sig");
  return result;
}

std::string signature_message(const Release& release) {
  version(release.version);
  if (!digest_valid(release.digest) || !release.bytes || release.bytes > kMaxPackageBytes)
    fail("invalid signed release metadata");
  return "slopfab-update-v1\n" + std::string(kRepository) + "\n" + release.version + "\nslopfab-" +
         release.version + "-windows-linux-x64.zip\n" + std::to_string(release.bytes) + "\n" +
         release.digest + "\n";
}

void verify_ed25519(const std::string& public_key_hex, const std::string& message,
                    const std::string& signature) {
  if (!digest_valid(public_key_hex) || signature.size() != 64)
    fail("invalid Ed25519 public key or signature length");
  std::array<uint8_t, 32> key{};
  for (size_t i = 0; i < key.size(); ++i)
    key[i] = uint8_t(std::stoul(public_key_hex.substr(i * 2, 2), nullptr, 16));
  if (crypto_ed25519_check(reinterpret_cast<const uint8_t*>(signature.data()), key.data(),
                           reinterpret_cast<const uint8_t*>(message.data()), message.size()) != 0)
    fail("release Ed25519 signature verification failed");
}

void verify_release_signature(const Release& release, const std::string& signature) {
  verify_ed25519(kUpdatePublicKey, signature_message(release), signature);
}

bool newer_version(const std::string& candidate, const std::string& installed) {
  return version(candidate) > version(installed);
}

std::string file_hash(const fs::path& path) {
  const auto digest = sha256_file(path.u8string());
  std::string result;
  for (auto byte : digest) {
    result += "0123456789abcdef"[byte >> 4];
    result += "0123456789abcdef"[byte & 15];
  }
  return result;
}

void verify_package(const fs::path& archive, const std::string& digest, uint64_t bytes) {
  if (!digest_valid(digest) || !bytes || bytes > kMaxPackageBytes ||
      fs::file_size(archive) != bytes || file_hash(archive) != digest)
    fail("package size or SHA256 checksum does not match GitHub");
}

void check_path(const fs::path& root, const fs::path& relative, bool directory) {
  if (relative.empty() || relative.is_absolute())
    fail("invalid installation path");
  if (link(root) || !fs::is_directory(root))
    fail("installation root must be a real directory");
  auto path = root;
  for (auto i = relative.begin(); i != relative.end(); ++i) {
    if (*i == "." || *i == ".." || i->empty())
      fail("invalid installation path");
    path /= *i;
    if (link(path))
      fail("refusing linked installation path: " + path.u8string());
    if (fs::exists(path)) {
      const bool want_directory = std::next(i) != relative.end() || directory;
      if (want_directory ? !fs::is_directory(path) : !fs::is_regular_file(path))
        fail("unexpected installation file type: " + path.u8string());
    }
  }
}

void extract_package(const fs::path& archive, const fs::path& directory) {
  const auto size = fs::file_size(archive);
  if (size < 22 || size > kMaxPackageBytes)
    fail("invalid ZIP size");
  std::vector<uint8_t> bytes(static_cast<size_t>(size));
  std::ifstream input(archive, std::ios::binary);
  if (!input.read(reinterpret_cast<char*>(bytes.data()), std::streamsize(bytes.size())))
    fail("cannot read package");
  size_t end = bytes.size() - 22;
  const size_t first = end > 65535 ? end - 65535 : 0;
  for (;;) {
    if (u32(bytes, end) == 0x06054b50 && end + 22 + u16(bytes, end + 20) == bytes.size())
      break;
    if (end == first)
      fail("ZIP directory not found");
    --end;
  }
  const size_t count = u16(bytes, end + 10);
  const size_t central = u32(bytes, end + 16);
  if (u16(bytes, end + 4) || u16(bytes, end + 6) || count != u16(bytes, end + 8) ||
      count != package_files().size() || central > end || u32(bytes, end + 12) != end - central)
    fail("unsupported ZIP directory (expected the standard slopfab bundle)");

  struct Entry {
    std::string name;
    size_t data, packed, unpacked;
    uint16_t method;
    uint32_t crc;
  };

  std::vector<Entry> entries;
  std::set<std::string> seen;
  uint64_t total = 0;
  size_t cursor = central;
  for (size_t i = 0; i < count; ++i) {
    if (cursor > end || end - cursor < 46 || u32(bytes, cursor) != 0x02014b50)
      fail("invalid ZIP entry");
    const size_t name_size = u16(bytes, cursor + 28);
    const size_t entry_size = 46 + name_size + u16(bytes, cursor + 30) + u16(bytes, cursor + 32);
    if (entry_size > end - cursor)
      fail("truncated ZIP entry");
    Entry e;
    e.name.assign(reinterpret_cast<const char*>(bytes.data() + cursor + 46), name_size);
    if (std::find(package_files().begin(), package_files().end(), e.name) ==
            package_files().end() ||
        !seen.insert(e.name).second)
      fail("unexpected or duplicate package path: " + e.name);
    const auto flags = u16(bytes, cursor + 8);
    e.method = u16(bytes, cursor + 10);
    e.crc = u32(bytes, cursor + 16);
    e.packed = u32(bytes, cursor + 20);
    e.unpacked = u32(bytes, cursor + 24);
    const auto mode = u32(bytes, cursor + 38) >> 16;
    if ((flags & ~0x080e) || (e.method != 0 && e.method != 8) ||
        ((mode & 0170000) && (mode & 0170000) != 0100000) || (u32(bytes, cursor + 38) & 0x10) ||
        u16(bytes, cursor + 34))
      fail("unsupported ZIP entry type");
    total += e.unpacked;
    if (!e.unpacked || total > kMaxPackageBytes)
      fail("empty or oversized package contents");
    const size_t local = u32(bytes, cursor + 42);
    if (local > central || central - local < 30 || u32(bytes, local) != 0x04034b50 ||
        u16(bytes, local + 6) != flags || u16(bytes, local + 8) != e.method ||
        u16(bytes, local + 26) != name_size)
      fail("invalid local ZIP header");
    e.data = local + 30 + name_size + u16(bytes, local + 28);
    if (e.data > central || e.packed > central - e.data ||
        std::string(reinterpret_cast<const char*>(bytes.data() + local + 30), name_size) != e.name)
      fail("invalid local ZIP path or size");
    entries.push_back(e);
    cursor += entry_size;
  }
  if (cursor != end)
    fail("unexpected ZIP directory contents");
  if (!fs::create_directory(directory))
    fail("staging directory already exists");
  for (const auto& e : entries) {
    std::vector<uint8_t> output(e.unpacked);
    if (e.method == 0) {
      if (e.packed != e.unpacked)
        fail("invalid stored ZIP size");
      std::copy_n(bytes.data() + e.data, e.packed, output.data());
    } else if (tinfl_decompress_mem_to_mem(output.data(), output.size(), bytes.data() + e.data,
                                           e.packed, 0) != output.size()) {
      fail("invalid DEFLATE data");
    }
    if (crc32(output) != e.crc)
      fail("ZIP checksum mismatch");
    if (e.name == "slopfab.exe" && (output.size() < 2 || output[0] != 'M' || output[1] != 'Z'))
      fail("package has no Windows executable");
    if (e.name == "slopfab" && (output.size() < 4 || output[0] != 0x7f || output[1] != 'E' ||
                                output[2] != 'L' || output[3] != 'F'))
      fail("package has no Linux executable");
    const auto path = directory / fs::u8path(e.name);
    fs::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(output.data()), std::streamsize(output.size()));
    out.close();
    if (!out)
      fail("cannot write staged package");
#ifndef _WIN32
    if (e.name == "slopfab")
      fs::permissions(path, fs::perms::owner_all | fs::perms::group_read | fs::perms::group_exec |
                                fs::perms::others_read | fs::perms::others_exec);
#endif
  }
}

void install_package(const fs::path& root, const fs::path& payload, const fs::path& backup) {
  for (const auto& name : package_files()) {
    check_path(root, fs::u8path(name));
    check_path(payload, fs::u8path(name));
    if (!fs::is_regular_file(payload / fs::u8path(name)))
      fail("missing staged file: " + name);
  }
  if (!fs::create_directory(backup))
    fail("backup directory already exists");

  struct Change {
    fs::path name;
    bool saved = false, installed = false;
  };

  std::vector<Change> changes;
  try {
    for (const auto& name : package_files()) {
      changes.push_back({fs::u8path(name)});
      auto& change = changes.back();
      const auto target = root / change.name;
      check_path(root, change.name);
      fs::create_directories(target.parent_path());
      if (fs::exists(target)) {
        fs::create_directories((backup / change.name).parent_path());
        fs::rename(target, backup / change.name);
        change.saved = true;
      }
      fs::rename(payload / change.name, target);
      change.installed = true;
    }
  } catch (const std::exception& error) {
    std::string recovery;
    for (auto i = changes.rbegin(); i != changes.rend(); ++i) {
      try {
        if (i->installed)
          fs::rename(root / i->name, payload / i->name);
        if (i->saved)
          fs::rename(backup / i->name, root / i->name);
      } catch (const std::exception& rollback) {
        recovery += std::string("; ") + rollback.what();
      }
    }
    fail(std::string("installation failed: ") + error.what() +
         (recovery.empty() ? "; previous files restored"
                           : "; recovery required from " + backup.u8string() + recovery));
  }
}
}
