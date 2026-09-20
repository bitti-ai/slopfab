#include "lora_grid.h"
#include "slopfab/lora.h"
#include "slopfab/json.h"
#include <set>
#include <mutex>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <winhttp.h>
#endif

namespace slopfab::detail {
namespace {
namespace fs = std::filesystem;

struct TemporaryDirectory {
  fs::path path;
  explicit TemporaryDirectory(const fs::path& parent) {
    static std::atomic<uint64_t> serial{0};
    for (int attempt = 0; attempt < 100; ++attempt) {
      const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
      auto candidate = parent / (".slopfab-lora-" + std::to_string(tick) + "-" + std::to_string(serial++));
      std::error_code error;
      if (fs::create_directory(candidate, error)) { path = std::move(candidate); return; }
      if (error) throw std::runtime_error("LoRA: cannot create temporary directory: " + error.message());
    }
    throw std::runtime_error("LoRA: cannot reserve temporary directory");
  }
  ~TemporaryDirectory() {
    // Only ever remove the uniquely created directory owned by this object.
    std::error_code error;
    fs::remove_all(path, error);
  }
};

uint64_t header_size(const SafeTensors& file) {
  uint64_t size = 0;
  std::memcpy(&size, file.mapping_base(), sizeof(size));
  return size; // SafeTensors::open already validated the mapped range.
}
Sha256Digest header_digest(const SafeTensors& file) {
  return sha256_bytes(file.mapping_base(), size_t(header_size(file)) + 8);
}

void download_grid(const fs::path& destination) {
#ifdef _WIN32
  struct InternetHandle {
    HINTERNET value;
    ~InternetHandle() { if (value) WinHttpCloseHandle(value); }
  };
  // Pin both the revision and bytes: an upstream replacement must not silently
  // change the embedded model data. This is the FL2VA timestep grid, 5.26 MiB.
  constexpr wchar_t resource[] =
      L"/deAPI-ai/minimax-h3-33b-int8/resolve/ee696877efb4553214cb8d920d5617fd3309b910/loras/h3_silu_temb_grid.safetensors";
  constexpr size_t expected_size = 5510600;
  const Sha256Digest expected_hash = {
      0x30,0xeb,0x3c,0x2c,0xc7,0xfb,0x6b,0x47,0x0d,0x97,0x17,0xff,0x84,0x0d,0x35,0x93,
      0x13,0xac,0x27,0xcd,0x64,0xb7,0x05,0xe3,0x2d,0xa1,0xba,0xa1,0x0f,0x72,0xd6,0xa8};
  std::fprintf(stderr, "LoRA: downloading FL2VA timestep grid for embedding (5.26 MiB)\n");
  InternetHandle session{WinHttpOpen(L"slopfab/0.1", WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY,
      WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0)};
  if (!session.value) throw std::runtime_error("LoRA: cannot initialize grid download");
  WinHttpSetTimeouts(session.value, 30000, 30000, 30000, 60000);
  InternetHandle connection{WinHttpConnect(session.value, L"huggingface.co", INTERNET_DEFAULT_HTTPS_PORT, 0)};
  if (!connection.value) throw std::runtime_error("LoRA: cannot connect for grid download");
  InternetHandle request{WinHttpOpenRequest(connection.value, L"GET", resource, nullptr,
      WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES, WINHTTP_FLAG_SECURE)};
  if (!request.value || !WinHttpSendRequest(request.value, WINHTTP_NO_ADDITIONAL_HEADERS, 0,
      WINHTTP_NO_REQUEST_DATA, 0, 0, 0) || !WinHttpReceiveResponse(request.value, nullptr))
    throw std::runtime_error("LoRA: grid download failed; for offline first use, place h3_silu_temb_grid.safetensors beside the LoRA");
  DWORD status = 0, size = sizeof(status);
  if (!WinHttpQueryHeaders(request.value, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
      WINHTTP_HEADER_NAME_BY_INDEX, &status, &size, WINHTTP_NO_HEADER_INDEX) || status != 200)
    throw std::runtime_error("LoRA: grid download returned HTTP " + std::to_string(status));
  std::vector<uint8_t> data(expected_size);
  size_t received = 0;
  for (;;) {
    uint8_t buffer[65536];
    DWORD count = 0;
    if (!WinHttpReadData(request.value, buffer, sizeof(buffer), &count))
      throw std::runtime_error("LoRA: grid download interrupted");
    if (!count) break;
    if (count > expected_size - received) throw std::runtime_error("LoRA: downloaded grid exceeds expected size");
    std::memcpy(data.data() + received, buffer, count);
    received += count;
  }
  if (received != expected_size || sha256_bytes(data.data(), data.size()) != expected_hash)
    throw std::runtime_error("LoRA: downloaded grid failed size/SHA-256 verification");
  std::ofstream out(destination, std::ios::binary);
  out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
  out.close();
  if (!out) throw std::runtime_error("LoRA: cannot stage downloaded grid");
#else
  (void)destination;
  throw std::runtime_error("LoRA: automatic grid download requires Windows; place h3_silu_temb_grid.safetensors beside the LoRA for first use");
#endif
}
}  // namespace

void LoraGrid::load(const SafeTensors& adapter, int width, bool allow_download) {
  std::string filename = "h3_silu_temb_grid.safetensors";
  std::string tensor_name = "silu_t_emb_grid";
  bool custom_asset = false;
  if (const auto it = adapter.metadata().find("slopfab.lora_grid"); it != adapter.metadata().end()) {
    custom_asset = true;
    const auto root = json::parse(it->second);
    if (!root.is_object()) throw std::runtime_error("LoRA: slopfab.lora_grid must be an object");
    const std::set<std::string> keys = {"version", "identity", "file", "tensor", "rows", "width"};
    for (const auto& item : root.as_object())
      if (!keys.count(item.first)) throw std::runtime_error("LoRA: unknown grid setting " + item.first);
    const auto* version = root.find("version");
    if (!version || version->as_number() != 1) throw std::runtime_error("LoRA: unsupported grid schema version");
    if (const auto* v = root.find("identity")) identity = v->as_string();
    if (const auto* v = root.find("file")) filename = v->as_string();
    if (const auto* v = root.find("tensor")) tensor_name = v->as_string();
    if (const auto* v = root.find("rows"); v && v->as_number() != 1025)
      throw std::runtime_error("LoRA: unsupported AdaLN grid row count");
    if (const auto* v = root.find("width"); v && v->as_number() != width)
      throw std::runtime_error("LoRA: AdaLN grid width does not match adapter");
    const fs::path relative = fs::u8path(filename);
    if (filename.empty() || tensor_name.empty() || relative.is_absolute() || relative.has_root_name())
      throw std::runtime_error("LoRA: grid asset must name a relative companion file and nonempty tensor");
    for (const auto& part : relative)
      if (part == "..") throw std::runtime_error("LoRA: grid asset cannot escape adapter directory");
  }
  auto copy = [&](const TensorView& source) {
    if (source.shape != std::vector<int64_t>{1025, width} ||
        (source.dtype != DType::kF32 && source.dtype != DType::kBF16 && source.dtype != DType::kF16))
      throw std::runtime_error("LoRA: embedded/companion AdaLN grid has invalid shape or dtype");
    tensor = source;
    const auto* begin = static_cast<const uint8_t*>(source.data);
    bytes.assign(begin, begin + source.nbytes);
    tensor.data = bytes.data();
  };
  if (const auto* embedded = adapter.find(kLoraGridTensor)) {
    copy(*embedded);
    return;
  }
  adapter_path = adapter.path();
  const auto path = fs::u8path(adapter_path);
  original_size = adapter.file_size();
  original_time = fs::last_write_time(path);
  original_header = header_digest(adapter);
  const auto companion = path.parent_path() / fs::u8path(filename);
  if (fs::is_regular_file(companion)) {
    SafeTensors archive; archive.open(companion.u8string());
    copy(archive.at(tensor_name));
  } else {
    if (!allow_download || custom_asset || width != 2688)
      throw std::runtime_error("LoRA: no matching embedded AdaLN grid; place h3_silu_temb_grid.safetensors beside the LoRA for first use: " + companion.u8string());
    TemporaryDirectory temporary(fs::temp_directory_path());
    const auto downloaded = temporary.path / "grid.safetensors";
    download_grid(downloaded);
    SafeTensors archive; archive.open(downloaded.u8string());
    copy(archive.at(tensor_name));
  }
  needs_embedding = true;
}

void LoraGrid::embed() const {
  if (!needs_embedding) return;
  const auto path = fs::absolute(fs::u8path(adapter_path));
  SafeTensors original; original.open(path.u8string());
  // Repeated adapters in a stack only need one write.
  if (const auto* present = original.find(kLoraGridTensor)) {
    if (present->shape != tensor.shape || present->dtype != tensor.dtype ||
        present->nbytes != bytes.size() || std::memcmp(present->data, bytes.data(), bytes.size()))
      throw std::runtime_error("LoRA: embedded grid changed during loading");
    return;
  }
  if (original.file_size() != original_size || fs::last_write_time(path) != original_time ||
      header_digest(original) != original_header)
    throw std::runtime_error("LoRA: adapter changed during loading; retry before embedding its grid");
  TemporaryDirectory temporary(path.parent_path()); // Same volume for atomic replacement.
  const auto staged = temporary.path / "adapter.safetensors";
  const size_t old_header = size_t(header_size(original));
  const auto* source = static_cast<const char*>(original.mapping_base());
  const size_t payload_size = original.file_size() - 8 - old_header;
  std::string header(source + 8, old_header);
  const auto end = header.find_last_not_of(" \t\r\n");
  if (end == std::string::npos || header[end] != '}') throw std::runtime_error("LoRA: invalid archive header");
  header.resize(end);
  header += ",\"" + std::string(kLoraGridTensor) + "\":{\"dtype\":\"" + dtype_name(tensor.dtype) +
      "\",\"shape\":[1025," + std::to_string(tensor.shape[1]) + "],\"data_offsets\":[" +
      std::to_string(payload_size) + "," + std::to_string(payload_size + bytes.size()) + "]}}";
  while (header.size() % 8) header += ' ';
  std::ofstream out(staged, std::ios::binary);
  const uint64_t new_header = header.size();
  out.write(reinterpret_cast<const char*>(&new_header), sizeof(new_header));
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  // Preserve all existing storage bytes, including dtype-specific bit patterns.
  for (size_t offset = 0; offset < payload_size;) {
    const size_t count = std::min<size_t>(1 << 20, payload_size - offset);
    out.write(source + 8 + old_header + offset, static_cast<std::streamsize>(count));
    if (!out) throw std::runtime_error("LoRA: cannot write embedded adapter; original left unchanged");
    offset += count;
  }
  out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  out.close();
  if (!out) throw std::runtime_error("LoRA: cannot finish embedded adapter; original left unchanged");
  {
    SafeTensors check; check.open(staged.u8string());
    if (check.tensor_count() != original.tensor_count() + 1 ||
        check.at(kLoraGridTensor).nbytes != bytes.size())
      throw std::runtime_error("LoRA: embedded adapter validation failed");
  }
  original.close(); // Windows denies replacement while a mapping is open.
#ifdef _WIN32
  if (!ReplaceFileW(path.c_str(), staged.c_str(), nullptr, 0, nullptr, nullptr))
    throw std::runtime_error("LoRA: cannot replace adapter while embedding grid (Windows error " +
                            std::to_string(GetLastError()) + "); close other readers and check write permissions");
#else
  fs::permissions(staged, fs::status(path).permissions());
  fs::rename(staged, path);
#endif
  std::fprintf(stderr, "LoRA: embedded timestep grid in %s; future loads need no companion file\n", adapter_path.c_str());
}
}  // namespace slopfab::detail

namespace slopfab {
void prepare_lora_grid(const std::string& adapter_path, int width, bool allow_download) {
  static std::mutex preparation;
  const std::lock_guard<std::mutex> lock(preparation);
  detail::LoraGrid grid;
  {
    SafeTensors adapter;
    adapter.open(adapter_path);
    grid.load(adapter, width, allow_download);
  }
  grid.embed();
}
}
