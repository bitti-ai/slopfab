#include "slopfab/safetensors.h"

#include <cstdlib>
#include <stdexcept>
#include <utility>

#include "slopfab/json.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace slopfab {
namespace {

// Header length prefix plus a sanity bound. Real headers run ~100 KB; anything
// beyond this indicates a corrupt or hostile file.
constexpr size_t kPrefixBytes = 8;
constexpr uint64_t kMaxHeaderBytes = 256ull << 20;

[[noreturn]] void fail(const std::string& path, const std::string& what) {
  throw std::runtime_error("safetensors: " + path + ": " + what);
}

// `std::getenv` is C4996 under /W4 on MSVC and this file is built with it.
bool env_flag(const char* name) {
#ifdef _MSC_VER
  size_t len = 0;
  char buf[8] = {};
  if (getenv_s(&len, buf, sizeof(buf), name) != 0)
    return false;
  return len != 0 && buf[0] == '1';
#else
  const char* v = std::getenv(name);
  return v != nullptr && v[0] == '1';
#endif
}

} // namespace

DType dtype_from_string(std::string_view name) {
  if (name == "BOOL")
    return DType::kBool;
  if (name == "U8")
    return DType::kU8;
  if (name == "I8")
    return DType::kI8;
  if (name == "I16")
    return DType::kI16;
  if (name == "I32")
    return DType::kI32;
  if (name == "I64")
    return DType::kI64;
  if (name == "F8_E4M3")
    return DType::kF8E4M3;
  if (name == "F8_E5M2")
    return DType::kF8E5M2;
  if (name == "F16")
    return DType::kF16;
  if (name == "BF16")
    return DType::kBF16;
  if (name == "F32")
    return DType::kF32;
  if (name == "F64")
    return DType::kF64;
  return DType::kUnknown;
}

const char* dtype_name(DType dt) {
  switch (dt) {
  case DType::kBool:
    return "BOOL";
  case DType::kU8:
    return "U8";
  case DType::kI8:
    return "I8";
  case DType::kI16:
    return "I16";
  case DType::kI32:
    return "I32";
  case DType::kI64:
    return "I64";
  case DType::kF8E4M3:
    return "F8_E4M3";
  case DType::kF8E5M2:
    return "F8_E5M2";
  case DType::kF16:
    return "F16";
  case DType::kBF16:
    return "BF16";
  case DType::kF32:
    return "F32";
  case DType::kF64:
    return "F64";
  case DType::kUnknown:
    break;
  }
  return "UNKNOWN";
}

size_t dtype_size(DType dt) {
  switch (dt) {
  case DType::kBool:
  case DType::kU8:
  case DType::kI8:
  case DType::kF8E4M3:
  case DType::kF8E5M2:
    return 1;
  case DType::kI16:
  case DType::kF16:
  case DType::kBF16:
    return 2;
  case DType::kI32:
  case DType::kF32:
    return 4;
  case DType::kI64:
  case DType::kF64:
    return 8;
  case DType::kUnknown:
    break;
  }
  return 0;
}

SafeTensors::~SafeTensors() {
  close();
}

SafeTensors::SafeTensors(SafeTensors&& other) noexcept {
  *this = std::move(other);
}

SafeTensors& SafeTensors::operator=(SafeTensors&& other) noexcept {
  if (this == &other)
    return *this;
  close();
  path_ = std::move(other.path_);
  base_ = other.base_;
  size_ = other.size_;
  tensors_ = std::move(other.tensors_);
  metadata_ = std::move(other.metadata_);
  lookup_prefix_ = std::move(other.lookup_prefix_);
  other.base_ = nullptr;
  other.size_ = 0;
#ifdef _WIN32
  file_handle_ = other.file_handle_;
  mapping_handle_ = other.mapping_handle_;
  other.file_handle_ = nullptr;
  other.mapping_handle_ = nullptr;
#else
  fd_ = other.fd_;
  other.fd_ = -1;
#endif
  return *this;
}

void SafeTensors::open(const std::string& path) {
  close();
  path_ = path;

#ifdef _WIN32
  // Widen via the ANSI->UTF16 path so non-ASCII paths work.
  const int wide_len = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
  if (wide_len <= 0)
    fail(path, "path is not valid UTF-8");
  std::wstring wide(static_cast<size_t>(wide_len), L'\0');
  MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wide.data(), wide_len);

  HANDLE file = CreateFileW(wide.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                            FILE_ATTRIBUTE_NORMAL, nullptr);
  if (file == INVALID_HANDLE_VALUE)
    fail(path, "cannot open file");
  file_handle_ = file;

  LARGE_INTEGER file_size{};
  if (!GetFileSizeEx(file, &file_size)) {
    close();
    fail(path, "cannot determine file size");
  }
  size_ = static_cast<size_t>(file_size.QuadPart);
  if (size_ < kPrefixBytes) {
    close();
    fail(path, "file is too small to be a safetensors archive");
  }

  HANDLE mapping = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (mapping == nullptr) {
    close();
    fail(path, "cannot create file mapping");
  }
  mapping_handle_ = mapping;

  base_ = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
  if (base_ == nullptr) {
    close();
    fail(path, "cannot map file into memory");
  }
#else
  fd_ = ::open(path.c_str(), O_RDONLY);
  if (fd_ < 0)
    fail(path, "cannot open file");

  struct stat st{};
  if (fstat(fd_, &st) != 0) {
    close();
    fail(path, "cannot stat file");
  }
  size_ = static_cast<size_t>(st.st_size);
  if (size_ < kPrefixBytes) {
    close();
    fail(path, "file is too small to be a safetensors archive");
  }

  void* mapped = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
  if (mapped == MAP_FAILED) {
    close();
    fail(path, "cannot map file into memory");
  }
  base_ = mapped;
#endif

  try {
    parse_header();
  } catch (...) {
    close();
    throw;
  }
}

void SafeTensors::close() {
  tensors_.clear();
  metadata_.clear();
  lookup_prefix_.clear();
#ifdef _WIN32
  if (base_ != nullptr)
    UnmapViewOfFile(base_);
  if (mapping_handle_ != nullptr)
    CloseHandle(static_cast<HANDLE>(mapping_handle_));
  if (file_handle_ != nullptr && file_handle_ != INVALID_HANDLE_VALUE) {
    CloseHandle(static_cast<HANDLE>(file_handle_));
  }
  mapping_handle_ = nullptr;
  file_handle_ = nullptr;
#else
  if (base_ != nullptr)
    munmap(base_, size_);
  if (fd_ >= 0)
    ::close(fd_);
  fd_ = -1;
#endif
  base_ = nullptr;
  size_ = 0;
}

bool SafeTensors::prefetch() const {
  return prefetch_range(base_, size_);
}

bool SafeTensors::prefetch_range(const void* begin, size_t bytes) const {
  if (base_ == nullptr || size_ == 0 || begin == nullptr || bytes == 0)
    return false;
  // Clamp to the mapping. A caller derives `begin` from a `TensorView`, so it
  // is inside by construction, but a hint that walks off the end of the view is
  // the one bug this cannot afford to have.
  const auto b = reinterpret_cast<uintptr_t>(base_);
  const auto q = reinterpret_cast<uintptr_t>(begin);
  if (q < b || q > b + size_)
    return false;
  const size_t avail = static_cast<size_t>(b + size_ - q);
  if (avail == 0)
    return false;
  if (bytes > avail)
    bytes = avail;

  // Exists so the same binary can be run both ways. Proving that a readahead
  // hint left the weight arena bit-identical needs an A/B, and an A/B across
  // two builds proves less than one across two runs of one build. Same shape
  // as SLOPFAB_NATIVE_NVFP4 in transformer.cpp.
  if (env_flag("SLOPFAB_NO_PREFETCH"))
    return false;
#ifdef _WIN32
  // PrefetchVirtualMemory is Windows 8+. Resolved at run time rather than
  // link time so a build that runs on something older degrades to demand
  // faulting instead of failing to start.
  using Fn = BOOL(WINAPI*)(HANDLE, ULONG_PTR, PWIN32_MEMORY_RANGE_ENTRY, ULONG);
  static const Fn prefetch_fn = [] {
    HMODULE k32 = GetModuleHandleW(L"kernel32.dll");
    return k32 == nullptr ? nullptr
                          : reinterpret_cast<Fn>(GetProcAddress(k32, "PrefetchVirtualMemory"));
  }();
  if (prefetch_fn == nullptr)
    return false;
  WIN32_MEMORY_RANGE_ENTRY range;
  range.VirtualAddress = const_cast<void*>(begin);
  range.NumberOfBytes = bytes;
  return prefetch_fn(GetCurrentProcess(), 1, &range, 0) != FALSE;
#else
  // POSIX spells the same hint MADV_WILLNEED. Same contract: advisory, and the
  // mapping is correct whether or not the kernel acts on it. `madvise` wants a
  // page-aligned start, so round down; the extra bytes are inside the mapping.
  const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  const uintptr_t aligned = q - (q % page);
  return madvise(reinterpret_cast<void*>(aligned), bytes + (q - aligned), MADV_WILLNEED) == 0;
#endif
}

void SafeTensors::prefix_extent(std::string_view prefix, const void** begin, size_t* bytes) const {
  if (begin != nullptr)
    *begin = nullptr;
  if (bytes != nullptr)
    *bytes = 0;
  if (base_ == nullptr)
    return;

  // `tensors_` is keyed by name, and name order is not offset order in general,
  // so this takes a min/max over the matching subset rather than trusting the
  // first and last key. The map is ordered, so the matching names are one
  // contiguous run of keys and the scan stops at the first key past the prefix.
  const uint8_t* lo = nullptr;
  const uint8_t* hi = nullptr;
  for (auto it = tensors_.lower_bound(std::string(prefix)); it != tensors_.end(); ++it) {
    if (it->first.compare(0, prefix.size(), prefix) != 0)
      break;
    const TensorView& v = it->second;
    if (v.data == nullptr)
      continue;
    const auto* p = static_cast<const uint8_t*>(v.data);
    if (lo == nullptr || p < lo)
      lo = p;
    if (hi == nullptr || p + v.nbytes > hi)
      hi = p + v.nbytes;
  }
  if (lo == nullptr || hi == nullptr || hi <= lo)
    return;
  if (begin != nullptr)
    *begin = lo;
  if (bytes != nullptr)
    *bytes = static_cast<size_t>(hi - lo);
}

void SafeTensors::parse_header() {
  const auto* bytes = static_cast<const uint8_t*>(base_);

  uint64_t header_len = 0;
  std::memcpy(&header_len, bytes, kPrefixBytes); // little-endian by spec
  if (header_len == 0)
    fail(path_, "header length is zero");
  if (header_len > kMaxHeaderBytes)
    fail(path_, "header length is implausibly large");
  if (header_len > size_ - kPrefixBytes)
    fail(path_, "header length exceeds file size");

  const std::string_view header_text(reinterpret_cast<const char*>(bytes + kPrefixBytes),
                                     static_cast<size_t>(header_len));
  const json::Value root = json::parse(header_text);
  if (!root.is_object())
    fail(path_, "header is not a JSON object");

  const size_t data_start = kPrefixBytes + static_cast<size_t>(header_len);
  const size_t data_bytes = size_ - data_start;

  for (const auto& [name, entry] : root.as_object()) {
    if (name == "__metadata__") {
      if (!entry.is_object())
        continue;
      for (const auto& [k, v] : entry.as_object()) {
        if (v.is_string())
          metadata_.emplace(k, v.as_string());
      }
      continue;
    }

    if (!entry.is_object())
      fail(path_, "tensor entry '" + name + "' is not an object");

    const json::Value* dtype_v = entry.find("dtype");
    const json::Value* shape_v = entry.find("shape");
    const json::Value* offsets_v = entry.find("data_offsets");
    if (dtype_v == nullptr || shape_v == nullptr || offsets_v == nullptr) {
      fail(path_, "tensor '" + name + "' is missing dtype, shape or data_offsets");
    }

    TensorView view;
    view.name = name;
    view.dtype = dtype_from_string(dtype_v->as_string());
    if (view.dtype == DType::kUnknown) {
      fail(path_, "tensor '" + name + "' has unsupported dtype " + dtype_v->as_string());
    }

    for (const json::Value& dim : shape_v->as_array()) {
      const int64_t d = dim.as_int();
      if (d < 0)
        fail(path_, "tensor '" + name + "' has a negative dimension");
      view.shape.push_back(d);
    }

    const json::Array& offsets = offsets_v->as_array();
    if (offsets.size() != 2) {
      fail(path_, "tensor '" + name + "' has malformed data_offsets");
    }
    const int64_t begin = offsets[0].as_int();
    const int64_t end = offsets[1].as_int();
    if (begin < 0 || end < begin) {
      fail(path_, "tensor '" + name + "' has an inverted byte range");
    }
    const auto span = static_cast<size_t>(end - begin);
    if (static_cast<size_t>(end) > data_bytes) {
      fail(path_, "tensor '" + name + "' extends past the end of the file");
    }

    // Cross-check the declared range against shape * element size. A mismatch
    // means we would misread the tensor, so it is fatal rather than a warning.
    const size_t expected = static_cast<size_t>(view.numel()) * dtype_size(view.dtype);
    if (expected != span) {
      fail(path_, "tensor '" + name + "' declares " + std::to_string(span) +
                      " bytes but its shape implies " + std::to_string(expected));
    }

    view.data = bytes + data_start + static_cast<size_t>(begin);
    view.nbytes = span;
    tensors_.emplace(name, std::move(view));
  }

  // Some ComfyUI releases (including H3 Singularity) wrap the entire state
  // dict. Only alias a uniform namespace: mixed archives must not silently
  // combine tensors from different models. This never copies weight data.
  const std::string prefix = "model.diffusion_model.";
  if (!tensors_.empty()) {
    bool uniform = true;
    for (const auto& item : tensors_) {
      if (item.first.compare(0, prefix.size(), prefix) != 0) {
        uniform = false;
        break;
      }
    }
    if (uniform)
      lookup_prefix_ = prefix;
  }
}

const TensorView* SafeTensors::find(std::string_view name) const {
  auto it = tensors_.find(std::string(name));
  if (it == tensors_.end() && !lookup_prefix_.empty()) {
    it = tensors_.find(lookup_prefix_ + std::string(name));
  }
  return it == tensors_.end() ? nullptr : &it->second;
}

const TensorView& SafeTensors::at(std::string_view name) const {
  const TensorView* v = find(name);
  if (v == nullptr) {
    throw std::runtime_error("safetensors: " + path_ + ": missing tensor '" + std::string(name) +
                             "'");
  }
  return *v;
}

} // namespace slopfab
