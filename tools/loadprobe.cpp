// Disk-side load probe: what a checkpoint's read pattern costs, repeatably.
//
// The problem this exists for is that "cold load" on Windows is not a
// measurement, it is a lottery. The same 12.5 GB transformer checkpoint timed
// 22.6 s and 60.7 s on two consecutive fresh runs — a 2.7x spread — because the
// only variable that matters, how much of the file the cache manager still
// holds, is invisible and uncontrolled. Optimising against that is optimising
// against noise.
//
// So this tool never asks the cache a question it cannot answer. Its primary
// mode opens the file with FILE_FLAG_NO_BUFFERING, which bypasses the cache
// entirely and therefore always measures the drive. That is repeatable to a
// few percent, needs no eviction, no privilege and no guessing, and it is what
// every number here is anchored to. The buffered and mapped modes are still
// available, and are still a lottery; they are reported with their spread and
// treated as corroboration rather than evidence.
//
// The unit of measurement is the extent list: the (offset, length) pairs of a
// checkpoint's tensors, replayed in a chosen order. Replaying the same bytes in
// name order and in file order, through the same code, on the same drive, in
// the same session isolates access order from everything else.
//
// Host-only. Links slopfab_core and touches no CUDA, so it can run while the
// card is busy.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "slopfab/safetensors.h"

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

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

struct Extent {
  std::string name;
  uint64_t begin = 0; // absolute file offset
  uint64_t bytes = 0;
};

// The extents a checkpoint's tensors occupy, in *name* order — which is the
// order `std::map<std::string, ...>` hands them to every loader in this
// project, and therefore the order the bytes are actually demanded in.
std::vector<Extent> extents_in_name_order(const slopfab::SafeTensors& st) {
  std::vector<Extent> out;
  out.reserve(st.tensor_count());
  const auto* base = static_cast<const uint8_t*>(st.mapping_base());
  for (const auto& kv : st.tensors()) {
    Extent e;
    e.name = kv.first;
    e.begin = static_cast<uint64_t>(static_cast<const uint8_t*>(kv.second.data) - base);
    e.bytes = kv.second.nbytes;
    out.push_back(std::move(e));
  }
  return out; // std::map already iterates in name order
}

std::vector<Extent> sorted_by_offset(std::vector<Extent> v) {
  std::sort(v.begin(), v.end(), [](const Extent& a, const Extent& b) {
    return a.begin < b.begin;
  });
  return v;
}

// --- ordering report --------------------------------------------------------

void report_order(const std::vector<Extent>& name_order, uint64_t file_size) {
  uint64_t total = 0;
  for (const Extent& e : name_order)
    total += e.bytes;

  const std::vector<Extent> file_order = sorted_by_offset(name_order);

  // Contiguity of the data section, which is what makes "file order" a
  // meaningful target: if the tensors did not tile the file there would be
  // nothing to sort into.
  size_t holes = 0;
  uint64_t prev_end = file_order.empty() ? 0 : file_order.front().begin;
  for (const Extent& e : file_order) {
    if (e.begin != prev_end)
      ++holes;
    prev_end = e.begin + e.bytes;
  }

  size_t seq = 0, fwd = 0, back = 0;
  uint64_t back_distance = 0, fwd_distance = 0;
  bool first = true;
  uint64_t cursor = 0;
  for (const Extent& e : name_order) {
    if (!first) {
      if (e.begin < cursor) {
        ++back;
        back_distance += cursor - e.begin;
      } else if (e.begin > cursor) {
        ++fwd;
        fwd_distance += e.begin - cursor;
      } else {
        ++seq;
      }
    }
    first = false;
    cursor = e.begin + e.bytes;
  }

  std::printf("tensors            %zu\n", name_order.size());
  std::printf("file size          %.3f GB (%llu bytes)\n", file_size / 1e9,
              static_cast<unsigned long long>(file_size));
  std::printf("tensor bytes       %.3f GB\n", total / 1e9);
  std::printf("holes in data      %zu (0 means the tensors tile the file)\n",
              holes ? holes - 1 : 0);
  std::printf("\nname-order traversal, which is what the loaders do:\n");
  std::printf("  continuations    %zu   (next tensor begins exactly where the last ended)\n", seq);
  std::printf("  forward skips    %zu   over %.3f GB\n", fwd, fwd_distance / 1e9);
  std::printf("  BACKWARD seeks   %zu   over %.3f GB cumulative\n", back, back_distance / 1e9);
  std::printf("  a sequential traversal of this file would have 0 of each.\n");
}

// --- read engines -----------------------------------------------------------

#ifdef _WIN32

std::wstring widen(const std::string& s) {
  const int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
  std::wstring w(static_cast<size_t>(n), L'\0');
  MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, w.data(), n);
  return w;
}

uint32_t sector_size_for(const std::string& path) {
  // "D:\Projects\..." -> "D:\". A UNC or relative path falls back to 4096,
  // which is a safe over-alignment on every drive this runs on.
  if (path.size() < 3 || path[1] != ':')
    return 4096;
  const std::wstring root = widen(path.substr(0, 3));
  DWORD spc = 0, bps = 0, freec = 0, totalc = 0;
  if (!GetDiskFreeSpaceW(root.c_str(), &spc, &bps, &freec, &totalc))
    return 4096;
  return bps == 0 ? 4096u : bps;
}

class AlignedBuffer {
public:
  void allocate(size_t bytes) {
    free();
    p_ = VirtualAlloc(nullptr, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    size_ = p_ ? bytes : 0;
  }

  ~AlignedBuffer() {
    free();
  }

  void free() {
    if (p_)
      VirtualFree(p_, 0, MEM_RELEASE);
    p_ = nullptr;
    size_ = 0;
  }

  uint8_t* get() const {
    return static_cast<uint8_t*>(p_);
  }

  size_t size() const {
    return size_;
  }

private:
  void* p_ = nullptr;
  size_t size_ = 0;
};

HANDLE open_read(const std::string& path, DWORD flags) {
  return CreateFileW(widen(path).c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                     flags, nullptr);
}

// Reads [offset, offset+bytes) into `dst`. `dst` and, when unbuffered, the
// offset and length must already satisfy the caller's alignment contract.
bool read_at(HANDLE h, uint64_t offset, uint8_t* dst, uint64_t bytes) {
  while (bytes > 0) {
    const DWORD chunk = static_cast<DWORD>(std::min<uint64_t>(bytes, 32u << 20));
    OVERLAPPED ov{};
    ov.Offset = static_cast<DWORD>(offset & 0xFFFFFFFFull);
    ov.OffsetHigh = static_cast<DWORD>(offset >> 32);
    DWORD got = 0;
    if (!ReadFile(h, dst, chunk, &got, &ov))
      return false;
    if (got == 0)
      return false; // past EOF
    offset += got;
    dst += got;
    bytes -= got;
  }
  return true;
}

// Replays an extent list. `unbuffered` bypasses the cache, which is the whole
// point: it measures the drive rather than whatever the cache manager happened
// to be holding. Each extent is widened to sector boundaries, so a run reads
// slightly more than the tensor bytes; that overhead is reported.
struct ReplayResult {
  double seconds = 0.0;
  uint64_t bytes_read = 0;
  uint64_t checksum = 0;
};

ReplayResult replay(const std::string& path, const std::vector<Extent>& order, bool unbuffered,
                    bool sequential_hint, bool checksum) {
  const uint32_t sector = unbuffered ? sector_size_for(path) : 1;
  DWORD flags = FILE_ATTRIBUTE_NORMAL;
  if (unbuffered)
    flags |= FILE_FLAG_NO_BUFFERING;
  if (sequential_hint)
    flags |= FILE_FLAG_SEQUENTIAL_SCAN;

  HANDLE h = open_read(path, flags);
  if (h == INVALID_HANDLE_VALUE) {
    std::fprintf(stderr, "loadprobe: cannot open %s (%lu)\n", path.c_str(), GetLastError());
    return {};
  }

  AlignedBuffer buf;
  buf.allocate(64u << 20);
  ReplayResult r;
  const auto t0 = Clock::now();
  for (const Extent& e : order) {
    uint64_t begin = e.begin;
    uint64_t end = e.begin + e.bytes;
    if (unbuffered) {
      begin -= begin % sector;
      end = ((end + sector - 1) / sector) * sector;
    }
    uint64_t left = end - begin;
    uint64_t at = begin;
    while (left > 0) {
      uint64_t n = std::min<uint64_t>(left, buf.size());
      if (unbuffered)
        n -= n % sector;
      if (n == 0)
        break;
      if (!read_at(h, at, buf.get(), n)) {
        // A sector-rounded tail can run past EOF; that is not an error.
        break;
      }
      if (checksum) {
        uint64_t hsh = r.checksum;
        const uint64_t* w = reinterpret_cast<const uint64_t*>(buf.get());
        for (uint64_t i = 0; i < n / 8; ++i)
          hsh = (hsh ^ w[i]) * 1099511628211ull;
        r.checksum = hsh;
      }
      r.bytes_read += n;
      at += n;
      left -= n;
    }
  }
  r.seconds = seconds_since(t0);
  CloseHandle(h);
  return r;
}

// A whole-file sequential read in fixed chunks. This is the control: the same
// drive, the same session, the same engine, with the access order removed.
ReplayResult sequential(const std::string& path, uint64_t file_size, size_t chunk_bytes,
                        bool unbuffered) {
  std::vector<Extent> one;
  for (uint64_t at = 0; at < file_size; at += chunk_bytes) {
    Extent e;
    e.begin = at;
    e.bytes = std::min<uint64_t>(chunk_bytes, file_size - at);
    one.push_back(e);
  }
  return replay(path, one, unbuffered, /*sequential_hint=*/!unbuffered, /*checksum=*/false);
}

// Touches an extent list through a memory mapping, which is exactly what the
// loaders do: `Uploader::copy` memcpys out of the mapping and every soft page
// fault on the way is charged to the load. Buffered by construction.
ReplayResult map_touch(const std::string& path, const std::vector<Extent>& order, bool prefetch) {
  HANDLE h = open_read(path, FILE_ATTRIBUTE_NORMAL);
  if (h == INVALID_HANDLE_VALUE)
    return {};
  LARGE_INTEGER fs{};
  GetFileSizeEx(h, &fs);
  HANDLE m = CreateFileMappingW(h, nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (m == nullptr) {
    CloseHandle(h);
    return {};
  }
  auto* base = static_cast<const uint8_t*>(MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0));
  if (base == nullptr) {
    CloseHandle(m);
    CloseHandle(h);
    return {};
  }

  ReplayResult r;
  const auto t0 = Clock::now();
  if (prefetch) {
    // One asynchronous whole-file read instead of a demand-fault storm. A hint:
    // it can fail or be ignored, and the traversal below is correct either way.
    WIN32_MEMORY_RANGE_ENTRY range;
    range.VirtualAddress = const_cast<uint8_t*>(base);
    range.NumberOfBytes = static_cast<SIZE_T>(fs.QuadPart);
    PrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0);
  }
  std::vector<uint8_t> stage(32u << 20);
  for (const Extent& e : order) {
    uint64_t left = e.bytes;
    const uint8_t* src = base + e.begin;
    while (left > 0) {
      const size_t n = static_cast<size_t>(std::min<uint64_t>(left, stage.size()));
      std::memcpy(stage.data(), src, n);
      r.checksum += stage[0] + stage[n - 1]; // keep the memcpy alive
      r.bytes_read += n;
      src += n;
      left -= n;
    }
  }
  r.seconds = seconds_since(t0);
  UnmapViewOfFile(base);
  CloseHandle(m);
  CloseHandle(h);
  return r;
}

// Drops a file's own cached pages. When a file is opened for non-cached access
// and no other handle holds a cached view of it, the cache manager flushes and
// purges its shared cache map — so a single unbuffered sector read evicts the
// whole file. That is a property of the Cache Manager rather than a documented
// contract, so `probe_cached_gbs` measures the result instead of trusting it;
// `--flood` remains as the fallback that cannot be argued with.
//
// It costs one 4 KB read instead of a 60 GB one, which is the difference
// between five cold samples and one.
bool evict_self(const std::string& path) {
  HANDLE h = open_read(path, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING);
  if (h == INVALID_HANDLE_VALUE)
    return false;
  AlignedBuffer buf;
  buf.allocate(64 << 10);
  const bool ok = read_at(h, 0, buf.get(), 64 << 10);
  CloseHandle(h);
  return ok;
}

// Pushes a file out of the standby list by reading unrelated data through the
// cache until more bytes have passed than the machine has RAM. Crude, but it
// needs no privilege.
//
// Deliberately *without* FILE_FLAG_SEQUENTIAL_SCAN: that flag asks the cache
// manager to retire the pages as soon as they are consumed, which makes the
// flood evict itself and leave the file it was meant to displace untouched.
// The first version of this had the flag and reported a fully cached file
// after reading 62 GB past it.
void flood(const std::vector<std::string>& paths, double gb_target) {
  if (paths.empty())
    return;
  std::vector<uint8_t> buf(8u << 20);
  uint64_t done = 0;
  const uint64_t target = static_cast<uint64_t>(gb_target * 1e9);
  const auto t0 = Clock::now();
  while (done < target) {
    bool any = false;
    for (const std::string& p : paths) {
      HANDLE h = open_read(p, FILE_ATTRIBUTE_NORMAL);
      if (h == INVALID_HANDLE_VALUE)
        continue;
      any = true;
      DWORD got = 0;
      while (ReadFile(h, buf.data(), static_cast<DWORD>(buf.size()), &got, nullptr) && got > 0) {
        done += got;
        if (done >= target)
          break;
      }
      CloseHandle(h);
      if (done >= target)
        break;
    }
    if (!any)
      break;
  }
  std::printf("  flood: read %.1f GB in %.1f s\n", done / 1e9, seconds_since(t0));
}

// Is `path` still in the cache? A buffered read of four scattered slices:
// served from RAM it runs at memory speed, served from the drive it runs at
// drive speed, and the two are an order of magnitude apart, so the answer is
// never ambiguous.
//
// Scattered rather than one slice at the tail, because the tail is whatever the
// last traversal touched most recently and is therefore the one region LRU is
// most likely to have kept. Measuring there says "cached" about a file that is
// 95% evicted. Four slices at 12/37/62/87% cost 512 MB of warmth out of 12.5 GB
// and cannot be gamed by traversal order.
double probe_cached_gbs(const std::string& path, uint64_t file_size, uint64_t slice_bytes) {
  HANDLE h = open_read(path, FILE_ATTRIBUTE_NORMAL);
  if (h == INVALID_HANDLE_VALUE)
    return 0.0;
  std::vector<uint8_t> buf(4u << 20);
  const uint64_t each = slice_bytes / 4;
  const auto t0 = Clock::now();
  uint64_t got_total = 0;
  for (int s = 0; s < 4; ++s) {
    uint64_t begin = file_size / 8 + (file_size / 4) * static_cast<uint64_t>(s);
    if (begin + each > file_size)
      begin = file_size - each;
    for (uint64_t at = begin; at < begin + each;) {
      const uint64_t n = std::min<uint64_t>(buf.size(), begin + each - at);
      if (!read_at(h, at, buf.data(), n))
        break;
      at += n;
      got_total += n;
    }
  }
  const double secs = seconds_since(t0);
  CloseHandle(h);
  return secs > 0 ? got_total / 1e9 / secs : 0.0;
}

#endif // _WIN32

void usage() {
  std::printf(
      "usage: slopfab_loadprobe <file.safetensors> [options]\n"
      "\n"
      "  --order                report name-order vs file-order traversal and exit\n"
      "  --mode <m>             seq | replay | map | evict   (default: the first three)\n"
      "                         evict drops the file's cached pages and exits, so another\n"
      "                         process can be measured cold. Exits non-zero if it failed.\n"
      "  --repeat <n>           samples per configuration (default 3)\n"
      "  --chunk <mb>           sequential control chunk size (default 8)\n"
      "  --buffered             use the cache for replay/seq instead of bypassing it\n"
      "  --cold                 evict the file's own cached pages before each sample\n"
      "                         (one unbuffered open; cheap). Only affects map/--buffered.\n"
      "  --flood <path>         also read this file to evict the cache; repeatable.\n"
      "                         Slower than --cold but relies on nothing subtle.\n"
      "  --flood-gb <n>         how many GB to push through (default 72)\n"
      "  --probe                report how cached the file is before each sample\n"
      "  --checksum             hash the bytes read, to prove two orders read the same\n");
}

} // namespace

int main(int argc, char** argv) {
#ifndef _WIN32
  (void)argc;
  (void)argv;
  std::fprintf(stderr, "loadprobe: Windows only for now\n");
  return 2;
#else
  if (argc < 2) {
    usage();
    return 2;
  }
  const std::string path = argv[1];
  bool order_only = false, buffered = false, checksum = false, cold = false, probe = false;
  std::string mode = "all";
  int repeat = 3;
  size_t chunk_mb = 8;
  double flood_gb = 72.0;
  std::vector<std::string> flood_paths;
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      return i + 1 < argc ? argv[++i] : std::string();
    };
    if (a == "--order")
      order_only = true;
    else if (a == "--buffered")
      buffered = true;
    else if (a == "--cold")
      cold = true;
    else if (a == "--probe")
      probe = true;
    else if (a == "--checksum")
      checksum = true;
    else if (a == "--mode")
      mode = next();
    else if (a == "--repeat")
      repeat = std::atoi(next().c_str());
    else if (a == "--chunk")
      chunk_mb = static_cast<size_t>(std::atoi(next().c_str()));
    else if (a == "--flood")
      flood_paths.push_back(next());
    else if (a == "--flood-gb")
      flood_gb = std::atof(next().c_str());
    else if (a == "--help" || a == "-h") {
      usage();
      return 0;
    } else {
      std::fprintf(stderr, "loadprobe: unknown option %s\n", a.c_str());
      return 2;
    }
  }

  std::vector<Extent> name_order;
  uint64_t file_size = 0;
  try {
    slopfab::SafeTensors st;
    st.open(path);
    file_size = st.file_size();
    name_order = extents_in_name_order(st);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "loadprobe: %s\n", e.what());
    return 1;
  }
  const std::vector<Extent> file_order = sorted_by_offset(name_order);

  std::printf("== %s ==\n", path.c_str());

  // Evict and exit. This is how a *different* process gets a cold cache to
  // measure against — `slopfab generate --bench-load` cannot drop its own file
  // and should not learn how. Reports the before and after rate so the run it
  // precedes can be called cold on evidence rather than on intent.
  if (mode == "evict") {
    std::printf("before: %.2f GB/s\n", probe_cached_gbs(path, file_size, 512ull << 20));
    if (!evict_self(path)) {
      std::fprintf(stderr, "loadprobe: eviction open failed\n");
      return 1;
    }
    const double after = probe_cached_gbs(path, file_size, 512ull << 20);
    evict_self(path); // the probe warmed 512 MB of it; take that back too
    std::printf("after:  %.2f GB/s%s\n", after,
                after < 3.0 ? "   (cold)" : "   (STILL CACHED - do not call the next run cold)");
    return after < 3.0 ? 0 : 1;
  }

  report_order(name_order, file_size);
  if (order_only)
    return 0;

  // Opening the file above faulted the header in; nothing else. Every sample
  // below reports its own number so the spread is visible rather than averaged
  // away — a mean with no spread is not a measurement on this machine.
  auto run = [&](const char* label, auto&& fn) {
    std::vector<double> gbs;
    std::printf("\n%s\n", label);
    for (int s = 0; s < repeat; ++s) {
      if (!flood_paths.empty())
        flood(flood_paths, flood_gb);
      if (cold && !evict_self(path))
        std::printf("  (eviction open failed)\n");
      if (probe) {
        std::printf("  cache probe (4 x 128 MB): %.2f GB/s\n",
                    probe_cached_gbs(path, file_size, 512ull << 20));
        if (cold)
          evict_self(path); // the probe warmed 512 MB; take it back
      }
      const ReplayResult r = fn();
      const double rate = r.bytes_read / 1e9 / r.seconds;
      gbs.push_back(rate);
      std::printf("  sample %d: %7.2f s   %6.3f GB read   %5.2f GB/s", s + 1, r.seconds,
                  r.bytes_read / 1e9, rate);
      if (checksum)
        std::printf("   fnv=%016llx", static_cast<unsigned long long>(r.checksum));
      std::printf("\n");
    }
    const double lo = *std::min_element(gbs.begin(), gbs.end());
    const double hi = *std::max_element(gbs.begin(), gbs.end());
    double sum = 0;
    for (double g : gbs)
      sum += g;
    std::printf("  n=%d  min %.2f  max %.2f  mean %.2f GB/s  spread %.2fx\n", repeat, lo, hi,
                sum / gbs.size(), lo > 0 ? hi / lo : 0.0);
  };

  const bool unbuf = !buffered;
  if (mode == "all" || mode == "seq") {
    run("sequential control (whole file, fixed chunks)", [&] {
      return sequential(path, file_size, chunk_mb << 20, unbuf);
    });
  }
  if (mode == "all" || mode == "replay") {
    run("replay, NAME order (what the loaders do today)", [&] {
      return replay(path, name_order, unbuf, false, checksum);
    });
    run("replay, FILE order (the same bytes, sorted by offset)", [&] {
      return replay(path, file_order, unbuf, false, checksum);
    });
  }
  // All four combinations of {name, file} order x {demand fault, prefetch}. The
  // two variables are separable only if measured separately: prefetch is
  // asynchronous, so a traversal that runs ahead of it in a different order can
  // still fault on every tensor and get none of its benefit.
  if (mode == "all" || mode == "map") {
    run("mapped touch, NAME order (what the loaders do today)", [&] {
      return map_touch(path, name_order, false);
    });
    run("mapped touch, FILE order", [&] {
      return map_touch(path, file_order, false);
    });
    run("mapped touch, NAME order + PrefetchVirtualMemory", [&] {
      return map_touch(path, name_order, true);
    });
    run("mapped touch, FILE order + PrefetchVirtualMemory", [&] {
      return map_touch(path, file_order, true);
    });
  }
  return 0;
#endif
}
