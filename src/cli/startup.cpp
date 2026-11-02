#include "startup.h"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <cstdlib>
#include <dlfcn.h>
#include <unistd.h>
#endif

namespace slopfab::cli {
namespace {
namespace fs = std::filesystem;

bool regular_file(const fs::path& path) {
  std::error_code error;
  return fs::is_regular_file(path, error);
}

#ifdef _WIN32
std::wstring environment(const wchar_t* name) {
  const DWORD length = GetEnvironmentVariableW(name, nullptr, 0);
  if (!length)
    return {};
  std::wstring value(length, L'\0');
  const DWORD written = GetEnvironmentVariableW(name, value.data(), length);
  if (!written || written >= length)
    return {};
  value.resize(written);
  return value;
}

bool nvidia_present() {
  // Detect hardware independently of CUDA so an absent toolkit/driver cannot
  // hide the very adapter for which installation guidance is needed.
  DISPLAY_DEVICEW device{};
  device.cb = sizeof(device);
  for (DWORD index = 0; EnumDisplayDevicesW(nullptr, index, &device, 0); ++index) {
    if (std::wstring(device.DeviceID).find(L"VEN_10DE") != std::wstring::npos ||
        std::wstring(device.DeviceString).find(L"NVIDIA") != std::wstring::npos)
      return true;
    device = {};
    device.cb = sizeof(device);
  }
  return false;
}

bool cuda_present() {
  // Match the Windows inference loader's supported toolkit locations without
  // loading cuBLAS or creating a CUDA context just to print startup help.
  std::vector<fs::path> roots;
  roots.emplace_back(environment(L"CUDA_PATH"));
  for (int major : {13, 12})
    for (int minor = 9; minor >= 0; --minor) {
      const auto suffix = std::to_wstring(major) + L"_" + std::to_wstring(minor);
      roots.emplace_back(environment((L"CUDA_PATH_V" + suffix).c_str()));
      auto program_files = environment(L"ProgramFiles");
      if (!program_files.empty())
        roots.push_back(fs::path(program_files) / L"NVIDIA GPU Computing Toolkit" / L"CUDA" /
                        (L"v" + std::to_wstring(major) + L"." + std::to_wstring(minor)));
    }
  for (auto root : roots) {
    if (root.empty() || !root.is_absolute())
      continue;
    if (root.filename() == L"x64")
      root = root.parent_path();
    if (root.filename() == L"bin")
      root = root.parent_path();
    for (const auto& bin : {root / L"bin" / L"x64", root / L"bin"})
      for (const char* major : {"13", "12"})
        if (regular_file(bin / (std::string("cublas64_") + major + ".dll")) &&
            regular_file(bin / (std::string("cublasLt64_") + major + ".dll")))
          return true;
  }
  return false;
}

bool ffmpeg_present(const fs::path& executable) {
  if (regular_file(executable.parent_path() / L"ffmpeg.exe"))
    return true;
  std::wistringstream path(environment(L"PATH"));
  std::wstring directory;
  while (std::getline(path, directory, L';')) {
    if (directory.size() >= 2 && directory.front() == L'"' && directory.back() == L'"')
      directory = directory.substr(1, directory.size() - 2);
    if (!directory.empty() && regular_file(fs::path(directory) / L"ffmpeg.exe"))
      return true;
  }
  return false;
}
#else
bool nvidia_present() {
  std::error_code error;
  for (fs::directory_iterator it("/sys/bus/pci/devices", error), end; !error && it != end;
       it.increment(error)) {
    std::string vendor, kind;
    std::ifstream(it->path() / "vendor") >> vendor;
    std::ifstream(it->path() / "class") >> kind;
    if (vendor == "0x10de" && kind.rfind("0x03", 0) == 0)
      return true;
  }
  // WSL and containers may expose the driver without PCI sysfs entries.
  void* driver = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
  if (!driver)
    return false;
  auto init = reinterpret_cast<int (*)(unsigned)>(dlsym(driver, "cuInit"));
  auto count = reinterpret_cast<int (*)(int*)>(dlsym(driver, "cuDeviceGetCount"));
  int devices = 0;
  bool found = init && count && init(0) == 0 && count(&devices) == 0 && devices > 0;
  dlclose(driver);
  return found;
}

bool cuda_present() {
  for (const char* library : {"libcublas.so.13", "libcublas.so.12"}) {
    void* handle = dlopen(library, RTLD_LAZY | RTLD_LOCAL);
    if (handle) {
      dlclose(handle);
      return true;
    }
  }
  return false;
}

bool ffmpeg_present(const fs::path& executable) {
  auto runnable = [](const fs::path& path) {
    return regular_file(path) && access(path.c_str(), X_OK) == 0;
  };
  if (runnable(executable.parent_path() / "ffmpeg"))
    return true;
  const char* value = std::getenv("PATH");
  std::istringstream path(value ? value : "");
  std::string directory;
  while (std::getline(path, directory, ':'))
    if (!directory.empty() && runnable(fs::path(directory) / "ffmpeg"))
      return true;
  return false;
}
#endif
}

StartupDependencies startup_dependencies(const std::string& executable) {
  StartupDependencies result;
  result.nvidia = nvidia_present();
  result.cuda = result.nvidia && cuda_present();
  result.ffmpeg = ffmpeg_present(std::filesystem::u8path(executable));
  return result;
}

const char* missing_startup_dependency(const StartupDependencies& dependencies,
                                       bool has_weight_paths) {
  if (dependencies.nvidia && !dependencies.cuda)
    return "cuda";
  if (!dependencies.ffmpeg)
    return "ffmpeg";
  if (!has_weight_paths)
    return "weight_paths";
  return "";
}

std::string startup_instructions(const StartupDependencies& dependencies, bool has_weight_paths) {
  const std::string missing = missing_startup_dependency(dependencies, has_weight_paths);
  if (missing == "cuda")
    return "Install the NVIDIA CUDA Toolkit, then run slopfab again.\n";
  if (missing == "ffmpeg")
    return "Install FFmpeg and add it to PATH, then run slopfab again.\n";
  if (missing == "weight_paths")
    return "Ask the user for a weight path, then save it with: slopfab settings set weight_paths PATH\n";
  return "slopfab generates videos, images and music.\n"
         "When running inside agent, add --agent parameter\n"
         "Generate: slopfab generate video|image|music --prompt TEXT --out FILE\n"
         "More: slopfab help; slopfab help COMMAND; slopfab schema; slopfab mods list\n";
}
}
