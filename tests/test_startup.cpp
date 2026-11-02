#include "../src/cli/startup.h"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <cstdlib>
#endif

using namespace slopfab::cli;

void require(bool value, const char* message) {
  if (!value)
    throw std::runtime_error(message);
}

int main() {
  const auto root = std::filesystem::temp_directory_path() /
                    ("slopfab-startup-" +
                     std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  try {
    for (bool nvidia : {false, true})
      for (bool cuda : {false, true})
        for (bool ffmpeg : {false, true}) {
          const StartupDependencies state{nvidia, cuda, ffmpeg};
          const auto text = startup_instructions(state);
          const std::string missing = missing_startup_dependency(state);
          if (nvidia && !cuda) {
            require(missing == "cuda", "CUDA must take priority on NVIDIA");
            require(text.find("CUDA") != std::string::npos &&
                        text.find("FFmpeg") == std::string::npos,
                    "missing CUDA must show only CUDA installation guidance");
          } else if (!ffmpeg) {
            require(missing == "ffmpeg", "missing FFmpeg was not reported");
            require(text.find("FFmpeg") != std::string::npos &&
                        text.find("CUDA") == std::string::npos,
                    "missing FFmpeg must show only FFmpeg installation guidance");
          } else {
            require(missing.empty(), "ready installation was blocked");
            require(std::count(text.begin(), text.end(), '\n') == 4,
                    "startup must have four lines");
            const auto second = text.find('\n') + 1;
            const auto third = text.find('\n', second) + 1;
            require(text.substr(second, third - second - 1) ==
                        "When running inside agent, add --agent parameter",
                    "incorrect second line");
            require(text.find("--agent", third) == std::string::npos, "examples repeat --agent");
          }
          if (!missing.empty())
            require(std::count(text.begin(), text.end(), '\n') == 1,
                    "installation guidance must be one line");
        }

    // Probe an isolated executable directory and PATH, then simulate adding and
    // removing an external installation. Nothing is executed or installed.
    std::filesystem::create_directories(root / "bin");
#ifdef _WIN32
    SetEnvironmentVariableW(L"PATH", L"");
    const auto filename = "ffmpeg.exe";
#else
    setenv("PATH", "", 1);
    const auto filename = "ffmpeg";
#endif
    const auto exe = (root / "slopfab.exe").u8string();
    require(!startup_dependencies(exe).ffmpeg, "FFmpeg unexpectedly found outside search paths");
    const auto external = root / "bin" / filename;
    std::ofstream(external).put('x');
#ifndef _WIN32
    std::filesystem::permissions(external, std::filesystem::perms::owner_exec,
                                 std::filesystem::perm_options::add);
#endif
#ifdef _WIN32
    SetEnvironmentVariableW(L"PATH", (root / "bin").c_str());
#else
    setenv("PATH", (root / "bin").c_str(), 1);
#endif
    require(startup_dependencies(exe).ffmpeg, "FFmpeg on PATH was not detected");
    std::filesystem::rename(external, root / filename);
    require(startup_dependencies(exe).ffmpeg, "FFmpeg beside executable was not detected");
    std::filesystem::remove(root / filename);
    require(!startup_dependencies(exe).ffmpeg, "dependency probe cached stale availability");
    std::filesystem::remove_all(root);
    std::cout << "PASS: compact startup, dependency priority, discovery and rechecking\n";
    return 0;
  } catch (const std::exception& e) {
    std::filesystem::remove_all(root);
    std::cerr << e.what() << '\n';
    return 1;
  }
}
