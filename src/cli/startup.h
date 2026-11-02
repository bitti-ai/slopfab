#pragma once
#include <string>

namespace slopfab::cli {
struct StartupDependencies {
  bool nvidia = false;
  bool cuda = false;
  bool ffmpeg = false;
};

StartupDependencies startup_dependencies(const std::string& executable);
const char* missing_startup_dependency(const StartupDependencies& dependencies);
std::string startup_instructions(const StartupDependencies& dependencies);
}
