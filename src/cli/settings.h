#pragma once
#include "agent.h"

namespace slopfab::cli {
// The existing per-user config is also the settings store. CLI-only: the DLL
// never reads or creates it.
void ensure_settings();
Value load_settings();
std::vector<std::filesystem::path> configured_weight_paths();
int settings_command(const std::vector<std::string>& args);
}
