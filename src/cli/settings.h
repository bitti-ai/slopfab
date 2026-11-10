#pragma once
#include "agent.h"

namespace slopfab::cli {
// The existing per-user config is also the settings store. CLI-only: the DLL
// never reads or creates it.
void ensure_settings();
Value load_settings();
// Atomically reserve today's automatic attempt across all CLI processes.
// Explicit checks record their date even when automatic updates are disabled.
bool begin_daily_update(const std::string& utc_date);
void record_update_check(const std::string& utc_date);
std::vector<std::filesystem::path> configured_weight_paths();
int settings_command(const std::vector<std::string>& args);
}
