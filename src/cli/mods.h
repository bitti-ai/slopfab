#pragma once
#include <string>
#include <vector>
#include "slopfab/json.h"

namespace slopfab::cli {
int mods_command(const std::vector<std::string>& args);
// No script executes during indexing or help.
json::Array mod_commands();
bool try_mod_command(const std::string& command, const std::vector<std::string>& args, bool help = false);
}
