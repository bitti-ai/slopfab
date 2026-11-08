#pragma once
#include <string>
#include <vector>

namespace slopfab::cli {
int update_command(const std::vector<std::string>& args);
int update_worker(const std::string& executable, const std::string& parent_pid);
}
