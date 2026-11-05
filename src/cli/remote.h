#pragma once
#include "agent.h"

namespace slopfab::cli {
int remote_command(const std::vector<std::string>& args);
// Internal SSH entry points run before stdout is redirected to the agent protocol.
int remote_worker(const std::string& executable, bool fetch);
}
