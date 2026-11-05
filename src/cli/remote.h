#pragma once
#include "agent.h"

namespace slopfab::cli {
int remote_command(const std::vector<std::string>& args);
int remote_execute(const std::string& host, const std::vector<std::string>& args,
                   const std::string& destination, const std::string& local_project,
                   const std::vector<std::string>& downloads);
int remote_fetch(const std::vector<std::string>& args);
// Internal SSH entry points run before stdout is redirected to the agent protocol.
int remote_worker(const std::string& executable, bool fetch);
}
