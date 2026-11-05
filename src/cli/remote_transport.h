#pragma once
#include "agent.h"

namespace slopfab::cli {
std::string remote_digest(const std::filesystem::path& path);
std::string remote_shell_quote(const std::string& value);
std::string remote_powershell(const std::string& script);
std::string remote_ps_string(const std::string& value);
std::vector<std::string> remote_ssh(const Value& connection, const std::string& command);
// Spool both streams, forward diagnostics and protocol events, and cancel the child on interruption.
int remote_process(const std::vector<std::string>& argv, const Value& input,
                   const std::filesystem::path& output, bool binary,
                   const std::function<void(const Value&)>& event = {});
Value remote_profile(const std::string& name);
Value remote_provision(const Value& profile);
}
