#pragma once
#include "agent.h"

namespace slopfab::cli {
int projects_command(const std::vector<std::string>& args);
// CLI-only bookkeeping; generation and the C ABI do not know about projects.
std::filesystem::path project_directory(const std::string& name, const std::string& output = {});
void prepare_project_generation(GenerateRequest& request, bool explicit_output, int count);
Value record_project_generation(const GenerateRequest& request, const Value& result);
}
