#pragma once
#include "commands.h"

namespace slopfab::cli {
// Normalize public media groups after global/remote options have been consumed.
// Returns the task constraint for a typed generate command.
std::string route_media_command(std::vector<std::string>&);
const CommandHelp* media_command_help(std::string_view);
bool hidden_media_alias(std::string_view);
std::string_view media_option_command(const std::vector<std::string>&);
}
