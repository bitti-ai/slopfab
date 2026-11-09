#include "media_commands.h"
#include "agent.h"
#include <algorithm>

namespace slopfab::cli {
namespace {
struct Route {
  const char* name;
  const char* target;
  const char* task;
};

const Route routes[] = {
    {"jobs submit", "submit", ""}, {"jobs status", "status", ""},
    {"jobs wait", "wait", ""}, {"jobs cancel", "cancel", ""}, {"jobs worker", "worker", ""},
    {"models list", "model-list", ""}, {"models inspect", "inspect", ""},
    {"models compare", "compare", ""}, {"models prepare-lora", "prepare-lora", ""},
    {"conditioning tokenize", "tokenize", ""}, {"conditioning encode-text", "encode-text", ""},
    {"conditioning encode-refmod", "encode-refmod", ""}, {"conditioning bundle-refmods", "bundle-refmods", ""},
    {"system doctor", "doctor", ""}, {"system devices", "devices", ""},
    {"system capabilities", "capabilities", ""}, {"system version", "version", ""},
    {"requests run", "generate", ""}, {"requests validate", "validate", ""},
    {"requests schema", "schema", ""}, {"remote fetch", "fetch", ""},
    {"video generate", "generate", "video"}, {"video regenerate", "regenerate", "video"},
    {"audio generate", "generate", "music"},
    {"image generate", "generate", "image"}, {"video frames", "frames", ""},
    {"video syncnet", "syncnet", ""},        {"video upscale", "upscale", ""},
    {"image upscale", "upscale", ""},        {"video upscale-latents", "upscale-latents", ""},
    {"video decode", "decode", ""},          {"video compare", "compare-y4m", ""},
    {"audio transcribe", "transcribe", ""},  {"audio separate", "separate", ""},
};

bool group(std::string_view s) {
  return s == "video" || s == "audio" || s == "image" || s == "jobs" ||
         s == "models" || s == "conditioning" || s == "system" || s == "requests" || s == "remote";
}

const Route* route(std::string_view s) {
  for (const auto& r : routes)
    if (s == r.name)
      return &r;
  return nullptr;
}
}

bool hidden_media_alias(std::string_view name) {
  if (name.find(' ') != std::string_view::npos)
    return true;
  for (const auto& r : routes)
    if (name == r.target)
      return true;
  return false;
}
std::string_view media_option_command(const std::vector<std::string>& args) {
  if (args.empty()) return {};
  if (args.size() >= 2 && group(args[0]))
    if (const auto* r = route(args[0] + " " + args[1])) return r->target;
  return args[0];
}

const CommandHelp* media_command_help(std::string_view name) {
  const auto* r = route(name);
  if (!r)
    return nullptr;
  const auto* base = find_command(r->target);
  if (!base)
    return nullptr;
  static thread_local std::string title, usage, detail;
  static thread_local CommandHelp result;
  title = r->name;
  usage = base->usage;
  auto prefix = std::string("slopfab ") + r->target;
  for (const auto& candidate : routes) {
    const auto candidate_prefix = std::string("slopfab ") + candidate.name;
    if (std::string_view(candidate.target) == r->target && usage.rfind(candidate_prefix, 0) == 0) {
      prefix = candidate_prefix;
      break;
    }
  }
  if (usage.rfind(prefix, 0) == 0)
    usage.replace(0, prefix.size(), "slopfab " + title);
  detail = base->detail;
  if (std::string_view(r->target) == "regenerate")
    detail += std::string("\nShared generation options:\n") + find_command("generate")->detail;
  if (*r->task) {
    auto pos = usage.find(" [video|image|music]");
    if (pos != std::string::npos)
      usage.erase(pos, 20);
    detail = "This command selects task '" + std::string(r->task) +
             "'. JSON requests must use that task. Shared generation options follow.\n" + detail;
  }
  result = {title.c_str(), usage.c_str(), base->summary, detail.c_str()};
  return &result;
}

std::string route_media_command(std::vector<std::string>& args) {
  if (args.empty())
    return {};
  if (args[0] == "help" && args.size() >= 3 && group(args[1])) {
    if (args.size() != 3)
      throw CliError("INVALID_REQUEST", "use help MEDIA COMMAND");
    args = {"help", args[1] + " " + args[2]};
    return {};
  }
  if (args[0] == "help" && args.size() == 2 && hidden_media_alias(args[1]))
    throw CliError("INVALID_REQUEST", "command moved to a group; run slopfab help");
  if (!group(args[0])) {
    if (hidden_media_alias(args[0]))
      throw CliError("INVALID_REQUEST", "command moved to a group; run slopfab help");
    return {};
  }
  if (args[0] == "remote" && (args.size() < 2 ||
      (args[1] != "fetch" && args[1] != "help"))) return {};
  const auto kind = args[0];
  if (args.size() == 1 || args[1] == "--help" || args[1] == "-h") {
    args = {"help", kind};
    return {};
  }
  if (args[1] == "help") {
    if (args.size() > 3)
      throw CliError("INVALID_REQUEST", "use MEDIA help COMMAND");
    args = {"help", args.size() == 3 ? kind + " " + args[2] : kind};
    return {};
  }
  const std::string name = kind + " " + args[1];
  const auto* r = route(name);
  if (!r)
    throw CliError("INVALID_REQUEST", "unknown command: " + name + "; see slopfab help " + kind);
  std::vector<char*> av;
  for (size_t i = 2; i < args.size(); ++i)
    av.push_back(args[i].data());
  if (wants_help(int(av.size()), av.data(), r->target)) {
    args = {"help", name};
    return {};
  }
  args.erase(args.begin(), args.begin() + 2);
  args.insert(args.begin(), r->target);
  return r->task;
}
} // namespace slopfab::cli
