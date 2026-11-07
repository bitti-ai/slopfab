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
    {"video generate", "generate", "video"}, {"audio generate", "generate", "music"},
    {"image generate", "generate", "image"}, {"video frames", "frames", ""},
    {"video syncnet", "syncnet", ""},        {"video motions", "motions", ""},
    {"image motions", "motions", ""},        {"video upscale", "upscale", ""},
    {"image upscale", "upscale", ""},        {"video upscale-latents", "upscale-latents", ""},
    {"video decode", "decode", ""},          {"video compare", "compare-y4m", ""},
    {"audio transcribe", "transcribe", ""},  {"audio separate", "separate", ""},
};

bool group(std::string_view s) {
  return s == "video" || s == "audio" || s == "image";
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
    if (name == r.target && name != "generate")
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
  if (usage.rfind(prefix, 0) == 0)
    usage.replace(0, prefix.size(), "slopfab " + title);
  detail = base->detail;
  if (*r->task) {
    auto pos = usage.find(" [video|image|music]");
    if (pos != std::string::npos)
      usage.erase(pos, 20);
    detail = "This command selects task '" + std::string(r->task) +
             "'. JSON requests must use that task. Shared generation options follow.\n" + detail;
  }
  if (name == "image motions") {
    usage = "slopfab image motions frame FILE.js --out IMAGE.png|.svg [options]";
    detail = "Use frame for a still image. Use video motions render for animation.\n" + detail;
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
  if (!group(args[0]))
    return {};
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
  if (kind == "image" && args[1] == "motions" && (args.size() < 3 || args[2] != "frame"))
    throw CliError("INVALID_REQUEST",
                   "image motions requires frame; use video motions render for animation");
  args.erase(args.begin(), args.begin() + 2);
  args.insert(args.begin(), r->target);
  return r->task;
}
} // namespace slopfab::cli
