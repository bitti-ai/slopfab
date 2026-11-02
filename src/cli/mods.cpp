#include "mods.h"
#include "agent.h"
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <map>

namespace slopfab::cli {
namespace {
namespace fs = std::filesystem;

std::string trim(const std::string& text) {
  const auto first = text.find_first_not_of(" \t\r\n");
  return first == std::string::npos
             ? ""
             : text.substr(first, text.find_last_not_of(" \t\r\n") - first + 1);
}

[[noreturn]] void invalid(const std::string& message) {
  throw CliError("INVALID_MOD", message, 2);
}

void check_utf8(const std::string& text) {
  for (size_t i = 0; i < text.size();) {
    const auto first = static_cast<unsigned char>(text[i++]);
    if (first < 0x80)
      continue;
    const int count = first >= 0xc2 && first <= 0xdf   ? 1
                      : first >= 0xe0 && first <= 0xef ? 2
                      : first >= 0xf0 && first <= 0xf4 ? 3
                                                       : 0;
    if (!count || i + count > text.size())
      invalid("MOD.md must contain valid UTF-8");
    unsigned codepoint = first & ((1u << (6 - count)) - 1);
    for (int n = 0; n < count; ++n) {
      const auto byte = static_cast<unsigned char>(text[i++]);
      if ((byte & 0xc0) != 0x80)
        invalid("MOD.md must contain valid UTF-8");
      codepoint = (codepoint << 6) | (byte & 0x3f);
    }
    if ((count == 2 && codepoint < 0x800) || (count == 3 && codepoint < 0x10000) ||
        (codepoint >= 0xd800 && codepoint <= 0xdfff) || codepoint > 0x10ffff)
      invalid("MOD.md must contain valid UTF-8");
  }
}

bool valid_name(const std::string& name) {
  return !name.empty() && name.size() <= 64 && name[0] >= 'a' && name[0] <= 'z' &&
         std::all_of(name.begin(), name.end(), [](char c) {
           return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-' || c == '_';
         });
}

// Mod metadata deliberately uses a small YAML schema: scalar fields, comments,
// quoted strings and literal/folded blocks. Reject unsupported YAML rather than
// silently treating nested structures, anchors or tags as instructions.
std::string scalar(std::string text) {
  text = trim(text);
  if (text.empty())
    return "";
  if (text[0] == '\'' || text[0] == '"') {
    const char quote = text[0];
    std::string value;
    for (size_t i = 1; i < text.size(); ++i) {
      if (quote == '"' && text[i] == '\\') {
        if (++i == text.size())
          invalid("unfinished quoted string");
        value += '\\';
        value += text[i];
      } else if (text[i] == quote) {
        if (quote == '\'' && i + 1 < text.size() && text[i + 1] == '\'') {
          value += '\'';
          ++i;
          continue;
        }
        auto tail = trim(text.substr(i + 1));
        if (!tail.empty() && tail[0] != '#')
          invalid("unexpected text after quoted value");
        if (quote == '"') {
          try {
            return parse_cli_json("\"" + value + "\"").as_string();
          } catch (const std::exception&) {
            invalid("double-quoted values must use JSON-compatible escapes");
          }
        }
        return value;
      } else {
        value += text[i];
      }
    }
    invalid("unfinished quoted string");
  }
  for (size_t i = 0; i < text.size(); ++i)
    if (text[i] == '#' && (i == 0 || text[i - 1] == ' ' || text[i - 1] == '\t')) {
      text.resize(i);
      break;
    }
  text = trim(text);
  if (!text.empty() && std::string("[]{}&*!|>@`").find(text[0]) != std::string::npos)
    invalid("metadata supports scalar values only; quote special characters");
  if (text.find(": ") != std::string::npos)
    invalid("quote metadata values containing a colon followed by a space");
  return text;
}

struct Mod {
  std::string name, description, prompt, content;
  fs::path path;
  bool enabled = true;
  Object metadata;

  Object summary() const {
    return {{"name", str(name)},
            {"description", str(description)},
            {"enabled", Value(enabled)},
            {"path", str(path.u8string())},
            {"directory", str(path.parent_path().u8string())},
            {"load_command", str("slopfab mods load " + name)}};
  }
};

Mod read_mod(const fs::path& path, const std::string& name) {
  if (!valid_name(name))
    invalid(
        "folder name must start with a-z and contain only a-z, 0-9, - or _ (maximum 64 characters)");
  std::ifstream file(path, std::ios::binary);
  if (!file)
    invalid("cannot read MOD.md");
  constexpr size_t limit = 1024 * 1024;
  std::string content(limit + 1, '\0');
  file.read(content.data(), std::streamsize(content.size()));
  content.resize(size_t(file.gcount()));
  if (file.bad())
    invalid("cannot read MOD.md");
  if (content.size() > limit)
    invalid("MOD.md exceeds 1 MiB");
  if (content.find('\0') != std::string::npos)
    invalid("MOD.md must be UTF-8 text without NUL characters");
  check_utf8(content);
  if (content.compare(0, 3, "\xef\xbb\xbf") == 0)
    content.erase(0, 3);
  size_t offset = 0;
  auto line = [&]() {
    const auto end = content.find('\n', offset);
    auto text = content.substr(offset, end == std::string::npos ? end : end - offset);
    offset = end == std::string::npos ? content.size() : end + 1;
    if (!text.empty() && text.back() == '\r')
      text.pop_back();
    return text;
  };
  if (trim(line()) != "---")
    invalid("MOD.md must begin with YAML front matter delimited by ---");
  std::vector<std::string> lines;
  bool closed = false;
  while (offset < content.size()) {
    auto text = line();
    if (text == "---") {
      closed = true;
      break;
    }
    lines.push_back(std::move(text));
  }
  if (!closed)
    invalid("missing closing --- after YAML metadata");
  Mod mod;
  mod.name = name;
  mod.path = path;
  mod.content = content;
  mod.prompt = content.substr(offset);
  if (trim(mod.prompt).empty())
    invalid("mod prompt must not be empty");
  for (size_t i = 0; i < lines.size(); ++i) {
    const auto& text = lines[i];
    if (trim(text).empty() || trim(text)[0] == '#')
      continue;
    const auto colon = text.find(':');
    if (colon == std::string::npos || text[0] == ' ' || text[0] == '\t')
      invalid("metadata must be a flat YAML mapping");
    auto key = trim(text.substr(0, colon));
    if (key != "name" && key != "description" && key != "enabled" && key != "author" &&
        key != "version")
      invalid("unknown metadata field: " + key);
    if (mod.metadata.count(key))
      invalid("duplicate metadata field: " + key);
    auto raw = trim(text.substr(colon + 1));
    // Block indicators may have a trailing YAML comment.
    auto marker = raw.substr(0, raw.find_first_of(" \t"));
    std::string value;
    if (marker == "|" || marker == ">" || marker == "|-" || marker == ">-" || marker == "|+" ||
        marker == ">+") {
      const auto tail = trim(raw.substr(marker.size()));
      if (!tail.empty() && tail[0] != '#')
        invalid("invalid block scalar indicator");
      std::vector<std::string> block;
      size_t indent = std::string::npos;
      while (i + 1 < lines.size()) {
        const auto& next = lines[i + 1];
        if (!trim(next).empty() && next[0] != ' ')
          break;
        ++i;
        const auto spaces = next.find_first_not_of(' ');
        if (spaces != std::string::npos) {
          if (indent == std::string::npos)
            indent = spaces;
          if (spaces < indent)
            invalid("inconsistent block scalar indentation");
          block.push_back(next.substr(indent));
        } else {
          block.emplace_back();
        }
      }
      for (size_t n = 0; n < block.size(); ++n) {
        value += block[n];
        // A blank line separates folded paragraphs with one newline, while
        // leading/trailing and more-indented blocks retain their line breaks.
        if (marker[0] == '>' && !block[n].empty() && block[n][0] != ' ' && n + 1 < block.size() &&
            block[n + 1].empty()) {
          auto next = std::find_if(block.begin() + n + 1, block.end(), [](const auto& part) {
            return !part.empty();
          });
          if (next != block.end() && (*next)[0] != ' ')
            continue;
        }
        const bool fold = marker[0] == '>' && n + 1 < block.size() && !block[n].empty() &&
                          !block[n + 1].empty() && block[n][0] != ' ' && block[n + 1][0] != ' ';
        value += fold ? ' ' : '\n';
      }
      if (marker.back() != '+') {
        while (!value.empty() && value.back() == '\n')
          value.pop_back();
        if (marker.back() != '-' && !value.empty())
          value += '\n';
      }
    } else {
      value = scalar(raw);
    }
    if (key == "enabled") {
      // YAML booleans must be unquoted, not string values such as "false".
      if (raw.empty() || (raw[0] != 't' && raw[0] != 'f') || (value != "true" && value != "false"))
        invalid("enabled must be true or false");
      mod.enabled = value == "true";
      mod.metadata[key] = Value(mod.enabled);
    } else {
      mod.metadata[key] = str(value);
    }
  }
  const auto n = mod.metadata.find("name"), d = mod.metadata.find("description");
  if (n == mod.metadata.end() || n->second.as_string() != name)
    invalid("metadata name must match the mod folder name");
  if (d == mod.metadata.end() || trim(d->second.as_string()).empty())
    invalid("description is required");
  mod.description = d->second.as_string();
  return mod;
}

std::vector<fs::path> roots() {
  std::vector<fs::path> paths;
  for (const auto& base : {fs::absolute(config_path()).parent_path(),
                           fs::u8path(agent().executable).parent_path(), fs::current_path()}) {
    auto path = fs::weakly_canonical(base / "mods");
    if (std::none_of(paths.begin(), paths.end(), [&](const auto& old) {
          std::error_code ec;
          return old == path || fs::equivalent(old, path, ec);
        }))
      paths.push_back(path);
  }
  return paths;
}

struct Index {
  Array roots, errors;
  std::map<std::string, std::vector<fs::path>> candidates;
  std::vector<Mod> mods;
};

Index index_mods() {
  Index index;
  for (const auto& root : roots()) {
    index.roots.push_back(str(root.u8string()));
    try {
      if (!fs::exists(root))
        continue;
      for (const auto& folder : fs::directory_iterator(root)) {
        if (!folder.is_directory() || !fs::is_regular_file(folder.path() / "MOD.md"))
          continue;
        const auto path = fs::canonical(folder.path() / "MOD.md");
        const auto name = folder.path().filename().u8string();
        auto& files = index.candidates[name];
        if (std::none_of(files.begin(), files.end(), [&](const auto& old) {
              return fs::equivalent(old, path);
            }))
          files.push_back(path);
      }
    } catch (const std::exception& e) {
      index.errors.push_back(Value(Object{{"code", str("MOD_SCAN_FAILED")},
                                          {"path", str(root.u8string())},
                                          {"message", str(e.what())}}));
    }
  }
  for (const auto& entry : index.candidates) {
    try {
      if (entry.second.size() != 1)
        throw CliError("MOD_AMBIGUOUS", "duplicate mod name; rename or remove one folder", 3);
      index.mods.push_back(read_mod(entry.second[0], entry.first));
    } catch (const std::exception& e) {
      const auto* cli = dynamic_cast<const CliError*>(&e);
      Array paths;
      for (const auto& path : entry.second)
        paths.push_back(str(path.u8string()));
      index.errors.push_back(Value(Object{{"name", str(entry.first)},
                                          {"paths", Value(paths)},
                                          {"code", str(cli ? cli->code : "INVALID_MOD")},
                                          {"message", str(e.what())}}));
    }
  }
  return index;
}
}

int mods_command(const std::vector<std::string>& args) {
  const bool list = args.empty() || args == std::vector<std::string>{"list"} ||
                    args == std::vector<std::string>{"index"};
  const bool all = args == std::vector<std::string>{"list", "--all"} ||
                   args == std::vector<std::string>{"index", "--all"};
  const bool load = args.size() == 2 && args[0] == "load";
  if (!list && !all && !load)
    throw CliError("INVALID_REQUEST", "use mods list [--all] or mods load NAME");
  if (load && !valid_name(args[1]))
    throw CliError("INVALID_REQUEST", "provide a mod name from slopfab mods list, not a path");
  auto index = index_mods();
  if (load) {
    for (const auto& error : index.errors)
      if (const auto* name = error.find("name"); name && name->as_string() == args[1])
        throw CliError(error.find("code")->as_string(), error.find("message")->as_string(),
                       error.find("code")->as_string() == "MOD_AMBIGUOUS" ? 3 : 2);
    for (const auto& mod : index.mods) {
      if (mod.name != args[1])
        continue;
      if (!mod.enabled)
        throw CliError("MOD_DISABLED", "mod is disabled; set enabled: true in its MOD.md", 4);
      if (agent().enabled) {
        auto data = mod.summary();
        data["metadata"] = Value(mod.metadata);
        data["prompt"] = str(mod.prompt);
        emit({{"type", str("result")}, {"ok", Value(true)}, {"mod", Value(data)}}, true);
      } else {
        std::fwrite(mod.content.data(), 1, mod.content.size(), stdout);
      }
      return 0;
    }
    throw CliError("MOD_NOT_FOUND", "mod not found; run slopfab mods list", 3);
  }
  Array mods;
  for (const auto& mod : index.mods)
    if (all || mod.enabled)
      mods.push_back(Value(mod.summary()));
  if (agent().enabled) {
    emit({{"type", str("result")},
          {"ok", Value(true)},
          {"mods", Value(mods)},
          {"roots", Value(index.roots)},
          {"errors", Value(index.errors)},
          {"next",
           str("Use slopfab mods load NAME to read a mod's instructions into your context.")}},
         true);
  } else {
    if (mods.empty())
      std::puts("No active mods. Add NAME/MOD.md under one of these locations:");
    for (const auto& root : index.roots)
      std::printf("Mods: %s\n", root.as_string().c_str());
    for (const auto& mod : mods)
      std::printf("%s%s: %s\n", mod.find("name")->as_string().c_str(),
                  mod.find("enabled")->as_bool() ? "" : " (disabled)",
                  mod.find("description")->as_string().c_str());
    for (const auto& error : index.errors)
      std::fprintf(stderr, "%s\n", json::stringify(error).c_str());
    std::puts("Use slopfab mods load NAME to read a mod's instructions into your context.");
  }
  return 0;
}
}
