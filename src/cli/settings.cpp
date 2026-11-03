#include "settings.h"
#include <algorithm>
#include <thread>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace slopfab::cli {
namespace {
namespace fs = std::filesystem;

fs::path settings_path() {
  return fs::absolute(config_path()).lexically_normal();
}

std::unique_ptr<FileLock> settings_lock() {
  auto path = settings_path();
  path += ".lock";
  for (int i = 0; i < 100; ++i) {
    auto lock = std::make_unique<FileLock>(path);
    if (lock->acquired())
      return lock;
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  throw CliError("SETTINGS_BUSY", "settings are being updated; retry", 7, {}, true);
}

bool same_path(const fs::path& a, const fs::path& b) {
#ifdef _WIN32
  return CompareStringOrdinal(a.c_str(), -1, b.c_str(), -1, TRUE) == CSTR_EQUAL;
#else
  return a == b;
#endif
}

Array normalize_paths(const Value& value, const fs::path& base) {
  if (!value.is_array())
    throw CliError("INVALID_REQUEST", "weight_paths must be an array of paths", 2, "weight_paths");
  Array paths;
  std::vector<fs::path> seen;
  for (const auto& item : value.as_array()) {
    if (!item.is_string() || item.as_string().empty())
      throw CliError("INVALID_REQUEST", "weight_paths entries must be nonempty strings", 2,
                     "weight_paths");
    auto path = fs::u8path(item.as_string());
    if (path.is_relative())
      path = base / path;
    path = fs::absolute(path).lexically_normal();
    // Drop a trailing separator so spelling alone does not create duplicates.
    if (path.filename().empty() && path != path.root_path())
      path = path.parent_path();
    if (std::none_of(seen.begin(), seen.end(), [&](const auto& old) {
          return same_path(old, path);
        })) {
      seen.push_back(path);
      paths.push_back(str(path.u8string()));
    }
  }
  return paths;
}

Value normalize_settings(const Value& input) {
  if (!input.is_object())
    throw CliError("INVALID_REQUEST", "settings must be a JSON object");
  auto object = input.as_object();
  for (const auto& field : object)
    if (field.first != "schema_version" && field.first != "weight_paths" &&
        field.first != "weights_directory" && field.first != "models" && field.first != "presets" &&
        field.first != "output_directory")
      throw CliError("INVALID_REQUEST", "unknown setting: " + field.first, 2, field.first);
  if (const auto* version = input.find("schema_version"))
    if (!version->is_number() || version->as_number() != 1)
      throw CliError("INVALID_REQUEST", "unsupported settings schema_version");
  for (const char* key : {"models", "presets"})
    if (const auto* value = input.find(key)) {
      if (!value->is_object())
        throw CliError("INVALID_REQUEST", std::string(key) + " must be an object", 2, key);
      for (const auto& entry : value->as_object())
        if (!entry.second.is_object())
          throw CliError("INVALID_REQUEST", std::string(key) + " entries must be objects", 2, key);
    }
  Value paths(Array{});
  if (const auto* value = input.find("weight_paths"))
    paths = *value;
  else if (const auto* legacy = input.find("weights_directory")) {
    if (!legacy->is_string() || legacy->as_string().empty())
      throw CliError("INVALID_REQUEST", "weights_directory must be a nonempty path");
    paths = Value(Array{*legacy});
  }
  object.erase("weights_directory");
  if (const auto* output = input.find("output_directory")) {
    if (!output->is_string() || output->as_string().empty())
      throw CliError("INVALID_REQUEST", "output_directory must be a nonempty path");
    auto path = fs::u8path(output->as_string());
    if (path.is_relative())
      path = settings_path().parent_path() / path;
    object["output_directory"] = str(fs::absolute(path).lexically_normal().u8string());
  }
  object["schema_version"] = num(1);
  object["weight_paths"] = Value(normalize_paths(paths, settings_path().parent_path()));
  return Value(std::move(object));
}

std::vector<std::string> key_parts(const std::string& key) {
  std::vector<std::string> parts;
  size_t begin = 0;
  for (;;) {
    const auto end = key.find('.', begin);
    const auto part = key.substr(begin, end == std::string::npos ? end : end - begin);
    if (part.empty())
      throw CliError("INVALID_REQUEST", "setting key contains an empty component", 2, key);
    parts.push_back(part);
    if (end == std::string::npos)
      return parts;
    begin = end + 1;
  }
}

const Value* lookup(const Value& settings, const std::vector<std::string>& parts) {
  const Value* result = &settings;
  for (const auto& part : parts) {
    result = result->find(part);
    if (!result)
      break;
  }
  return result;
}

Value update(Value current, const std::vector<std::string>& parts, size_t index,
             const Value* replacement) {
  if (!current.is_object())
    throw CliError("INVALID_REQUEST", "setting parent must be an object");
  auto object = current.as_object();
  if (index + 1 == parts.size()) {
    if (replacement)
      object[parts[index]] = *replacement;
    else
      object.erase(parts[index]);
  } else {
    const auto* child = current.find(parts[index]);
    object[parts[index]] = update(child ? *child : Value(Object{}), parts, index + 1, replacement);
  }
  return Value(std::move(object));
}
}

Value load_settings() {
  if (!std::filesystem::exists(settings_path()))
    return normalize_settings(Value(Object{}));
  return normalize_settings(read_json(settings_path()));
}

void ensure_settings() {
  const auto path = settings_path();
  if (std::filesystem::exists(path)) {
    auto original = read_json(path);
    auto normalized = normalize_settings(original);
    if (json::stringify(original) == json::stringify(normalized))
      return;
  }
  auto lock = settings_lock();
  // Re-read under the lock: simultaneous first launches or migration must not
  // overwrite a setting written by another process.
  atomic_json(path, load_settings());
}

std::vector<std::filesystem::path> configured_weight_paths() {
  auto settings = load_settings();
  std::vector<std::filesystem::path> paths;
  for (const auto& value : settings.find("weight_paths")->as_array())
    paths.push_back(std::filesystem::u8path(value.as_string()));
  return paths;
}

int settings_command(const std::vector<std::string>& args) {
  if (args.empty())
    throw CliError("INVALID_REQUEST",
                   "use settings get [KEY], set KEY VALUE, or remove KEY [PATH]");
  const auto& action = args[0];
  if (action != "get" && action != "set" && action != "remove")
    throw CliError("INVALID_REQUEST", "unknown settings action: " + action);
  if (action == "get" && args.size() == 1) {
    emit({{"type", str("result")},
          {"ok", Value(true)},
          {"path", str(settings_path().u8string())},
          {"settings", load_settings()}},
         true);
    return 0;
  }
  if (args.size() < 2)
    throw CliError("INVALID_REQUEST", "provide a setting key");
  const auto& key = args[1];
  auto parts = key_parts(key);
  if (action == "get") {
    if (args.size() != 2)
      throw CliError("INVALID_REQUEST", "settings get accepts one key");
    auto settings = load_settings();
    const auto* value = lookup(settings, parts);
    if (!value)
      throw CliError("SETTING_NOT_FOUND", "setting not found: " + key, 3, key);
    emit({{"type", str("result")},
          {"ok", Value(true)},
          {"path", str(settings_path().u8string())},
          {"key", str(key)},
          {"value", *value}},
         true);
    return 0;
  }
  if ((parts[0] != "weight_paths" && parts[0] != "models" && parts[0] != "presets" &&
       parts[0] != "output_directory") ||
      ((parts[0] == "weight_paths" || parts[0] == "output_directory") && parts.size() != 1))
    throw CliError("INVALID_REQUEST",
                   "writable settings: weight_paths, output_directory, models, presets (schema_version is read-only)",
                   2, key);
  auto lock = settings_lock();
  auto settings = load_settings();
  Value value;
  bool removed = false;
  if (action == "set") {
    if (args.size() < 3)
      throw CliError("INVALID_REQUEST", "settings set requires a value");
    if (key == "output_directory") {
      if (args.size() != 3 || args[2].empty())
        throw CliError("INVALID_REQUEST", "output_directory requires one path");
      value = str(fs::absolute(fs::u8path(args[2])).lexically_normal().u8string());
    } else if (key == "weight_paths") {
      if (args.size() == 3 && !args[2].empty() && args[2][0] == '[') {
        try {
          value = parse_cli_json(args[2]);
        } catch (const std::exception& e) {
          throw CliError("INVALID_REQUEST", e.what(), 2, key);
        }
      } else {
        Array paths;
        for (size_t i = 2; i < args.size(); ++i)
          paths.push_back(str(args[i]));
        value = Value(paths);
      }
      value = Value(normalize_paths(value, std::filesystem::current_path()));
    } else {
      if (args.size() != 3)
        throw CliError("INVALID_REQUEST", "setting value must be one JSON argument");
      try {
        value = parse_cli_json(args[2]);
      } catch (const std::exception& e) {
        throw CliError("INVALID_REQUEST", e.what(), 2, key);
      }
    }
    settings = update(settings, parts, 0, &value);
  } else if (key == "weight_paths") {
    Array remaining;
    auto current = settings.find("weight_paths")->as_array();
    Array requested;
    for (size_t i = 2; i < args.size(); ++i)
      requested.push_back(str(args[i]));
    auto remove = normalize_paths(Value(requested), std::filesystem::current_path());
    for (const auto& entry : current) {
      bool erase = args.size() == 2;
      for (const auto& path : remove)
        if (same_path(std::filesystem::u8path(entry.as_string()),
                      std::filesystem::u8path(path.as_string())))
          erase = true;
      if (erase)
        removed = true;
      else
        remaining.push_back(entry);
    }
    value = Value(remaining);
    settings = update(settings, parts, 0, &value);
  } else {
    if (args.size() != 2)
      throw CliError("INVALID_REQUEST", "only weight_paths supports removing individual paths");
    removed = lookup(settings, parts) != nullptr;
    if (removed)
      settings = update(settings, parts, 0, nullptr);
  }
  settings = normalize_settings(settings);
  atomic_json(settings_path(), settings);
  const auto* result = lookup(settings, parts);
  emit({{"type", str("result")},
        {"ok", Value(true)},
        {"path", str(settings_path().u8string())},
        {"key", str(key)},
        {"value", result ? *result : Value()},
        {"removed", Value(removed)}},
       true);
  return 0;
}
}
