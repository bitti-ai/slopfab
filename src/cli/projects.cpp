#include "projects.h"
#include "settings.h"
#include "project_template.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <set>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <shellapi.h>
#else
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif

namespace slopfab::cli {
namespace {
namespace fs = std::filesystem;
using Options = std::map<std::string, std::string>;

[[noreturn]] void invalid(const std::string& message) {
  throw CliError("INVALID_PROJECT", message, 2);
}

void identifier(const std::string& name) {
  if (name.empty() || name.size() > 80 || name.front() == '-' || name.front() == '_' ||
      name.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789-_") != std::string::npos)
    invalid(
        "names and item IDs use 1..80 lowercase letters, digits, - or _; start with a letter or digit");
  static const std::set<std::string> reserved = {
      "con",  "prn",  "aux",  "nul",  "com1", "com2", "com3", "com4", "com5", "com6", "com7",
      "com8", "com9", "lpt1", "lpt2", "lpt3", "lpt4", "lpt5", "lpt6", "lpt7", "lpt8", "lpt9"};
  if (reserved.count(name))
    invalid("reserved filesystem name: " + name);
}

fs::path output_root(const std::string& override_path) {
  auto settings = load_settings();
  auto configured = settings.find("output_directory");
  return fs::weakly_canonical(fs::absolute(fs::u8path(!override_path.empty() ? override_path
                                                      : configured ? configured->as_string()
                                                                   : "output")));
}

// Check resolved paths as well as lexical paths, including existing symlinks.
fs::path inside(const fs::path& root, const fs::path& path) {
  auto base = fs::weakly_canonical(fs::absolute(root));
  auto resolved = fs::weakly_canonical(path.is_absolute() ? path : base / path);
  auto relative = resolved.lexically_relative(base);
  if (relative.empty() || relative == "." || *relative.begin() == "..")
    invalid("path must stay inside " + base.u8string());
  return resolved;
}

std::string field(const Value& value, const char* key, const std::string& fallback = {}) {
  auto v = value.find(key);
  return v ? v->as_string() : fallback;
}

bool flag(const Value& value, const char* key) {
  auto v = value.find(key);
  return v && v->as_bool();
}

std::string extension(const fs::path& path) {
  auto ext = path.extension().u8string();
  std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
    return char(std::tolower(c));
  });
  return ext;
}

std::string media_kind(const fs::path& path) {
  const auto ext = extension(path);
  if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".webp" || ext == ".gif" ||
      ext == ".ppm")
    return "image";
  if (ext == ".mp4" || ext == ".webm" || ext == ".mov" || ext == ".y4m")
    return "video";
  if (ext == ".wav" || ext == ".mp3" || ext == ".ogg" || ext == ".flac" || ext == ".m4a")
    return "music";
  invalid("unsupported media extension: " + ext);
}

Value naming_rules() {
  return Value(Object{
      {"manifest",
       str("project.json is the source of truth. The project HTML page uses items[].file and items[].preview as exact project-relative paths; it does not scan folders.")},
      {"imports",
       str("Imported files: assets/ITEM_ID/vVERSION/source.EXT and preview.EXT. IDs use lowercase letters, digits, - or _. Versions are positive integers; old files are retained.")},
      {"generation",
       str("generate --project NAME chooses and registers assets/ITEM_ID/vVERSION/source.EXT. Batches append -001, -002, etc. Custom --out paths are relative to the project and must stay inside it; never overwrite a registered file.")},
      {"final_delivery",
       str("Always copy the final generation result directly into the configured output folder as NAME.EXT, using the project name and appropriate media extension (for example, output/NAME.mp4, output/NAME.png or output/NAME.wav). Refresh this delivery copy when the final result changes; retain the registered version inside the project folder.")},
      {"registration",
       str("Use projects add NAME --file PATH for a new item; projects update NAME ID --file PATH or --preview PATH for a replacement. Registration writes project.json and regenerates NAME.html in the output directory. Simply placing files in a folder does not register them.")},
      {"previews",
       str("PNG/JPEG/WebP/GIF, MP4/WebM and WAV/MP3/Ogg are browser-friendly. For Y4M/PPM, use external ffmpeg to create a preview, then projects update NAME ID --preview FILE. Slopfab does not invoke FFmpeg.")}});
}

Value load(const fs::path& directory) {
  auto value = read_json(inside(directory, "project.json"));
  if (!value.is_object() || !value.find("schema_version") ||
      value.find("schema_version")->as_number() != 1 || !value.find("items") ||
      !value.find("items")->is_array() || !value.find("revision"))
    invalid("unsupported project.json; expected schema_version 1, revision and items");
  if (!value.find("name") || !value.find("name")->is_string() || !value.find("title") ||
      !value.find("title")->is_string())
    invalid("project name and title must be strings");
  identifier(field(value, "name"));
  auto valid_counter = [](const Value* v) {
    return v && v->is_number() && std::isfinite(v->as_number()) && v->as_number() >= 1 &&
           v->as_number() < 1000000000 && std::floor(v->as_number()) == v->as_number();
  };
  if (!valid_counter(value.find("revision")))
    invalid("invalid project revision");
  std::set<std::string> ids;
  for (const auto& item : value.find("items")->as_array()) {
    auto id = field(item, "id");
    identifier(id);
    if (!ids.insert(id).second)
      invalid("duplicate item ID: " + id);
    const auto kind = field(item, "kind");
    if (kind != "image" && kind != "video" && kind != "music" && kind != "clip")
      invalid("invalid item kind");
    for (const char* key : {"file", "preview"}) {
      const auto path = field(item, key);
      if (path.empty()) {
        if (std::string(key) == "file")
          invalid("item has no file");
        else
          continue;
      }
      if (fs::u8path(path).is_absolute() || path.find('\\') != std::string::npos)
        invalid("item paths must be relative, using forward slashes");
      inside(directory, fs::u8path(path));
      if (media_kind(fs::u8path(path)) != (kind == "clip" ? "video" : kind))
        invalid("media type does not match item kind");
    }
    if (!valid_counter(item.find("version")))
      invalid("item version must be positive");
    if (!item.find("history") || !item.find("history")->is_array())
      invalid("item history must be an array");
  }
  for (const auto& item : value.find("items")->as_array()) {
    const auto parent = field(item, "parent");
    if (field(item, "kind") == "clip" && parent.empty())
      invalid("clips require a parent video");
    if (parent.empty())
      continue;
    if (field(item, "kind") != "clip")
      invalid("only clips have parent videos");
    bool found = false;
    for (const auto& candidate : value.find("items")->as_array())
      if (field(candidate, "id") == parent && field(candidate, "kind") == "video")
        found = true;
    if (!found)
      invalid("missing parent video: " + parent);
  }
  return value;
}

void atomic_text(const fs::path& path, const std::string& text) {
  auto temporary = path;
  temporary += "." + unique_id() + ".tmp";
  try {
    std::ofstream stream(temporary, std::ios::binary);
    stream.write(text.data(), std::streamsize(text.size()));
    stream.close();
    if (!stream)
      throw CliError("OUTPUT_FAILED", "cannot write " + path.u8string(), 6);
#ifdef _WIN32
    if (!MoveFileExW(temporary.c_str(), path.c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
      throw CliError("OUTPUT_FAILED", "cannot publish " + path.u8string(), 6);
#else
    fs::rename(temporary, path);
#endif
  } catch (...) {
    std::error_code error;
    fs::remove(temporary, error);
    throw;
  }
}

Value context(const fs::path& directory, const Value& manifest) {
  auto object = manifest.as_object();
  object["directory"] = str(directory.u8string());
  object["relative_directory"] = str(directory.filename().u8string());
  object["manifest_path"] = str((directory / "project.json").u8string());
  object["index_path"] = str(project_index_path(directory).u8string());
  object["naming_rules"] = naming_rules();
  return Value(object);
}

void render(const fs::path& directory, const Value& manifest) {
  // Escape '<' before putting JSON in an inert script block, including </script>.
  auto data = json::stringify(context(directory, manifest));
  std::string escaped;
  for (char c : data)
    escaped += c == '<' ? "\\u003c" : std::string(1, c);
  std::string html = kProjectTemplate;
  const std::string marker = "__SLOPFAB_PROJECT_DATA__";
  html.replace(html.find(marker), marker.size(), escaped);
  atomic_text(project_index_path(directory), html);
  std::string guide =
      "# " + field(manifest, "name") +
      "\n\nOpen ../" + directory.filename().u8string() + ".html to review media and copy edit prompts into your agent.\n\n## File naming and registration\n\n";
  const auto rules = naming_rules();
  for (const auto& rule : rules.as_object())
    guide += "- " + rule.second.as_string() + "\n";
  guide +=
      "\n## Agent workflow\n\nRead project.json before editing; IDs are stable and array entries carry version, prompt, generation recipe, parent and position. Clip order is ascending position. Removed items stay in the manifest and retain their files.\n\nUse slopfab help projects for commands. After regenerating a clip or removing/restoring/reordering it, rebuild its parent video with external FFmpeg and register the replacement using projects update. needs_rebuild marks an outdated parent. The page only prepares prompts; it never runs commands.\n\nGeneration recipes record arguments, working directory, resolved plan and artifacts. Resolve relative model/reference paths against that working directory. Use a new output and seed when asked to regenerate. Preserve other items. --if-revision can reject stale project edits.\n";
  atomic_text(inside(directory, "PROJECT.md"), guide);
}

void open_index(const fs::path& directory) {
  const auto path = project_index_path(directory);
  if (!fs::is_regular_file(path))
    throw CliError("INPUT_NOT_FOUND",
                   "project HTML page not found; run projects render " + directory.filename().u8string(), 3);
#ifdef _WIN32
  const auto result = reinterpret_cast<INT_PTR>(
      ShellExecuteW(nullptr, L"open", path.c_str(), nullptr, nullptr, SW_SHOWNORMAL));
  if (result <= 32)
    throw CliError("OPEN_FAILED", "cannot open project in default browser (Windows error " +
                                     std::to_string(result) + "): " + path.u8string(), 6);
#else
#ifdef __APPLE__
  std::string opener = "open";
#else
  std::string opener = "xdg-open";
#endif
  std::string filename = path.string();
  char* arguments[] = {opener.data(), filename.data(), nullptr};
  posix_spawn_file_actions_t actions;
  int error = posix_spawn_file_actions_init(&actions);
  if (error)
    throw CliError("OPEN_FAILED", "cannot prepare browser launcher: " + std::string(std::strerror(error)), 6);
  // Keep launcher output off the CLI's JSON stream and avoid reading its stdin.
  error = posix_spawn_file_actions_addopen(&actions, STDIN_FILENO, "/dev/null", O_RDONLY, 0);
  if (!error)
    error = posix_spawn_file_actions_adddup2(&actions, STDERR_FILENO, STDOUT_FILENO);
  pid_t child = -1;
  if (!error)
    error = posix_spawnp(&child, arguments[0], &actions, nullptr, arguments, environ);
  posix_spawn_file_actions_destroy(&actions);
  if (error)
    throw CliError("OPEN_FAILED", "cannot launch " + opener + ": " + std::strerror(error), 6);
  int status = 0;
  pid_t waited;
  do {
    waited = waitpid(child, &status, 0);
  } while (waited < 0 && errno == EINTR);
  if (waited < 0 || !WIFEXITED(status) || WEXITSTATUS(status) != 0)
    throw CliError("OPEN_FAILED", "cannot open project in default browser: " + path.u8string(), 6);
#endif
}

void save(const fs::path& directory, Object object) {
  object["revision"] = num(object.at("revision").as_number() + 1);
  if (json::stringify(Value(object)).size() >= 16 * 1024 * 1024)
    invalid("project manifest exceeds 16 MiB; archive older history before adding more items");
  atomic_json(inside(directory, "project.json"), Value(object));
  render(directory, Value(object));
}

size_t item_index(const Array& items, const std::string& id) {
  for (size_t i = 0; i < items.size(); ++i)
    if (field(items[i], "id") == id)
      return i;
  throw CliError("PROJECT_ITEM_NOT_FOUND", "project item not found: " + id, 3);
}

void check_revision(const Value& project, const Options& options) {
  auto expected = options.find("--if-revision");
  if (expected != options.end() && expected->second != json::stringify(*project.find("revision")))
    throw CliError("PROJECT_CHANGED", "project changed; read projects show again before editing", 7,
                   {}, true);
}

std::string import_file(const fs::path& directory, const fs::path& source, const std::string& id,
                        int version, const char* stem, bool keep_local = false) {
  if (!fs::is_regular_file(source))
    throw CliError("INPUT_NOT_FOUND", "media file not found: " + source.u8string(), 3);
  auto absolute = fs::canonical(source);
  if (keep_local)
    return inside(directory, absolute).lexically_relative(directory).generic_u8string();
  auto relative = fs::path("assets") / id / ("v" + std::to_string(version)) /
                  (std::string(stem) + extension(source));
  auto destination = inside(directory, relative);
  // Agents may follow PROJECT.md and write the canonical filename themselves.
  // Registering that same file must not try to copy it over itself.
  if (fs::exists(destination) && fs::equivalent(source, destination))
    return relative.generic_u8string();
  fs::create_directories(destination.parent_path());
  if (fs::exists(destination))
    throw CliError("OUTPUT_EXISTS", "project asset already exists: " + destination.u8string(), 6);
  fs::copy_file(source, destination, fs::copy_options::none);
  return relative.generic_u8string();
}

Value change_item(const fs::path& directory, const std::string& action, const std::string& id,
                  const Options& options, const Value& generation = Value(),
                  bool keep_local = false, int expected_version = 0) {
  if (!fs::is_regular_file(directory / "project.json"))
    throw CliError("INPUT_NOT_FOUND", "project not found; use projects create first", 3);
  FileLock lock(inside(directory, ".project.lock"));
  if (!lock.acquired())
    throw CliError("PROJECT_BUSY", "project is being updated; retry", 7, {}, true);
  auto project = load(directory);
  check_revision(project, options);
  auto items = project.find("items")->as_array();
  const bool adding = action == "add";
  Object item;
  size_t index = items.size();
  int version = 1;
  if (adding) {
    for (const auto& old : items)
      if (field(old, "id") == id)
        invalid("item ID already exists: " + id);
    item = {{"id", str(id)},
            {"version", num(1)},
            {"title", str(id)},
            {"removed", Value(false)},
            {"history", Value(Array{})}};
  } else {
    index = item_index(items, id);
    item = items[index].as_object();
    if (expected_version &&
        (item.at("version").as_number() != expected_version || flag(Value(item), "removed")))
      throw CliError(
          "PROJECT_CHANGED",
          "item changed during generation; output retained, inspect and register manually", 7, {},
          true);
    auto history = item.at("history").as_array();
    auto old = item;
    old.erase("history");
    history.emplace_back(old);
    item["history"] = Value(history);
    version = int(item.at("version").as_number()) + 1;
    item["version"] = num(version);
  }
  if (action == "remove" || action == "restore") {
    item["removed"] = Value(action == "remove");
  } else {
    auto source = options.find("--file");
    if (adding && source == options.end())
      invalid("projects add requires --file PATH");
    if (adding) {
      auto kind =
          options.count("--kind") ? options.at("--kind") : media_kind(fs::u8path(source->second));
      if (kind != "video" && kind != "image" && kind != "music" && kind != "clip")
        invalid("kind must be image, video, music or clip");
      item["kind"] = str(kind);
      if (options.count("--parent"))
        item["parent"] = str(options.at("--parent"));
      if (kind == "clip") {
        if (!item.count("parent"))
          invalid("clips require --parent VIDEO_ID");
        const auto& parent = items[item_index(items, item.at("parent").as_string())];
        if (field(parent, "kind") != "video" || flag(parent, "removed"))
          invalid("parent must be an active video");
        item["position"] = num(double(items.size()));
      } else if (item.count("parent"))
        invalid("--parent requires --kind clip");
    }
    const auto kind = item.at("kind").as_string() == "clip" ? "video" : item.at("kind").as_string();
    // Validate every supplied file before copying any of them.
    for (const char* key : {"--file", "--preview"})
      if (options.count(key)) {
        const auto path = fs::u8path(options.at(key));
        if (media_kind(path) != kind)
          invalid(std::string(key) + " type must match item kind");
        if (!fs::is_regular_file(path))
          throw CliError("INPUT_NOT_FOUND", "media file not found: " + path.u8string(), 3);
      }
    for (const char* key : {"title", "prompt"})
      if (options.count(std::string("--") + key))
        item[key] = str(options.at(std::string("--") + key));
    for (const char* key : {"position", "start", "end"})
      if (options.count(std::string("--") + key)) {
        if (item.at("kind").as_string() != "clip")
          invalid(std::string("--") + key + " requires a clip");
        const auto& raw = options.at(std::string("--") + key);
        size_t used = 0;
        double n = std::stod(raw, &used);
        if (used != raw.size() || !std::isfinite(n) || n < 0)
          invalid(std::string(key) + " must be a nonnegative number");
        item[key] = num(n);
      }
    if (item.count("end") &&
        item.at("end").as_number() <= (item.count("start") ? item.at("start").as_number() : 0))
      invalid("clip end must be greater than start");
    if (options.count("--request")) {
      auto request = read_json(fs::u8path(options.at("--request")));
      if (!request.is_object())
        invalid("--request requires a generation request JSON object");
      item["request"] = request;
      item["request_directory"] = str(fs::current_path().u8string());
    }
    if (!generation.is_null())
      item["generation"] = generation;
    if (source != options.end()) {
      item["file"] = str(
          import_file(directory, fs::u8path(source->second), id, version, "source", keep_local));
      item.erase("preview");
      item["needs_rebuild"] = Value(false);
      if (!adding && generation.is_null())
        item.erase("generation");
    }
    if (options.count("--preview"))
      item["preview"] =
          str(import_file(directory, fs::u8path(options.at("--preview")), id, version, "preview"));
    if (source != options.end() || options.count("--preview"))
      item["media_revision"] = num(project.find("revision")->as_number() + 1);
  }
  if (index == items.size())
    items.emplace_back(item);
  else
    items[index] = Value(item);
  const auto parent = field(Value(item), "parent");
  if (!parent.empty()) {
    auto at = item_index(items, parent);
    auto updated = items[at].as_object();
    updated["needs_rebuild"] = Value(true);
    items[at] = Value(updated);
  }
  auto object = project.as_object();
  object["items"] = Value(items);
  save(directory, object);
  return Value(item);
}
}

std::filesystem::path project_directory(const std::string& name, const std::string& output) {
  identifier(name);
  const auto root = output_root(output);
  return inside(root, fs::u8path(name));
}

std::filesystem::path project_index_path(const fs::path& directory) {
  return inside(directory.parent_path(), directory.filename().u8string() + ".html");
}

int projects_command(const std::vector<std::string>& args) {
  std::vector<std::string> positional;
  Options options;
  const std::set<std::string> allowed = {
      "--output-dir", "--title", "--file", "--preview", "--kind",    "--id",         "--parent",
      "--position",   "--start", "--end",  "--prompt",  "--request", "--if-revision"};
  for (size_t i = 0; i < args.size(); ++i) {
    if (args[i].rfind("--", 0) != 0) {
      positional.push_back(args[i]);
      continue;
    }
    auto key = args[i];
    if (!allowed.count(key) || i + 1 == args.size() || options.count(key))
      invalid("unknown, missing or repeated option: " + key);
    options[key] = args[++i];
  }
  if (positional.empty())
    invalid(
        "use projects create, list, show, open, add, update, remove, restore or render; slopfab help projects");
  const auto action = positional[0];
  const bool edit = action == "update" || action == "remove" || action == "restore";
  const bool known = edit || action == "list" || action == "create" || action == "show" ||
                     action == "render" || action == "add" || action == "open";
  if (!known || positional.size() != (action == "list" ? 1u : edit ? 3u : 2u))
    invalid("wrong project action or arguments; run slopfab help projects");
  std::set<std::string> permitted = {"--output-dir"};
  if (action == "create")
    permitted.insert("--title");
  if (action == "add" || action == "update") {
    for (const char* key : {"--title", "--file", "--preview", "--position", "--start", "--end",
                            "--prompt", "--request", "--if-revision"})
      permitted.insert(key);
    if (action == "add")
      for (const char* key : {"--id", "--kind", "--parent"})
        permitted.insert(key);
  }
  if (action == "remove" || action == "restore")
    permitted.insert("--if-revision");
  for (const auto& option : options)
    if (!permitted.count(option.first))
      invalid(option.first + " is not supported by projects " + action);
  const auto output = options.count("--output-dir") ? options.at("--output-dir") : "";
  Object response{{"type", str("result")}, {"ok", Value(true)}};
  if (action == "list") {
    Array projects;
    const auto root = output_root(output);
    if (fs::is_directory(root))
      for (const auto& entry : fs::directory_iterator(root)) {
        if (!entry.is_directory() || !fs::is_regular_file(entry.path() / "project.json"))
          continue;
        auto directory = project_directory(entry.path().filename().u8string(), output);
        const auto manifest = load(directory);
        projects.emplace_back(Object{{"name", *manifest.find("name")},
                                     {"title", *manifest.find("title")},
                                     {"directory", str(directory.u8string())},
                                     {"index_path", str(project_index_path(directory).u8string())}});
      }
    response["projects"] = Value(projects);
    response["output_directory"] = str(root.u8string());
  } else {
    const auto directory = project_directory(positional[1], output);
    if (action == "create") {
      fs::create_directories(directory.parent_path());
      if (!fs::create_directory(directory))
        throw CliError("OUTPUT_EXISTS",
                       "project folder already exists; use projects show " + positional[1], 6);
      Object manifest{
          {"schema_version", num(1)},
          {"revision", num(1)},
          {"name", str(positional[1])},
          {"title", str(options.count("--title") ? options.at("--title") : positional[1])},
          {"items", Value(Array{})},
          {"naming_rules", naming_rules()}};
      atomic_json(directory / "project.json", Value(manifest));
      render(directory, Value(manifest));
    } else if (action == "render") {
      if (!fs::is_regular_file(directory / "project.json"))
        throw CliError("INPUT_NOT_FOUND", "project not found; use projects create first", 3);
      FileLock lock(inside(directory, ".project.lock"));
      if (!lock.acquired())
        throw CliError("PROJECT_BUSY", "project is being updated; retry", 7, {}, true);
      render(directory, load(directory));
    } else if (action == "add" || edit) {
      std::string id = action == "add" ? options.count("--id") ? options.at("--id")
                                                               : "item-" + unique_id().substr(0, 12)
                                       : positional[2];
      identifier(id);
      response["item"] = change_item(directory, action, id, options);
    }
    response["project"] = context(directory, load(directory));
    if (action == "open") {
      open_index(directory);
      response["opened"] = Value(true);
    }
  }
  emit(std::move(response), true);
  return 0;
}

void prepare_project_generation(GenerateRequest& request, bool explicit_output, int count) {
  if (agent().project.empty())
    return;
  const auto directory = project_directory(agent().project, agent().project_output);
  const auto project = load(directory);
  agent().project_output = directory.parent_path().u8string();
  agent().project_manifest_revision = int(project.find("revision")->as_number());
  if (!agent().project_item.empty()) {
    if (count != 1)
      invalid("--project-item requires --count 1");
    const auto& items = project.find("items")->as_array();
    const auto& item = items[item_index(items, agent().project_item)];
    if (flag(item, "removed"))
      invalid("restore the item before regenerating it");
    auto kind = field(item, "kind");
    if ((kind == "clip" ? "video" : kind) != media_kind(fs::u8path(request.out_path)))
      invalid("generation task must match project item kind");
    agent().project_revision = int(item.find("version")->as_number());
  } else {
    agent().project_new_id = "item-" + unique_id().substr(0, 12);
  }
  const auto id = agent().project_item.empty() ? agent().project_new_id : agent().project_item;
  auto path = explicit_output
                  ? fs::u8path(request.out_path)
                  : fs::path("assets") / id / ("v" + std::to_string(agent().project_revision + 1)) /
                        ("source" + extension(fs::u8path(request.out_path)));
  request.out_path = inside(directory, path).u8string();
  if (agent().overwrite)
    invalid("project generation keeps versions; omit --overwrite and choose a new output path");
}

Value record_project_generation(const GenerateRequest& request, const Value& result) {
  if (agent().project.empty())
    return Value();
  const auto directory = project_directory(agent().project, agent().project_output);
  const auto index = int(result.find("index")->as_number());
  auto id = agent().project_item.empty() ? agent().project_new_id : agent().project_item;
  if (index)
    id += "-" + std::to_string(index + 1);
  Object recipe{{"arguments", Value(agent().generation_arguments)},
                {"prompts", Value(Object{{"prompt", str(request.prompt)},
                                          {"negative_prompt", str(request.negative_prompt)},
                                          {"lyrics", str(request.music.lyrics)}})},
                {"working_directory", str(fs::current_path().u8string())},
                {"plan", *result.find("plan")},
                {"artifacts", *result.find("artifacts")}};
  Options options{{"--file", request.out_path}, {"--prompt", request.prompt}};
  if (!agent().project_item.empty())
    options["--if-revision"] = std::to_string(agent().project_manifest_revision);
  const auto item = change_item(directory, agent().project_item.empty() ? "add" : "update", id,
                                options, Value(recipe), true, agent().project_revision);
  const auto ext = extension(fs::u8path(request.out_path));
  const bool preview_required = ext == ".y4m" || ext == ".ppm";
  return Value(Object{
      {"id", str(id)},
      {"version", *item.find("version")},
      {"preview_required", Value(preview_required)},
      {"next",
       str(preview_required
               ? "Create a browser preview with external FFmpeg (MP4/WebM for Y4M, PNG for PPM); include the generated WAV soundtrack when present. Register it with projects update NAME ID --preview FILE, then open index_path."
               : "Open index_path to review the result and copy edit prompts into your agent.")},
      {"manifest_path", str((directory / "project.json").u8string())},
      {"index_path", str(project_index_path(directory).u8string())}});
}

Value register_remote_generation(const fs::path& directory, const std::string& id,
                                 const std::string& source, const Value& recipe) {
  identifier(id);
  // A successful registration may precede a local receipt update. Retrying a
  // transfer must not create another item or overwrite subsequent local edits.
  auto project = load(directory);
  for (const auto& item : project.find("items")->as_array()) {
    if (field(item, "id") != id) continue;
    const auto* generation = item.find("generation");
    if (generation && field(*generation, "remote_receipt") == field(recipe, "remote_receipt"))
      return item;
    invalid("remote import item ID is already in use: " + id);
  }
  return change_item(directory, "add", id, Options{{"--file", source}}, recipe);
}
}
