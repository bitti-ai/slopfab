#include "remote.h"
#include "remote_transport.h"
#include "commands.h"
#include "agent_instructions.h"
#include "settings.h"
#include "projects.h"
#include "script_host.h"
#include "slopfab/sha256.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iostream>
#include <set>
#include <thread>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

namespace slopfab::cli {
namespace fs = std::filesystem;

std::string field(const Value& v, const char* key, const std::string& fallback = {}) {
  const auto* p = v.find(key);
  return p ? p->as_string() : fallback;
}

std::string remote_digest(const fs::path& path) {
  static const char hex[] = "0123456789abcdef";
  std::string result;
  for (auto byte : sha256_file(path.u8string())) {
    result += hex[byte >> 4];
    result += hex[byte & 15];
  }
  return result;
}

std::string base64(const std::string& bytes) {
  static const char alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  unsigned bits = 0;
  int count = 0;
  for (unsigned char c : bytes) {
    bits = (bits << 8) | c;
    count += 8;
    while (count >= 6) {
      count -= 6;
      out += alphabet[(bits >> count) & 63];
    }
  }
  if (count)
    out += alphabet[(bits << (6 - count)) & 63];
  while (out.size() % 4)
    out += '=';
  return out;
}

void no_nul(const std::string& s) {
  if (s.find('\0') != std::string::npos)
    throw CliError("INVALID_REQUEST", "remote arguments cannot contain NUL");
}

std::string remote_shell_quote(const std::string& s) {
  no_nul(s);
  std::string out = "'";
  for (char c : s)
    out += c == '\'' ? "'\\''" : std::string(1, c);
  return out + "'";
}

std::vector<std::string> remote_ssh(const Value& connection, const std::string& command) {
  const auto host = field(connection, "host");
  if (host.empty() || host.front() == '-' ||
      host.find_first_not_of(
          "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-@:%[]") !=
          std::string::npos)
    throw CliError("INVALID_REQUEST", "--host must be an SSH alias or [user@]hostname");
  return {field(connection, "ssh", "ssh"),
          "-T",
          "-o",
          "BatchMode=yes",
          "-o",
          "ConnectTimeout=15",
          "-o",
          "ServerAliveInterval=15",
          "-o",
          "ServerAliveCountMax=3",
          host,
          command};
}

std::vector<std::string> ssh_arguments(const Value& connection, bool fetch) {
  auto exe = field(connection, "executable", "slopfab");
  no_nul(exe);
  if (exe.empty())
    throw CliError("INVALID_REQUEST", "remote executable is empty");
  const auto platform = field(connection, "platform", "posix");
  const std::string action = fetch ? "fetch" : "run";
  std::string command;
  if (platform == "windows") {
    // Start the native process with inherited binary handles. A PowerShell pipeline
    // would decode/re-encode stdout and corrupt downloaded media.
    const std::string script =
        "$ErrorActionPreference='Stop';$p=New-Object System.Diagnostics.Process;"
        "$p.StartInfo.UseShellExecute=$false;$p.StartInfo.FileName="
        "[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('" +
        base64(exe) + "'));$p.StartInfo.Arguments='_ssh-" + action +
        "';[void]$p.Start();$p.WaitForExit();exit $p.ExitCode";
    std::string wide;
    for (char c : script) {
      wide += c;
      wide += '\0';
    }
    command = "powershell.exe -NoLogo -NoProfile -NonInteractive -EncodedCommand " + base64(wide);
  } else if (platform == "posix" || platform == "linux") {
    command = "exec " + remote_shell_quote(exe) + " _ssh-" + action;
  } else {
    throw CliError("INVALID_REQUEST", "--platform must be posix or windows");
  }
  return remote_ssh(connection, command);
}

struct Scratch {
  fs::path directory = fs::temp_directory_path() / ("slopfab-ssh-" + unique_id());

  Scratch() {
    fs::create_directory(directory);
  }

  ~Scratch() {
    std::error_code ec;
    fs::remove_all(directory, ec);
  }
};

// Tail spooled streams while the child is running. ScriptProcesses handles
// cancellation and process cleanup without blocking on either SSH pipe.
int remote_process(const std::vector<std::string>& argv, const Value& input, const fs::path& output,
                   bool fetch, const std::function<void(const Value&)>& event) {
  Scratch scratch;
  auto errors = scratch.directory / "stderr.log";
  ScriptProcesses processes([] {
    if (cancelled())
      throw_cancelled();
  });
  auto options = input.as_object();
  options["stdoutFile"] = str(output.u8string());
  options["stderrFile"] = str(errors.u8string());
  const auto child = processes.spawn(argv, Value(options));
  std::ifstream out, err(errors, std::ios::binary);
  if (!fetch)
    out.open(output, std::ios::binary);
  std::string pending;
  auto drain = [&] {
    char bytes[8192];
    err.clear();
    while (err.read(bytes, sizeof(bytes)) || err.gcount()) {
      std::fwrite(bytes, 1, size_t(err.gcount()), stderr);
      std::fflush(stderr);
    }
    if (fetch)
      return;
    out.clear();
    while (out.read(bytes, sizeof(bytes)) || out.gcount()) {
      pending.append(bytes, size_t(out.gcount()));
      size_t end;
      while ((end = pending.find('\n')) != std::string::npos) {
        if (end > 16 * 1024 * 1024)
          throw CliError("REMOTE_PROTOCOL", "remote event exceeds 16 MiB", 6);
        auto line = pending.substr(0, end);
        pending.erase(0, end + 1);
        if (!line.empty() && line.back() == '\r')
          line.pop_back();
        if (!line.empty()) {
          Value value;
          try {
            value = parse_cli_json(line);
          } catch (const std::exception& e) {
            throw CliError("REMOTE_PROTOCOL", std::string("invalid remote event: ") + e.what(), 6);
          }
          if (!value.is_object())
            throw CliError("REMOTE_PROTOCOL", "remote event must be an object", 6);
          event(value);
        }
      }
      if (pending.size() > 16 * 1024 * 1024)
        throw CliError("REMOTE_PROTOCOL", "remote event exceeds 16 MiB", 6);
    }
  };
  for (;;) {
    auto status = processes.poll(int(child.find("id")->as_number()), false);
    drain();
    if (status.find("done")->as_bool()) {
      if (!pending.empty())
        throw CliError("REMOTE_PROTOCOL", "incomplete remote event", 6);
      return int(status.find("exitCode")->as_number());
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
  }
}

int exchange(const Value& connection, const Value& request, const fs::path& output, bool fetch,
             const std::function<void(const Value&)>& event = {}) {
  return remote_process(ssh_arguments(connection, fetch),
                        Value(Object{{"stdin", str(json::stringify(request))}}), output, fetch,
                        event);
}

std::string remote_ps_string(const std::string& value) {
  no_nul(value);
  return "[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('" + base64(value) + "'))";
}

std::string remote_powershell(const std::string& script) {
  // Callers construct ASCII scripts, with any UTF-8 strings separately encoded.
  std::string wide;
  for (char c : "$ProgressPreference='SilentlyContinue';" + script) {
    wide += c;
    wide += '\0';
  }
  return "powershell.exe -NoLogo -NoProfile -NonInteractive -EncodedCommand " + base64(wide);
}

std::string filename(std::string path) {
  std::replace(path.begin(), path.end(), '\\', '/');
  auto name = path.substr(path.find_last_of('/') + 1);
  // The same portable output contract on Windows and Linux. Never interpret
  // remote paths as local paths, drive names, ADS, or traversal components.
  if (name.empty() || name == "." || name == ".." || name.back() == '.' || name.back() == ' ' ||
      name.find_first_of(":<>\"|?*") != std::string::npos ||
      std::any_of(name.begin(), name.end(), [](unsigned char c) {
        return c < 32;
      }))
    throw CliError("REMOTE_PROTOCOL", "nonportable remote artifact filename", 6);
  std::string stem = name.substr(0, name.find('.'));
  std::transform(stem.begin(), stem.end(), stem.begin(), [](unsigned char c) {
    return char(std::tolower(c));
  });
  if (stem == "con" || stem == "prn" || stem == "aux" || stem == "nul" ||
      (stem.size() == 4 && (stem.substr(0, 3) == "com" || stem.substr(0, 3) == "lpt") &&
       stem[3] >= '0' && stem[3] <= '9'))
    throw CliError("REMOTE_PROTOCOL", "reserved remote artifact filename", 6);
  return name;
}

fs::path local_path(const fs::path& root, const std::string& relative) {
  auto p = fs::u8path(relative);
  if (p.is_absolute() || p.has_root_name() || relative.find('\\') != std::string::npos)
    throw CliError("REMOTE_PROTOCOL", "invalid local artifact path", 6);
  for (const auto& part : p)
    filename(part.u8string());
  auto resolved = fs::weakly_canonical(root / p);
  auto rel = resolved.lexically_relative(fs::weakly_canonical(root));
  if (rel.empty() || rel == "." || *rel.begin() == "..")
    throw CliError("REMOTE_PROTOCOL", "artifact escapes local download directory", 6);
  return resolved;
}

Value localize(const Value& value, const std::map<std::string, std::string>& paths) {
  if (value.is_string()) {
    auto found = paths.find(value.as_string());
    return found == paths.end() ? value : str(found->second);
  }
  if (value.is_array()) {
    Array result;
    for (const auto& child : value.as_array())
      result.push_back(localize(child, paths));
    return Value(result);
  }
  if (value.is_object()) {
    Object result;
    for (const auto& child : value.as_object())
      result[child.first] = localize(child.second, paths);
    return Value(result);
  }
  return value;
}

Value collect(const Value& receipt, const fs::path& directory, bool provisioned = false) {
  auto manifest = receipt.as_object();
  const auto& connection = manifest.at("connection");
  auto artifacts = manifest.at("artifacts").as_array();
  // Validate the entire manifest before starting any transfers.
  std::set<fs::path> destinations;
  for (const auto& artifact : artifacts) {
    auto path = local_path(directory, field(artifact, "local"));
    if (!destinations.insert(path).second)
      throw CliError("REMOTE_PROTOCOL", "duplicate local artifact path", 6);
    const auto* bytes = artifact.find("bytes");
    const auto hash = field(artifact, "sha256");
    if (field(artifact, "remote").empty() || !bytes || !bytes->is_number() ||
        !std::isfinite(bytes->as_number()) || bytes->as_number() < 0 ||
        bytes->as_number() > 9007199254740991.0 ||
        std::floor(bytes->as_number()) != bytes->as_number() || hash.size() != 64 ||
        hash.find_first_not_of("0123456789abcdef") != std::string::npos)
      throw CliError("REMOTE_PROTOCOL", "invalid artifact metadata", 6);
  }
  for (auto& artifact : artifacts) {
    auto entry = artifact.as_object();
    auto path = local_path(directory, entry.at("local").as_string());
    if (entry.count("downloaded") && entry.at("downloaded").as_bool()) {
      if (!fs::is_regular_file(path) ||
          double(fs::file_size(path)) != entry.at("bytes").as_number() ||
          remote_digest(path) != entry.at("sha256").as_string())
        throw CliError("OUTPUT_EXISTS", "downloaded artifact changed: " + path.u8string(), 6);
      continue;
    }
    if (fs::exists(path)) {
      // Recover a crash after publication but before the receipt was updated.
      if (!fs::is_regular_file(path) ||
          double(fs::file_size(path)) != entry.at("bytes").as_number() ||
          remote_digest(path) != entry.at("sha256").as_string())
        throw CliError("OUTPUT_EXISTS", "refusing to replace local artifact: " + path.u8string(),
                       6);
      entry["downloaded"] = Value(true);
      artifact = Value(entry);
      manifest["artifacts"] = Value(artifacts);
      atomic_json(directory / "remote.json", Value(manifest));
      continue;
    }
    fs::create_directories(path.parent_path());
    const auto temporary = path.parent_path() / (".download-" + unique_id() + ".tmp");
    try {
      if (!provisioned) {
        manifest["connection"] = remote_provision(manifest.at("connection"));
        provisioned = true;
      }
      const int status = exchange(
          connection, Value(Object{{"path", entry.at("remote")}, {"bytes", entry.at("bytes")}}),
          temporary, true);
      if (status || double(fs::file_size(temporary)) != entry.at("bytes").as_number() ||
          remote_digest(temporary) != entry.at("sha256").as_string())
        throw CliError("TRANSFER_FAILED",
                       "SSH download failed; retry with slopfab fetch " +
                           (directory / "remote.json").u8string(),
                       6, {}, true);
      // Hard-link publication is atomic and fails if another writer claimed the
      // destination. The temporary file is always on the destination filesystem.
      fs::create_hard_link(temporary, path);
      fs::remove(temporary);
    } catch (...) {
      std::error_code ec;
      fs::remove(temporary, ec);
      throw;
    }
    entry["downloaded"] = Value(true);
    artifact = Value(entry);
    manifest["artifacts"] = Value(artifacts);
    atomic_json(directory / "remote.json", Value(manifest));
    emit({{"type", str("download")}, {"path", str(path.u8string())}, {"bytes", entry.at("bytes")}});
  }
  if (manifest.count("local_project")) {
    Array items;
    const auto project = fs::u8path(manifest.at("local_project").as_string());
    const auto* response = &manifest.at("remote_result");
    if (const auto* job = response->find("job"))
      if (const auto* result = job->find("result"))
        response = result;
    const auto* results = response->find("results");
    if (results) {
      size_t index = 0;
      for (const auto& result : results->as_array()) {
        const auto* plan = result.find("plan");
        if (!plan)
          continue;
        std::string source;
        Array files;
        std::set<std::string> result_paths;
        if (const auto* result_files = result.find("artifacts"))
          for (const auto& file : result_files->as_array())
            result_paths.insert(field(file, "path"));
        for (const auto& artifact : artifacts) {
          if (!result_paths.count(field(artifact, "remote")))
            continue;
          const auto local = local_path(directory, field(artifact, "local")).u8string();
          if (field(artifact, "remote") == field(*plan, "output"))
            source = local;
          files.emplace_back(Object{{"path", str(local)},
                                    {"remote_path", str(field(artifact, "remote"))},
                                    {"media_type", str(field(artifact, "media_type"))}});
        }
        if (source.empty())
          throw CliError("REMOTE_PROTOCOL", "primary project artifact is missing", 6);
        const auto id = "remote-" + directory.filename().u8string().substr(0, 12) + "-" +
                        std::to_string(++index);
        Value recipe(Object{{"remote_receipt", str((directory / "remote.json").u8string())},
                            {"host", str(field(connection, "host"))},
                            {"working_directory", manifest.at("remote_directory")},
                            {"arguments", manifest.at("arguments")},
                            {"plan", *plan},
                            {"artifacts", Value(files)}});
        items.push_back(register_remote_generation(project, id, source, recipe));
      }
    }
    manifest["project_items"] = Value(items);
    manifest["index_path"] = str(project_index_path(project).u8string());
  }
  std::map<std::string, std::string> paths;
  auto remote_root = field(Value(manifest), "remote_directory");
  std::replace(remote_root.begin(), remote_root.end(), '\\', '/');
  if (!remote_root.empty() && remote_root.back() != '/')
    remote_root += '/';
  for (const auto& artifact : artifacts) {
    auto remote = field(artifact, "remote");
    const auto local = local_path(directory, field(artifact, "local")).u8string();
    paths[remote] = local;
    std::replace(remote.begin(), remote.end(), '\\', '/');
    paths[remote] = local;
    if (!remote_root.empty() && remote.rfind(remote_root, 0) == 0) {
      auto relative = remote.substr(remote_root.size());
      paths[relative] = local;
      paths["./" + relative] = local;
      std::replace(relative.begin(), relative.end(), '/', '\\');
      paths[relative] = local;
      paths[".\\" + relative] = local;
    }
  }
  // Preserve the regular command's response fields for callers using the same
  // CLI locally and remotely. Keep original server paths in remote_result.
  for (const auto& entry : manifest.at("remote_result").as_object())
    if (!manifest.count(entry.first))
      manifest[entry.first] = localize(entry.second, paths);
  manifest["type"] = str("result");
  manifest["ok"] = Value(true);
  manifest["receipt"] = str((directory / "remote.json").u8string());
  // Persist the finished project registration as well as the transfer state.
  atomic_json(directory / "remote.json", Value(manifest));
  return Value(manifest);
}

int remote_fetch(const std::vector<std::string>& args) {
  start_request();
  if (args.size() == 1) {
    const auto receipt = fs::absolute(fs::u8path(args[0]));
    if (receipt.filename() != "remote.json")
      throw CliError("INVALID_REQUEST", "fetch expects a remote.json receipt");
    FileLock lock(receipt.parent_path() / ".remote.lock");
    if (!lock.acquired())
      throw CliError("REMOTE_BUSY", "download already running", 7);
    auto value = read_json(receipt);
    if (!value.find("schema_version") || value.find("schema_version")->as_number() != 1)
      throw CliError("INVALID_REQUEST", "unsupported remote receipt");
    emit(collect(value, receipt.parent_path()).as_object(), true);
    return 0;
  }
  throw CliError("INVALID_REQUEST", "use slopfab fetch PATH/remote.json");
}

int remote_execute(const std::string& host, const std::vector<std::string>& args,
                   const std::string& destination, const std::string& local_project,
                   const std::vector<std::string>& downloads) {
  start_request();
  if (args.empty() || args[0] == "remote" || args[0] == "fetch" || args[0].rfind("_ssh-", 0) == 0 ||
      args[0] == "worker")
    throw CliError("INVALID_REQUEST", "select a regular command for remote execution");
  if (args.size() > 1 && args[0] == "projects" && args[1] == "open")
    throw CliError(
        "INVALID_REQUEST",
        "projects open launches a local browser; use projects show --host NAME or open a downloaded project locally");
  fs::path project;
  if (!local_project.empty()) {
    if (args[0] != "generate" && args[0] != "wait" && args[0] != "status")
      throw CliError("INVALID_REQUEST", "--local-project applies to generation results");
    project = project_directory(local_project);
    read_json(project / "project.json");
  }
  const auto connection = remote_provision(remote_profile(host)).as_object();
  const auto remote_directory = connection.at("directory").as_string();
  Array arguments, requested_files;
  for (const auto& arg : args) {
    no_nul(arg);
    arguments.push_back(str(arg));
  }
  for (const auto& file : downloads) {
    no_nul(file);
    requested_files.push_back(str(file));
  }
  const auto directory =
      fs::absolute(fs::u8path(destination.empty() ? "output/remote" : destination)) / unique_id();
  fs::create_directories(directory);
  FileLock lock(directory / ".remote.lock");
  Scratch scratch;
  Value terminal;
  Value ready;
  emit({{"type", str("remote_started")},
        {"directory", str(directory.u8string())},
        {"host", connection.at("host")}});
  const auto status =
      exchange(Value(connection),
               Value(Object{{"directory", str(remote_directory)},
                            {"arguments", Value(arguments)},
                            {"downloads", Value(requested_files)}}),
               scratch.directory / "events.ndjson", false, [&](const Value& event) {
                 const auto type = field(event, "type");
                 if (type == "remote_manifest")
                   ready = event;
                 else if (type == "remote_response") {
                   if (!event.find("response") || !event.find("response")->is_object())
                     throw CliError("REMOTE_PROTOCOL", "invalid remote response", 6);
                   terminal = *event.find("response");
                 } else if (type == "result" || type == "error")
                   terminal = event;
                 else
                   emit({{"type", str("remote_event")}, {"event", event}});
               });
  if (status || terminal.is_null() || (terminal.find("ok") && !terminal.find("ok")->as_bool())) {
    Object error{
        {"type", str("error")},
        {"ok", Value(false)},
        {"remote", terminal},
        {"error",
         Value(Object{{"code", str("REMOTE_FAILED")},
                      {"message",
                       str("remote command or SSH failed; inspect remote outputs before retrying")},
                      {"exit_code", num(status)}})}};
    emit(std::move(error), true);
    return status > 0 && status < 256 ? status : 1;
  }
  if (ready.is_null() || !ready.find("artifacts") || !ready.find("artifacts")->is_array())
    throw CliError("REMOTE_PROTOCOL",
                   "remote did not report an artifact manifest; update both executables", 6);
  Array artifacts;
  std::map<std::string, size_t> groups;
  std::set<std::string> remote_paths;
  for (const auto& artifact : ready.find("artifacts")->as_array()) {
    const auto path = field(artifact, "path");
    if (!remote_paths.insert(path).second)
      continue;
    const auto bytes = artifact.find("bytes");
    if (!bytes || !bytes->is_number() || !std::isfinite(bytes->as_number()) ||
        bytes->as_number() < 0 || bytes->as_number() > 9007199254740991.0 ||
        std::floor(bytes->as_number()) != bytes->as_number())
      throw CliError("REMOTE_PROTOCOL", "invalid artifact size", 6);
    const auto hash = field(artifact, "sha256");
    if (hash.size() != 64 || hash.find_first_not_of("0123456789abcdef") != std::string::npos)
      throw CliError("REMOTE_PROTOCOL", "invalid artifact checksum", 6);
    auto normalized = path;
    std::replace(normalized.begin(), normalized.end(), '\\', '/');
    auto parent = normalized.substr(0, normalized.find_last_of('/'));
    auto group = groups.emplace(parent, groups.size()).first->second;
    // Preserve video/audio sidecars together; distinguish equal basenames from
    // different remote directories without interpreting a remote absolute path.
    artifacts.emplace_back(Object{{"remote", str(path)},
                                  {"bytes", *bytes},
                                  {"local", str(std::to_string(group) + "/" + filename(path))},
                                  {"media_type", str(field(artifact, "media_type"))},
                                  {"sha256", str(hash)},
                                  {"downloaded", Value(false)}});
  }
  Object receipt_fields{{"schema_version", num(1)},
                        {"connection", Value(connection)},
                        {"remote_directory", str(remote_directory)},
                        {"arguments", Value(arguments)},
                        {"remote_result", terminal},
                        {"artifacts", Value(artifacts)}};
  if (!project.empty())
    receipt_fields["local_project"] = str(project.u8string());
  Value receipt(receipt_fields);
  atomic_json(directory / "remote.json", receipt);
  emit(collect(receipt, directory, true).as_object(), true);
  return 0;
}

// Traverse only result containers and explicit artifact descriptors. Never copy
// model/reference paths or scan the working directory for changed files.
void result_artifacts(const Value& value, std::set<std::string>& paths) {
  if (value.is_array()) {
    for (const auto& child : value.as_array())
      result_artifacts(child, paths);
    return;
  }
  if (!value.is_object())
    return;
  if (value.find("media_type") && value.find("path") && value.find("path")->is_string())
    paths.insert(field(value, "path"));
  if (const auto* output = value.find("output"); output && output->is_string())
    paths.insert(output->as_string());
  for (const char* key : {"artifacts", "results", "result", "data", "job"})
    if (const auto* child = value.find(key))
      result_artifacts(*child, paths);
}

int remote_worker(const std::string& executable, bool fetch) {
  try {
    std::string input;
    char bytes[65536];
    while (std::cin.read(bytes, sizeof(bytes)) || std::cin.gcount()) {
      input.append(bytes, size_t(std::cin.gcount()));
      if (input.size() > 16 * 1024 * 1024)
        throw CliError("INVALID_REQUEST", "remote request exceeds 16 MiB");
    }
    auto request = parse_cli_json(input);
    if (fetch) {
#ifdef _WIN32
      _setmode(_fileno(stdout), _O_BINARY);
#endif
      const auto path = fs::u8path(field(request, "path"));
      if (!path.is_absolute() || !fs::is_regular_file(path) || !request.find("bytes") ||
          double(fs::file_size(path)) != request.find("bytes")->as_number())
        throw CliError("INPUT_NOT_FOUND", "remote artifact missing or size changed", 3);
      std::ifstream file(path, std::ios::binary);
      if (!file)
        throw CliError("INPUT_NOT_FOUND", "cannot open remote artifact", 3);
      while (file.read(bytes, sizeof(bytes)) || file.gcount())
        if (std::fwrite(bytes, 1, size_t(file.gcount()), stdout) != size_t(file.gcount()))
          throw CliError("TRANSFER_FAILED", "artifact stream closed", 6);
      if (file.bad() || std::fflush(stdout))
        throw CliError("TRANSFER_FAILED", "artifact stream failed", 6);
      return 0;
    }
    const auto directory = fs::u8path(field(request, "directory"));
    if (!directory.is_absolute())
      throw CliError("INVALID_REQUEST", "remote directory must be absolute");
    fs::current_path(directory);
    initialize_agent(true, executable);
    ensure_agent_instructions();
    ensure_settings();
    std::vector<std::string> arguments;
    const auto* values = request.find("arguments");
    if (!values)
      throw CliError("INVALID_REQUEST", "missing generation arguments");
    for (const auto& value : values->as_array()) {
      no_nul(value.as_string());
      arguments.push_back(value.as_string());
    }
    if (arguments.empty() || arguments[0] == "remote" || arguments[0] == "fetch" ||
        arguments[0] == "worker" || arguments[0].rfind("_ssh-", 0) == 0)
      throw CliError("INVALID_REQUEST", "invalid remote command");
    std::vector<std::string> child{executable, "--agent"};
    child.insert(child.end(), arguments.begin(), arguments.end());
    Scratch scratch;
    Value response;
    std::set<std::string> paths;
    const int status = remote_process(child, Value(Object{}), scratch.directory / "events.ndjson",
                                      false, [&](const Value& event) {
                                        response = event;
                                        if (field(event, "type") == "artifact")
                                          result_artifacts(event, paths);
                                        emit(event.as_object());
                                      });
    if (response.is_null())
      throw CliError("REMOTE_PROTOCOL", "command produced no response", 6);
    emit({{"type", str("remote_response")}, {"ok", Value(status == 0)}, {"response", response}});
    if (status)
      return status;
    result_artifacts(response, paths);
    // Older diagnostic writers return text rather than artifact metadata. Their
    // explicit output flags are known outputs, unlike arbitrary input paths.
    const std::set<std::string> legacy{"decode",      "upscale",       "upscale-latents",
                                       "encode-text", "encode-refmod", "bundle-refmods",
                                       "prepare-lora"};
    if (legacy.count(arguments[0])) {
      bool has_output = false;
      for (size_t i = 1; i + 1 < arguments.size(); ++i) {
        const auto option = arguments[i];
        if (option == "--out" || option == "--output" || option == "--dump") {
          paths.insert(arguments[++i]);
          has_output = true;
        } else if (option_takes_value(option, arguments[0]))
          ++i;
      }
      if (arguments[0] == "decode" && !has_output)
        paths.insert("out.y4m");
    }
    if (const auto* files = request.find("downloads"))
      for (const auto& file : files->as_array())
        paths.insert(file.as_string());
    Array artifacts;
    for (const auto& name : paths) {
      no_nul(name);
      const auto path = fs::absolute(fs::u8path(name));
      if (!fs::is_regular_file(path))
        throw CliError("OUTPUT_FAILED", "reported remote output is missing: " + path.u8string(), 6);
      artifacts.emplace_back(Object{{"path", str(path.u8string())},
                                    {"bytes", num(double(fs::file_size(path)))},
                                    {"sha256", str(remote_digest(path))},
                                    {"media_type", str("application/octet-stream")}});
    }
    emit({{"type", str("remote_manifest")}, {"artifacts", Value(artifacts)}}, true);
    return 0;
  } catch (const std::exception& e) {
    if (fetch) {
      std::fprintf(stderr, "slopfab remote fetch: %s\n", e.what());
      return 6;
    }
    if (!agent().enabled)
      initialize_agent(true, executable);
    return report_exception(e);
  }
}
}
