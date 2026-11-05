#include "remote.h"
#include "remote_transport.h"
#include <algorithm>
#include <cctype>
#include <fstream>
#include <set>

namespace slopfab::cli {
namespace {
namespace fs = std::filesystem;

std::string get(const Value& value, const char* key, const std::string& fallback = {}) {
  const auto* v = value.find(key);
  return v ? v->as_string() : fallback;
}

fs::path profiles_path() {
  auto path = fs::absolute(config_path());
  path += ".remotes.json";
  return path;
}

Value profiles() {
  const auto path = profiles_path();
  return fs::exists(path) ? read_json(path) : Value(Object{});
}

void identifier(const std::string& name) {
  if (name.empty() || name.size() > 80 || !std::isalnum(static_cast<unsigned char>(name[0])) ||
      name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-") !=
          std::string::npos)
    throw CliError(
        "INVALID_REQUEST",
        "remote name must start with a letter/digit and contain only letters, digits, ., _ or -");
}

Value normalize(Value value) {
  auto object = value.as_object();
  auto directory = get(value, "directory");
  auto platform = get(value, "platform");
  if (platform.empty())
    platform = directory.size() > 2 && directory[1] == ':' ? "windows" : "linux";
  if (platform == "posix")
    platform = "linux";
  if (platform != "windows" && platform != "linux")
    throw CliError("INVALID_REQUEST", "platform must be windows or linux");
  if (directory.find('\0') != std::string::npos || directory.find('\n') != std::string::npos ||
      directory.find('\r') != std::string::npos)
    throw CliError("INVALID_REQUEST", "invalid remote directory");
  if (platform == "windows")
    std::replace(directory.begin(), directory.end(), '\\', '/');
  const bool absolute = platform == "windows"
                            ? directory.size() >= 3 &&
                                  std::isalpha(static_cast<unsigned char>(directory[0])) &&
                                  directory[1] == ':' && directory[2] == '/'
                            : !directory.empty() && directory[0] == '/';
  if (!absolute)
    throw CliError("INVALID_REQUEST",
                   "remote directory must be an absolute path on the remote platform");
  while (directory.size() > (platform == "windows" ? 3u : 1u) && directory.back() == '/')
    directory.pop_back();
  object["directory"] = str(directory);
  object["platform"] = str(platform);
  remote_ssh(Value(object), ""); // Validate host and SSH arguments before connecting.
  return Value(object);
}

fs::path bundled_binary(const Value& connection) {
  const auto platform = get(connection, "platform");
  const auto name = platform == "windows" ? "slopfab.exe" : "slopfab";
  const auto override = get(connection, "binary_directory");
  auto executable = fs::absolute(fs::u8path(agent().executable));
  std::vector<fs::path> roots;
  if (!override.empty())
    roots.push_back(fs::u8path(override));
  else {
    roots.push_back(executable.parent_path());
    roots.push_back(executable.parent_path().parent_path());
  }
  for (const auto& root : roots) {
    auto binary = root / platform / name;
    if (fs::is_regular_file(binary))
      return binary;
  }
  throw CliError("REMOTE_BINARY_MISSING",
                 "package is missing " + platform + "/" + name +
                     "; use the dual-platform package or remote setup --binary-dir PACKAGE_ROOT",
                 3);
}

struct Temporary {
  fs::path path = fs::temp_directory_path() / ("slopfab-provision-" + unique_id() + ".log");

  ~Temporary() {
    std::error_code ec;
    fs::remove(path, ec);
  }
};
}

Value remote_profile(const std::string& name) {
  identifier(name);
  const auto all = profiles();
  const auto* value = all.find(name);
  if (!value)
    throw CliError(
        "REMOTE_NOT_FOUND",
        "unknown remote: " + name + "; run slopfab remote setup " + name + " --directory PATH", 3);
  return normalize(*value);
}

Value remote_provision(const Value& profile) {
  auto connection = normalize(profile).as_object();
  const auto binary = bundled_binary(Value(connection));
  const bool windows = connection.at("platform").as_string() == "windows";
  {
    std::ifstream input(binary, std::ios::binary);
    char magic[4]{};
    input.read(magic, 4);
    if (!input ||
        (windows ? !(magic[0] == 'M' && magic[1] == 'Z')
                 : !(magic[0] == '\x7f' && magic[1] == 'E' && magic[2] == 'L' && magic[3] == 'F')))
      throw CliError("REMOTE_BINARY_INVALID", "bundled binary does not match the remote platform",
                     3);
  }
  const auto hash = remote_digest(binary);
  const auto root = connection.at("directory").as_string();
  const auto directory = root + "/.slopfab/bin/" + hash;
  const auto executable = directory + (windows ? "/slopfab.exe" : "/slopfab");
  connection["executable"] = str(executable);
  connection["binary_sha256"] = str(hash);
  const auto q = remote_shell_quote;
  std::string probe;
  if (windows) {
    probe = remote_powershell(
        "$ErrorActionPreference='Stop';$root=" + remote_ps_string(root) +
        ";$exe=" + remote_ps_string(executable) +
        ";if(!(Test-Path -LiteralPath $root -PathType Container)){throw 'Remote project directory does not exist'};"
        "$current=(Test-Path -LiteralPath $exe -PathType Leaf) -and ((Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash -eq '" +
        hash +
        "');if($current){[Console]::Out.WriteLine('{\"current\":true}')}else{[Console]::Out.WriteLine('{\"current\":false}')}");
  } else {
    probe =
        "set -eu; test -d " + q(root) + "; if test -x " + q(executable) +
        " && test \"$(sha256sum -- " + q(executable) + " | cut -d ' ' -f 1)\" = " + q(hash) +
        "; then printf '%s\\n' '{\"current\":true}'; else printf '%s\\n' '{\"current\":false}'; fi";
  }
  Temporary output;
  bool current = false, received = false;
  const int checked = remote_process(remote_ssh(Value(connection), probe), Value(Object{}),
                                     output.path, false, [&](const Value& response) {
                                       if (const auto* v = response.find("current")) {
                                         current = v->as_bool();
                                         received = true;
                                       }
                                     });
  if (checked || !received)
    throw CliError("REMOTE_SETUP_FAILED",
                   "cannot verify remote executable; check SSH, project directory and platform",
                   checked ? checked : 6);
  if (current)
    return Value(connection);

  // Each binary has its own directory: old workers continue running their
  // executable, including on Windows. Only a verified upload is published.
  const auto pending = directory + "/.upload-" + unique_id();
  std::string install;
  if (windows) {
    install = remote_powershell(
        "$ErrorActionPreference='Stop';$dir=" + remote_ps_string(directory) +
        ";$tmp=" + remote_ps_string(pending) + ";$exe=" + remote_ps_string(executable) +
        ";[void][IO.Directory]::CreateDirectory($dir);try {"
        "$out=[IO.File]::Open($tmp,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None);"
        "try{[Console]::OpenStandardInput().CopyTo($out)}finally{$out.Dispose()};"
        "if((Get-FileHash -LiteralPath $tmp -Algorithm SHA256).Hash -ne '" +
        hash +
        "'){throw 'Binary checksum mismatch'};"
        "if(Test-Path -LiteralPath $exe){"
        "if((Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash -ne '" +
        hash +
        "'){$backup=$tmp+'.old';[IO.File]::Replace($tmp,$exe,$backup);[IO.File]::Delete($backup)}"
        "}else{[IO.File]::Move($tmp,$exe)}"
        "}finally{if(Test-Path -LiteralPath $tmp){[IO.File]::Delete($tmp)}}");
  } else {
    install = "set -eu; umask 077; mkdir -p -- " + q(directory) + "; tmp=" + q(pending) +
              "; trap 'rm -f -- \"$tmp\"' EXIT HUP INT TERM; cat > \"$tmp\"; "
              "test \"$(sha256sum -- \"$tmp\" | cut -d ' ' -f 1)\" = " +
              q(hash) + "; chmod 700 \"$tmp\"; mv -f -- \"$tmp\" " + q(executable);
  }
  emit({{"type", str("remote_update")}, {"host", connection.at("host")}, {"sha256", str(hash)}});
  const int installed =
      remote_process(remote_ssh(Value(connection), install),
                     Value(Object{{"stdinFile", str(binary.u8string())}}), output.path, true);
  if (installed)
    throw CliError("REMOTE_UPDATE_FAILED",
                   "remote executable upload failed; existing versions were retained", installed);
  return Value(connection);
}

int remote_command(const std::vector<std::string>& args) {
  start_request();
  if (args == std::vector<std::string>{"list"}) {
    emit({{"type", str("result")}, {"ok", Value(true)}, {"remotes", profiles()}}, true);
    return 0;
  }
  if (args.size() < 2)
    throw CliError("INVALID_REQUEST", "use remote setup|show|update|remove NAME, or remote list");
  const auto& action = args[0];
  const auto& name = args[1];
  identifier(name);
  if ((action == "show" || action == "update") && args.size() == 2) {
    auto connection = remote_profile(name);
    if (action == "update")
      connection = remote_provision(connection);
    emit(
        {{"type", str("result")}, {"ok", Value(true)}, {"name", str(name)}, {"remote", connection}},
        true);
    return 0;
  }
  if (action != "setup" && !(action == "remove" && args.size() == 2))
    throw CliError("INVALID_REQUEST",
                   "remote manages hosts; use COMMAND --host NAME for remote execution");
  Value profile;
  if (action == "setup") {
    Object options{{"host", str(name)}};
    std::set<std::string> seen;
    for (size_t i = 2; i < args.size(); ++i) {
      const auto option = args[i];
      if (++i == args.size() || !seen.insert(option).second)
        throw CliError("INVALID_REQUEST", "missing or duplicate remote option: " + option);
      const auto& value = args[i];
      if (option == "--host")
        options["host"] = str(value);
      else if (option == "--directory")
        options["directory"] = str(value);
      else if (option == "--platform")
        options["platform"] = str(value);
      else if (option == "--ssh")
        options["ssh"] = str(value);
      else if (option == "--binary-dir")
        options["binary_directory"] = str(fs::absolute(fs::u8path(value)).u8string());
      else
        throw CliError("INVALID_REQUEST", "unknown remote setup option: " + option);
    }
    profile = normalize(Value(options));
    remote_provision(profile);
  }
  auto path = profiles_path();
  fs::create_directories(path.parent_path());
  auto lock_path = path;
  lock_path += ".lock";
  FileLock lock(lock_path);
  if (!lock.acquired())
    throw CliError("REMOTE_BUSY", "remote profiles are being updated", 7);
  auto all = profiles().as_object();
  if (action == "remove") {
    if (!all.erase(name))
      throw CliError("REMOTE_NOT_FOUND", "unknown remote: " + name, 3);
  } else
    all[name] = profile;
  atomic_json(path, Value(all));
  emit({{"type", str("result")}, {"ok", Value(true)}, {"name", str(name)}, {"remote", profile}},
       true);
  return 0;
}
}
