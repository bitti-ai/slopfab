#include "update.h"
#include "update_package.h"
#include "agent.h"
#include "commands.h"
#include "script_host.h"
#include <fstream>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace slopfab::cli {
namespace fs = std::filesystem;

namespace {
const char* executable_name() {
#ifdef _WIN32
  return "slopfab.exe";
#else
  return "slopfab";
#endif
}

void check_cancelled() {
  if (cancelled())
    throw_cancelled();
}

Value http_options(uint64_t timeout) {
  return Value(
      Object{{"timeoutMs", num(double(timeout))},
             {"maxBytes", num(4 * 1024 * 1024)},
             {"headers", Value(Object{{"Accept", str("application/vnd.github+json")},
                                      {"X-GitHub-Api-Version", str("2022-11-28")},
                                      {"User-Agent", str(std::string("slopfab/") + kVersion)}})}});
}

Object result(const std::string& status, const updater::Release& release) {
  return {{"type", str("result")},
          {"ok", Value(true)},
          {"status", str(status)},
          {"current_version", str(kVersion)},
          {"latest_version", str(release.version)},
          {"package_url", str(release.url)}};
}

void install(const fs::path& root, const fs::path& stage, const Value& plan,
             const fs::path& payload) {
  updater::check_path(root, fs::u8path(executable_name()));
  if (updater::file_hash(root / executable_name()) != script_string(plan, "previous_sha256"))
    throw CliError("UPDATE_CONFLICT",
                   "Installed executable changed while the update was staged; run update again");
  updater::install_package(root, payload, stage / "backup");
}
#ifdef _WIN32
void launch_helper(const fs::path& stage) {
  const auto helper = stage / "updater.exe";
  // Paths cannot contain quotes; the executable path ends in a filename, not a backslash.
  std::wstring command =
      L"\"" + helper.wstring() + L"\" _update-install " + std::to_wstring(GetCurrentProcessId());
  STARTUPINFOW startup{};
  startup.cb = sizeof(startup);
  startup.dwFlags = STARTF_USESHOWWINDOW;
  startup.wShowWindow = SW_HIDE;
  PROCESS_INFORMATION process{};
  auto flags = CREATE_NO_WINDOW | CREATE_BREAKAWAY_FROM_JOB;
  BOOL ok = CreateProcessW(helper.c_str(), command.data(), nullptr, nullptr, FALSE, flags, nullptr,
                           stage.c_str(), &startup, &process);
  if (!ok && GetLastError() == ERROR_ACCESS_DENIED) {
    command =
        L"\"" + helper.wstring() + L"\" _update-install " + std::to_wstring(GetCurrentProcessId());
    ok = CreateProcessW(helper.c_str(), command.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW,
                        nullptr, stage.c_str(), &startup, &process);
  }
  if (!ok)
    throw CliError("UPDATE_FAILED", "Cannot launch the update helper (Windows error " +
                                        std::to_string(GetLastError()) + ")");
  CloseHandle(process.hThread);
  CloseHandle(process.hProcess);
}
#endif
}

int update_command(const std::vector<std::string>& args) {
  const bool check = args == std::vector<std::string>{"--check"};
  if (!args.empty() && !check)
    throw CliError("INVALID_REQUEST", "usage: slopfab update [--check]");
  const auto response = script_http_request(
      "https://api.github.com/repos/" + std::string(updater::kRepository) + "/releases/latest",
      http_options(30000), {}, check_cancelled);
  const auto status = response.find("status")->as_int();
  if (status == 404)
    throw CliError("UPDATE_NOT_AVAILABLE",
                   "No public latest release is available at github.com/bitti-ai/slopfab");
  if (status == 403 || status == 429)
    throw CliError("UPDATE_RATE_LIMITED",
                   "GitHub denied the release check or its rate limit was reached; try again later",
                   2, {}, true);
  if (status != 200)
    throw CliError("UPDATE_FAILED", "GitHub release check returned HTTP " + std::to_string(status));
  const auto release = updater::parse_release(json::parse(script_string(response, "body")));
  const bool available = updater::newer_version(release.version, kVersion);
  const auto executable = fs::canonical(fs::u8path(agent().executable));
  const auto root = executable.parent_path();
  if (check || !available) {
    auto output = result(available ? "available" : "up_to_date", release);
    output["update_available"] = Value(available);
    // A Windows update is completed asynchronously. Expose its outcome on subsequent checks.
    updater::check_path(root, ".slopfab-update-result.json");
    if (fs::exists(root / ".slopfab-update-result.json"))
      output["last_update"] = read_json(root / ".slopfab-update-result.json");
    emit(std::move(output), true);
    return 0;
  }
  if (executable.filename() != executable_name())
    throw CliError("UPDATE_FAILED",
                   std::string("Run update from the executable named ") + executable_name());
#if !defined(_M_X64) && !defined(__x86_64__)
  throw CliError("UPDATE_FAILED", "Published update packages require an x64 executable");
#endif
  updater::check_path(root, ".slopfab-update.lock");
  FileLock lock(root / ".slopfab-update.lock");
  if (!lock.acquired())
    throw CliError("UPDATE_BUSY", "Another update is in progress");
  updater::check_path(root, ".slopfab-update-result.json");
  for (const auto& name : updater::package_files())
    updater::check_path(root, fs::u8path(name));
  const auto stage = root / fs::u8path(".slopfab-update-" + unique_id());
  if (!fs::create_directory(stage))
    throw CliError("UPDATE_FAILED", "Cannot create update staging directory");
  // Keep staging owner-only on POSIX, including the helper's installation plan.
#ifndef _WIN32
  fs::permissions(stage, fs::perms::owner_all);
#endif
  const auto archive = stage / "package.zip";
  try {
    emit({{"type", str("progress")},
          {"stage", str("update_download")},
          {"message", str("Downloading slopfab " + release.version)},
          {"bytes", num(double(release.bytes))}});
    std::ofstream file(archive, std::ios::binary);
    if (!file)
      throw CliError("UPDATE_FAILED", "Cannot write update package");
    uint64_t received = 0;
    HttpStream stream;
    stream.headers = [](int code, const Object&) {
      if (code != 200)
        throw CliError("UPDATE_FAILED", "Package download returned HTTP " + std::to_string(code));
    };
    stream.write = [&](const char* data, size_t bytes) {
      if (bytes > release.bytes - received)
        throw CliError("UPDATE_FAILED", "Package exceeds its declared size");
      file.write(data, std::streamsize(bytes));
      if (!file)
        throw CliError("UPDATE_FAILED", "Cannot write update package");
      received += bytes;
    };
    script_http_request(release.url, http_options(600000), {}, check_cancelled, &stream);
    file.close();
    if (!file)
      throw CliError("UPDATE_FAILED", "Cannot finish writing update package");
    updater::verify_package(archive, release.digest, release.bytes);
    updater::extract_package(archive, stage / "payload");
    Value plan(Object{{"version", str(release.version)},
                      {"sha256", str(release.digest)},
                      {"bytes", num(double(release.bytes))},
                      {"previous_sha256", str(updater::file_hash(executable))}});
    atomic_json(stage / "plan.json", plan);
    check_cancelled();
#ifdef _WIN32
    fs::copy_file(executable, stage / "updater.exe");
    atomic_json(root / ".slopfab-update-result.json",
                Value(Object{{"status", str("scheduled")},
                             {"version", str(release.version)},
                             {"staging", str(stage.u8string())}}));
    launch_helper(stage);
    auto output = result("scheduled", release);
    output["message"] =
        str("Downloaded and verified. Installation finishes after this process exits.");
    output["status_file"] = str((root / ".slopfab-update-result.json").u8string());
#else
    install(root, stage, plan, stage / "payload");
    atomic_json(root / ".slopfab-update-result.json",
                Value(Object{{"status", str("installed")},
                             {"version", str(release.version)},
                             {"backup", str((stage / "backup").u8string())}}));
    std::error_code ignored;
    fs::remove(archive, ignored);
    auto output = result("installed", release);
    output["message"] = str("Update installed. The next command uses the new version.");
#endif
    output["backup"] = str((stage / "backup").u8string());
    emit(std::move(output), true);
    return 0;
  } catch (const std::exception& error) {
    // Leave staging for diagnosis/recovery; never recursively delete user paths.
    atomic_json(root / ".slopfab-update-result.json",
                Value(Object{{"status", str("failed")},
                             {"version", str(release.version)},
                             {"error", str(error.what())},
                             {"staging", str(stage.u8string())}}));
    throw;
  }
}

int update_worker(const std::string& executable, const std::string& parent_pid) {
#ifdef _WIN32
  fs::path root, stage;
  bool can_report = false;
  try {
    const auto helper = fs::canonical(fs::u8path(executable));
    stage = helper.parent_path();
    root = stage.parent_path();
    if (helper.filename() != "updater.exe" ||
        stage.filename().u8string().find(".slopfab-update-") != 0)
      throw std::runtime_error("invalid update helper location");
    updater::check_path(root, stage.filename(), true);
    updater::check_path(root, ".slopfab-update-result.json");
    can_report = true;
    if (parent_pid.empty() || parent_pid.find_first_not_of("0123456789") != std::string::npos)
      throw std::runtime_error("invalid updater parent PID");
    const auto pid = std::stoull(parent_pid);
    if (!pid || pid > MAXDWORD || pid == GetCurrentProcessId())
      throw std::runtime_error("invalid updater parent PID");
    HANDLE parent = OpenProcess(SYNCHRONIZE, FALSE, DWORD(pid));
    if (parent) {
      const auto waited = WaitForSingleObject(parent, 120000);
      CloseHandle(parent);
      if (waited != WAIT_OBJECT_0)
        throw std::runtime_error("timed out waiting for the original process to exit");
    } else if (GetLastError() != ERROR_INVALID_PARAMETER) {
      throw std::runtime_error("cannot wait for the original process");
    }
    updater::check_path(root, ".slopfab-update.lock");
    FileLock lock(root / ".slopfab-update.lock");
    if (!lock.acquired())
      throw std::runtime_error("another update is in progress; run update again");
    updater::check_path(stage, "plan.json");
    updater::check_path(stage, "package.zip");
    const auto plan = read_json(stage / "plan.json");
    updater::verify_package(stage / "package.zip", script_string(plan, "sha256"),
                            script_uint(plan, "bytes", 0));
    // Re-extract the verified archive so changes to the parent's payload cannot be installed.
    updater::extract_package(stage / "package.zip", stage / "install");
    install(root, stage, plan, stage / "install");
    atomic_json(root / ".slopfab-update-result.json",
                Value(Object{{"status", str("installed")},
                             {"version", str(script_string(plan, "version"))},
                             {"backup", str((stage / "backup").u8string())}}));
    std::error_code ignored;
    fs::remove(stage / "package.zip", ignored);
    // These are exact generated file paths; retain the backup and helper for recovery.
    for (const auto& name : updater::package_files()) {
      try {
        updater::check_path(stage, fs::u8path("payload/" + name));
        fs::remove(stage / "payload" / fs::u8path(name), ignored);
      } catch (...) {
        // Cleanup must not turn a completed installation into a reported failure.
      }
    }
    return 0;
  } catch (const std::exception& error) {
    if (can_report) {
      try {
        updater::check_path(root, ".slopfab-update-result.json");
        atomic_json(root / ".slopfab-update-result.json",
                    Value(Object{{"status", str("failed")},
                                 {"error", str(error.what())},
                                 {"staging", str(stage.u8string())}}));
      } catch (...) {
      }
    }
    return 2;
  }
#else
  (void)executable;
  (void)parent_pid;
  return 2;
#endif
}
}
