#include "update.h"
#include "settings.h"
#include <cstdio>
#include <cstdlib>
#include <ctime>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <io.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace slopfab::cli {
namespace fs = std::filesystem;

std::string update_check_date() {
  const auto now = std::time(nullptr);
  std::tm utc{};
#ifdef _WIN32
  if (gmtime_s(&utc, &now) != 0) throw std::runtime_error("cannot read UTC date");
#else
  if (!gmtime_r(&now, &utc)) throw std::runtime_error("cannot read UTC date");
#endif
  char date[11];
  if (!std::strftime(date, sizeof(date), "%Y-%m-%d", &utc))
    throw std::runtime_error("cannot format UTC date");
  return date;
}

void schedule_daily_update() noexcept {
  try {
    if (const char* disabled = std::getenv("SLOPFAB_NO_AUTO_UPDATE");
        disabled && std::string(disabled) == "1") return;
    const auto executable = fs::canonical(fs::u8path(agent().executable));
    const auto settings = fs::absolute(config_path()).lexically_normal();
    if (!begin_daily_update(update_check_date())) return;
#ifdef _WIN32
    // Both arguments are absolute file paths, so neither ends in a backslash
    // or contains a quote. No console or caller-owned handles are inherited.
    const std::wstring command = L"\"" + executable.wstring() + L"\" _update-auto \"" +
                                 settings.wstring() + L"\"";
    auto actual = command;
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESHOWWINDOW;
    startup.wShowWindow = SW_HIDE;
    PROCESS_INFORMATION process{};
    BOOL launched = CreateProcessW(executable.c_str(), actual.data(), nullptr, nullptr, FALSE,
        CREATE_NO_WINDOW | CREATE_BREAKAWAY_FROM_JOB, nullptr, nullptr, &startup, &process);
    if (!launched && GetLastError() == ERROR_ACCESS_DENIED) {
      actual = command;
      launched = CreateProcessW(executable.c_str(), actual.data(), nullptr, nullptr, FALSE,
          CREATE_NO_WINDOW, nullptr, nullptr, &startup, &process);
    }
    if (!launched) throw std::runtime_error("cannot launch automatic update worker");
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
#else
    // Prepare storage before fork; use only async-signal-safe calls in the
    // child because inference libraries may still have background threads.
    auto exe = executable.u8string(), config = settings.u8string();
    char mode[] = "_update-auto";
    char* args[] = {exe.data(), mode, config.data(), nullptr};
    const int null = open("/dev/null", O_RDWR | O_CLOEXEC);
    if (null < 0) throw std::runtime_error("cannot redirect automatic update worker");
    const auto pid = fork();
    if (pid == 0) {
      if (setsid() < 0 || dup2(null, 0) < 0 || dup2(null, 1) < 0 || dup2(null, 2) < 0)
        _exit(127);
      if (null > 2) close(null);
      execv(exe.c_str(), args);
      _exit(127);
    }
    close(null);
    if (pid < 0) throw std::runtime_error("cannot launch automatic update worker");
#endif
  } catch (const std::exception& error) {
    // Never change the foreground command's result or its machine protocol.
    std::fprintf(stderr, "slopfab: automatic update not started: %s\n", error.what());
  }
}

int automatic_update_worker(const std::string& executable, const std::string& settings) {
  try {
    agent().config_path = settings;
    initialize_agent(false, executable);
    const auto current = load_settings();
    if (const auto* enabled = current.find("auto_update"); enabled && !enabled->as_bool())
      return 0;
    const auto log = fs::absolute(config_path()).parent_path() / "auto-update.log";
#ifdef _WIN32
    FILE* opened = nullptr;
    if (_wfreopen_s(&opened, log.c_str(), L"wb", stdout) != 0 ||
        _wfreopen_s(&opened, L"NUL", L"wb", stderr) != 0 ||
        _dup2(_fileno(stdout), _fileno(stderr)) != 0)
      return 2;
#else
    if (!std::freopen(log.c_str(), "w", stdout) || dup2(fileno(stdout), fileno(stderr)) < 0)
      return 2;
#endif
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::fprintf(stdout, "Automatic update check: %s UTC\n", update_check_date().c_str());
    // Reuse signature verification, checksums, backups and the Windows handoff.
    // Unlike `--check`, this also installs a newer release when available.
    return update_command({});
  } catch (const std::exception& error) {
    std::fprintf(stderr, "slopfab: automatic update failed: %s\n", error.what());
    return 2;
  }
}
} // namespace slopfab::cli
