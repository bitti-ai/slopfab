#include "agent.h"
#include "commands.h"
#include "slopfab/sha256.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <thread>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <io.h>
#else
#include <unistd.h>
#include <fcntl.h>
#endif

namespace slopfab::cli {
namespace {
bool terminal(const std::string& state) {
  return state == "succeeded" || state == "failed" || state == "cancelled" ||
         state == "interrupted";
}

std::filesystem::path job_dir(const std::string& id) {
  if (id.size() != 32 || id.find_first_not_of("0123456789abcdef") != std::string::npos)
    throw CliError("INVALID_REQUEST", "job ID must contain 32 lowercase hexadecimal digits", 2,
                   "job_id");
  return jobs_path() / id;
}

double now() {
  return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

std::unique_ptr<FileLock> store_lock() {
  for (int i = 0; i < 100; ++i) {
    auto lock = std::make_unique<FileLock>(jobs_path() / "store.lock");
    if (lock->acquired())
      return lock;
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  throw CliError("QUEUE_BUSY", "job store is busy; retry", 7, {}, true);
}

std::vector<std::pair<std::filesystem::path, Value>> jobs() {
  std::vector<std::pair<std::filesystem::path, Value>> result;
  std::filesystem::create_directories(jobs_path());
  for (const auto& d : std::filesystem::directory_iterator(jobs_path()))
    if (d.is_directory() && std::filesystem::is_regular_file(d.path() / "job.json"))
      result.emplace_back(d.path(), read_json(d.path() / "job.json"));
  std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) {
    return a.second.find("created_at")->as_number() < b.second.find("created_at")->as_number();
  });
  return result;
}

void launch_worker() {
  const auto exe = std::filesystem::absolute(std::filesystem::u8path(agent().executable));
  const std::vector<std::string> args = {
      exe.u8string(), "worker",
      "--jobs-dir",   std::filesystem::absolute(jobs_path()).u8string(),
      "--config",     std::filesystem::absolute(config_path()).u8string()};
  std::filesystem::create_directories(jobs_path());
  // Do not pass a nonexistent optional config as an explicitly required one.
  auto actual = args;
  if (!std::filesystem::exists(config_path()))
    actual.resize(4);
#ifdef _WIN32
  std::wstring command;
  for (const auto& a : actual) {
    if (!command.empty())
      command += L' ';
    command += L'"';
    size_t slashes = 0;
    for (wchar_t c : std::filesystem::u8path(a).wstring()) {
      if (c == L'\\') {
        ++slashes;
        continue;
      }
      command.append(c == L'"' ? slashes * 2 + 1 : slashes, L'\\');
      slashes = 0;
      command += c;
    }
    command.append(slashes * 2, L'\\');
    command += L'"';
  }
  STARTUPINFOW startup{};
  startup.cb = sizeof(startup);
  startup.dwFlags = STARTF_USESHOWWINDOW;
  startup.wShowWindow = SW_HIDE;
  PROCESS_INFORMATION process{};
  if (!CreateProcessW(exe.c_str(), command.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW,
                      nullptr, nullptr, &startup, &process))
    throw CliError("WORKER_START_FAILED", "cannot start local worker; run slopfab worker manually",
                   1);
  CloseHandle(process.hThread);
  CloseHandle(process.hProcess);
#else
  pid_t pid = fork();
  if (pid < 0)
    throw CliError("WORKER_START_FAILED", "fork failed", 1);
  if (pid == 0) {
    setsid();
    int fd = open("/dev/null", O_RDWR);
    if (fd >= 0) {
      dup2(fd, 0);
      dup2(fd, 1);
      dup2(fd, 2);
      if (fd > 2)
        close(fd);
    }
    std::vector<char*> argv;
    for (auto& a : actual)
      argv.push_back(a.data());
    argv.push_back(nullptr);
    execv(exe.c_str(), argv.data());
    _exit(127);
  }
#endif
}

int worker(double idle_seconds) {
  auto lock = std::make_unique<FileLock>(jobs_path() / "worker.lock");
  if (!lock->acquired())
    return 0;
  const auto log = jobs_path() / "worker.log";
#ifdef _WIN32
  FILE* ignored = nullptr;
  if (_wfreopen_s(&ignored, log.c_str(), L"ab", stdout) != 0 ||
      _wfreopen_s(&ignored, L"NUL", L"ab", stderr) != 0 ||
      _dup2(_fileno(stdout), _fileno(stderr)) != 0)
    throw CliError("OUTPUT_FAILED", "cannot open worker diagnostics", 6);
#else
  (void)freopen(log.c_str(), "a", stdout);
  (void)freopen(log.c_str(), "a", stderr);
#endif
  // A dead process cannot retain the OS lock. Recover interrupted jobs without
  // rerunning them: generation may already have published some artifacts.
  {
    auto store = store_lock();
    for (auto& entry : jobs())
      if (entry.second.find("state")->as_string() == "running") {
        auto state = entry.second.as_object();
        state["state"] = str("interrupted");
        state["finished_at"] = num(now());
        state["error"] =
            str("worker exited during generation; inspect artifacts before resubmitting");
        atomic_json(entry.first / "job.json", Value(state));
      }
  }
  auto idle = std::chrono::steady_clock::now();
  for (;;) {
    if (cancelled() || std::filesystem::exists(jobs_path() / "stop"))
      break;
    std::filesystem::path directory;
    Value record;
    {
      auto store = store_lock();
      for (auto& item : jobs())
        if (item.second.find("state")->as_string() == "queued") {
          directory = item.first;
          auto state = item.second.as_object();
          state["state"] = str("running");
          state["started_at"] = num(now());
          record = Value(state);
          atomic_json(directory / "job.json", record);
          break;
        }
    }
    if (directory.empty()) {
      if (std::chrono::duration<double>(std::chrono::steady_clock::now() - idle).count() >=
          idle_seconds) {
        auto store = store_lock();
        bool queued = false;
        for (const auto& item : jobs())
          if (item.second.find("state")->as_string() == "queued")
            queued = true;
        if (!queued) {
          lock.reset();
          return 0;
        }
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(250));
      continue;
    }
    idle = std::chrono::steady_clock::now();
    agent().job_id = directory.filename().u8string();
    agent().worker = true;
    agent().event_file = (directory / "events.ndjson").u8string();
    agent().cancel_file = (directory / "cancel").u8string();
    agent().terminal = false;
    auto original = std::filesystem::current_path();
    auto original_config = agent().config_path;
    int status = 1;
    try {
      std::filesystem::current_path(std::filesystem::u8path(record.find("cwd")->as_string()));
      if (const auto* c = record.find("config"))
        agent().config_path = c->as_string();
      std::vector<std::string> args;
      for (const auto& a : record.find("arguments")->as_array())
        args.push_back(a.as_string());
      status = run_generation_arguments(args);
      if (status && !agent().terminal)
        status = report_exception(CliError(status == 2 ? "INVALID_REQUEST" : "GENERATION_FAILED",
                                           "generation failed; see worker diagnostics", status));
    } catch (const std::exception& e) {
      status = report_exception(e);
    }
    std::filesystem::current_path(original);
    agent().config_path = original_config;
    {
      auto store = store_lock();
      auto state = read_json(directory / "job.json").as_object();
      const auto* err = agent().last_event.find("error");
      const bool busy = err && err->find("code") && err->find("code")->as_string() == "GPU_BUSY";
      state["state"] = str(busy                               ? "queued"
                           : status == 0                      ? "succeeded"
                           : (status == 130 || status == 124) ? "cancelled"
                                                              : "failed");
      state["exit_code"] = num(status);
      state["result"] = agent().last_event;
      if (!busy)
        state["finished_at"] = num(now());
      atomic_json(directory / "job.json", Value(state));
    }
    agent().event_file.clear();
    agent().cancel_file.clear();
    agent().job_id.clear();
    agent().deadline_seconds = 0;
    if (status == 7)
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }
  return 0;
}
}

int job_command(const std::string& command, const std::vector<std::string>& args) {
  if (command == "worker") {
    if (args == std::vector<std::string>{"--stop"}) {
      atomic_json(jobs_path() / "stop", Value(Object{}));
      emit({{"type", str("result")}, {"ok", Value(true)}, {"stop_requested", Value(true)}}, true);
      return 0;
    }
    double idle = 60;
    if (!args.empty()) {
      if (args.size() != 2 || args[0] != "--idle-seconds")
        throw CliError("INVALID_REQUEST", "worker accepts --idle-seconds N");
      size_t end = 0;
      idle = std::stod(args[1], &end);
      if (end != args[1].size() || !std::isfinite(idle) || idle < 1 || idle > 86400)
        throw CliError("INVALID_REQUEST", "idle seconds must be in [1,86400]");
    }
    agent().started = std::chrono::steady_clock::now();
    return worker(idle);
  }
  if (command == "submit") {
    std::string key;
    std::vector<std::string> generation;
    for (size_t i = 0; i < args.size(); ++i) {
      if (args[i] == "--idempotency-key") {
        if (!key.empty() || ++i == args.size())
          throw CliError("INVALID_REQUEST", "provide one idempotency key");
        key = args[i];
        if (key.empty() || key.size() > 256)
          throw CliError("INVALID_REQUEST", "idempotency key must contain 1..256 bytes");
      } else
        generation.push_back(args[i]);
    }
    auto prepared = prepare_arguments(generation);
    // Persist normalized arguments and all frontend controls, never a stdin/file
    // dependency whose content can disappear before the worker reads it.
    if (!agent().task.empty())
      prepared.insert(prepared.begin(), agent().task);
    if (agent().overwrite)
      prepared.push_back("--overwrite");
    if (agent().deadline_seconds > 0)
      prepared.insert(prepared.end(), {"--deadline", std::to_string(agent().deadline_seconds)});
    if (agent().requested_duration > 0)
      prepared.insert(prepared.end(), {"--duration", std::to_string(agent().requested_duration)});
    if (agent().check_ready)
      prepared.push_back("--check-ready");
    Array arguments;
    for (const auto& a : prepared)
      arguments.push_back(str(a));
    const auto cwd = std::filesystem::current_path().u8string();
    const auto cfg = std::filesystem::exists(config_path())
                         ? std::filesystem::absolute(config_path()).u8string()
                         : "";
    Value saved;
    {
      auto lock = store_lock();
      size_t queued = 0;
      std::error_code ignored;
      std::filesystem::remove(jobs_path() / "stop", ignored);
      for (const auto& entry : jobs()) {
        const auto& job = entry.second;
        if (!terminal(job.find("state")->as_string()))
          ++queued;
        if (!key.empty() && job.find("idempotency_key")->as_string() == key) {
          if (json::stringify(*job.find("arguments")) != json::stringify(Value(arguments)) ||
              job.find("cwd")->as_string() != cwd || job.find("config")->as_string() != cfg)
            throw CliError("IDEMPOTENCY_CONFLICT", "key already belongs to a different request", 2,
                           "idempotency_key");
          saved = job;
          break;
        }
      }
      if (saved.is_null()) {
        if (queued >= 32)
          throw CliError("QUEUE_FULL", "queue capacity is 32; retry later", 7, {}, true);
        auto id = unique_id();
        auto directory = job_dir(id);
        saved = Value(Object{
            {"job_id", str(id)},
            {"state", str("queued")},
            {"created_at", num(now())},
            {"cwd", str(cwd)},
            {"config", str(cfg)},
            {"arguments", Value(arguments)},
            {"idempotency_key", str(key)},
            {"events_path", str(std::filesystem::absolute(directory / "events.ndjson").u8string())},
            {"diagnostics_path",
             str(std::filesystem::absolute(jobs_path() / "worker.log").u8string())}});
        atomic_json(directory / "job.json", saved);
      }
    }
    launch_worker();
    emit({{"type", str("result")}, {"ok", Value(true)}, {"job", saved}}, true);
    return 0;
  }
  if (args.empty())
    throw CliError("INVALID_REQUEST", command + " requires JOB_ID");
  auto directory = job_dir(args[0]);
  double timeout = 60;
  if (args.size() > 1) {
    if (command != "wait" || args.size() != 3 || args[1] != "--timeout")
      throw CliError("INVALID_REQUEST", "unexpected job arguments");
    size_t end = 0;
    timeout = std::stod(args[2], &end);
    if (end != args[2].size() || !std::isfinite(timeout) || timeout < 0 || timeout > 86400)
      throw CliError("INVALID_REQUEST", "timeout must be in [0,86400]");
  }
  if (command == "cancel") {
    auto lock = store_lock();
    auto record = read_json(directory / "job.json").as_object();
    const auto state = record.at("state").as_string();
    if (!terminal(state)) {
      atomic_json(directory / "cancel", Value(Object{}));
      if (state == "queued") {
        record["state"] = str("cancelled");
        record["finished_at"] = num(now());
      }
      record["cancel_requested"] = Value(true);
      atomic_json(directory / "job.json", Value(record));
    }
    emit({{"type", str("result")}, {"ok", Value(true)}, {"job", Value(record)}}, true);
    return 0;
  }
  const auto start = std::chrono::steady_clock::now();
  for (;;) {
    auto record = read_json(directory / "job.json");
    if (record.find("state")->as_string() == "running") {
      FileLock worker_lock(jobs_path() / "worker.lock");
      if (worker_lock.acquired()) {
        auto lock = store_lock();
        record = read_json(directory / "job.json");
        if (record.find("state")->as_string() == "running") {
          auto state = record.as_object();
          state["state"] = str("interrupted");
          state["finished_at"] = num(now());
          state["error"] = str("worker exited; inspect artifacts before resubmitting");
          record = Value(state);
          atomic_json(directory / "job.json", record);
        }
      }
    }
    const auto state = record.find("state")->as_string();
    bool expired =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() >= timeout;
    if (command == "status" || terminal(state) || expired) {
      emit({{"type", str("result")},
            {"ok", Value(true)},
            {"timed_out", Value(command == "wait" && !terminal(state))},
            {"job", record}},
           true);
      return command == "wait" && !terminal(state) ? 8 : 0;
    }
    if (cancelled())
      throw CliError("CANCELLED", "wait cancelled; the job continues until explicitly cancelled",
                     130);
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
  }
}
}
