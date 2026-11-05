#include "script_host.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <thread>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <cerrno>
#include <csignal>
#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif

namespace slopfab::cli {
namespace fs = std::filesystem;

uint64_t script_uint(const Value& options, const char* key, uint64_t fallback) {
  const auto* value = options.find(key);
  if (!value)
    return fallback;
  const double n = value->as_number();
  if (!std::isfinite(n) || n < 0 || n > 9007199254740991.0 || std::floor(n) != n)
    throw std::runtime_error(std::string(key) + " must be a nonnegative safe integer");
  return uint64_t(n);
}

namespace {
void no_nul(const std::string& value) {
  if (value.find('\0') != std::string::npos)
    throw std::runtime_error("process strings cannot contain NUL");
}

std::string read_capture(const fs::path& path, uint64_t limit) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    throw std::runtime_error("cannot read process output");
  const auto size = fs::file_size(path);
  if (size > limit)
    throw std::runtime_error(
        "process output exceeds maxOutputBytes; redirect stdoutFile/stderrFile");
  std::string out(size_t(size), '\0');
  in.read(out.data(), std::streamsize(out.size()));
  if (in.bad())
    throw std::runtime_error("cannot read process output");
  return out;
}
#ifdef _WIN32
std::wstring wide(const std::string& s) {
  no_nul(s);
  return fs::u8path(s).wstring();
}

std::wstring quote(const std::string& s) {
  std::wstring out = L"\"";
  size_t slashes = 0;
  for (wchar_t c : wide(s)) {
    if (c == L'\\') {
      ++slashes;
      continue;
    }
    out.append(c == L'"' ? slashes * 2 + 1 : slashes, L'\\');
    slashes = 0;
    out += c;
  }
  out.append(slashes * 2, L'\\');
  return out + L'"';
}

struct Handle {
  HANDLE value = INVALID_HANDLE_VALUE;

  ~Handle() {
    if (value != INVALID_HANDLE_VALUE && value)
      CloseHandle(value);
  }
};
#endif
}

struct ScriptProcesses::Impl {
  struct Child {
    fs::path directory, out, err;
    bool capture_out = true, capture_err = true, done = false;
    uint64_t limit = 16 * 1024 * 1024, timeout = 0;
    int exit_code = 0;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
#ifdef _WIN32
    Handle process, job;
#else
    pid_t pid = -1;
#endif
    void terminate() noexcept {
#ifdef _WIN32
      if (job.value != INVALID_HANDLE_VALUE && job.value)
        TerminateJobObject(job.value, 137);
#else
      if (pid > 0)
        ::kill(-pid, SIGKILL);
#endif
    }

    ~Child() {
      // Also stop descendants after a parent has exited.
      terminate();
#ifdef _WIN32
      if (process.value != INVALID_HANDLE_VALUE)
        WaitForSingleObject(process.value, INFINITE);
#else
      if (pid > 0 && !done)
        while (waitpid(pid, nullptr, 0) < 0 && errno == EINTR) {
        }
#endif
      std::error_code ec;
      fs::remove_all(directory, ec);
    }

    bool poll() {
      if (done)
        return true;
#ifdef _WIN32
      if (WaitForSingleObject(process.value, 0) == WAIT_TIMEOUT)
        return false;
      DWORD status;
      if (!GetExitCodeProcess(process.value, &status))
        throw std::runtime_error("cannot read process status");
      exit_code = int(status);
#else
      int status = 0;
      const auto result = waitpid(pid, &status, WNOHANG);
      if (result == 0 || (result < 0 && errno == EINTR))
        return false;
      if (result < 0)
        throw std::runtime_error("cannot wait for process");
      exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
#endif
      done = true;
      return true;
    }
  };

  std::function<void()> check;
  std::map<int, std::unique_ptr<Child>> children;
  int next = 1;
};

ScriptProcesses::ScriptProcesses(std::function<void()> check) : impl_(new Impl) {
  impl_->check = std::move(check);
}

ScriptProcesses::~ScriptProcesses() = default;

Value ScriptProcesses::spawn(const std::vector<std::string>& argv, const Value& options) {
  impl_->check();
  if (argv.empty() || argv[0].empty())
    throw std::runtime_error("spawn needs a nonempty argv array");
  for (const auto& s : argv)
    no_nul(s);
  auto child = std::make_unique<Impl::Child>();
  child->directory = fs::temp_directory_path() / ("slopfab-js-" + unique_id());
  if (!fs::create_directory(child->directory))
    throw std::runtime_error("cannot create process spool directory");
  child->limit = script_uint(options, "maxOutputBytes", child->limit);
  child->timeout = script_uint(options, "timeoutMs", 0);
  const auto cwd_text = script_string(options, "cwd", fs::current_path().u8string());
  no_nul(cwd_text);
  const auto cwd = fs::absolute(fs::u8path(cwd_text));
  if (!fs::is_directory(cwd))
    throw std::runtime_error("process cwd is not a directory");
  const auto out = script_string(options, "stdoutFile"), err = script_string(options, "stderrFile");
  no_nul(out);
  no_nul(err);
  child->capture_out = out.empty();
  child->capture_err = err.empty();
  child->out = out.empty() ? child->directory / "stdout" : fs::absolute(cwd / fs::u8path(out));
  child->err = err.empty() ? child->directory / "stderr" : fs::absolute(cwd / fs::u8path(err));
  if (child->out == child->err)
    throw std::runtime_error("stdoutFile and stderrFile must differ");
  const auto input_file = script_string(options, "stdinFile");
  no_nul(input_file);
  if (!input_file.empty() && options.find("stdin"))
    throw std::runtime_error("stdin and stdinFile are mutually exclusive");
  const auto input = input_file.empty() ? child->directory / "stdin"
                                        : fs::absolute(cwd / fs::u8path(input_file));
  if (!input_file.empty() && (input == child->out || input == child->err))
    throw std::runtime_error("stdinFile must differ from output files");
  if (input_file.empty()) {
    std::ofstream f(input, std::ios::binary);
    const auto data = script_string(options, "stdin");
    f.write(data.data(), std::streamsize(data.size()));
    f.close();
    if (!f)
      throw std::runtime_error("cannot write process input");
  }
  std::map<std::string, std::string> environment;
#ifdef _WIN32
  auto block = GetEnvironmentStringsW();
  if (!block)
    throw std::runtime_error("cannot read environment");
  for (const wchar_t* p = block; *p; p += wcslen(p) + 1) {
    const auto entry = fs::path(p).u8string();
    const auto eq = entry.find('=', entry[0] == '=' ? 1 : 0);
    if (eq != std::string::npos)
      environment[entry.substr(0, eq)] = entry.substr(eq + 1);
  }
  FreeEnvironmentStringsW(block);
#else
  for (char** p = environ; *p; ++p) {
    const std::string entry(*p);
    const auto eq = entry.find('=');
    if (eq != std::string::npos)
      environment[entry.substr(0, eq)] = entry.substr(eq + 1);
  }
#endif
  if (const auto* env = options.find("env"))
    for (const auto& item : env->as_object()) {
      no_nul(item.first);
      if (item.first.empty() || item.first.find('=') != std::string::npos)
        throw std::runtime_error("invalid environment key");
#ifdef _WIN32
      for (auto it = environment.begin(); it != environment.end();) {
        if (_stricmp(it->first.c_str(), item.first.c_str()) == 0)
          it = environment.erase(it);
        else
          ++it;
      }
#endif
      if (item.second.is_null())
        environment.erase(item.first);
      else {
        no_nul(item.second.as_string());
        environment[item.first] = item.second.as_string();
      }
    }
#ifdef _WIN32
  SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
  Handle in, output, error;
  in.value = CreateFileW(input.c_str(), GENERIC_READ, FILE_SHARE_READ, &sa, OPEN_EXISTING,
                         FILE_ATTRIBUTE_NORMAL, nullptr);
  output.value = CreateFileW(child->out.c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
                             &sa, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
  error.value = CreateFileW(child->err.c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
                            &sa, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (in.value == INVALID_HANDLE_VALUE || output.value == INVALID_HANDLE_VALUE ||
      error.value == INVALID_HANDLE_VALUE)
    throw std::runtime_error("cannot open process streams");
  std::wstring command, envblock;
  for (const auto& s : argv) {
    if (!command.empty())
      command += L' ';
    command += quote(s);
  }
  for (const auto& e : environment) {
    envblock += wide(e.first + "=" + e.second);
    envblock += L'\0';
  }
  envblock += L'\0';
  STARTUPINFOEXW si{};
  si.StartupInfo.cb = sizeof(si);
  si.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
  si.StartupInfo.hStdInput = in.value;
  si.StartupInfo.hStdOutput = output.value;
  si.StartupInfo.hStdError = error.value;
  SIZE_T size = 0;
  InitializeProcThreadAttributeList(nullptr, 1, 0, &size);
  std::vector<unsigned char> storage(size);
  si.lpAttributeList = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(storage.data());
  if (!InitializeProcThreadAttributeList(si.lpAttributeList, 1, 0, &size))
    throw std::runtime_error("cannot initialize process attributes");
  HANDLE inherited[] = {in.value, output.value, error.value};
  const bool attributes =
      UpdateProcThreadAttribute(si.lpAttributeList, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, inherited,
                                sizeof(inherited), nullptr, nullptr) != 0;
  PROCESS_INFORMATION pi{};
  const bool ok =
      attributes && CreateProcessW(nullptr, command.data(), nullptr, nullptr, TRUE,
                                   CREATE_NO_WINDOW | CREATE_SUSPENDED |
                                       CREATE_UNICODE_ENVIRONMENT | EXTENDED_STARTUPINFO_PRESENT,
                                   envblock.data(), cwd.c_str(), &si.StartupInfo, &pi);
  DeleteProcThreadAttributeList(si.lpAttributeList);
  if (!ok)
    throw std::runtime_error("cannot launch " + argv[0] + " (Windows error " +
                             std::to_string(GetLastError()) + ")");
  child->process.value = pi.hProcess;
  Handle thread;
  thread.value = pi.hThread;
  child->job.value = CreateJobObjectW(nullptr, nullptr);
  JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
  limits.BasicLimitInformation.LimitFlags =
      JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_BREAKAWAY_OK;
  if (!child->job.value ||
      !SetInformationJobObject(child->job.value, JobObjectExtendedLimitInformation, &limits,
                               sizeof(limits)) ||
      !AssignProcessToJobObject(child->job.value, pi.hProcess)) {
    TerminateProcess(pi.hProcess, 137);
    throw std::runtime_error("cannot contain child process");
  }
  if (ResumeThread(pi.hThread) == DWORD(-1))
    throw std::runtime_error("cannot start child process");
  const double pid = pi.dwProcessId;
#else
  std::vector<std::string> envstrings;
  for (const auto& e : environment)
    envstrings.push_back(e.first + "=" + e.second);
  std::vector<char*> args, env;
  for (const auto& s : argv)
    args.push_back(const_cast<char*>(s.c_str()));
  args.push_back(nullptr);
  for (auto& s : envstrings)
    env.push_back(s.data());
  env.push_back(nullptr);
  posix_spawn_file_actions_t actions;
  posix_spawn_file_actions_init(&actions);
  posix_spawnattr_t attr;
  posix_spawnattr_init(&attr);
  int error = posix_spawn_file_actions_addchdir_np(&actions, cwd.c_str());
  if (!error)
    error = posix_spawn_file_actions_addopen(&actions, 0, input.c_str(), O_RDONLY, 0600);
  if (!error)
    error = posix_spawn_file_actions_addopen(&actions, 1, child->out.c_str(),
                                             O_WRONLY | O_CREAT | O_TRUNC, 0600);
  if (!error)
    error = posix_spawn_file_actions_addopen(&actions, 2, child->err.c_str(),
                                             O_WRONLY | O_CREAT | O_TRUNC, 0600);
  posix_spawnattr_setflags(&attr, POSIX_SPAWN_SETPGROUP);
  posix_spawnattr_setpgroup(&attr, 0);
  if (!error)
    error = posix_spawnp(&child->pid, args[0], &actions, &attr, args.data(), env.data());
  posix_spawn_file_actions_destroy(&actions);
  posix_spawnattr_destroy(&attr);
  if (error)
    throw std::runtime_error("cannot launch " + argv[0] + ": " + std::to_string(error));
  const double pid = child->pid;
#endif
  const int id = impl_->next++;
  impl_->children[id] = std::move(child);
  return Value(Object{{"id", num(id)}, {"pid", num(pid)}});
}

Value ScriptProcesses::poll(int id, bool wait) {
  auto it = impl_->children.find(id);
  if (it == impl_->children.end())
    throw std::runtime_error("unknown process id");
  auto& child = *it->second;
  do {
    impl_->check();
    if (child.poll()) {
      return Value(
          Object{{"done", Value(true)},
                 {"exitCode", num(child.exit_code)},
                 {"stdout", str(child.capture_out ? read_capture(child.out, child.limit) : "")},
                 {"stderr", str(child.capture_err ? read_capture(child.err, child.limit) : "")}});
    }
    if (child.timeout && std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::steady_clock::now() - child.start)
                                 .count() >= int64_t(child.timeout)) {
      child.terminate();
      throw std::runtime_error("process timeout");
    }
    if ((child.capture_out && fs::file_size(child.out) > child.limit) ||
        (child.capture_err && fs::file_size(child.err) > child.limit)) {
      child.terminate();
      throw std::runtime_error("process output exceeds maxOutputBytes; redirect to files");
    }
    if (wait)
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
  } while (wait);
  return Value(Object{{"done", Value(false)}});
}

void ScriptProcesses::kill(int id) {
  auto it = impl_->children.find(id);
  if (it == impl_->children.end())
    throw std::runtime_error("unknown process id");
  it->second->terminate();
}
}
