#include "pipe_process.h"
#include <algorithm>
#include <filesystem>
#include <chrono>
#include <mutex>
#include <stdexcept>
#include <thread>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <cerrno>
#include <csignal>
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace slopfab::cli {
std::string media_tool(const char* name, const char* executable) {
  std::string file = name;
#ifdef _WIN32
  file += ".exe";
#endif
  for (const auto& dir :
       {std::filesystem::absolute(std::filesystem::u8path(executable)).parent_path(),
        std::filesystem::current_path() / "external" / "ffmpeg" / "bin"})
    if (std::filesystem::is_regular_file(dir / file))
      return (dir / file).u8string();
  return file;
}

struct PipeProcess::Impl {
  bool writing = false, finished = false;
  std::mutex process_mutex;
#ifdef _WIN32
  HANDLE pipe = nullptr, process = nullptr;

  ~Impl() {
    if (pipe)
      CloseHandle(pipe);
    if (process) {
      if (!finished)
        TerminateProcess(process, 1);
      WaitForSingleObject(process, INFINITE);
      CloseHandle(process);
    }
  }
#else
  int pipe = -1;
  pid_t process = -1;

  ~Impl() {
    if (pipe >= 0)
      ::close(pipe);
    if (process > 0 && !finished) {
      kill(process, SIGKILL);
      while (waitpid(process, nullptr, 0) < 0 && errno == EINTR) {
      }
    }
  }
#endif
};

PipeProcess::PipeProcess(const std::vector<std::string>& args, bool write)
    : impl_(std::make_unique<Impl>()) {
  if (args.empty())
    throw std::invalid_argument("empty process arguments");
  impl_->writing = write;
#ifdef _WIN32
  auto quote = [](const std::string& s) {
    std::wstring out = L"\"";
    size_t slashes = 0;
    for (wchar_t c : std::filesystem::u8path(s).wstring()) {
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
  };
  SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
  HANDLE reader = nullptr, writer = nullptr;
  if (!CreatePipe(&reader, &writer, &sa, 0))
    throw std::runtime_error("cannot create media pipe");
  impl_->pipe = write ? writer : reader;
  HANDLE child = write ? reader : writer;
  SetHandleInformation(impl_->pipe, HANDLE_FLAG_INHERIT, 0);
  HANDLE null = CreateFileW(L"NUL", GENERIC_READ | GENERIC_WRITE,
                            FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, OPEN_EXISTING, 0, nullptr);
  HANDLE err = nullptr;
  DuplicateHandle(GetCurrentProcess(), GetStdHandle(STD_ERROR_HANDLE), GetCurrentProcess(), &err, 0,
                  TRUE, DUPLICATE_SAME_ACCESS);
  if (!err)
    err = null;
  STARTUPINFOEXW si{};
  si.StartupInfo.cb = sizeof(si);
  si.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
  si.StartupInfo.hStdInput = write ? child : null;
  si.StartupInfo.hStdOutput = write ? null : child;
  si.StartupInfo.hStdError = err;
  SIZE_T size = 0;
  InitializeProcThreadAttributeList(nullptr, 1, 0, &size);
  std::vector<unsigned char> storage(size);
  si.lpAttributeList = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(storage.data());
  BOOL init = InitializeProcThreadAttributeList(si.lpAttributeList, 1, 0, &size);
  HANDLE inherited[3] = {child, null, err};
  BOOL attributes = init && UpdateProcThreadAttribute(
                                si.lpAttributeList, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, inherited,
                                (err == null ? 2 : 3) * sizeof(HANDLE), nullptr, nullptr);
  std::wstring command;
  for (const auto& s : args) {
    if (!command.empty())
      command += L' ';
    command += quote(s);
  }
  PROCESS_INFORMATION pi{};
  BOOL ok = attributes && CreateProcessW(nullptr, command.data(), nullptr, nullptr, TRUE,
                                         CREATE_NO_WINDOW | EXTENDED_STARTUPINFO_PRESENT, nullptr,
                                         nullptr, &si.StartupInfo, &pi);
  if (init)
    DeleteProcThreadAttributeList(si.lpAttributeList);
  CloseHandle(child);
  CloseHandle(null);
  if (err != null)
    CloseHandle(err);
  if (!ok)
    throw std::runtime_error("cannot launch " + args.front() + "; install FFmpeg/ffprobe");
  impl_->process = pi.hProcess;
  CloseHandle(pi.hThread);
#else
  int fds[2];
  if (::pipe(fds))
    throw std::runtime_error("cannot create media pipe");
  fcntl(fds[0], F_SETFD, FD_CLOEXEC);
  fcntl(fds[1], F_SETFD, FD_CLOEXEC);
  impl_->pipe = write ? fds[1] : fds[0];
  std::vector<char*> argv;
  for (const auto& s : args)
    argv.push_back(const_cast<char*>(s.c_str()));
  argv.push_back(nullptr);
  pid_t child = fork();
  if (child == 0) {
    if (dup2(write ? fds[0] : fds[1], write ? STDIN_FILENO : STDOUT_FILENO) < 0)
      _exit(127);
    ::close(fds[0]);
    ::close(fds[1]);
    execvp(argv[0], argv.data());
    _exit(127);
  }
  ::close(write ? fds[0] : fds[1]);
  if (child < 0)
    throw std::runtime_error("cannot fork media process");
  impl_->process = child;
  if (write)
    signal(SIGPIPE, SIG_IGN);
#endif
}

PipeProcess::~PipeProcess() = default;

size_t PipeProcess::read(void* data, size_t count) {
  if (impl_->writing || impl_->finished)
    throw std::logic_error("invalid media pipe read");
  size_t done = 0;
  while (done < count) {
#ifdef _WIN32
    DWORD n = 0;
    if (!ReadFile(impl_->pipe, static_cast<char*>(data) + done,
                  DWORD(std::min(count - done, size_t(1 << 20))), &n, nullptr)) {
      if (GetLastError() == ERROR_BROKEN_PIPE)
        break;
      throw std::runtime_error("media pipe read failed");
    }
#else
    auto n = ::read(impl_->pipe, static_cast<char*>(data) + done, count - done);
    if (n < 0 && errno == EINTR)
      continue;
    if (n < 0)
      throw std::runtime_error("media pipe read failed");
#endif
    if (!n)
      break;
    done += size_t(n);
  }
  return done;
}

void PipeProcess::write(const void* data, size_t count) {
  if (!impl_->writing || impl_->finished)
    throw std::logic_error("invalid media pipe write");
  size_t done = 0;
  while (done < count) {
#ifdef _WIN32
    DWORD n = 0;
    if (!WriteFile(impl_->pipe, static_cast<const char*>(data) + done,
                   DWORD(std::min(count - done, size_t(1 << 20))), &n, nullptr))
      throw std::runtime_error("media encoder pipe failed");
#else
    auto n = ::write(impl_->pipe, static_cast<const char*>(data) + done, count - done);
    if (n < 0 && errno == EINTR)
      continue;
    if (n < 0)
      throw std::runtime_error("media encoder pipe failed");
#endif
    if (!n)
      throw std::runtime_error("media encoder pipe closed");
    done += size_t(n);
  }
}

void PipeProcess::finish() {
  std::unique_lock<std::mutex> lock(impl_->process_mutex);
  if (impl_->finished)
    return;
#ifdef _WIN32
  CloseHandle(impl_->pipe);
  impl_->pipe = nullptr;
  while (WaitForSingleObject(impl_->process, 0) == WAIT_TIMEOUT) {
    lock.unlock();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    lock.lock();
  }
  DWORD status = 1;
  GetExitCodeProcess(impl_->process, &status);
  impl_->finished = true;
  if (status)
    throw std::runtime_error("FFmpeg/ffprobe exited with code " + std::to_string(status));
#else
  ::close(impl_->pipe);
  impl_->pipe = -1;
  int status = 0;
  pid_t result;
  for (;;) {
    result = waitpid(impl_->process, &status, WNOHANG);
    if (result > 0 || (result < 0 && errno != EINTR))
      break;
    lock.unlock();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    lock.lock();
  }
  impl_->finished = true;
  if (result < 0 || !WIFEXITED(status) || WEXITSTATUS(status))
    throw std::runtime_error("FFmpeg/ffprobe failed");
#endif
}

void PipeProcess::cancel() noexcept {
  std::lock_guard<std::mutex> lock(impl_->process_mutex);
  if (impl_->finished)
    return;
#ifdef _WIN32
  if (impl_->process)
    TerminateProcess(impl_->process, 1);
#else
  if (impl_->process > 0)
    kill(impl_->process, SIGKILL);
#endif
}
}
