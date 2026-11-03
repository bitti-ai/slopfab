#include "agent_instructions.h"
#include "agent.h"
#include "agent_instructions_template.h"
#include <cstring>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace slopfab::cli {
void ensure_agent_instructions() {
  if (!agent().enabled)
    return;
  const auto path = std::filesystem::current_path() / "AGENTS.md";
  const size_t length = std::strlen(kAgentInstructions);
  // Exclusive creation also protects an existing empty file or symlink and
  // ensures only one concurrent first launch emits the instructions.
#ifdef _WIN32
  HANDLE file = CreateFileW(path.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr, CREATE_NEW,
                            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
  if (file == INVALID_HANDLE_VALUE) {
    const auto error = GetLastError();
    if (error == ERROR_FILE_EXISTS || error == ERROR_ALREADY_EXISTS)
      return;
    throw CliError("OUTPUT_FAILED", "cannot create agent instructions: " + path.u8string(), 6);
  }
  DWORD written = 0;
  bool ok = WriteFile(file, kAgentInstructions, DWORD(length), &written, nullptr) &&
            written == length && FlushFileBuffers(file);
  if (!CloseHandle(file))
    ok = false;
  if (!ok) {
    DeleteFileW(path.c_str());
    throw CliError("OUTPUT_FAILED", "cannot write agent instructions: " + path.u8string(), 6);
  }
#else
  const int file = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
  if (file < 0) {
    if (errno == EEXIST)
      return;
    throw CliError("OUTPUT_FAILED", "cannot create agent instructions: " + path.u8string(), 6);
  }
  size_t written = 0;
  while (written < length) {
    const auto count = write(file, kAgentInstructions + written, length - written);
    if (count < 0 && errno == EINTR)
      continue;
    if (count <= 0)
      break;
    written += size_t(count);
  }
  bool ok = written == length && fsync(file) == 0;
  if (close(file) != 0)
    ok = false;
  if (!ok) {
    unlink(path.c_str());
    throw CliError("OUTPUT_FAILED", "cannot write agent instructions: " + path.u8string(), 6);
  }
#endif
  agent().startup_context =
      Value(Object{{"path", str(path.u8string())}, {"instructions", str(kAgentInstructions)}});
}
}
