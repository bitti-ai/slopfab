#include "agent_instructions.h"
#include "agent.h"
#include "agent_instructions_template.h"
#include <charconv>
#include <cstring>
#include <fstream>
#include <limits>
#include <string_view>
#include <thread>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace slopfab::cli {
namespace {
uint64_t instruction_version(std::string_view line) {
  line = line.substr(0, line.find('\n'));
  if (!line.empty() && line.back() == '\r')
    line.remove_suffix(1);
  if (line.empty() || line.find_first_not_of("0123456789") != std::string_view::npos)
    return 0;
  uint64_t version = 0;
  const auto result = std::from_chars(line.data(), line.data() + line.size(), version);
  // A future version too large for our counter must never be downgraded.
  return result.ec == std::errc::result_out_of_range ? std::numeric_limits<uint64_t>::max() : version;
}

bool needs_update(const std::filesystem::path& path, uint64_t bundled_version) {
  std::error_code error;
  const auto status = std::filesystem::symlink_status(path, error);
  if (error && error != std::errc::no_such_file_or_directory)
    throw CliError("OUTPUT_FAILED", "cannot inspect agent instructions: " + path.u8string(), 6);
  if (status.type() == std::filesystem::file_type::not_found)
    return true;
  if (!std::filesystem::is_regular_file(status))
    return false; // Includes dangling symlinks and directories.
  std::ifstream input(path, std::ios::binary);
  if (!input)
    throw CliError("OUTPUT_FAILED", "cannot read agent instructions: " + path.u8string(), 6);
  std::string line;
  std::getline(input, line);
  if (input.bad())
    throw CliError("OUTPUT_FAILED", "cannot read agent instructions: " + path.u8string(), 6);
  return instruction_version(line) < bundled_version;
}

void write_instructions(const std::filesystem::path& path) {
  const size_t length = std::strlen(kAgentInstructions);
  // Write a unique temporary file completely before replacing the old version.
#ifdef _WIN32
  HANDLE file = CreateFileW(path.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr, CREATE_NEW,
                            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
  if (file == INVALID_HANDLE_VALUE) {
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
}
} // namespace

void ensure_agent_instructions() {
  if (!agent().enabled)
    return;
  const auto path = std::filesystem::current_path() / "AGENTS.md";
  const auto version = instruction_version(kAgentInstructions);
  if (!version)
    throw CliError("OUTPUT_FAILED", "bundled agent instructions need a positive version", 6);
  auto lock_path = path;
  lock_path += ".lock";
  std::unique_ptr<FileLock> lock;
  for (int attempt = 0; attempt < 100; ++attempt) {
    lock = std::make_unique<FileLock>(lock_path);
    if (lock->acquired())
      break;
    lock.reset();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  if (!lock)
    throw CliError("OUTPUT_FAILED", "agent instructions are being updated; retry", 6);
  // Read under the lock too: Windows readers must release the old file before
  // replacement, and only the process publishing a new version emits it.
  if (!needs_update(path, version))
    return;
  auto temporary = path;
  temporary += "." + unique_id() + ".tmp";
  try {
    write_instructions(temporary);
#ifdef _WIN32
    if (!MoveFileExW(temporary.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
      throw CliError("OUTPUT_FAILED", "cannot update agent instructions: " + path.u8string(), 6);
#else
    std::filesystem::rename(temporary, path);
#endif
  } catch (...) {
    std::error_code error;
    std::filesystem::remove(temporary, error);
    throw;
  }
  agent().startup_context =
      Value(Object{{"path", str(path.u8string())}, {"instructions", str(kAgentInstructions)}});
}
}
