#pragma once
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace slopfab::cli {
// One-way binary child-process pipe. Arguments never pass through a shell.
class PipeProcess {
public:
  PipeProcess(const std::vector<std::string>& args, bool write);
  ~PipeProcess();
  PipeProcess(const PipeProcess&) = delete;
  size_t read(void*, size_t);
  void write(const void*, size_t);
  void finish();
  // Thread-safe: terminate the child to unblock a concurrent read/write/finish.
  // The owner must join its I/O worker before destroying this object.
  void cancel() noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

std::string media_tool(const char* name, const char* executable);
}
