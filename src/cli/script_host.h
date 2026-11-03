#pragma once
#include "agent.h"
#include <functional>
#include <memory>

namespace slopfab::cli {
inline std::string script_string(const Value& options, const char* key, std::string fallback = {}) {
  const auto* value = options.find(key);
  return value ? value->as_string() : fallback;
}
inline bool script_bool(const Value& options, const char* key, bool fallback = false) {
  const auto* value = options.find(key);
  return value ? value->as_bool() : fallback;
}
uint64_t script_uint(const Value& options, const char* key, uint64_t fallback);
// Child output is spooled to files: both streams can fill without pipe deadlock.
// Every live process belongs to this object and is terminated on destruction.
class ScriptProcesses {
public:
  explicit ScriptProcesses(std::function<void()> check);
  ~ScriptProcesses();
  Value spawn(const std::vector<std::string>& argv, const Value& options);
  Value poll(int id, bool wait);
  void kill(int id);
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
Value script_http(const std::string& url, const Value& options, const std::string& destination,
                  const std::function<void()>& check);
}
