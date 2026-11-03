#pragma once
#include "agent.h"
#include <functional>
#include <memory>

namespace slopfab::cli {
// Per-JavaScript-runtime renderer/encoder ownership; no C API dependency.
class Motions {
public:
  explicit Motions(std::function<void()> check);
  ~Motions();
  Value call(const std::string& operation, const Array& args);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

int motions_command(const std::vector<std::string>& args);
}
