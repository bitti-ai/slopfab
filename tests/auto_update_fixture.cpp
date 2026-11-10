// Exercise production scheduling/settings/process code without a public release.
#include "../src/cli/update.h"
#include "../src/cli/settings.h"
#include <fstream>
#include <iostream>
#include <thread>

namespace slopfab::cli {
int update_command(const std::vector<std::string>& args) {
  if (!args.empty()) throw std::runtime_error("automatic update must install, not only check");
  record_update_check(update_check_date());
  const auto root = config_path().parent_path();
  std::ofstream(root / "attempts", std::ios::app) << "attempt\n";
  // A blocking download must not keep the foreground process or its pipes open.
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
  while (!std::filesystem::exists(root / "release")) {
    if (std::chrono::steady_clock::now() >= deadline)
      throw std::runtime_error("fixture release timed out");
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  if (std::filesystem::exists(root / "fail")) {
    std::ofstream(root / "finished") << "failed";
    throw std::runtime_error("fixture offline");
  }
  std::ofstream(root / "installed") << "new release";
  std::ofstream(root / "finished") << "installed";
  return 0;
}
}

int main(int argc, char** argv) {
  using namespace slopfab::cli;
  try {
    if (argc == 3 && std::string(argv[1]) == "_update-auto")
      return automatic_update_worker(argv[0], argv[2]);
    if (argc < 3) return 2;
    agent().config_path = argv[2];
    initialize_agent(false, std::filesystem::absolute(argv[0]).u8string());
    ensure_settings();
    const std::string operation = argv[1];
    if (operation == "launch") {
      DailyUpdateOnExit check{true};
      std::cout << "foreground result\n";
    } else if (operation == "claim" && argc == 4) {
      std::cout << (begin_daily_update(argv[3]) ? "claimed" : "skipped");
    } else if (operation == "record" && argc == 4) {
      record_update_check(argv[3]);
    } else return 2;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what();
    return 2;
  }
}
