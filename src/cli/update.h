#pragma once
#include <string>
#include <vector>

namespace slopfab::cli {
int update_command(const std::vector<std::string>& args);
int update_worker(const std::string& executable, const std::string& parent_pid);
std::string update_check_date();
void schedule_daily_update() noexcept;
int automatic_update_worker(const std::string& executable, const std::string& settings);

struct DailyUpdateOnExit {
  bool enabled;
  ~DailyUpdateOnExit() { if (enabled) schedule_daily_update(); }
};
}
