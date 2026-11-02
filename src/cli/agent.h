#pragma once
#include "slopfab/json.h"
#include "slopfab/generate.h"
#include <chrono>
#include <filesystem>
#include <functional>
#include <exception>
#include <memory>
#include <stdexcept>

namespace slopfab::cli {
using Object = json::Object;
using Array = json::Array;
using Value = json::Value;

inline Value str(const std::string& s) {
  return Value(s);
}

inline Value num(double d) {
  return Value(d);
}

struct CliError : std::runtime_error {
  std::string code, field;
  int exit_code;
  bool retryable;

  CliError(std::string c, std::string message, int status = 2, std::string f = {},
           bool retry = false)
      : std::runtime_error(message), code(std::move(c)), field(std::move(f)), exit_code(status),
        retryable(retry) {
  }
};

struct AgentContext {
  bool enabled = false, terminal = false, overwrite = false, check_ready = false;
  bool worker = false;
  std::string executable, task, config_path, jobs_path, cancel_file, event_file, job_id;
  double deadline_seconds = 0, requested_duration = 0;
  std::chrono::steady_clock::time_point started;
  Array results;
  Value last_event;
};

AgentContext& agent();
void initialize_agent(bool enabled, const std::string& executable);
void emit(Object event, bool terminal = false);
int report_exception(const std::exception& error);
[[noreturn]] void cli_fail(int status, const char* format, ...);
[[noreturn]] void throw_cancelled();
Value parse_cli_json(const std::string& text);
std::filesystem::path runtime_path();
int finish_command(int status);
bool cancelled();
bool progress(RunStage stage, int step, int total, void*);
void start_request();
std::string unique_id();
std::filesystem::path config_path();
std::filesystem::path jobs_path();
Value read_json(const std::filesystem::path& path);
void atomic_json(const std::filesystem::path& path, const Value& value);

class FileLock {
public:
  explicit FileLock(const std::filesystem::path& path);
  ~FileLock();
  bool acquired() const;
  FileLock(const FileLock&) = delete;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

std::vector<std::string> prepare_arguments(const std::vector<std::string>& args);
Value request_schema();
Value capabilities();
int discovery_command(const std::string& command, const std::vector<std::string>& args);
int job_command(const std::string& command, const std::vector<std::string>& args);
void resolve_model_alias(GenerateRequest& request, const std::string& name);
Value plan_json(const GenerateRequest&, const GeneratePlan&);
void check_models_ready(const GenerateRequest&, bool need_text);
int run_generation_arguments(const std::vector<std::string>& args);

// The command layer owns delivery. No changes to the DLL's sample contract.
struct Delivery {
  std::string output;
  video::FrameRate fps;
  Array artifacts;
  std::exception_ptr error;
  std::vector<std::filesystem::path> reservations;
  explicit Delivery(const std::string& path);
  ~Delivery();
  void reserve(const std::string& path);
  static bool samples(RunSamples&, void*);
};
}
