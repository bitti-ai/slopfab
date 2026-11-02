#include "slopfab/text/clip_projection.h"
#include "agent.h"
#include "commands.h"
#include "native_media.h"
#include "slopfab/audio/wav.h"
#include "slopfab/model_profile.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <csignal>
#include <cstdarg>
#include <set>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <random>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <io.h>
#else
#include <unistd.h>
#include <sys/file.h>
#include <fcntl.h>
#endif

namespace slopfab::cli {
namespace {
AgentContext context;
FILE* protocol = nullptr;
std::mutex output_mutex;
std::atomic<bool> interrupted{false};
static_assert(std::atomic<bool>::is_always_lock_free);

void interrupt(int) {
  interrupted.store(true);
}
#ifdef _WIN32
BOOL WINAPI console_interrupt(DWORD type) {
  if (type == CTRL_C_EVENT || type == CTRL_BREAK_EVENT || type == CTRL_CLOSE_EVENT) {
    interrupted.store(true);
    return TRUE;
  }
  return FALSE;
}
#endif
const char* stage_name(RunStage s) {
  switch (s) {
  case RunStage::kStarting:
    return "starting";
  case RunStage::kReferences:
    return "references";
  case RunStage::kConditioning:
    return "conditioning";
  case RunStage::kTransformerLoad:
    return "loading_model";
  case RunStage::kDenoising:
    return "denoising";
  case RunStage::kVideoDecode:
    return "video_decode";
  case RunStage::kAudioDecode:
    return "audio_decode";
  case RunStage::kDelivering:
    return "writing";
  case RunStage::kFinished:
    return "finished";
  case RunStage::kUpscaling:
    return "upscaling";
  }
  return "unknown";
}
}

AgentContext& agent() {
  return context;
}

void initialize_agent(bool enabled, const std::string& executable) {
  context.enabled = enabled;
  context.executable = executable;
  std::signal(SIGINT, interrupt);
  std::signal(SIGTERM, interrupt);
#ifdef _WIN32
  SetConsoleCtrlHandler(console_interrupt, TRUE);
#endif
  if (enabled && !protocol) {
    std::fflush(stdout);
#ifdef _WIN32
    protocol = _fdopen(_dup(_fileno(stdout)), "wb");
    _dup2(_fileno(stderr), _fileno(stdout));
#else
    // A detached worker must not inherit the caller's response pipe: agents
    // wait for EOF after submit exits, even while the worker stays alive.
    const int descriptor = fcntl(fileno(stdout), F_DUPFD_CLOEXEC, 3);
    if (descriptor >= 0) {
      protocol = fdopen(descriptor, "w");
      if (!protocol)
        close(descriptor);
    }
    dup2(fileno(stderr), fileno(stdout));
#endif
    if (!protocol)
      throw std::runtime_error("cannot open agent output stream");
  }
}

void emit(Object event, bool terminal) {
  std::lock_guard<std::mutex> lock(output_mutex);
  event["schema_version"] = num(1);
  if (!context.job_id.empty())
    event["job_id"] = str(context.job_id);
  const auto value = Value(std::move(event));
  context.last_event = value;
  if (terminal)
    context.terminal = true;
  const auto line = json::stringify(value) + "\n";
  if (!context.event_file.empty()) {
    std::ofstream f(std::filesystem::u8path(context.event_file), std::ios::binary | std::ios::app);
    f << line;
    f.flush();
    if (!f)
      throw CliError("OUTPUT_FAILED", "cannot write job events", 6);
  }
  if (context.enabled) {
    FILE* target = protocol ? protocol : stdout;
    std::fwrite(line.data(), 1, line.size(), target);
    std::fflush(target);
  } else if (terminal) {
    std::fwrite(line.data(), 1, line.size(), stdout);
  }
}

int report_exception(const std::exception& error) {
  const auto* typed = dynamic_cast<const CliError*>(&error);
  std::string code = typed ? typed->code : "GENERATION_FAILED";
  int status = typed ? typed->exit_code : 1;
  bool retry = typed && typed->retryable;
  if (!typed && (dynamic_cast<const std::invalid_argument*>(&error) ||
                 dynamic_cast<const std::out_of_range*>(&error))) {
    code = "INVALID_REQUEST";
    status = 2;
  }
  std::string message = error.what();
  if (!typed && (message.find("out of memory") != std::string::npos ||
                 message.find("OUT_OF_MEMORY") != std::string::npos ||
                 dynamic_cast<const std::bad_alloc*>(&error))) {
    code = "OUT_OF_MEMORY";
    status = 5;
  }
  emit({{"type", str("error")},
        {"ok", Value(false)},
        {"error",
         Value(Object{{"code", str(code)},
                      {"message", str(message)},
                      {"field", typed && !typed->field.empty() ? str(typed->field) : Value()},
                      {"retryable", Value(retry)}})},
        {"results", Value(context.results)}},
       true);
  if (!context.enabled)
    std::fprintf(stderr, "slopfab: %s\n", error.what());
  return status;
}

[[noreturn]] void cli_fail(int status, const char* format, ...) {
  char message[4096];
  va_list args;
  va_start(args, format);
  std::vsnprintf(message, sizeof(message), format, args);
  va_end(args);
  throw CliError(status == 2 ? "INVALID_REQUEST" : "UNSUPPORTED_CAPABILITY", message,
                 status == 2 ? 2 : 4);
}

[[noreturn]] void throw_cancelled() {
  const bool timeout =
      context.deadline_seconds > 0 &&
      std::chrono::duration<double>(std::chrono::steady_clock::now() - context.started).count() >=
          context.deadline_seconds;
  throw CliError(timeout ? "DEADLINE_EXCEEDED" : "CANCELLED",
                 timeout ? "generation deadline exceeded" : "generation cancelled",
                 timeout ? 124 : 130);
}

std::filesystem::path runtime_path() {
#ifdef _WIN32
  if (const char* env = std::getenv("LOCALAPPDATA"))
    return std::filesystem::u8path(env) / "slopfab";
#else
  if (const char* env = std::getenv("XDG_RUNTIME_DIR"))
    return std::filesystem::u8path(env) / "slopfab";
  if (const char* env = std::getenv("HOME"))
    return std::filesystem::u8path(env) / ".cache" / "slopfab";
#endif
  return std::filesystem::temp_directory_path() / "slopfab";
}

Value parse_cli_json(const std::string& text) {
  auto parsed = json::parse(text);
  // The checkpoint JSON reader permits duplicate keys for legacy metadata.
  // Requests reject them locally without changing DLL parsing behavior.
  std::vector<std::set<std::string>> objects;
  for (size_t i = 0; i < text.size(); ++i) {
    if (text[i] == '{')
      objects.emplace_back();
    else if (text[i] == '}')
      objects.pop_back();
    else if (text[i] == '"') {
      const size_t begin = i++;
      while (i < text.size() && text[i] != '"') {
        if (text[i] == '\\')
          ++i;
        ++i;
      }
      auto value = json::parse(text.substr(begin, i - begin + 1)).as_string();
      if (value.find('\0') != std::string::npos)
        throw CliError("INVALID_REQUEST", "JSON strings cannot contain NUL");
      size_t next = i + 1;
      while (next < text.size() && std::isspace(static_cast<unsigned char>(text[next])))
        ++next;
      if (next < text.size() && text[next] == ':' && !objects.back().insert(value).second)
        throw CliError("INVALID_REQUEST", "duplicate JSON key: " + value, 2, value);
    }
  }
  return parsed;
}

int finish_command(int status) {
  if (!context.terminal && context.enabled) {
    if (status)
      return report_exception(CliError(status == 2 ? "INVALID_REQUEST" : "COMMAND_FAILED",
                                       "Command failed; diagnostic details are on stderr.",
                                       status));
    emit({{"type", str("result")}, {"ok", Value(true)}}, true);
  }
  return status;
}

void start_request() {
  context.started = std::chrono::steady_clock::now();
  context.results.clear();
  context.terminal = false;
}

bool cancelled() {
  return interrupted.load() ||
         (!context.cancel_file.empty() &&
          std::filesystem::exists(std::filesystem::u8path(context.cancel_file))) ||
         (context.deadline_seconds > 0 &&
          std::chrono::duration<double>(std::chrono::steady_clock::now() - context.started)
                  .count() >= context.deadline_seconds);
}

bool progress(RunStage stage, int step, int total, void*) {
  // Heartbeats are deliberately rate limited; stage transitions are immediate.
  static auto last = std::chrono::steady_clock::time_point{};
  static RunStage previous = RunStage::kFinished;
  auto now = std::chrono::steady_clock::now();
  if ((context.enabled || context.worker) &&
      (stage != previous || now - last >= std::chrono::milliseconds(250) || step == total)) {
    emit({{"type", str("progress")},
          {"stage", str(stage_name(stage))},
          {"completed", step >= 0 ? num(step) : Value()},
          {"total", total > 0 ? num(total) : Value()},
          {"elapsed_seconds", num(std::chrono::duration<double>(now - context.started).count())}});
    previous = stage;
    last = now;
  }
  return !cancelled();
}

std::string unique_id() {
  char text[40];
  std::snprintf(text, sizeof(text), "%016llx%016llx",
                static_cast<unsigned long long>(random_seed()),
                static_cast<unsigned long long>(random_seed()));
  return text;
}

std::filesystem::path config_path() {
  if (!context.config_path.empty())
    return std::filesystem::u8path(context.config_path);
  if (const char* env = std::getenv("SLOPFAB_CONFIG"))
    return std::filesystem::u8path(env);
#ifdef _WIN32
  if (const char* env = std::getenv("LOCALAPPDATA"))
    return std::filesystem::u8path(env) / "slopfab" / "config.json";
#else
  if (const char* env = std::getenv("XDG_CONFIG_HOME"))
    return std::filesystem::u8path(env) / "slopfab" / "config.json";
  if (const char* env = std::getenv("HOME"))
    return std::filesystem::u8path(env) / ".config" / "slopfab" / "config.json";
#endif
  return std::filesystem::absolute(context.executable).parent_path() / "slopfab.json";
}

std::filesystem::path jobs_path() {
  return context.jobs_path.empty() ? config_path().parent_path() / "jobs"
                                   : std::filesystem::u8path(context.jobs_path);
}

Value read_json(const std::filesystem::path& path) {
  std::string text;
#ifdef _WIN32
  struct File {
    HANDLE h;

    ~File() {
      if (h != INVALID_HANDLE_VALUE)
        CloseHandle(h);
    }
  } file{CreateFileW(path.c_str(), GENERIC_READ,
                     FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
                     FILE_ATTRIBUTE_NORMAL, nullptr)};

  if (file.h == INVALID_HANDLE_VALUE)
    throw CliError("INPUT_NOT_FOUND", "cannot read " + path.u8string(), 3);
  LARGE_INTEGER length{};
  if (!GetFileSizeEx(file.h, &length) || length.QuadPart < 0 || length.QuadPart > 16 * 1024 * 1024)
    throw CliError("INVALID_REQUEST", "JSON file exceeds 16 MiB");
  text.resize(size_t(length.QuadPart));
  DWORD read = 0;
  if (!ReadFile(file.h, text.data(), DWORD(text.size()), &read, nullptr) || read != text.size())
    throw CliError("INPUT_NOT_FOUND", "cannot read complete JSON file: " + path.u8string(), 3);
#else
  std::ifstream f(path, std::ios::binary);
  if (!f)
    throw CliError("INPUT_NOT_FOUND", "cannot read " + path.u8string(), 3);
  if (std::filesystem::file_size(path) > 16 * 1024 * 1024)
    throw CliError("INVALID_REQUEST", "JSON file exceeds 16 MiB");
  text.assign(std::istreambuf_iterator<char>(f), {});
#endif
  if (text.compare(0, 3, "\xef\xbb\xbf") == 0)
    text.erase(0, 3);
  try {
    return parse_cli_json(text);
  } catch (const std::exception& e) {
    throw CliError("INVALID_REQUEST", e.what());
  }
}

void atomic_json(const std::filesystem::path& path, const Value& value) {
  if (!path.parent_path().empty())
    std::filesystem::create_directories(path.parent_path());
  auto tmp = path;
  tmp += "." + unique_id() + ".tmp";
  try {
    std::ofstream f(tmp, std::ios::binary);
    f << json::stringify(value) << '\n';
    f.close();
    if (!f)
      throw CliError("OUTPUT_FAILED", "cannot write " + path.u8string(), 6);
#ifdef _WIN32
    if (!MoveFileExW(tmp.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
      throw CliError("OUTPUT_FAILED", "cannot publish " + path.u8string(), 6);
#else
    std::filesystem::rename(tmp, path);
#endif
  } catch (...) {
    std::error_code ec;
    std::filesystem::remove(tmp, ec);
    throw;
  }
}

struct FileLock::Impl {
#ifdef _WIN32
  HANDLE handle = INVALID_HANDLE_VALUE;
#else
  int fd = -1;
#endif
  bool held = false;
};

FileLock::FileLock(const std::filesystem::path& path) : impl_(new Impl) {
  std::filesystem::create_directories(path.parent_path());
#ifdef _WIN32
  impl_->handle =
      CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
                  nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
  OVERLAPPED ov{};
  if (impl_->handle != INVALID_HANDLE_VALUE)
    impl_->held = LockFileEx(impl_->handle, LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY, 0,
                             1, 0, &ov) != 0;
#else
  impl_->fd = open(path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
  impl_->held = impl_->fd >= 0 && flock(impl_->fd, LOCK_EX | LOCK_NB) == 0;
#endif
}

FileLock::~FileLock() {
#ifdef _WIN32
  if (impl_->handle != INVALID_HANDLE_VALUE)
    CloseHandle(impl_->handle);
#else
  if (impl_->fd >= 0)
    close(impl_->fd);
#endif
}

bool FileLock::acquired() const {
  return impl_->held;
}

Value plan_json(const GenerateRequest& r, const GeneratePlan& p) {
  Array warnings, sources;
  if (int64_t(p.canvas_width) * p.canvas_height > p.geometry.trained_max_pixels)
    warnings.push_back(
        str("Requested canvas exceeds the trained area; memory and quality may vary."));
  for (const auto& source : p.sampling_sources)
    sources.push_back(str(source));
  return Value(Object{
      {"model", str(request_model_family(r))},
      {"steps", num(p.num_inference_steps)},
      {"sampler",
       str(p.music_flow_sampler.empty() ? sampler::sampler_name(p.sampler) : p.music_flow_sampler)},
      {"sampling_sources", Value(sources)},
      {"warnings", Value(warnings)},
      {"seed", str(std::to_string(r.seed))},
      {"width", num(p.canvas_width)},
      {"height", num(p.canvas_height)},
      {"frames", num(p.aligned_frames)},
      {"duration_seconds", num(p.duration_seconds)},
      {"requested_duration_seconds",
       context.requested_duration > 0 ? num(context.requested_duration) : Value()},
      {"output", str(std::filesystem::absolute(std::filesystem::u8path(r.out_path)).u8string())},
      {"transformer", str(r.transformer_path)},
      {"text_encoder", str(r.text_encoder_path)},
      {"clip_projection", str(r.clip_projection_path)},
      {"video_vae", str(r.video_decode_path())},
      {"audio_vae", str(r.audio_vae_path)},
      {"description", str(describe_plan(r, p))}});
}

void check_models_ready(const GenerateRequest& r, bool need_text) {
  try {
    ensure_generate_models(r, need_text);
  } catch (const std::exception& e) {
    throw CliError("MODEL_MISSING", e.what(), 3);
  }
  for (const auto& p : {r.transformer_path, r.text_encoder_path, r.video_vae_path,
                        r.video_decode_path(), r.audio_vae_path, r.clip_projection_path})
    if (!p.empty() && !std::filesystem::is_regular_file(std::filesystem::u8path(p)))
      throw CliError("MODEL_MISSING", "checkpoint does not exist: " + p, 3);
  if (need_text && !r.clip_projection_path.empty()) {
    try {
      text::ClipProjection projection(r.clip_projection_path);
      SafeTensors encoder;
      encoder.open(r.text_encoder_path);
      (void)text::validate_clip_student(encoder, projection);
    } catch (const std::exception& e) {
      throw CliError("INVALID_REQUEST", e.what(), 2, "clip_projection");
    }
  }
}

Delivery::Delivery(const std::string& path)
    : output(std::filesystem::absolute(std::filesystem::u8path(path)).u8string()) {
}

Delivery::~Delivery() {
  for (const auto& p : reservations) {
    std::error_code ec;
    std::filesystem::remove(p, ec);
  }
}

void Delivery::reserve(const std::string& name) {
  auto path = std::filesystem::u8path(name);
  try {
    std::filesystem::create_directories(path.parent_path());
  } catch (const std::exception& e) {
    throw CliError("OUTPUT_FAILED", e.what(), 6);
  }
  auto lock = path;
  lock += ".slopfab-reserved";
  if (!std::filesystem::create_directory(lock))
    throw CliError("OUTPUT_EXISTS", "output is reserved: " + name, 6);
  reservations.push_back(lock);
  if (!context.overwrite && std::filesystem::exists(path))
    throw CliError("OUTPUT_EXISTS", "output already exists: " + name, 6);
}

bool Delivery::samples(RunSamples& s, void* user) {
  auto& d = *static_cast<Delivery*>(user);

  struct Pending {
    std::filesystem::path temp, final;
    Value metadata;
  };

  std::vector<Pending> pending;
  auto write = [&](const std::string& name, const std::string& mime, Object info,
                   const auto& writer) {
    auto final = std::filesystem::u8path(name);
    auto tmp = final;
    tmp += "." + unique_id() + ".tmp";
    info["path"] = str(name);
    info["media_type"] = str(mime);
    pending.push_back({tmp, final, Value()});
    writer(tmp.u8string());
    info["bytes"] = num(double(std::filesystem::file_size(tmp)));
    pending.back().metadata = Value(std::move(info));
  };
  try {
    if (cancelled())
      throw_cancelled();
    if (s.video && !s.video->empty()) {
      if (std::filesystem::u8path(d.output).extension() == ".png" ||
          std::filesystem::u8path(d.output).extension() == ".ppm") {
        if (s.frames != 1)
          throw CliError("OUTPUT_FAILED", "image delivery received multiple frames", 6);
        size_t plane = size_t(s.width) * s.height;
        std::vector<uint8_t> rgb(plane * 3);
        for (size_t p = 0; p < plane; ++p)
          for (size_t c = 0; c < 3; ++c)
            rgb[p * 3 + c] =
                uint8_t(std::lround(std::clamp((*s.video)[c * plane + p], 0.0f, 1.0f) * 255));
        const bool png = std::filesystem::u8path(d.output).extension() == ".png";
        write(d.output, png ? "image/png" : "image/x-portable-pixmap",
              {{"width", num(s.width)}, {"height", num(s.height)}}, [&](const std::string& p) {
                if (png)
                  write_png(p, rgb.data(), s.width, s.height);
                else
                  video::write_ppm(p, *s.video, 1, s.height, s.width, 0);
              });
      } else {
        write(d.output, "video/x-yuv4mpeg",
              {{"width", num(s.width)},
               {"height", num(s.height)},
               {"frames", num(s.frames)},
               {"fps", num(double(d.fps.numerator) / d.fps.denominator)},
               {"duration_seconds", num(double(s.frames) * d.fps.denominator / d.fps.numerator)}},
              [&](const std::string& p) {
                video::write_y4m(p, *s.video, s.frames, s.height, s.width, d.fps, d.converter);
              });
      }
    }
    if (s.audio && !s.audio->empty()) {
      auto p = std::filesystem::u8path(d.output);
      p.replace_extension(".wav");
      write(p.u8string(), "audio/wav",
            {{"sample_rate", num(s.audio_sample_rate)},
             {"channels", num(s.audio_channels)},
             {"duration_seconds",
              num(double(s.audio->size()) / s.audio_channels / s.audio_sample_rate)}},
            [&](const std::string& name) {
              audio::write_wav(name, *s.audio, s.audio_channels, s.audio_sample_rate);
            });
    }
    for (const auto& file : pending) {
#ifdef _WIN32
      if (!MoveFileExW(file.temp.c_str(), file.final.c_str(),
                       context.overwrite ? MOVEFILE_REPLACE_EXISTING : 0))
        throw CliError("OUTPUT_FAILED", "cannot publish artifact: " + file.final.u8string(), 6);
#else
      if (context.overwrite)
        std::filesystem::rename(file.temp, file.final);
      else {
        std::filesystem::create_hard_link(file.temp, file.final);
        std::filesystem::remove(file.temp);
      }
#endif
      d.artifacts.push_back(file.metadata);
    }
  } catch (...) {
    d.error = std::current_exception();
    try {
      std::rethrow_exception(d.error);
    } catch (const CliError&) {
    } catch (const std::exception& e) {
      d.error = std::make_exception_ptr(CliError("OUTPUT_FAILED", e.what(), 6));
    }
    for (const auto& f : pending) {
      std::error_code ec;
      std::filesystem::remove(f.temp, ec);
    }
    std::rethrow_exception(d.error);
  }
  return true;
}
}
