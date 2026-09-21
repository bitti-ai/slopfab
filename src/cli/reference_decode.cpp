#include "reference_decode.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

#include "slopfab/json.h"

#if SLOPFAB_WITH_FFMPEG
#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#else
#include <cerrno>
#include <csignal>
#include <sys/wait.h>
#include <unistd.h>
#endif
#endif

namespace slopfab::cli {
#if SLOPFAB_WITH_FFMPEG
namespace {
using Sink = std::function<void(const uint8_t*, size_t)>;

std::string tool_path(const char* name, const char* executable) {
  std::string filename = name;
#if defined(_WIN32)
  filename += ".exe";
#endif
  const auto exe_dir = std::filesystem::absolute(std::filesystem::u8path(executable)).parent_path();
  for (const auto& directory :
       {exe_dir, std::filesystem::current_path() / "external" / "ffmpeg" / "bin"}) {
    const auto candidate = directory / filename;
    if (std::filesystem::is_regular_file(candidate))
      return candidate.u8string();
  }
  return filename; // Native process API searches PATH, never a command shell.
}

#if defined(_WIN32)
struct Handle {
  HANDLE value = nullptr;

  ~Handle() {
    if (value && value != INVALID_HANDLE_VALUE)
      CloseHandle(value);
  }

  void close() {
    if (value && value != INVALID_HANDLE_VALUE)
      CloseHandle(value);
    value = nullptr;
  }
};

std::wstring quote(const std::string& argument) {
  const auto wide = std::filesystem::u8path(argument).wstring();
  std::wstring result = L"\"";
  size_t slashes = 0;
  for (wchar_t c : wide) {
    if (c == L'\\') {
      ++slashes;
      continue;
    }
    result.append(c == L'"' ? slashes * 2 + 1 : slashes, L'\\');
    slashes = 0;
    result += c;
  }
  result.append(slashes * 2, L'\\');
  return result + L'"';
}

void run(const std::vector<std::string>& args, const Sink& sink) {
  SECURITY_ATTRIBUTES security{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
  Handle reader, writer, null_input, errors, process, thread;
  if (!CreatePipe(&reader.value, &writer.value, &security, 0) ||
      !SetHandleInformation(reader.value, HANDLE_FLAG_INHERIT, 0))
    throw std::runtime_error("reference decoder: cannot create pipe");
  null_input.value = CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
                                 &security, OPEN_EXISTING, 0, nullptr);
  // Use an inherited stderr when available; a detached CLI has no console.
  HANDLE stderr_handle = GetStdHandle(STD_ERROR_HANDLE);
  if (stderr_handle && stderr_handle != INVALID_HANDLE_VALUE)
    DuplicateHandle(GetCurrentProcess(), stderr_handle, GetCurrentProcess(), &errors.value, 0, TRUE,
                    DUPLICATE_SAME_ACCESS);
  if (!errors.value)
    errors.value = CreateFileW(L"NUL", GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, &security,
                               OPEN_EXISTING, 0, nullptr);
  STARTUPINFOW startup{};
  startup.cb = sizeof(startup);
  startup.dwFlags = STARTF_USESTDHANDLES;
  startup.hStdInput = null_input.value;
  startup.hStdOutput = writer.value;
  startup.hStdError = errors.value;
  std::wstring command;
  for (const auto& arg : args) {
    if (!command.empty())
      command += L' ';
    command += quote(arg);
  }
  PROCESS_INFORMATION info{};
  if (!CreateProcessW(nullptr, command.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr,
                      nullptr, &startup, &info))
    throw std::runtime_error("reference decoder: cannot launch " + args.front() +
                             " (install FFmpeg and ffprobe)");
  process.value = info.hProcess;
  thread.value = info.hThread;
  writer.close();
  try {
    std::array<uint8_t, 65536> bytes;
    DWORD count = 0;
    for (;;) {
      if (!ReadFile(reader.value, bytes.data(), static_cast<DWORD>(bytes.size()), &count,
                    nullptr)) {
        if (GetLastError() == ERROR_BROKEN_PIPE)
          break;
        throw std::runtime_error("reference decoder: pipe read failed");
      }
      if (!count)
        break;
      sink(bytes.data(), count);
    }
    WaitForSingleObject(process.value, INFINITE);
    DWORD status = 1;
    if (!GetExitCodeProcess(process.value, &status) || status != 0)
      throw std::runtime_error("reference decoder: " + args.front() + " failed");
  } catch (...) {
    TerminateProcess(process.value, 1);
    WaitForSingleObject(process.value, INFINITE);
    throw;
  }
}
#else
void run(const std::vector<std::string>& args, const Sink& sink) {
  std::vector<char*> argv;
  for (const auto& arg : args)
    argv.push_back(const_cast<char*>(arg.c_str()));
  argv.push_back(nullptr);
  int descriptors[2];
  if (pipe(descriptors))
    throw std::runtime_error("reference decoder: cannot create pipe");
  const pid_t child = fork();
  if (child < 0) {
    close(descriptors[0]);
    close(descriptors[1]);
    throw std::runtime_error("reference decoder: fork failed");
  }
  if (child == 0) {
    close(descriptors[0]);
    if (dup2(descriptors[1], STDOUT_FILENO) < 0)
      _exit(127);
    close(descriptors[1]);
    execvp(argv[0], argv.data());
    _exit(127);
  }
  close(descriptors[1]);
  try {
    std::array<uint8_t, 65536> bytes;
    for (;;) {
      const auto count = read(descriptors[0], bytes.data(), bytes.size());
      if (count < 0 && errno == EINTR)
        continue;
      if (count < 0)
        throw std::runtime_error("reference decoder: pipe read failed");
      if (count == 0)
        break;
      sink(bytes.data(), static_cast<size_t>(count));
    }
  } catch (...) {
    close(descriptors[0]);
    kill(child, SIGKILL);
    while (waitpid(child, nullptr, 0) < 0 && errno == EINTR) {
    }
    throw;
  }
  close(descriptors[0]);
  int status = 0;
  pid_t waited;
  do {
    waited = waitpid(child, &status, 0);
  } while (waited < 0 && errno == EINTR);
  if (waited < 0 || !WIFEXITED(status) || WEXITSTATUS(status) != 0)
    throw std::runtime_error("reference decoder: " + args.front() +
                             " failed (install FFmpeg and ffprobe)");
}
#endif

double duration_of(const json::Value* value) {
  if (!value)
    return 0;
  const auto* duration = value->find("duration");
  if (!duration || !duration->is_string() || duration->as_string() == "N/A")
    return 0;
  return std::stod(duration->as_string());
}

double start_of(const json::Value* value) {
  if (!value)
    return 0;
  const auto* start = value->find("start_time");
  if (!start || !start->is_string() || start->as_string() == "N/A")
    return 0;
  const double result = std::stod(start->as_string());
  if (!std::isfinite(result))
    throw std::runtime_error("reference media: invalid stream start time");
  return result;
}

} // namespace
#endif

ReferenceMedia decode_reference_file(const std::string& path, bool video, const char* executable) {
#if !SLOPFAB_WITH_FFMPEG
  (void)path;
  (void)video;
  (void)executable;
  throw std::runtime_error(
      "reference file decoding requires SLOPFAB_WITH_FFMPEG=ON; DLL hosts can supply decoded frames/PCM");
#else
  const auto input = std::filesystem::absolute(std::filesystem::u8path(path));
  if (!std::filesystem::is_regular_file(input))
    throw std::runtime_error("reference media file not found: " + path);
  std::string metadata;
  run({tool_path("ffprobe", executable), "-v", "error", "-show_entries",
       "format=duration:stream=index,codec_type,width,height,duration,start_time,sample_rate,channels",
       "-of", "json", input.u8string()},
      [&](const uint8_t* bytes, size_t count) {
        if (metadata.size() + count > 1024 * 1024)
          throw std::runtime_error("reference decoder: excessive metadata");
        metadata.append(reinterpret_cast<const char*>(bytes), count);
      });
  const auto info = json::parse(metadata);
  const auto* streams = info.find("streams");
  if (!streams)
    throw std::runtime_error("reference media: no streams");
  const json::Value* picture = nullptr;
  const json::Value* sound = nullptr;
  for (const auto& stream : streams->as_array()) {
    const auto* kind = stream.find("codec_type");
    if (kind && kind->as_string() == "video" && !picture)
      picture = &stream;
    if (kind && kind->as_string() == "audio" && !sound)
      sound = &stream;
  }
  const auto* selected = video ? picture : sound;
  if (!selected)
    throw std::runtime_error(video ? "reference media: no video stream"
                                   : "reference media: no audio stream");
  double duration = duration_of(selected);
  if (duration == 0)
    duration = duration_of(info.find("format"));
  if (!std::isfinite(duration) || duration < 2 || duration > 15)
    throw std::runtime_error(
        "reference media: file duration must be known and between 2 and 15 seconds; trim the source first");
  const std::string ffmpeg = tool_path("ffmpeg", executable);
  ReferenceMedia result;
  if (video) {
    const auto* width_field = picture->find("width");
    const auto* height_field = picture->find("height");
    if (!width_field || !height_field)
      throw std::runtime_error("reference media: missing video dimensions");
    const auto width64 = width_field->as_int(), height64 = height_field->as_int();
    if (width64 <= 0 || height64 <= 0 || width64 > 8192 || height64 > 8192)
      throw std::runtime_error("reference media: decoder supports dimensions up to 8192 pixels");
    const int width = static_cast<int>(width64), height = static_cast<int>(height64);
    result = ReferenceMedia::video(duration);
    const size_t frame_bytes = static_cast<size_t>(width) * height * 3;
    if (std::ceil(duration * 24) * static_cast<double>(frame_bytes) > 2.0 * 1024 * 1024 * 1024)
      throw std::runtime_error(
          "reference decoder: decoded video exceeds 2 GiB; resize the source first");
    std::vector<uint8_t> frame(frame_bytes);
    size_t filled = 0, index = 0;
    run({ffmpeg, "-v", "error", "-nostdin", "-noautorotate", "-i", input.u8string(), "-map",
         "0:v:0", "-an", "-vf", "setpts=PTS-STARTPTS,fps=24", "-t", std::to_string(duration),
         "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"},
        [&](const uint8_t* bytes, size_t count) {
          while (count) {
            const size_t take = std::min(count, frame_bytes - filled);
            std::memcpy(frame.data() + filled, bytes, take);
            filled += take;
            bytes += take;
            count -= take;
            if (filled == frame_bytes) {
              const double timestamp = static_cast<double>(index++) / 24;
              if (timestamp < duration)
                result.append_frame(frame.data(), frame.size(), width, height, size_t(width) * 3, 3,
                                    timestamp);
              else
                throw std::runtime_error("reference decoder: too many video frames");
              filled = 0;
            }
          }
        });
    if (filled)
      throw std::runtime_error("reference decoder: truncated RGB frame");
  }
  if (sound) {
    // Preserve the native rate and channel count. This keeps FFmpeg's resampler
    // and mono upmix gain out of the model's shared preprocessing contract.
    const auto* rate_field = sound->find("sample_rate");
    const auto* channels_field = sound->find("channels");
    if (!rate_field || !channels_field)
      throw std::runtime_error("reference audio: missing PCM format");
    const int rate = std::stoi(rate_field->as_string());
    const auto channels64 = channels_field->as_int();
    if (rate <= 0 || (channels64 != 1 && channels64 != 2))
      throw std::runtime_error(
          "reference audio: expected a positive native rate and mono or stereo sound");
    const int channels = static_cast<int>(channels64);
    const double relative_start = video ? start_of(sound) - start_of(picture) : 0;
    const double start = std::max(0.0, relative_start);
    if (start >= duration)
      throw std::runtime_error("reference audio: soundtrack starts after the video ends");
    const double audio_duration = duration - start;
    const size_t max_floats = static_cast<size_t>(std::floor(audio_duration * rate)) * channels;
    if (max_floats > 256 * 1024 * 1024 / sizeof(float))
      throw std::runtime_error("reference audio: decoded PCM exceeds 256 MiB");
    const std::string audio_filter =
        relative_start < 0
            ? "atrim=start=" + std::to_string(-relative_start) + ",asetpts=PTS-STARTPTS"
            : "asetpts=PTS-STARTPTS";
    std::vector<uint8_t> pcm_bytes;
    run({ffmpeg, "-v", "error", "-nostdin", "-i", input.u8string(), "-map", "0:a:0", "-vn", "-af",
         audio_filter, "-t", std::to_string(audio_duration), "-f", "f32le", "pipe:1"},
        [&](const uint8_t* bytes, size_t count) {
          if (pcm_bytes.size() + count > (max_floats + channels) * sizeof(float))
            throw std::runtime_error("reference decoder: too much audio");
          pcm_bytes.insert(pcm_bytes.end(), bytes, bytes + count);
        });
    if (pcm_bytes.empty() || pcm_bytes.size() % (channels * sizeof(float)))
      throw std::runtime_error("reference decoder: incomplete PCM audio");
    std::vector<float> samples(std::min(max_floats, pcm_bytes.size() / sizeof(float)));
    std::memcpy(samples.data(), pcm_bytes.data(), samples.size() * sizeof(float));
    // Lossy codecs can overshoot full scale. The public PCM boundary is [-1,1].
    for (float& sample : samples) {
      if (!std::isfinite(sample))
        throw std::runtime_error("reference decoder: non-finite audio sample");
      sample = std::clamp(sample, -1.0f, 1.0f);
    }
    if (video)
      result.set_audio(samples.data(), samples.size(), channels, rate, start);
    else
      result = ReferenceMedia::audio(samples.data(), samples.size(), channels, rate);
  }
  result.validate();
  return result;
#endif
}

} // namespace slopfab::cli
