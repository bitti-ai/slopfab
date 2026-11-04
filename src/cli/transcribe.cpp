#include "transcribe.h"
#include "agent.h"
#include "native_media.h"
#include "pipe_process.h"
#include "slopfab/transcribe.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <set>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace slopfab::cli {
namespace {
Value result_json(const whisper::Result& r) {
  Array segments;
  for (const auto& s : r.segments) {
    Array words;
    for (const auto& w : s.words)
      words.emplace_back(
          Object{{"start", num(w.start)}, {"end", num(w.end)}, {"text", str(w.text)}});
    Object o{{"start", num(s.start)},
             {"end", num(s.end)},
             {"text", str(s.text)},
             {"average_logprob", num(s.average_logprob)},
             {"no_speech_probability", num(s.no_speech_probability)}};
    if (!s.words.empty())
      o["words"] = Value(words);
    segments.emplace_back(o);
  }
  return Value(Object{{"model", str(r.model)},
                      {"backend", str(r.backend)},
                      {"language", str(r.language)},
                      {"duration", num(r.duration)},
                      {"text", str(r.text)},
                      {"segments", Value(segments)}});
}

void write_output(const std::filesystem::path& path, const std::string& contents) {
  auto temporary = path;
  temporary += "." + unique_id() + ".tmp";
  try {
    std::ofstream out(temporary, std::ios::binary);
    out.write(contents.data(), std::streamsize(contents.size()));
    out.close();
    if (!out)
      throw CliError("OUTPUT_FAILED", "cannot write transcript", 6);
#ifdef _WIN32
    if (!MoveFileExW(temporary.c_str(), path.c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
      throw CliError("OUTPUT_FAILED", "cannot publish transcript", 6);
#else
    std::filesystem::rename(temporary, path);
#endif
  } catch (...) {
    std::error_code ec;
    std::filesystem::remove(temporary, ec);
    throw;
  }
}
}

int transcribe_command(const std::vector<std::string>& args) {
  whisper::Options options;
  std::string input, output, format, ffmpeg;
  std::set<std::string> seen;
  for (size_t i = 0; i < args.size(); ++i) {
    const auto& a = args[i];
    if (a.rfind("--", 0) != 0) {
      if (!input.empty())
        throw CliError("INVALID_REQUEST", "transcribe takes one input file");
      input = a;
      continue;
    }
    if (!seen.insert(a).second)
      throw CliError("INVALID_REQUEST", "repeated option: " + a);
    if (a == "--word-timestamps") {
      options.word_timestamps = true;
      continue;
    }
    if (a == "--no-context") {
      options.condition_on_previous_text = false;
      continue;
    }
    if (a == "--overwrite") {
      agent().overwrite = true;
      continue;
    }
    if (a != "--model" && a != "--variant" && a != "--backend" && a != "--language" &&
        a != "--prompt" && a != "--out" && a != "--format" && a != "--ffmpeg" && a != "--beam-size")
      throw CliError("INVALID_REQUEST", "unknown transcribe option: " + a);
    if (++i == args.size() || args[i].empty())
      throw CliError("INVALID_REQUEST", "missing value for " + a);
    const auto& v = args[i];
    if (a == "--model")
      options.model = v;
    else if (a == "--variant")
      options.variant = v;
    else if (a == "--backend")
      options.backend = v;
    else if (a == "--language")
      options.language = v;
    else if (a == "--prompt")
      options.prompt = v;
    else if (a == "--out")
      output = v;
    else if (a == "--format")
      format = v;
    else if (a == "--ffmpeg")
      ffmpeg = v;
    else {
      if (v.find_first_not_of("0123456789") != std::string::npos || v.size() > 2)
        throw CliError("INVALID_REQUEST", "beam size must be 1..10");
      options.beam_size = std::stoi(v);
    }
  }
  if (input.empty() || options.model.empty())
    throw CliError("INVALID_REQUEST",
                   "use transcribe FILE --model MODEL.safetensors; see slopfab help transcribe");
  if (options.beam_size < 1 || options.beam_size > 10)
    throw CliError("INVALID_REQUEST", "beam size must be 1..10");
  if (options.backend != "auto" && options.backend != "cuda" && options.backend != "vulkan" &&
      options.backend != "cpu")
    throw CliError("INVALID_REQUEST", "backend must be auto, cuda, vulkan or cpu");
  auto source = std::filesystem::absolute(std::filesystem::u8path(input));
  if (!std::filesystem::is_regular_file(source))
    throw CliError("INPUT_NOT_FOUND", "audio file not found: " + input, 3);
  auto weights = std::filesystem::absolute(std::filesystem::u8path(options.model));
  if (std::filesystem::is_directory(weights))
    weights /= "model.safetensors";
  if (!std::filesystem::is_regular_file(weights))
    throw CliError("INPUT_NOT_FOUND", "Whisper weights not found: " + weights.u8string(), 3);
  auto destination = output.empty() ? std::filesystem::path{}
                                    : std::filesystem::absolute(std::filesystem::u8path(output));
  if (!output.empty()) {
    if (std::filesystem::weakly_canonical(destination) ==
            std::filesystem::weakly_canonical(source) ||
        std::filesystem::weakly_canonical(destination) ==
            std::filesystem::weakly_canonical(weights))
      throw CliError("INVALID_REQUEST", "transcript output must differ from input and weights");
    if (format.empty()) {
      format = destination.extension().u8string();
      if (!format.empty())
        format.erase(0, 1);
    }
  }
  if (format.empty())
    format = "json";
  if (format != "json" && format != "txt" && format != "srt" && format != "vtt")
    throw CliError("INVALID_REQUEST", "format must be json, txt, srt or vtt");
  start_request();
  Delivery delivery(output.empty() ? "." : output);
  if (!output.empty())
    delivery.reserve(destination.u8string());
  auto check = []() {
    if (cancelled())
      throw_cancelled();
  };
  check();
  std::vector<float> pcm;
  int rate = 16000, channels = 1;
  auto extension = source.extension().u8string();
  std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) {
    return char(std::tolower(c));
  });
  if (extension == ".wav") {
    auto wav = read_wav(source.u8string());
    pcm = std::move(wav.interleaved);
    rate = wav.sample_rate;
    channels = wav.channels;
  } else {
    PipeProcess child({ffmpeg.empty() ? media_tool("ffmpeg", "ffmpeg") : ffmpeg, "-nostdin", "-v",
                       "error", "-i", source.u8string(), "-vn", "-ac", "1", "-ar", "16000", "-f",
                       "f32le", "pipe:1"},
                      false);
    std::vector<uint8_t> bytes;
    uint8_t chunk[65536];
    for (;;) {
      check();
      auto n = child.read(chunk, sizeof(chunk));
      if (!n)
        break;
      if (bytes.size() + n > 512ull * 1024 * 1024)
        throw CliError("INVALID_REQUEST", "decoded audio exceeds 512 MiB");
      bytes.insert(bytes.end(), chunk, chunk + n);
    }
    child.finish();
    if (bytes.size() % 4 || bytes.empty())
      throw CliError("INVALID_REQUEST", "decoder returned invalid PCM");
    pcm.resize(bytes.size() / 4);
    std::memcpy(pcm.data(), bytes.data(), bytes.size());
  }
  std::string previous;
  double previous_fraction = -1;
  options.progress = [&](double fraction, const std::string& stage) {
    check();
    if (agent().enabled && (stage != previous || fraction != previous_fraction) &&
        stage != "decoder" && stage != "encoder")
      emit({{"type", str("progress")}, {"stage", str(stage)}, {"fraction", num(fraction)}});
    previous = stage;
    previous_fraction = fraction;
    return true;
  };
  whisper::Result result;
  try {
    result = whisper::transcribe(pcm, rate, channels, options);
  } catch (const std::invalid_argument& e) {
    throw CliError("INVALID_REQUEST", e.what());
  }
  auto data = result_json(result);
  std::string contents = format == "json"  ? json::stringify(data) + "\n"
                         : format == "txt" ? result.text + "\n"
                                           : whisper::subtitles(result, format == "vtt");
  if (!output.empty())
    write_output(destination, contents);
  if (agent().enabled) {
    Object response{{"type", str("result")}, {"ok", Value(true)}, {"data", data}};
    if (!output.empty())
      response["output"] = str(destination.u8string());
    emit(std::move(response), true);
  } else
    std::fputs(contents.c_str(), stdout);
  return 0;
}
}
