#include "separate.h"
#include "agent.h"
#include "native_media.h"
#include "pipe_process.h"
#include "slopfab/audio/wav.h"
#include "slopfab/separate.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <set>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace slopfab::cli {
namespace {
void check_cancelled() {
  if (cancelled())
    throw_cancelled();
}

double number(const std::string& text, const std::string& option) {
  try {
    size_t consumed = 0;
    double value = std::stod(text, &consumed);
    if (consumed == text.size() && std::isfinite(value))
      return value;
  } catch (const std::exception&) {
  }
  throw CliError("INVALID_REQUEST", "invalid number for " + option);
}

std::vector<float> decode_audio(const std::filesystem::path& input, const std::string& ffmpeg) {
  std::string extension = input.extension().u8string();
  std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) {
    return char(std::tolower(c));
  });
  if (extension == ".wav") {
    auto wave = read_wav(input.u8string(), false);
    if (wave.sample_rate == 44100 && (wave.channels == 1 || wave.channels == 2)) {
      if (wave.channels == 2)
        return std::move(wave.interleaved);
      std::vector<float> stereo(wave.interleaved.size() * 2);
      for (size_t i = 0; i < wave.interleaved.size(); ++i)
        stereo[i * 2] = stereo[i * 2 + 1] = wave.interleaved[i];
      return stereo;
    }
  }
  PipeProcess child({ffmpeg.empty() ? media_tool("ffmpeg", "ffmpeg") : ffmpeg, "-nostdin", "-v",
                     "error", "-i", input.u8string(), "-vn", "-ac", "2", "-ar", "44100", "-f",
                     "f32le", "pipe:1"},
                    false);
  std::vector<uint8_t> bytes;
  uint8_t chunk[65536];
  for (;;) {
    check_cancelled();
    auto n = child.read(chunk, sizeof(chunk));
    if (!n)
      break;
    if (bytes.size() + n > (512ull << 20))
      throw CliError("INVALID_REQUEST", "decoded audio exceeds 512 MiB");
    bytes.insert(bytes.end(), chunk, chunk + n);
  }
  child.finish();
  if (bytes.empty() || bytes.size() % 8)
    throw CliError("INVALID_REQUEST", "decoder returned invalid stereo PCM");
  std::vector<float> pcm(bytes.size() / sizeof(float));
  std::memcpy(pcm.data(), bytes.data(), bytes.size());
  return pcm;
}
} // namespace

int separate_command(const std::vector<std::string>& args) {
  demucs::Options options;
  std::string input, output, ffmpeg, two_stems, backend = "cpu";
  std::set<std::string> seen;
  for (size_t i = 0; i < args.size(); ++i) {
    const auto& a = args[i];
    if (a.rfind("--", 0) != 0) {
      if (!input.empty())
        throw CliError("INVALID_REQUEST", "separate takes one input file");
      input = a;
      continue;
    }
    if (!seen.insert(a).second)
      throw CliError("INVALID_REQUEST", "repeated option: " + a);
    if (a == "--overwrite") {
      agent().overwrite = true;
      continue;
    }
    if (a != "--model" && a != "--out" && a != "--two-stems" && a != "--segment" &&
        a != "--overlap" && a != "--ffmpeg" && a != "--backend")
      throw CliError("INVALID_REQUEST", "unknown separate option: " + a);
    if (++i == args.size() || args[i].empty() || args[i].rfind("--", 0) == 0)
      throw CliError("INVALID_REQUEST", "missing value for " + a);
    const auto& value = args[i];
    if (a == "--model")
      options.model = value;
    else if (a == "--out")
      output = value;
    else if (a == "--two-stems")
      two_stems = value;
    else if (a == "--segment")
      options.segment = number(value, a);
    else if (a == "--overlap")
      options.overlap = number(value, a);
    else if (a == "--ffmpeg")
      ffmpeg = value;
    else
      backend = value;
  }
  if (input.empty() || options.model.empty())
    throw CliError("INVALID_REQUEST",
                   "use separate FILE --model MODEL.safetensors; see slopfab help audio separate");
  if (backend != "auto" && backend != "cpu")
    throw CliError("INVALID_REQUEST", "Demucs currently supports --backend cpu or auto (CPU)");
  const std::vector<std::string> sources{"drums", "bass", "other", "vocals"};
  auto selected = std::find(sources.begin(), sources.end(), two_stems);
  if (!two_stems.empty() && selected == sources.end())
    throw CliError("INVALID_REQUEST", "two-stems must be vocals, drums, bass or other");
  try {
    demucs::validate_options(options);
  } catch (const std::invalid_argument& e) {
    throw CliError("INVALID_REQUEST", e.what());
  }
  auto source = std::filesystem::absolute(std::filesystem::u8path(input));
  if (!std::filesystem::is_regular_file(source))
    throw CliError("INPUT_NOT_FOUND", "audio file not found: " + input, 3);
  auto weights = std::filesystem::absolute(std::filesystem::u8path(options.model));
  if (std::filesystem::is_directory(weights))
    weights /= "model.safetensors";
  if (!std::filesystem::is_regular_file(weights))
    throw CliError("INPUT_NOT_FOUND", "Demucs weights not found: " + weights.u8string(), 3);
  options.model = weights.u8string();
  auto destination = std::filesystem::absolute(
      output.empty() ? std::filesystem::path("output") / (source.stem().u8string() + "-stems")
                     : std::filesystem::u8path(output));
  std::vector<std::string> names = sources;
  if (!two_stems.empty())
    names = {two_stems, two_stems == "vocals" ? "instrumental" : "no_" + two_stems};
  std::vector<std::filesystem::path> paths, temporary;
  for (const auto& name : names) {
    auto path = destination / (name + ".wav");
    auto canonical = std::filesystem::weakly_canonical(path);
    if (canonical == std::filesystem::weakly_canonical(source) ||
        canonical == std::filesystem::weakly_canonical(weights))
      throw CliError("INVALID_REQUEST", "stem output must differ from input and weights");
    paths.push_back(path);
  }
  start_request();
  Delivery delivery(destination.u8string());
  for (const auto& path : paths)
    delivery.reserve(path.u8string());
  check_cancelled();
  auto pcm = decode_audio(source, ffmpeg);
  options.progress = [&](double fraction, const std::string& stage) {
    check_cancelled();
    if (agent().enabled)
      emit({{"type", str("progress")}, {"stage", str(stage)}, {"fraction", num(fraction)}});
    return true;
  };
  demucs::Result result;
  try {
    result = demucs::separate(pcm, options);
  } catch (const std::invalid_argument& e) {
    throw CliError("INVALID_REQUEST", e.what());
  }
  if (!two_stems.empty()) {
    auto stem = std::move(result.stems[size_t(selected - sources.begin())]);
    // Subtraction guarantees an exactly reconstructing pair, including model
    // residuals, instead of independently normalizing or clipping the tracks.
    for (size_t i = 0; i < pcm.size(); ++i)
      pcm[i] -= stem[i];
    result.stems.clear();
    result.stems.push_back(std::move(stem));
    result.stems.push_back(std::move(pcm));
  }
  Array artifacts;
  try {
    for (size_t i = 0; i < paths.size(); ++i) {
      check_cancelled();
      auto temp = paths[i];
      temp += "." + unique_id() + ".tmp";
      temporary.push_back(temp);
      audio::write_wav(temp.u8string(), result.stems[i], 2, 44100, audio::SampleFormat::kFloat32);
    }
    check_cancelled();
    for (size_t i = 0; i < paths.size(); ++i) {
#ifdef _WIN32
      if (!MoveFileExW(temporary[i].c_str(), paths[i].c_str(),
                       MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        throw CliError("OUTPUT_FAILED", "cannot publish stem: " + paths[i].u8string(), 6);
#else
      std::filesystem::rename(temporary[i], paths[i]);
#endif
      artifacts.emplace_back(Object{{"path", str(paths[i].u8string())},
                                    {"stem", str(names[i])},
                                    {"media_type", str("audio/wav")},
                                    {"bytes", num(double(std::filesystem::file_size(paths[i])))}});
    }
  } catch (...) {
    for (const auto& path : temporary) {
      std::error_code ec;
      std::filesystem::remove(path, ec);
    }
    throw;
  }
  if (agent().enabled)
    emit({{"type", str("result")},
          {"ok", Value(true)},
          {"artifacts", Value(artifacts)},
          {"data", Value(Object{{"model", str("htdemucs")},
                                {"backend", str("cpu")},
                                {"sample_rate", num(44100)},
                                {"channels", num(2)},
                                {"frames", num(double(result.stems[0].size() / 2))},
                                {"directory", str(destination.u8string())}})}},
         true);
  else
    for (const auto& path : paths)
      std::printf("%s\n", path.u8string().c_str());
  return 0;
}
} // namespace slopfab::cli
