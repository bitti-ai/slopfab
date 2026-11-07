#include "syncnet.h"
#include "agent.h"
#include "pipe_process.h"
#include "../syncnet/network.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <deque>
#include <fstream>
#include <set>
#include <thread>

namespace slopfab::cli {
namespace {
namespace sn = slopfab::syncnet;

void check() {
  if (cancelled())
    throw_cancelled();
}

struct Reader {
  PipeProcess pipe;
  std::atomic<bool> done{false};
  std::thread watcher;

  explicit Reader(const std::vector<std::string>& args)
      : pipe(args, false), watcher([this] {
          while (!done) {
            if (cancelled()) {
              pipe.cancel();
              break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
          }
        }) {
  }

  ~Reader() {
    done = true;
    watcher.join();
  }

  size_t read(void* data, size_t size) {
    check();
    size_t n = 0;
    while (n < size) {
      size_t got = pipe.read(static_cast<char*>(data) + n, size - n);
      if (!got)
        break;
      n += got;
      check();
    }
    return n;
  }

  void finish() {
    check();
    pipe.finish();
    check();
  }
};

struct Spool {
  std::filesystem::path path =
      std::filesystem::temp_directory_path() / ("slopfab-syncnet-" + unique_id() + ".rgb");
  std::fstream file;

  Spool() : file(path, std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc) {
    if (!file)
      throw CliError("OUTPUT_FAILED", "cannot create SyncNet temporary video", 6);
  }

  ~Spool() {
    file.close();
    std::error_code ec;
    std::filesystem::remove(path, ec);
  }
};

double number(const std::string& v, double lo, double hi, bool integer = false) {
  size_t used = 0;
  double n = 0;
  try {
    n = std::stod(v, &used);
  } catch (...) {
    used = 0;
  }
  if (used != v.size() || !used || !std::isfinite(n) || n < lo || n > hi ||
      (integer && n != std::floor(n)))
    throw CliError("INVALID_REQUEST", "numeric option out of range: " + v);
  return n;
}

Value probe(const std::string& executable, const std::string& input) {
  Reader r({executable, "-v", "error", "-show_streams", "-of", "json", input});
  std::string text;
  std::array<char, 4096> bytes;
  while (size_t n = r.read(bytes.data(), bytes.size())) {
    text.append(bytes.data(), n);
    if (text.size() > 1024 * 1024)
      throw CliError("INVALID_INPUT", "excessive media metadata", 3);
  }
  r.finish();
  return json::parse(text);
}

Array box_json(const sn::Box& b) {
  return {num(b.x1), num(b.y1), num(b.x2), num(b.y2)};
}
}

int syncnet_command(const std::vector<std::string>& args) {
  std::string input, model, detector, output, ffmpeg, ffprobe, backend = "auto";
  double start = 0, duration = 30, min_track = 1.5, threshold = .9;
  int shift = 15, detect_every = 5, detect_size = 320;
  bool cropped = false;
  std::set<std::string> seen;
  for (size_t i = 0; i < args.size(); ++i) {
    const auto& a = args[i];
    if (a.rfind("--", 0) != 0) {
      if (!input.empty())
        throw CliError("INVALID_REQUEST", "syncnet takes one input video");
      input = a;
      continue;
    }
    if (!seen.insert(a).second)
      throw CliError("INVALID_REQUEST", "repeated option: " + a);
    if (a == "--cropped") {
      cropped = true;
      continue;
    }
    if (a == "--overwrite") {
      agent().overwrite = true;
      continue;
    }
    if (a != "--model" && a != "--face-model" && a != "--out" && a != "--backend" &&
        a != "--ffmpeg" && a != "--ffprobe" && a != "--start" && a != "--duration" &&
        a != "--min-track" && a != "--vshift" && a != "--detect-every" && a != "--detect-size" &&
        a != "--face-threshold")
      throw CliError("INVALID_REQUEST", "unknown syncnet option: " + a);
    if (++i == args.size() || args[i].empty())
      throw CliError("INVALID_REQUEST", "missing value for " + a);
    const auto& v = args[i];
    if (a == "--model")
      model = v;
    else if (a == "--face-model")
      detector = v;
    else if (a == "--out")
      output = v;
    else if (a == "--backend")
      backend = v;
    else if (a == "--ffmpeg")
      ffmpeg = v;
    else if (a == "--ffprobe")
      ffprobe = v;
    else if (a == "--start")
      start = number(v, 0, 86400);
    else if (a == "--duration")
      duration = number(v, .24, 600);
    else if (a == "--min-track")
      min_track = number(v, .24, 600);
    else if (a == "--vshift")
      shift = int(number(v, 0, 250, true));
    else if (a == "--detect-every")
      detect_every = int(number(v, 1, 10, true));
    else if (a == "--detect-size")
      detect_size = int(number(v, 128, 1280, true));
    else
      threshold = number(v, .5, .9999);
  }
  if (input.empty() || model.empty() || (!cropped && detector.empty()))
    throw CliError("INVALID_REQUEST",
                   "use video syncnet VIDEO --model syncnet_v2.model "
                   "--face-model sfd_face.pth (or --cropped for an existing face crop)");
  if (cropped && !detector.empty())
    throw CliError("INVALID_REQUEST", "--cropped excludes --face-model");
  if (backend != "auto" && backend != "cpu" && backend != "vulkan")
    throw CliError("INVALID_REQUEST", "backend must be auto, cpu or vulkan");
  auto source = std::filesystem::absolute(std::filesystem::u8path(input));
  if (!std::filesystem::is_regular_file(source))
    throw CliError("INPUT_NOT_FOUND", "video not found", 3);
  input = source.u8string();
  if (output.empty())
    output =
        (std::filesystem::path("output") / (source.stem().u8string() + "-syncnet.json")).u8string();
  auto destination = std::filesystem::absolute(std::filesystem::u8path(output));
  for (const auto& p : {input, model, detector})
    if (!p.empty() && std::filesystem::weakly_canonical(destination) ==
                          std::filesystem::weakly_canonical(std::filesystem::u8path(p)))
      throw CliError("INVALID_REQUEST", "report output must differ from video and models");
  start_request();
  sn::Weights weights, face_weights;
  try {
    weights = sn::load_weights(model);
    if (!cropped)
      face_weights = sn::load_weights(detector, true);
  } catch (const std::exception& e) {
    throw CliError("INVALID_MODEL", e.what(), 3);
  }
  sn::Network network(std::move(weights), backend, check);
  std::unique_ptr<sn::Network> face;
  if (!cropped)
    face = std::make_unique<sn::Network>(std::move(face_weights), network.backend(), check);
  Delivery delivery(output);
  delivery.reserve(destination.u8string());
  if (ffmpeg.empty())
    ffmpeg = media_tool("ffmpeg", agent().executable.c_str());
  if (ffprobe.empty())
    ffprobe = media_tool("ffprobe", agent().executable.c_str());
  auto metadata = probe(ffprobe, input);
  const Value* video = nullptr;
  bool has_audio = false;
  if (const auto* streams = metadata.find("streams"))
    for (const auto& stream : streams->as_array()) {
      auto type = stream.find("codec_type");
      if (!type)
        continue;
      if (type->as_string() == "video" && !video)
        video = &stream;
      if (type->as_string() == "audio")
        has_audio = true;
    }
  if (!video || !has_audio)
    throw CliError("INVALID_INPUT", "SyncNet requires both video and audio streams", 3);
  if (!video->find("width") || !video->find("height"))
    throw CliError("INVALID_INPUT", "video dimensions are missing", 3);
  int width = int(video->find("width")->as_int()), height = int(video->find("height")->as_int());
  if (width < 1 || height < 1 || width > 16384 || height > 16384)
    throw CliError("INVALID_INPUT", "invalid video dimensions", 3);
  if (const auto* side = video->find("side_data_list"))
    for (const auto& item : side->as_array())
      if (const auto* rotation = item.find("rotation"))
        if (std::abs(int(rotation->as_number())) % 180 == 90)
          std::swap(width, height);
  double origin = 0;
  if (const auto* t = video->find("start_time"))
    origin = number(t->as_string(), -86400, 86400);
  double scale = std::min(1., 1280. / std::max(width, height));
  width = std::max(1, int(std::round(width * scale)));
  height = std::max(1, int(std::round(height * scale)));
  const std::string pts = "PTS-(" + std::to_string(origin + start) + ")/TB";
  std::vector<int16_t> pcm;
  {
    Reader r({ffmpeg,    "-nostdin",
              "-v",      "error",
              "-copyts", "-i",
              input,     "-map",
              "0:a:0",   "-vn",
              "-af",     "asetpts=" + pts + ",aresample=16000:async=1:first_pts=0",
              "-t",      std::to_string(duration),
              "-ac",     "1",
              "-ar",     "16000",
              "-f",      "s16le",
              "pipe:1"});
    std::array<int16_t, 4096> data;
    while (size_t bytes = r.read(data.data(), data.size() * 2)) {
      if (bytes % 2)
        throw CliError("INVALID_INPUT", "truncated audio sample", 3);
      pcm.insert(pcm.end(), data.begin(), data.begin() + bytes / 2);
      if (pcm.size() > size_t(std::ceil(duration * 16000)) + 16000)
        throw CliError("INVALID_INPUT", "audio exceeds requested duration", 3);
    }
    r.finish();
  }
  if (pcm.size() < 3840)
    throw CliError("INVALID_INPUT", "audio is too short for SyncNet", 3);
  double energy = 0;
  for (auto v : pcm)
    energy += double(v) * v;
  const double rms = std::sqrt(energy / pcm.size());
  if (rms < 1)
    throw CliError("INVALID_INPUT", "audio is silent; synchronization cannot be assessed", 3);
  auto features = sn::mfcc(pcm);
  Spool spool;
  sn::Tracker tracker;
  const size_t frame_bytes = size_t(width) * height * 3;
  std::vector<uint8_t> bytes(frame_bytes);
  sn::Image frame(height, width, 3);
  std::array<double, 48> previous{};
  int count = 0, scenes = 0;
  {
    Reader r({ffmpeg, "-nostdin", "-v", "error", "-copyts", "-i", input, "-map", "0:v:0", "-an",
              "-vf",
              "setpts=" + pts + ",fps=25:start_time=0,scale=" + std::to_string(width) + ":" +
                  std::to_string(height) + ":flags=bilinear",
              "-t", std::to_string(duration), "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"});
    while (size_t n = r.read(bytes.data(), bytes.size())) {
      if (n != frame_bytes)
        throw CliError("INVALID_INPUT", "truncated video frame", 3);
      if (count >= int(std::ceil(duration * 25)) + 1 ||
          uint64_t(count + 1) * frame_bytes > 8ull * 1024 * 1024 * 1024)
        throw CliError("INVALID_INPUT",
                       "temporary video exceeds 8 GiB; select a shorter --duration", 3);
      spool.file.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
      if (!spool.file)
        throw CliError("OUTPUT_FAILED", "cannot write temporary video (disk full)", 6);
      std::array<double, 48> histogram{};
      for (size_t p = 0; p < bytes.size(); p += 3 * 16)
        for (size_t c = 0; c < 3; ++c)
          histogram[c * 16 + bytes[p + c] / 16] += 1.;
      double sum = 0;
      for (double v : histogram)
        sum += v;
      double difference = 0;
      for (size_t i = 0; i < 48; ++i) {
        histogram[i] /= sum;
        difference += std::abs(histogram[i] - previous[i]);
      }
      bool cut = count > 0 && difference > .6;
      previous = histogram;
      if (cut)
        ++scenes;
      if (cropped)
        tracker.update(count, {{0, 0, float(width), float(height), 1}}, cut);
      else if (count % detect_every == 0 || cut) {
        for (size_t i = 0; i < bytes.size(); ++i)
          frame.data[i] = bytes[i];
        float factor = std::min(1.f, float(detect_size) / std::max(width, height));
        int dw = std::max(32, int(std::round(width * factor))),
            dh = std::max(32, int(std::round(height * factor)));
        auto small = sn::resize_crop(frame, 0, 0, float(width), float(height), dh, dw);
        auto boxes = sn::detect(*face, small, float(threshold));
        for (auto& b : boxes) {
          b.x1 *= float(width) / dw;
          b.x2 *= float(width) / dw;
          b.y1 *= float(height) / dh;
          b.y2 *= float(height) / dh;
        }
        if (boxes.size() > 32)
          boxes.resize(32);
        tracker.update(count, boxes, cut);
      }
      ++count;
      if (count % 25 == 0) {
        if (agent().enabled)
          emit({{"type", str("progress")},
                {"stage", str("face_tracking")},
                {"seconds", num(count / 25.)}});
        else
          std::fprintf(stderr, "SyncNet: tracked %.1f seconds\n", count / 25.);
      }
    }
    r.finish();
  }
  spool.file.flush();
  int usable = std::min(count, int(pcm.size() / 640));
  Array tracks, skipped;
  for (const auto& track : tracker.tracks()) {
    check();
    auto crops = sn::track_boxes(track);
    int first = track.observations.front().frame;
    int length = std::min(int(crops.size()), usable - first);
    if (length < std::max(int(std::ceil(min_track * 25)), 2 * shift + 6)) {
      skipped.emplace_back(
          Object{{"id", num(track.id)}, {"reason", str("track_too_short_for_search")}});
      continue;
    }
    std::vector<std::vector<float>> visual, sound;
    std::deque<sn::Image> window;
    for (int f = 0; f < length - 1; ++f) {
      check();
      spool.file.seekg(std::streamoff(uint64_t(first + f) * frame_bytes));
      spool.file.read(reinterpret_cast<char*>(bytes.data()), std::streamsize(frame_bytes));
      if (!spool.file)
        throw CliError("INVALID_INPUT", "cannot read temporary video", 3);
      for (size_t i = 0; i < bytes.size(); ++i)
        frame.data[i] = bytes[i];
      auto b = cropped ? sn::Box{0, 0, float(width), float(height), 1} : crops[f];
      if (b.x2 <= b.x1 || b.y2 <= b.y1 || b.x2 - b.x1 > 8192 || b.y2 - b.y1 > 8192)
        throw CliError("INVALID_MODEL", "invalid detected crop", 3);
      window.push_back(sn::resize_crop(frame, b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1, 224, 224));
      if (window.size() < 5)
        continue;
      sn::Image lip(224, 224, 15), audio(13, 20, 1);
      for (int p = 0; p < 224 * 224; ++p)
        for (int c = 0; c < 3; ++c)
          for (int t = 0; t < 5; ++t)
            lip.data[size_t(p) * 15 + c * 5 + t] =
                std::round(window[t].data[size_t(p) * 3 + 2 - c]);
      int at = (first + f - 4) * 4;
      for (int c = 0; c < 13; ++c)
        for (int t = 0; t < 20; ++t)
          audio.data[c * 20 + t] = features.at(size_t(at + t) * 13 + c);
      visual.push_back(network.embedding(std::move(lip), false));
      sound.push_back(network.embedding(std::move(audio), true));
      window.pop_front();
      if (visual.size() % 25 == 0) {
        if (agent().enabled)
          emit({{"type", str("progress")},
                {"stage", str("syncnet")},
                {"track", num(track.id)},
                {"windows", num(double(visual.size()))}});
        else
          std::fprintf(stderr, "SyncNet: track %d, %zu windows\n", track.id, visual.size());
      }
    }
    auto s = sn::score(visual, sound, shift);
    Array distances, boxes;
    for (double d : s.shift_distances)
      distances.push_back(num(d));
    for (const auto& o : track.observations)
      boxes.emplace_back(Object{{"time", num(start + o.frame / 25.)},
                                {"box", Value(box_json(o.box))},
                                {"score", num(o.box.score)}});
    tracks.emplace_back(Object{{"id", num(track.id)},
                               {"start", num(start + first / 25.)},
                               {"end", num(start + (first + length) / 25.)},
                               {"windows", num(double(visual.size()))},
                               {"offset_frames", num(s.offset_frames)},
                               {"audio_shift_ms", num(s.offset_frames * 40)},
                               {"lse_d", num(s.distance)},
                               {"lse_c", num(s.confidence)},
                               {"zero_shift_distance", num(s.zero_shift_distance)},
                               {"search_boundary", Value(s.search_boundary)},
                               {"shift_distances", Value(distances)},
                               {"detections", Value(boxes)}});
  }
  if (tracks.empty())
    throw CliError(
        "NO_VALID_TRACKS",
        "no face tracks long enough for this search; try a larger --detect-size, longer --duration or smaller --vshift",
        3);
  Object report{
      {"schema_version", num(1)},
      {"input", str(input)},
      {"backend", str(network.backend())},
      {"model_sha256", str("961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442")},
      {"fps", num(25)},
      {"sample_rate", num(16000)},
      {"start", num(start)},
      {"duration", num(usable / 25.)},
      {"requested_duration", num(duration)},
      {"analysis_width", num(width)},
      {"analysis_height", num(height)},
      {"scene_cuts", num(scenes)},
      {"vshift", num(shift)},
      {"audio_rms_pcm16", num(rms)},
      {"offset_convention", str("negative audio_shift_ms advances audio; positive delays audio")},
      {"score_note",
       str("Lower LSE-D and higher LSE-C indicate a stronger match, not accuracy or probability. Scores can be unreliable for silence, non-speech or occluded mouths.")},
      {"tracks", Value(tracks)},
      {"skipped_tracks", Value(skipped)}};
  check();
  atomic_json(destination, Value(report));
  if (agent().enabled)
    emit({{"type", str("result")},
          {"ok", Value(true)},
          {"analysis", Value(report)},
          {"output", str(destination.u8string())},
          {"artifacts", Value(Array{Value(Object{{"kind", str("analysis")},
                                                 {"format", str("json")},
                                                 {"path", str(destination.u8string())}})})}},
         true);
  else
    std::puts(json::stringify(Value(report)).c_str());
  return 0;
}
} // namespace slopfab::cli
