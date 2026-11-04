#include "frames.h"

namespace slopfab::cli {
namespace fs = std::filesystem;

namespace {
[[noreturn]] void invalid(const std::string& message) {
  throw CliError("INVALID_REQUEST", message);
}

fs::path path(const std::string& s) {
  return fs::absolute(fs::u8path(s)).lexically_normal();
}
}

static int frames_command_impl(const std::vector<std::string>& args) {
  Object o;
  std::vector<std::string> generation;
  const bool generate = !args.empty() && args[0] == "generate";
  for (size_t i = generate ? 1 : 0; i < args.size(); ++i) {
    const auto& a = args[i];
    if (a == "--" && generate) {
      generation.assign(args.begin() + i + 1, args.end());
      break;
    }
    if (a == "--overwrite") {
      o["overwrite"] = Value(true);
      continue;
    }
    if (a.rfind("--", 0) != 0 && !generate && !o.count("input")) {
      o["input"] = str(a);
      continue;
    }
    if (i + 1 >= args.size())
      invalid("missing value for " + a);
    auto value = args[++i];
    if (a == "--out")
      o["output"] = str(value);
    else if (a == "--ffmpeg" || a == "--ffprobe")
      o[a.substr(2)] = str(value);
    else if (a == "--grid") {
      auto x = value.find('x');
      if (x == std::string::npos)
        invalid("grid must be COLSxROWS");
      o["columns"] = parse_cli_json(value.substr(0, x));
      o["rows"] = parse_cli_json(value.substr(x + 1));
    } else if (a == "--frame" || a == "--time" || a == "--width" || a == "--start" || a == "--end")
      o[a.substr(2)] = parse_cli_json(value);
    else
      invalid("unknown frames option: " + a);
  }
  auto check = [] {
    if (cancelled())
      throw_cancelled();
  };
  if (generate) {
    if (generation.empty())
      invalid("video frames generate needs -- followed by generation arguments");
    validate_frame_options(Value(o));
    // Capture once per generated output; automatic names also support --count batches.
    if (!o.count("output"))
      invalid("video frames generate requires --out review.png");
    if (agent().samples_observer)
      invalid("a sample observer is already installed");
    int count = 0;

    struct Reset {
      ~Reset() {
        agent().samples_observer = {};
      }
    } reset;

    agent().samples_observer = [&](RunSamples& samples, Delivery& delivery) {
      Object current = o;
      if (count++) {
        auto p = path(current["output"].as_string());
        p = p.parent_path() /
            (p.stem().u8string() + "-" + std::to_string(count) + p.extension().u8string());
        current["output"] = str(p.u8string());
      }
      if (path(current["output"].as_string()) == path(delivery.output))
        invalid("review output must differ from video output");
      delivery.artifacts.push_back(
          frames_from_samples(samples, delivery.fps, Value(current), check));
    };
    return finish_command(run_generation_arguments(generation));
  }
  auto result = extract_frames(Value(o), check);
  emit({{"type", str("result")}, {"ok", Value(true)}, {"data", result}}, true);
  return 0;
}

int frames_command(const std::vector<std::string>& args) {
  try {
    return frames_command_impl(args);
  } catch (const CliError&) {
    throw;
  } catch (const std::exception& e) {
    throw CliError("FRAME_EXTRACTION_FAILED", e.what(), 6);
  }
}
}
