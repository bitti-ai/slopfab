#include "prompting.h"
#include "agent.h"
#include "prompting_catalog.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <locale>
#include <set>
#include <sstream>

namespace slopfab::cli {
namespace {
std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
    return char(std::tolower(c));
  });
  return s;
}

[[noreturn]] void invalid(const std::string& message) {
  throw CliError("INVALID_REQUEST", message + "; run slopfab help prompts");
}

std::string family(const std::string& name) {
  GenerateRequest request;
  resolve_model_alias(request, name);
  // Guide discovery must not open weights, even for a configured alias.
  const auto& model = request.model_family;
  for (const char* supported : {"h3", "krea2", "moss", "fibo2", "yue2", "minimax_music3"})
    if (model == supported)
      return model;
  invalid("unsupported model: " + name);
}

void replace(std::string& text, const std::string& key, const std::string& value) {
  size_t pos = 0;
  while ((pos = text.find(key, pos)) != std::string::npos) {
    text.replace(pos, key.size(), value);
    pos += value.size();
  }
}

Object guide(const Value& catalog, const std::string& model, const std::string& use_case) {
  for (const auto& item : catalog.find("guides")->as_array()) {
    if (item.find("model")->as_string() != model || item.find("use_case")->as_string() != use_case)
      continue;
    auto result = item.as_object();
    result["model_guidance"] = *catalog.find("models")->find(model);
    for (const char* key : {"version", "reviewed", "notice", "workflow"})
      result[key] = *catalog.find(key);
    result["next"] = str(
        "Inspect references, fill remaining template placeholders, then validate the generation request. Examples are illustrations, not descriptions of your files.");
    return result;
  }
  invalid("no guide for " + model + "/" + use_case + "; use prompts list --model " + model);
}

struct Selection {
  std::string model, task, first, last, edit, duration, shots = "1";
  std::vector<std::string> images, videos, audio;
  bool instrumental = false, layout = false, continuation = false;
  bool timing = false;
};

Selection parse_selection(const std::vector<std::string>& args) {
  Selection s;
  std::set<std::string> seen;
  for (size_t i = 1; i < args.size(); ++i) {
    const auto& flag = args[i];
    const bool repeated =
        flag == "--reference-image" || flag == "--reference-video" || flag == "--reference-audio";
    if (!repeated && !seen.insert(flag).second)
      invalid("duplicate option: " + flag);
    if (flag == "--instrumental") {
      s.instrumental = true;
      continue;
    }
    if (flag == "--layout") {
      s.layout = true;
      continue;
    }
    if (flag == "--continuation") {
      s.continuation = true;
      continue;
    }
    if (flag != "--model" && flag != "--task" && flag != "--first-frame" &&
        flag != "--last-frame" && flag != "--edit-image" && flag != "--duration" &&
        flag != "--shots" && !repeated)
      invalid("unknown prompt selection option: " + flag);
    if (++i == args.size() || args[i].empty())
      invalid("missing value for " + flag);
    const auto& value = args[i];
    if (flag == "--model")
      s.model = family(value);
    else if (flag == "--task")
      s.task = lower(value);
    else if (flag == "--first-frame")
      s.first = value;
    else if (flag == "--last-frame")
      s.last = value;
    else if (flag == "--edit-image")
      s.edit = value;
    else if (flag == "--reference-image")
      s.images.push_back(value);
    else if (flag == "--reference-video")
      s.videos.push_back(value);
    else if (flag == "--reference-audio")
      s.audio.push_back(value);
    else {
      size_t consumed = 0;
      double number = 0;
      try {
        number = std::stod(value, &consumed);
      } catch (const std::exception&) {
        invalid("invalid number for " + flag);
      }
      if (consumed != value.size() || !std::isfinite(number) || number <= 0 ||
          (flag == "--shots" && (number != std::floor(number) || number > 1000)))
        invalid("invalid positive number for " + flag);
      if (flag == "--shots")
        s.shots = std::to_string(int(number));
      else {
        std::ostringstream out;
        out.imbue(std::locale::classic());
        out << std::fixed << std::setprecision(2) << number;
        s.duration = out.str();
        if (s.duration == "0.00")
          invalid("duration rounds to zero");
      }
      s.timing = true;
    }
  }
  if (s.model.empty())
    invalid("prompts select requires --model");
  const bool music = s.model == "yue2" || s.model == "minimax_music3";
  if (s.task.empty())
    s.task = music ? "music" : s.model == "h3" ? "video" : "image";
  if ((music && s.task != "music") ||
      (!music && s.task != "image" && !(s.model == "h3" && s.task == "video")))
    invalid("unsupported task for " + s.model);
  return s;
}

Object select_guide(const Value& catalog, const Selection& s) {
  const bool endpoints = !s.first.empty() || !s.last.empty();
  const bool refs = !s.images.empty() || !s.videos.empty() || !s.audio.empty();
  std::string use_case;
  if (s.task == "music") {
    if (endpoints || refs || !s.edit.empty() || s.layout || s.continuation || s.timing)
      invalid("music guides accept --instrumental, without visual, timing or reference selectors");
    use_case = s.instrumental ? "instrumental" : "song";
  } else if (s.task == "image") {
    if (endpoints || !s.videos.empty() || !s.audio.empty() || s.continuation || s.instrumental ||
        s.timing)
      invalid("image guides do not accept video endpoints, audio or video timing selectors");
    if (s.layout && (s.model != "moss" || refs || !s.edit.empty()))
      invalid("layout requires MOSS without image references");
    if ((!s.edit.empty() || refs) && s.model != "h3" && s.model != "fibo2")
      invalid("this model has no local image-reference/edit path");
    if (!s.edit.empty() && refs)
      invalid("use one edit source, or ordinary references separately");
    if (s.model == "fibo2" && s.images.size() > 4)
      invalid("Fibo supports at most four references");
    use_case = s.layout                                    ? "layout"
               : !s.edit.empty()                           ? "edit"
               : s.model == "fibo2" && s.images.size() > 1 ? "multi-reference"
               : s.model == "fibo2" && refs                ? "edit"
                                                           : "image";
  } else {
    if (s.instrumental || s.layout || !s.edit.empty())
      invalid("incompatible video guide selector");
    // Keep endpoint numbering unambiguous. Mixed roles need explicit Ref2VA authorship.
    if (endpoints && refs)
      invalid(
          "mixed endpoint and other reference roles require prompts get h3 ref2va; assign every role explicitly");
    if (s.continuation && (endpoints || refs))
      invalid(
          "load the continuation and endpoint/reference guides separately for a combined workflow");
    if (!s.audio.empty() && s.images.empty() && s.videos.empty())
      invalid("H3 reference audio requires a visual reference");
    use_case = s.continuation                        ? "continuation"
               : !s.first.empty() && !s.last.empty() ? "fl2va"
               : !s.first.empty()                    ? "i2va"
               : !s.last.empty()                     ? "l2va"
               : refs                                ? "ref2va"
                                                     : "t2va";
  }
  auto result = guide(catalog, s.model, use_case);
  Array bindings;
  int picture = 0, video = 0, audio = 0;
  auto bind = [&](const std::string& path, const std::string& label, const std::string& role,
                  const std::string& option) {
    bindings.emplace_back(Object{{"path", str(path)},
                                 {"label", str(label)},
                                 {"role", str(role)},
                                 {"generation_option", str(option)}});
  };
  auto image_label = [&]() {
    ++picture;
    if (s.model == "fibo2")
      return s.images.size() > 1 ? "image_" + std::to_string(picture) : std::string("image");
    return "<Picture " + std::to_string(picture) + ">";
  };
  if (!s.first.empty())
    bind(s.first, image_label(), "intended opening", "--reference-image");
  for (const auto& path : s.images)
    bind(path, image_label(), "assign identity/style/content role", "--reference-image");
  if (!s.last.empty())
    bind(s.last, image_label(), "ending anchor", "--end-anchor");
  for (const auto& path : s.videos)
    bind(path, "<Video " + std::to_string(++video) + ">", "assign reference role",
         "--reference-video");
  for (const auto& path : s.audio)
    bind(path, "<Audio " + std::to_string(++audio) + ">", "assign reference role",
         "--reference-audio");
  if (!s.edit.empty())
    bind(s.edit, s.model == "fibo2" ? "image" : "source image", "edit source", "--edit-image");
  result["reference_bindings"] = Value(bindings);
  result["selection_note"] = str(
      "Selectors declare intent only: files are not opened and no generation is run. Inspect media and validate checkpoint/conditioning compatibility separately.");
  auto prompt = result.at("prompt_template").as_string();
  replace(prompt, "{{last_shot}}", s.shots);
  if (!s.duration.empty())
    replace(prompt, "{{duration}}", s.duration);
  result["prompt_template"] = str(prompt);
  if (endpoints || s.timing)
    result["timing_note"] = str(
        "--duration must be the validated sampling-window duration, not cumulative joined output length. Default --shots is 1; fill unresolved duration before generation and keep cut times inside the window.");
  if (s.model == "h3" && s.task == "image" && refs)
    result["reference_note"] = str(
        "Keep Picture roles explicit in the still-image brief; ordinary references require compatible Ref2VA weights.");
  return result;
}
} // namespace

int prompts_command(const std::vector<std::string>& args) {
  const auto catalog = json::parse(kPromptingCatalog);
  Object data;
  if (args.empty() || args[0] == "list") {
    std::string filter;
    if (args.size() != 0 && args.size() != 1) {
      if (args.size() != 3 || args[1] != "--model")
        invalid("use prompts list [--model NAME]");
      filter = family(args[2]);
    }
    Array guides;
    for (const auto& item : catalog.find("guides")->as_array()) {
      const auto& model = item.find("model")->as_string();
      const auto& use_case = item.find("use_case")->as_string();
      if (!filter.empty() && filter != model)
        continue;
      guides.emplace_back(Object{{"model", str(model)},
                                 {"use_case", str(use_case)},
                                 {"title", *item.find("title")},
                                 {"get", str("slopfab prompts get " + model + " " + use_case)}});
    }
    data = {
        {"guides", Value(guides)},
        {"version", *catalog.find("version")},
        {"next",
         str("Retrieve a guide with prompts get MODEL USE_CASE, or select by reference roles with prompts select. Run help prompts for selectors.")}};
  } else if (args[0] == "get") {
    if (args.size() != 3)
      invalid("use prompts get MODEL USE_CASE");
    data = guide(catalog, family(args[1]), lower(args[2]));
  } else if (args[0] == "select") {
    data = select_guide(catalog, parse_selection(args));
  } else
    invalid("unknown prompts action: " + args[0]);
  emit({{"type", str("result")}, {"ok", Value(true)}, {"data", Value(data)}}, true);
  return 0;
}
} // namespace slopfab::cli
