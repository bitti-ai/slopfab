#include "agent.h"
#include "startup.h"
#include "settings.h"
#include "mods.h"
#include "commands.h"
#include "slopfab/model_profile.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#if SLOPFAB_WITH_CUDA
#include "slopfab/cuda/device.h"
#endif
#if SLOPFAB_WITH_VULKAN
#include "slopfab/vulkan/runtime.h"
#endif

namespace slopfab::cli {
namespace {
void keys(const Value& v, std::initializer_list<const char*> allowed, const std::string& field) {
  if (!v.is_object())
    throw CliError("INVALID_REQUEST", field + " must be an object", 2, field);
  for (const auto& item : v.as_object()) {
    bool found = false;
    for (auto k : allowed)
      if (item.first == k)
        found = true;
    if (!found)
      throw CliError("INVALID_REQUEST", "unknown field: " + field + "." + item.first, 2,
                     field + "." + item.first);
  }
}

std::string scalar(const Value& v, const std::string& field) {
  if (v.is_string())
    return v.as_string();
  if (v.is_number() && std::isfinite(v.as_number()))
    return json::stringify(v);
  throw CliError("INVALID_REQUEST", field + " must be a string or finite number", 2, field);
}

Value config() {
  return load_settings();
}

Value read_request(const std::string& path) {
  if (path != "-")
    return read_json(std::filesystem::u8path(path));
  std::string text;
  char buffer[8192];
  while (std::cin.read(buffer, sizeof(buffer)) || std::cin.gcount()) {
    text.append(buffer, size_t(std::cin.gcount()));
    if (text.size() > 16 * 1024 * 1024)
      throw CliError("INVALID_REQUEST", "stdin request exceeds 16 MiB");
  }
  if (text.compare(0, 3, "\xef\xbb\xbf") == 0)
    text.erase(0, 3);
  try {
    return parse_cli_json(text);
  } catch (const std::exception& e) {
    throw CliError("INVALID_REQUEST", e.what());
  }
}

std::vector<std::string> request_args(Value request) {
  keys(request,
       {"schema_version", "task", "model", "preset", "prompt", "negative_prompt", "lyrics",
        "duration_seconds", "resolution", "seed", "count", "output", "references", "loras",
        "options", "deadline_seconds", "project", "project_item", "output_directory", "driving_audio"},
       "request");
  auto version = request.find("schema_version");
  if (!version || !version->is_number() || version->as_number() != 1)
    throw CliError("INVALID_REQUEST", "schema_version must be 1", 2, "schema_version");
  for (const char* key : {"model", "preset", "prompt", "negative_prompt", "lyrics", "project",
                          "project_item", "output_directory", "driving_audio"})
    if (const auto* v = request.find(key))
      if (!v->is_string())
        throw CliError("INVALID_REQUEST", std::string(key) + " must be a string", 2, key);
  for (const char* key : {"duration_seconds", "deadline_seconds", "count"})
    if (const auto* v = request.find(key))
      if (!v->is_number() || !std::isfinite(v->as_number()) || v->as_number() <= 0 ||
          (std::string(key) == "count" &&
           (std::floor(v->as_number()) != v->as_number() || v->as_number() > 1000000)))
        throw CliError("INVALID_REQUEST",
                       std::string(key) + " must be a positive number (count is an integer)", 2,
                       key);
  auto object = request.as_object();
  if (const auto* preset = request.find("preset")) {
    auto cfg = config();
    auto presets = cfg.find("presets");
    auto entry = presets ? presets->find(preset->as_string()) : nullptr;
    if (!entry)
      throw CliError("INVALID_REQUEST", "unknown preset: " + preset->as_string(), 2, "preset");
    keys(*entry,
         {"task", "model", "prompt", "negative_prompt", "lyrics", "duration_seconds", "resolution",
          "seed", "count", "options", "deadline_seconds"},
         "preset");
    auto merged = entry->as_object();
    for (const auto& item : object)
      merged[item.first] = item.second;
    object = std::move(merged);
    request = Value(object);
  }
  auto task = request.find("task");
  if (!task || !task->is_string() ||
      (task->as_string() != "image" && task->as_string() != "video" &&
       task->as_string() != "music"))
    throw CliError("INVALID_REQUEST", "task must be video, image or music", 2, "task");
  std::vector<std::string> args = {task->as_string()};
  const std::pair<const char*, const char*> simple[] = {{"model", "--model"},
                                                        {"project", "--project"},
                                                        {"project_item", "--project-item"},
                                                        {"output_directory", "--output-dir"},
                                                        {"prompt", "--prompt"},
                                                        {"negative_prompt", "--negative-prompt"},
                                                        {"lyrics", "--lyrics"},
                                                        {"driving_audio", "--driving-audio"},
                                                        {"duration_seconds", "--duration"},
                                                        {"seed", "--seed"},
                                                        {"count", "--count"},
                                                        {"deadline_seconds", "--deadline"}};
  for (const auto& field : simple)
    if (const auto* v = request.find(field.first)) {
      if (std::string(field.first) == "seed" && !v->is_string())
        throw CliError("INVALID_REQUEST", "seed must be a decimal string", 2, "seed");
      args.push_back(field.second);
      args.push_back(scalar(*v, field.first));
    }
  if (const auto* v = request.find("resolution")) {
    keys(*v, {"width", "height"}, "resolution");
    if (!v->find("width") || !v->find("height"))
      throw CliError("INVALID_REQUEST", "resolution requires width and height", 2, "resolution");
    for (const char* axis : {"width", "height"}) {
      const auto* n = v->find(axis);
      if (!n->is_number() || !std::isfinite(n->as_number()) || n->as_number() <= 0 ||
          std::floor(n->as_number()) != n->as_number())
        throw CliError("INVALID_REQUEST", "resolution dimensions must be positive integers", 2,
                       "resolution");
    }
    args.insert(args.end(), {"--resolution", scalar(*v->find("width"), "width") + "x" +
                                                 scalar(*v->find("height"), "height")});
  }
  if (const auto* v = request.find("output")) {
    keys(*v, {"path", "format", "overwrite"}, "output");
    if (const auto* path = v->find("path")) {
      args.insert(args.end(), {"--out", path->as_string()});
      if (const auto* format = v->find("format"))
        if (std::filesystem::u8path(path->as_string()).extension() != "." + format->as_string())
          throw CliError("INVALID_REQUEST", "output path extension must match format", 2,
                         "output.format");
    } else if (v->find("format"))
      throw CliError("INVALID_REQUEST", "output.format requires output.path", 2, "output.path");
    if (const auto* overwrite = v->find("overwrite"))
      if (overwrite->as_bool())
        args.push_back("--overwrite");
  }
  if (const auto* refs = request.find("references"))
    for (const auto& ref : refs->as_array()) {
      keys(ref, {"type", "path"}, "references");
      if (!ref.find("type") || !ref.find("path"))
        throw CliError("INVALID_REQUEST", "reference requires type and path");
      const auto kind = ref.find("type")->as_string();
      if (kind != "image" && kind != "video" && kind != "audio" && kind != "refmod")
        throw CliError("INVALID_REQUEST", "unknown reference type");
      args.insert(args.end(), {kind == "refmod" ? "--refmod" : "--reference-" + kind,
                               ref.find("path")->as_string()});
    }
  if (const auto* loras = request.find("loras"))
    for (const auto& lora : loras->as_array()) {
      keys(lora, {"path", "strength"}, "loras");
      if (!lora.find("path"))
        throw CliError("INVALID_REQUEST", "LoRA requires path");
      args.insert(args.end(), {"--lora", lora.find("path")->as_string()});
      if (const auto* strength = lora.find("strength"))
        args.insert(args.end(), {"--lora-strength", scalar(*strength, "strength")});
    }
  if (const auto* opts = request.find("options")) {
    if (!opts->is_object())
      throw CliError("INVALID_REQUEST", "options must be an object", 2, "options");
    const std::set<std::string> forbidden = {"request",    "help",        "agent",
                                             "config",     "jobs-dir",    "preset",
                                             "model",      "prompt",      "negative-prompt",
                                             "lyrics",     "duration",    "music-duration",
                                             "resolution", "seed",        "count",
                                             "out",        "overwrite",   "deadline",
                                             "dry-run",    "check-ready", "task",
                                             "bench-load", "project",     "project-item",
                                             "output-dir"};
    for (const auto& item : opts->as_object()) {
      const auto& key = item.first;
      if (key.empty() || key[0] == '-' || forbidden.count(key))
        throw CliError("INVALID_REQUEST", "use top-level fields instead of options." + key, 2,
                       "options." + key);
      const std::string flag = "--" + key;
      const std::string help = find_command("generate")->detail;
      const auto location = help.find(flag);
      if (location == std::string::npos ||
          (location + flag.size() < help.size() && help[location + flag.size()] != ' ' &&
           help[location + flag.size()] != '\n'))
        throw CliError("INVALID_REQUEST", "unknown option: " + key, 2, "options." + key);
      auto append = [&](const Value& value) {
        if (value.type() == json::Type::Bool) {
          if (option_takes_value(flag))
            throw CliError("INVALID_REQUEST", key + " requires a value", 2, "options." + key);
          if (value.as_bool())
            args.push_back(flag);
          return;
        }
        if (!option_takes_value(flag))
          throw CliError("INVALID_REQUEST", key + " requires a boolean", 2, "options." + key);
        if (key == "music-settings") {
          const auto settings = value.is_object() ? value : parse_cli_json(value.as_string());
          if (request.find("lyrics") && settings.find("lyrics"))
            throw CliError("INVALID_REQUEST", "lyrics conflicts with options.music-settings.lyrics",
                           2, "lyrics");
          if (request.find("duration_seconds") && settings.find("duration"))
            throw CliError("INVALID_REQUEST",
                           "duration conflicts with options.music-settings.duration", 2,
                           "duration_seconds");
          // Music settings replace a complete object in the legacy parser, so
          // append top-level lyrics again after applying the advanced settings.
          args.push_back("--" + key);
          args.push_back(json::stringify(settings));
          if (const auto* lyrics = request.find("lyrics"))
            args.insert(args.end(), {"--lyrics", lyrics->as_string()});
          return;
        }
        args.push_back("--" + key);
        args.push_back(value.is_object() ? json::stringify(value) : scalar(value, key));
      };
      if (item.second.is_array())
        for (const auto& v : item.second.as_array())
          append(v);
      else
        append(item.second);
    }
  }
  return args;
}
}

void resolve_model_alias(GenerateRequest& request, const std::string& name) {
  auto cfg = config();
  const auto* models = cfg.find("models");
  const auto* model = models ? models->find(name) : nullptr;
  if (!model) {
    request.model_family = name;
    return;
  }
  keys(
      *model,
      {"family", "transformer", "text_encoder", "clip_projection", "vae", "audio_vae", "tokenizer"},
      "models." + name);
  const auto* family = model->find("family");
  if (!family)
    throw CliError("INVALID_REQUEST", "model alias requires family", 2, "models." + name);
  request.model_family = family->as_string();
  for (auto field : std::initializer_list<std::pair<const char*, std::string*>>{
           {"transformer", &request.transformer_path},
           {"text_encoder", &request.text_encoder_path},
           {"clip_projection", &request.clip_projection_path},
           {"vae", &request.video_vae_path},
           {"audio_vae", &request.audio_vae_path},
           {"tokenizer", &request.tokenizer_path}}) {
    if (const auto* v = model->find(field.first)) {
      auto path = std::filesystem::u8path(v->as_string());
      if (path.is_relative())
        path = config_path().parent_path() / path;
      *field.second = std::filesystem::absolute(path).lexically_normal().u8string();
    }
  }
}

std::vector<std::string> prepare_arguments(const std::vector<std::string>& input,
                                         const std::string& task) {
  std::vector<std::string> args = input;
  size_t request_pos = args.size();
  for (size_t i = 0; i < args.size(); ++i) {
    if (args[i] == "--request") {
      if (request_pos != args.size() || i + 1 == args.size())
        throw CliError("INVALID_REQUEST", "provide one --request FILE or -");
      request_pos = i;
    }
    if (option_takes_value(args[i]) && i + 1 < args.size())
      ++i;
  }
  if (request_pos < args.size()) {
    std::vector<std::string> controls;
    for (size_t i = 0; i < args.size(); ++i) {
      if (i == request_pos) {
        ++i;
        continue;
      }
      if (args[i] == "--dry-run" || args[i] == "--check-ready")
        controls.push_back(args[i]);
      else
        throw CliError(
            "INVALID_REQUEST",
            "--request cannot be combined with generation flags; put settings in the request");
    }
    try {
      args = request_args(read_request(args[request_pos + 1]));
    } catch (const CliError&) {
      throw;
    } catch (const std::exception& e) {
      throw CliError("INVALID_REQUEST", e.what());
    }
    args.insert(args.end(), controls.begin(), controls.end());
  }
  if (!task.empty()) {
    if (!args.empty() && (args[0] == "video" || args[0] == "image" || args[0] == "music")) {
      if (args[0] != task)
        throw CliError("INVALID_REQUEST", "request task conflicts with media command", 2, "task");
    } else args.insert(args.begin(), task);
  }
  agent().task.clear();
  agent().deadline_seconds = 0;
  agent().requested_duration = 0;
  agent().overwrite = false;
  agent().check_ready = false;
  agent().project.clear();
  agent().project_output.clear();
  agent().project_item.clear();
  agent().project_new_id.clear();
  agent().project_revision = 0;
  agent().project_manifest_revision = 0;
  agent().generation_arguments.clear();
  agent().reference_attachments.clear();
  for (const auto& argument : args)
    agent().generation_arguments.push_back(str(argument));
  if (!args.empty() && (args[0] == "image" || args[0] == "video" || args[0] == "music")) {
    agent().task = args[0];
    args.erase(args.begin());
  }
  bool model = false, frames = false;
  std::vector<std::string> result;
  std::set<std::string> singletons;
  for (size_t i = 0; i < args.size(); ++i) {
    const auto& a = args[i];
    if (a.rfind("--", 0) == 0)
      singletons.insert(a);
    auto value = [&]() {
      if (i + 1 == args.size())
        throw CliError("INVALID_REQUEST", "missing value for " + a, 2, a);
      return args[++i];
    };
    if (a == "--project" || a == "--project-item" || a == "--output-dir") {
      auto& destination = a == "--project"        ? agent().project
                          : a == "--project-item" ? agent().project_item
                                                  : agent().project_output;
      if (!destination.empty())
        throw CliError("INVALID_REQUEST", "repeated " + a);
      destination = value();
      if (destination.empty())
        throw CliError("INVALID_REQUEST", "empty " + a);
      continue;
    }
    if (a == "--overwrite") {
      agent().overwrite = true;
      continue;
    }
    if (a == "--check-ready") {
      agent().check_ready = true;
      continue;
    }
    if (a == "--duration" || a == "--deadline") {
      const auto text = value();
      size_t used = 0;
      double n = std::stod(text, &used);
      if (used != text.size() || !std::isfinite(n) || n <= 0 || n > 86400)
        throw CliError("INVALID_REQUEST", a + " must be in (0,86400]", 2, a);
      if (a == "--deadline")
        agent().deadline_seconds = n;
      else
        agent().requested_duration = n;
      continue;
    }
    if (a == "--preset") {
      throw CliError("INVALID_REQUEST",
                     "presets are selected with the preset field of a JSON request");
    }
    if (a == "--model")
      model = true;
    if (a == "--frames")
      frames = true;
    result.push_back(a);
    if (option_takes_value(a))
      result.push_back(value());
  }
  if (agent().requested_duration > 0 && (frames || singletons.count("--music-duration")))
    throw CliError("INVALID_REQUEST", "duration conflicts with --frames or --music-duration");
  if (agent().project.empty() && (!agent().project_item.empty() || !agent().project_output.empty()))
    throw CliError("INVALID_REQUEST", "--project-item and --output-dir require --project");
  if (agent().task == "image" &&
      (frames || agent().requested_duration > 0 || singletons.count("--music-duration") ||
       singletons.count("--lyrics")))
    throw CliError("INVALID_REQUEST", "image requests do not accept duration, frames or lyrics");
  if (!model && !agent().task.empty()) {
    result.insert(result.begin(), {"--model", agent().task == "image"   ? "krea2"
                                              : agent().task == "music" ? "yue2"
                                                                        : "h3"});
  }
  return result;
}

Value request_schema() {
  Object properties;
  auto typed = [](const char* type) {
    return Value(Object{{"type", str(type)}});
  };
  properties["schema_version"] = Value(Object{{"const", num(1)}});
  properties["task"] =
      Value(Object{{"enum", Value(Array{str("video"), str("image"), str("music")})}});
  for (const char* key : {"model", "preset", "prompt", "negative_prompt", "lyrics", "project",
                          "project_item", "output_directory"})
    properties[key] = typed("string");
  properties["seed"] =
      Value(Object{{"type", str("string")},
                   {"pattern", str("^[0-9]+$")},
                   {"description", str("Unsigned 64-bit integer, encoded as a decimal string.")}});
  properties["driving_audio"] = Value(Object{
      {"type", str("string")}, {"minLength", num(1)},
      {"description", str("WAV attached at time zero to exactly one video reference. Trim excess samples or pad silence to the clip duration; no source slicing.")}});
  for (const char* key : {"duration_seconds", "deadline_seconds"})
    properties[key] = Value(
        Object{{"type", str("number")}, {"exclusiveMinimum", num(0)}, {"maximum", num(86400)}});
  properties["count"] = Value(Object{{"type", str("integer")}, {"minimum", num(1)}});
  properties["resolution"] = Value(Object{
      {"type", str("object")},
      {"additionalProperties", Value(false)},
      {"required", Value(Array{str("width"), str("height")})},
      {"properties", Value(Object{{"width", typed("integer")}, {"height", typed("integer")}})}});
  properties["output"] = Value(Object{
      {"type", str("object")},
      {"additionalProperties", Value(false)},
      {"properties",
       Value(Object{{"path", typed("string")},
                    {"format", Value(Object{{"enum", Value(Array{str("png"), str("ppm"), str("y4m"),
                                                                 str("wav")})}})},
                    {"overwrite", typed("boolean")}})}});
  properties["references"] = Value(
      Object{{"type", str("array")},
             {"items",
              Value(Object{
                  {"type", str("object")},
                  {"additionalProperties", Value(false)},
                  {"required", Value(Array{str("type"), str("path")})},
                  {"properties",
                   Value(Object{
                       {"type", Value(Object{{"enum", Value(Array{str("image"), str("video"),
                                                                  str("audio"), str("refmod")})}})},
                       {"path", typed("string")}})}})}});
  properties["loras"] = Value(Object{
      {"type", str("array")},
      {"items", Value(Object{{"type", str("object")},
                             {"additionalProperties", Value(false)},
                             {"required", Value(Array{str("path")})},
                             {"properties", Value(Object{{"path", typed("string")},
                                                         {"strength", typed("number")}})}})}});
  properties["options"] = Value(Object{
      {"type", str("object")},
      {"description",
       str("Advanced generate options without leading --. Use booleans for switches, arrays for repeated options, objects for JSON settings. Unknown CLI options are rejected. See generate --help.")}});
  return Value(
      Object{{"$schema", str("https://json-schema.org/draft/2020-12/schema")},
             {"type", str("object")},
             {"additionalProperties", Value(false)},
             {"required", Value(Array{str("schema_version")})},
             {"anyOf", Value(Array{Value(Object{{"required", Value(Array{str("task")})}}),
                                   Value(Object{{"required", Value(Array{str("preset")})}})})},
             {"properties", Value(properties)}});
}

Value capabilities() {
  Array backends;
#if SLOPFAB_WITH_CUDA
  backends.push_back(str("cuda"));
#endif
#if SLOPFAB_WITH_VULKAN
  backends.push_back(str("vulkan"));
#endif
  Array models;
  for (const char* family : {"h3", "krea2", "moss", "fibo2", "yue2", "minimax_music3"}) {
    const std::string f = family;
    Array tasks = f == "h3" ? Array{str("video"), str("image")}
                            : Array{str(f == "yue2" || f == "minimax_music3" ? "music" : "image")};
    Array supported;
    for (const auto& b : backends)
      if (b.as_string() == "cuda" || (f != "moss" && f != "fibo2"))
        supported.push_back(b);
    models.push_back(Value(Object{{"family", str(f)},
                                  {"tasks", Value(tasks)},
                                  {"backends", Value(supported)},
                                  {"canvas_multiple", num(model_profile(f).canvas_multiple)},
                                  {"music_duration_max_seconds",
                                   f == "yue2" || f == "minimax_music3" ? num(300) : Value()}}));
  }
  return Value(Object{
      {"version", str(kVersion)},
      {"updates", Value(Object{{"command", str("slopfab update")},
                                {"check_command", str("slopfab update --check")},
                                {"source", str("https://github.com/bitti-ai/slopfab/releases/latest")},
                                {"automatic", Value(false)}})},
      {"protocol_version", num(1)},
      {"remote", Value(Object{{"command", str("slopfab COMMAND --host NAME")},
          {"setup", str("slopfab remote setup NAME --directory PATH")}, {"automatic_binary_update", Value(true)}, {"transport", str("OpenSSH")}, {"platforms", Value(Array{str("linux"), str("windows")})},
          {"transfer", str("result artifacts only; no project or model synchronization")},
          {"retry", str("slopfab remote fetch PATH/remote.json")}, {"cli_only", Value(true)}})},
      {"driving_audio", Value(Object{
          {"option", str("--driving-audio FILE.wav")}, {"request_field", str("driving_audio")},
          {"video_format", str("Y4M")}, {"audio_format", str("mono/stereo PCM16/24/32 or float32 WAV")},
          {"video_references", num(1)}, {"start_seconds", num(0)},
          {"duration_policy", str("trim_end_or_pad_silence")}, {"source_slicing", Value(false)},
          {"preserve_option", str("--preserve-driving-audio")},
          {"pin_target_audio", Value(true)}, {"include_reference_audio_when_pinned", Value(false)}})},
      {"regenerate_2k", Value(Object{{"option", str("--regenerate-2k up3|up4|hf8")},
          {"frames", num(124)}, {"fps", num(24)}, {"default_width", num(2560)}, {"default_height", num(1440)},
          {"base", str("H3 Ref2VA")}, {"hyperflow_required", Value(true)},
          {"audio", str("fixed published silence latents; mux original soundtrack after rendering")},
          {"lms_sharp_default", num(0)}})},
      {"viggle_remake", Value(Object{{"option", str("--remake swap|edit")},
          {"steps", num(6)}, {"reference_images", str("1..4 painted frames")},
          {"conditioning", str("text-only H3 prompt labels; source video first")},
          {"source_audio", str("pinned through audio VAE; silence for silent sources")},
          {"merged_model_needs_lora", Value(false)}, {"timeline_pins", Value(false)}})},
      {"syncnet", Value(Object{{"command", str("slopfab video syncnet VIDEO --model syncnet_v2.model --face-model sfd_face.pth")},
          {"model_sha256", str("961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442")},
          {"automatic_face_tracking", Value(true)}, {"fps", num(25)},
          {"s3fd_safetensors", Value(true)}, {"cli_only", Value(true)}})},
      {"media_commands", Value(Object{{"video", Value(Array{str("generate"), str("frames"), str("syncnet"), str("upscale"), str("upscale-latents"), str("decode"), str("compare")})},
          {"audio", Value(Array{str("generate"), str("transcribe"), str("separate")})},
          {"image", Value(Array{str("generate"), str("upscale")})}})},
      {"separation", Value(Object{{"command", str("slopfab audio separate FILE --model MODEL.safetensors")},
          {"model", str("htdemucs")}, {"backends", Value(Array{str("cpu")})},
          {"sources", Value(Array{str("drums"), str("bass"), str("other"), str("vocals")})},
          {"sample_rate", num(44100)}, {"format", str("float32 WAV")},
          {"two_stems", Value(true)}})},
      {"transcription", Value(Object{{"command",str("slopfab audio transcribe FILE --model MODEL.safetensors")},
          {"model",str("whisper")}, {"multilingual",Value(true)}, {"word_timestamps",Value(true)},
          {"embedded_tokenizer",Value(true)}, {"weights",str("Hugging Face safetensors, F16/BF16/F32")},
          {"formats",Value(Array{str("json"),str("txt"),str("srt"),str("vtt")})},
          {"cli_only",Value(true)}})},
      {"video_frames", Value(Object{{"command",str("slopfab video frames VIDEO --out IMAGE.png")},
          {"javascript",str("slopfab.frames.extract / fromRGB / generate")},
          {"generation_ram_capture",Value(true)}, {"native_y4m",Value(true)},
          {"compressed_video",str("external FFmpeg and ffprobe")}, {"cli_only",Value(true)}})},
      {"motions", Value(Object{{"renderer",str("ThorVG 1.1.3")}, {"runtime",str("embedded JavaScript")},
        {"command",str("slopfab motions render|frame FILE.js --out FILE")},
        {"formats",Value(Array{str("svg"),str("png"),str("mp4"),str("mov"),str("webm")})},
        {"overlay",Value(true)}, {"video_encoder",str("external FFmpeg")}, {"cli_only",Value(true)}})},
      {"scripting", Value(Object{{"engine",str("QuickJS-NG 0.18.0")}, {"api_version",num(1)},
          {"command",str("slopfab run FILE.js -- ARGS")},
          {"apis",Value(Array{str("filesystem"),str("binary files"),str("paths"),str("processes"),
              str("HTTP/downloads"),str("CLI"),str("environment"),str("timers"),str("SHA-256")})}})},
      {"mod_commands",Value(mod_commands())},
      {"backends", Value(backends)},
      {"models", Value(models)},
      {"prompting", Value(Object{{"index", str("slopfab prompts list")},
                                  {"get", str("slopfab prompts get MODEL USE_CASE")},
                                  {"select", str("slopfab prompts select --model NAME; see help prompts for reference-role selectors")},
                                  {"offline", Value(true)}})},
      {"output_formats", Value(Array{str("png"), str("ppm"), str("y4m"), str("wav")})},
      {"reference_formats",
       Value(Array{str("platform images"), str("Y4M BT.709 limited 8-bit 4:2:0"),
                   str("WAV PCM16/24/32 or float32")})},
      {"progress", str("NDJSON; progress events followed by exactly one result or error")},
      {"execution", str("one generation per local worker; CLI processes share a GPU lock")},
      {"external_conversion",
       str("Use a separate ffmpeg installation to encode MP4, MP3 or other formats.")},
      {"h3_clip_projection",
       Value(Object{
           {"option", str("--clip-projection FILE")},
           {"encoders", Value(Array{str("Qwen3-VL-4B"), str("Qwen3-VL-8B")})},
           {"conditioning", str("text only; BF16 streaming student, CPU ridge/MLP projection")}})},
      {"mods",
       Value(Object{{"index_command", str("slopfab mods list")},
                    {"load_command", str("slopfab mods load NAME")},
                    {"format", str("mods/NAME/MOD.md: YAML front matter and Markdown prompt; command + script register a JavaScript CLI command")}})},
      {"projects",
       Value(Object{
           {"help", str("slopfab help projects")},
           {"create", str("slopfab projects create NAME")},
           {"open", str("slopfab projects open NAME")},
           {"update", str("slopfab projects update NAME")},
           {"editable_page_source", str("project.json only; HTML and PROJECT.md are generated")},
           {"automatic_page_update", Value(true)},
           {"generate", str("requests run --project NAME; --project-item ID replaces an item")},
           {"manifest", str("project.json")},
           {"preview", str("NAME.html: offline media review and copyable agent prompts")},
           {"naming",
            str("PROJECT.md and projects show describe exact manifest paths and naming rules")}})},
      {"schema_command", str("slopfab requests schema")}});
}

int discovery_command(const std::string& command, const std::vector<std::string>& args) {
  if (command == "models" && args != std::vector<std::string>{"list"})
    throw CliError("INVALID_REQUEST", "use models list");
  if (command != "models" && !args.empty())
    throw CliError("INVALID_REQUEST", "unexpected discovery arguments");
  Value data;
  if (command == "schema")
    data = request_schema();
  else if (command == "capabilities")
    data = capabilities();
  else if (command == "models") {
    Array models;
    auto cfg = config();
    std::vector<std::string> names = {"h3", "krea2", "moss", "fibo2", "yue2", "minimax_music3"};
    if (const auto* aliases = cfg.find("models"))
      for (const auto& m : aliases->as_object())
        names.push_back(m.first);
    for (const auto& name : names) {
      GenerateRequest r;
      resolve_model_alias(r, name);
      bool ready = true;
      std::string reason;
      try {
        discover_generate_checkpoints(r, agent().executable.c_str());
        check_models_ready(r, true);
      } catch (const std::exception& e) {
        ready = false;
        reason = e.what();
      }
      models.push_back(Value(Object{{"name", str(name)},
                                    {"family", str(request_model_family(r))},
                                    {"files_present", Value(ready)},
                                    {"reason", str(reason)},
                                    {"transformer", str(r.transformer_path)},
                                    {"text_encoder", str(r.text_encoder_path)},
                                    {"clip_projection", str(r.clip_projection_path)},
                                    {"vae", str(r.video_vae_path)},
                                    {"audio_vae", str(r.audio_vae_path)}}));
    }
    data = Value(Object{{"config_path", str(config_path().u8string())},
                        {"models", Value(models)},
                        {"weight_paths", *cfg.find("weight_paths")},
                        {"presets", cfg.find("presets") ? *cfg.find("presets") : Value(Object{})}});
  } else {
    const auto dependencies = startup_dependencies(agent().executable);
    const bool has_weights = !configured_weight_paths().empty();
    Array devices, diagnostics;
#if SLOPFAB_WITH_CUDA
    try {
      for (int i = 0; i < slopfab::cuda::device_count(); ++i) {
        auto d = slopfab::cuda::query_device(i);
        devices.push_back(Value(Object{{"backend", str("cuda")},
                                       {"index", num(i)},
                                       {"name", str(d.name)},
                                       {"free_bytes", num(double(d.free_memory))},
                                       {"total_bytes", num(double(d.total_memory))}}));
      }
    } catch (const std::exception& e) {
      diagnostics.push_back(str(e.what()));
    }
#endif
#if SLOPFAB_WITH_VULKAN
    try {
      auto instance = slopfab::vulkan::Instance::create();
      auto ds = instance.enumerate_devices();
      for (size_t i = 0; i < ds.size(); ++i)
        devices.push_back(Value(Object{{"backend", str("vulkan")},
                                       {"index", num(double(i))},
                                       {"name", str(ds[i].info().name)}}));
    } catch (const std::exception& e) {
      diagnostics.push_back(str(e.what()));
    }
#endif
    data = Value(Object{
        {"gpu_available", Value(!devices.empty())},
        {"devices", Value(devices)},
        {"diagnostics", Value(diagnostics)},
        {"config_path", str(config_path().u8string())},
        {"jobs_path", str(jobs_path().u8string())},
        {"ffmpeg_required", Value(false)},
        {"ffmpeg_available", Value(dependencies.ffmpeg)},
        {"cuda_installed", Value(dependencies.cuda)},
        {"weight_paths", *load_settings().find("weight_paths")},
        {"startup_ready",
         Value(std::string(missing_startup_dependency(dependencies, has_weights)).empty())},
        {"next", str(std::string(missing_startup_dependency(dependencies, has_weights)).empty()
                         ? "Use models list, then validate --request FILE --check-ready."
                         : startup_instructions(dependencies, has_weights))}});
  }
  emit({{"type", str("result")}, {"ok", Value(true)}, {"data", data}}, true);
  return 0;
}

int run_generation_arguments(const std::vector<std::string>& args, const std::string& task) {
  start_request();
  auto prepared = prepare_arguments(args, task);
  std::vector<char*> argv;
  for (auto& a : prepared)
    argv.push_back(a.data());
  return cmd_generate(int(argv.size()), argv.data(), agent().executable.c_str());
}
}
