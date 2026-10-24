#include "slopfab/w4a8.h"

#include <stdexcept>
#include <cmath>
#include "slopfab/tensor_convert.h"

#include "slopfab/json.h"

namespace slopfab {
namespace {

std::string state_name(const std::string& weight_name) {
  constexpr const char* suffix = ".weight";
  constexpr size_t suffix_size = 7;
  if (weight_name.size() >= suffix_size &&
      weight_name.compare(weight_name.size() - suffix_size, suffix_size, suffix) == 0) {
    return weight_name.substr(0, weight_name.size() - suffix_size) + ".comfy_quant";
  }
  return weight_name + ".comfy_quant";
}

json::Value parse_state(const TensorView& view, const std::string& name, const char* consumer) {
  if (view.dtype != DType::kU8 || view.shape.size() != 1) {
    throw std::runtime_error(std::string(consumer) + ": '" + name + "' is not rank-1 U8 JSON");
  }
  try {
    std::string text(static_cast<const char*>(view.data), view.nbytes);
    while (!text.empty() && (text.back() == '\0' || text.back() == ' ' || text.back() == '\n'))
      text.pop_back();
    return text.empty() ? json::Value() : json::parse(text);
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string(consumer) + ": invalid W4A8 state '" + name + "' (" +
                             e.what() + ")");
  }
}

const json::Value* parameter(const json::Value& root, const char* key) {
  if (const json::Value* value = root.find(key))
    return value;
  if (const json::Value* params = root.find("params"))
    return params->find(key);
  return nullptr;
}

} // namespace

json::Value quantization_state(const SafeTensors& checkpoint, const std::string& weight_name) {
  if (const auto* local = checkpoint.find(state_name(weight_name)))
    return parse_state(*local, state_name(weight_name), "quantization");
  const auto found = checkpoint.metadata().find("_quantization_metadata");
  if (found == checkpoint.metadata().end())
    return {};
  const auto root = json::parse(found->second);
  const auto* version = root.find("format_version");
  if (!version || version->as_string() != "1.0")
    throw std::runtime_error("unsupported checkpoint quantization metadata version");
  const auto* layers = root.find("layers");
  if (!layers || !layers->is_object())
    throw std::runtime_error("invalid quantization layers metadata");
  auto name = weight_name;
  if (name.size() >= 7 && name.compare(name.size() - 7, 7, ".weight") == 0)
    name.resize(name.size() - 7);
  const auto* layer = layers->find(name);
  return layer ? *layer : json::Value();
}

bool is_w4a8_weight(const SafeTensors& checkpoint, const std::string& weight_name) {
  const auto root = quantization_state(checkpoint, weight_name);
  const auto* format = root.find("format");
  return format && format->as_string() == "asym_w4a8_int8";
}

W4A8State read_w4a8_state(const SafeTensors& checkpoint, const std::string& weight_name,
                          const char* consumer) {
  const auto root = quantization_state(checkpoint, weight_name);
  const json::Value* format = root.find("format");
  if (!format || format->as_string() != "asym_w4a8_int8") {
    throw std::runtime_error(std::string(consumer) + ": unsupported W4A8 format for '" +
                             weight_name + "'");
  }
  const json::Value* group = parameter(root, "group_size");
  const json::Value* convrot = parameter(root, "convrot_groupsize");
  if (!group || !convrot) {
    throw std::runtime_error(std::string(consumer) + ": incomplete W4A8 state for '" + weight_name +
                             "'");
  }
  if (const auto* enabled = parameter(root, "convrot"); enabled && !enabled->as_bool())
    throw std::runtime_error("W4A8 requires ConvRot");
  W4A8State state;
  state.group_size = static_cast<int>(group->as_int());
  state.convrot_group_size = static_cast<int>(convrot->as_int());
  if (state.group_size != 16 || state.convrot_group_size != 256) {
    throw std::runtime_error(std::string(consumer) + ": unsupported W4A8 layout for '" +
                             weight_name + "' (requires group_size 16 and convrot_groupsize 256)");
  }
  return state;
}

void validate_w4a8_weight(const SafeTensors& st, const std::string& name, int out, int in) {
  const auto state = read_w4a8_state(st, name, "transformer");
  const auto& w = st.at(name);
  const auto p = name.substr(0, name.size() - 7);
  const auto& group = st.at(p + ".weight_s_rel");
  const auto& channel = st.at(p + ".weight_s_channel");
  const auto& book = st.at(p + ".weight_codebook");
  if (out <= 0 || in <= 0 || in % state.convrot_group_size || w.dtype != DType::kI8 ||
      w.shape != std::vector<int64_t>{out, in / 2} || group.dtype != DType::kF8E4M3 ||
      group.shape != std::vector<int64_t>{out, in / state.group_size} ||
      channel.dtype != DType::kF32 || channel.shape != std::vector<int64_t>{out} ||
      book.dtype != DType::kF32 || book.shape != std::vector<int64_t>{16})
    throw std::runtime_error("W4A8: incompatible packed weight or auxiliaries: " + name);
  if (st.find(p + ".weight_correction"))
    throw std::runtime_error("W4A8: asymmetric correction is unsupported: " + name);
  for (float v : to_f32(channel))
    if (!std::isfinite(v) || v <= 0)
      throw std::runtime_error("W4A8: invalid channel scale: " + name);
  for (float v : to_f32(book))
    if (!std::isfinite(v))
      throw std::runtime_error("W4A8: non-finite codebook: " + name);
  const auto* bytes = static_cast<const uint8_t*>(group.data);
  for (size_t i = 0; i < group.nbytes; ++i)
    if (bytes[i] & 128 || (bytes[i] & 127) == 127)
      throw std::runtime_error("W4A8: invalid group scale: " + name);
}

} // namespace slopfab
