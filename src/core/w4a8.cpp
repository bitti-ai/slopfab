#include "vidfab/w4a8.h"

#include <stdexcept>

#include "vidfab/json.h"

namespace vidfab {
namespace {

std::string state_name(const std::string& weight_name) {
  constexpr const char* suffix = ".weight";
  constexpr size_t suffix_size = 7;
  if (weight_name.size() >= suffix_size &&
      weight_name.compare(weight_name.size() - suffix_size, suffix_size,
                          suffix) == 0) {
    return weight_name.substr(0, weight_name.size() - suffix_size) +
           ".comfy_quant";
  }
  return weight_name + ".comfy_quant";
}

json::Value parse_state(const TensorView& view, const std::string& name,
                        const char* consumer) {
  if (view.dtype != DType::kU8 || view.shape.size() != 1) {
    throw std::runtime_error(std::string(consumer) + ": '" + name +
                             "' is not rank-1 U8 JSON");
  }
  try {
    return json::parse(
        std::string(static_cast<const char*>(view.data), view.nbytes));
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string(consumer) +
                             ": invalid W4A8 state '" + name + "' (" +
                             e.what() + ")");
  }
}

const json::Value* parameter(const json::Value& root, const char* key) {
  if (const json::Value* value = root.find(key)) return value;
  if (const json::Value* params = root.find("params")) return params->find(key);
  return nullptr;
}

}  // namespace

bool is_w4a8_weight(const SafeTensors& checkpoint,
                    const std::string& weight_name) {
  const TensorView* view = checkpoint.find(state_name(weight_name));
  if (!view) return false;
  const json::Value root = parse_state(*view, state_name(weight_name), "W4A8");
  const json::Value* format = root.find("format");
  return format && format->as_string() == "asym_w4a8_int8";
}

W4A8State read_w4a8_state(const SafeTensors& checkpoint,
                          const std::string& weight_name,
                          const char* consumer) {
  const std::string name = state_name(weight_name);
  const TensorView* view = checkpoint.find(name);
  if (!view) {
    throw std::runtime_error(std::string(consumer) + ": missing W4A8 state '" +
                             name + "'");
  }
  const json::Value root = parse_state(*view, name, consumer);
  const json::Value* format = root.find("format");
  if (!format || format->as_string() != "asym_w4a8_int8") {
    throw std::runtime_error(std::string(consumer) +
                             ": unsupported W4A8 format for '" + weight_name +
                             "'");
  }
  const json::Value* group = parameter(root, "group_size");
  const json::Value* convrot = parameter(root, "convrot_groupsize");
  if (!group || !convrot) {
    throw std::runtime_error(std::string(consumer) +
                             ": incomplete W4A8 state for '" + weight_name +
                             "'");
  }
  W4A8State state;
  state.group_size = static_cast<int>(group->as_int());
  state.convrot_group_size = static_cast<int>(convrot->as_int());
  if (state.group_size != 16 || state.convrot_group_size != 256) {
    throw std::runtime_error(std::string(consumer) +
                             ": unsupported W4A8 layout for '" + weight_name +
                             "' (requires group_size 16 and convrot_groupsize 256)");
  }
  return state;
}

}  // namespace vidfab
