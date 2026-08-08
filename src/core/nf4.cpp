#include "vidfab/nf4.h"

#include <cmath>
#include <stdexcept>

#include "vidfab/json.h"

namespace vidfab {

namespace {
std::string state_name(const std::string& weight_name) {
  return weight_name + ".quant_state.bitsandbytes__nf4";
}
}  // namespace

bool is_nf4_weight(const SafeTensors& checkpoint, const std::string& weight_name) {
  return checkpoint.find(state_name(weight_name)) != nullptr;
}

NF4State read_nf4_state(const SafeTensors& checkpoint, const std::string& weight_name,
                        const char* consumer) {
  const std::string name = state_name(weight_name);
  const TensorView* view = checkpoint.find(name);
  if (view == nullptr || view->dtype != DType::kU8 || view->shape.size() != 1)
    throw std::runtime_error(std::string(consumer) + ": '" + name + "' is not rank-1 U8 JSON");
  json::Value root;
  try {
    root = json::parse(std::string(static_cast<const char*>(view->data), view->nbytes));
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string(consumer) + ": invalid NF4 state '" + name +
                             "' (" + e.what() + ")");
  }
  auto get = [&](const char* key) -> const json::Value& {
    const json::Value* value = root.find(key);
    if (!value) throw std::runtime_error(std::string(consumer) + ": NF4 state missing '" + key + "'");
    return *value;
  };
  if (get("quant_type").as_string() != "nf4" || get("nested_dtype").as_string() != "float32")
    throw std::runtime_error(std::string(consumer) + ": unsupported NF4 contract for '" + weight_name + "'");
  NF4State state;
  state.block_size = static_cast<int>(get("blocksize").as_int());
  state.nested_block_size = static_cast<int>(get("nested_blocksize").as_int());
  state.nested_offset = static_cast<float>(get("nested_offset").as_number());
  for (const json::Value& dim : get("shape").as_array()) state.shape.push_back(dim.as_int());
  if (state.block_size != 64 || state.nested_block_size != 256 || !std::isfinite(state.nested_offset))
    throw std::runtime_error(std::string(consumer) + ": unsupported NF4 block layout for '" + weight_name + "'");
  return state;
}

}  // namespace vidfab
