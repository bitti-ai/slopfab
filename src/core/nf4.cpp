#include "slopfab/nf4.h"

#include <cmath>
#include <stdexcept>
#include <limits>

#include "slopfab/json.h"

namespace slopfab {

namespace {
std::string state_name(const std::string& weight_name) {
  return weight_name + ".quant_state.bitsandbytes__nf4";
}
} // namespace

bool is_nf4_weight(const SafeTensors& checkpoint, const std::string& weight_name) {
  return checkpoint.find(state_name(weight_name)) != nullptr;
}

NF4State read_nf4_state(const SafeTensors& checkpoint, const std::string& weight_name,
                        const char* consumer, bool require_bfloat16) {
  const std::string name = state_name(weight_name);
  const TensorView* view = checkpoint.find(name);
  if (view == nullptr || view->dtype != DType::kU8 || view->shape.size() != 1)
    throw std::runtime_error(std::string(consumer) + ": '" + name + "' is not rank-1 U8 JSON");
  json::Value root;
  try {
    root = json::parse(std::string(static_cast<const char*>(view->data), view->nbytes));
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string(consumer) + ": invalid NF4 state '" + name + "' (" +
                             e.what() + ")");
  }
  auto get = [&](const char* key) -> const json::Value& {
    const json::Value* value = root.find(key);
    if (!value)
      throw std::runtime_error(std::string(consumer) + ": NF4 state missing '" + key + "'");
    return *value;
  };
  if (get("quant_type").as_string() != "nf4" || get("nested_dtype").as_string() != "float32")
    throw std::runtime_error(std::string(consumer) + ": unsupported NF4 contract for '" +
                             weight_name + "'");
  NF4State state;
  if (const auto* dtype = root.find("dtype"))
    state.source_dtype = dtype->as_string();
  if (require_bfloat16 && state.source_dtype != "bfloat16")
    throw std::runtime_error(std::string(consumer) + ": NF4 source dtype must be bfloat16 for '" +
                             weight_name + "'");
  auto integer = [&](const json::Value& value, int64_t maximum) {
    const double number = value.as_number();
    if (!std::isfinite(number) || number <= 0 || std::floor(number) != number ||
        number >= static_cast<double>(maximum))
      throw std::runtime_error(std::string(consumer) + ": invalid NF4 integer for '" + weight_name +
                               "'");
    return static_cast<int64_t>(number);
  };
  state.block_size = static_cast<int>(integer(get("blocksize"), std::numeric_limits<int>::max()));
  state.nested_block_size =
      static_cast<int>(integer(get("nested_blocksize"), std::numeric_limits<int>::max()));
  state.nested_offset = static_cast<float>(get("nested_offset").as_number());
  for (const json::Value& dim : get("shape").as_array())
    state.shape.push_back(integer(dim, std::numeric_limits<int64_t>::max()));
  if (state.shape.empty())
    throw std::runtime_error(std::string(consumer) + ": NF4 shape must not be empty");
  if (state.block_size != 64 || state.nested_block_size != 256 ||
      !std::isfinite(state.nested_offset))
    throw std::runtime_error(std::string(consumer) + ": unsupported NF4 block layout for '" +
                             weight_name + "'");
  return state;
}

} // namespace slopfab
