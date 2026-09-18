#include "slopfab/int8_weight.h"

#include <cmath>
#include <limits>
#include <stdexcept>

#include "slopfab/json.h"

namespace slopfab {

Int8WeightState read_int8_weight(const SafeTensors& checkpoint,
                                const std::string& name, const char* consumer) {
  const auto fail = [&](const std::string& reason) {
    return std::runtime_error(std::string(consumer) + ": INT8 weight '" + name +
                              "': " + reason);
  };
  const TensorView& weight = checkpoint.at(name);
  if (weight.dtype != DType::kI8 || weight.shape.size() != 2 ||
      weight.shape[0] <= 0 || weight.shape[1] <= 0 ||
      weight.shape[0] > std::numeric_limits<int>::max() ||
      weight.shape[1] > std::numeric_limits<int>::max())
    throw fail("expected a rank-2 I8 matrix");
  const std::string base = name.size() >= 7 &&
      name.compare(name.size() - 7, 7, ".weight") == 0
      ? name.substr(0, name.size() - 7) : name;
  const TensorView* tag = checkpoint.find(base + ".comfy_quant");
  if (!tag || tag->dtype != DType::kU8 || tag->shape.size() != 1)
    throw fail("missing rank-1 U8 comfy_quant metadata");
  std::string text(static_cast<const char*>(tag->data), tag->nbytes);
  while (!text.empty() && text.back() == '\0') text.pop_back();
  json::Value root;
  try {
    root = json::parse(text);
  } catch (const std::exception& e) {
    throw fail(std::string("invalid comfy_quant JSON: ") + e.what());
  }
  const json::Value* format = root.find("format");
  if (!format || !format->is_string() || format->as_string() != "int8_tensorwise")
    throw fail("requires int8_tensorwise format");
  Int8WeightState result;
  result.rows = static_cast<int>(weight.shape[0]);
  result.columns = static_cast<int>(weight.shape[1]);
  const json::Value* rotation = root.find("convrot");
  if (rotation && rotation->as_bool()) {
    const json::Value* group = root.find("convrot_groupsize");
    const double size = group ? group->as_number() : 256;
    if (size != 4 && size != 16 && size != 64 && size != 256)
      throw fail("ConvRot group must be 4, 16, 64 or 256");
    // The quantizer skips rotation when a row does not contain whole groups.
    if (result.columns % static_cast<int>(size) == 0)
      result.rotation_group = static_cast<int>(size);
  }
  const TensorView* scale = checkpoint.find(base + ".weight_scale");
  if (!scale || scale->dtype != DType::kF32 ||
      (scale->shape != std::vector<int64_t>{result.rows, 1} &&
       scale->shape != std::vector<int64_t>{result.rows}))
    throw fail("requires one F32 weight_scale per output row");
  result.codes = static_cast<const int8_t*>(weight.data);
  result.scales = static_cast<const float*>(scale->data);
  for (int row = 0; row < result.rows; ++row) {
    if (!std::isfinite(result.scales[row]) || result.scales[row] < 0)
      throw fail("weight_scale must be finite and nonnegative");
  }
  return result;
}

std::vector<uint16_t> unpack_int8_weight(const Int8WeightState& state,
                                        bool canonicalize_subnormals) {
  const int group = state.rotation_group;
  const float normalization = 1.0f / std::sqrt(static_cast<float>(group));
  std::vector<uint16_t> output(static_cast<size_t>(state.rows) * state.columns);
  for (int row = 0; row < state.rows; ++row) {
    for (int column = 0; column < state.columns; column += group) {
      const size_t offset = static_cast<size_t>(row) * state.columns + column;
      float values[256];
      for (int i = 0; i < group; ++i) values[i] = state.codes[offset + i];
      for (int stride = 1; stride < group; stride *= 4) {
        for (int base = 0; base < group; base += 4 * stride) {
          for (int i = 0; i < stride; ++i) {
            const int j = base + i;
            const float a = values[j], b = values[j + stride];
            const float c = values[j + 2 * stride], d = values[j + 3 * stride];
            values[j] = a + b + c - d;
            values[j + stride] = a + b - c + d;
            values[j + 2 * stride] = a - b + c + d;
            values[j + 3 * stride] = -a + b + c + d;
          }
        }
      }
      const float factor = state.scales[row] * normalization;
      for (int i = 0; i < group; ++i) {
        uint16_t bits = f32_to_f16(values[i] * factor);
        if (canonicalize_subnormals && (bits & 0x7c00u) == 0)
          bits &= 0x8000u;
        output[offset + i] = bits;
      }
    }
  }
  return output;
}

}  // namespace slopfab
