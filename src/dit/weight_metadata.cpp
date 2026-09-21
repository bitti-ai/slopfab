#include "weight_metadata.h"
#include "slopfab/json.h"
#include <stdexcept>

namespace slopfab::dit::detail {

// The blob is the checkpoint's own statement of what its bytes mean, so it is
// parsed rather than pattern-matched. A substring search cannot tell a format
// it does not implement from one it does — it just fails to find its needle and
// carries on — and this is the file's only description of layouts that are
// otherwise indistinguishable by inspection.
QuantTag read_comfy_quant(const SafeTensors& st, const std::string& name) {
  QuantTag tag;
  const TensorView* v = st.find(name + ".comfy_quant");
  if (v == nullptr)
    return tag;

  std::string text(static_cast<const char*>(v->data), v->nbytes);
  // ComfyUI writes the blob as a byte tensor, which may be NUL-padded to a
  // whole number of elements.
  while (!text.empty() && (text.back() == '\0' || text.back() == ' ' || text.back() == '\n')) {
    text.pop_back();
  }
  if (text.empty())
    return tag;

  json::Value root;
  try {
    root = json::parse(text);
  } catch (const std::exception& e) {
    throw std::runtime_error("transformer: '" + name + ".comfy_quant' is not valid JSON (" +
                             e.what() + "): " + text);
  }
  if (const json::Value* f = root.find("format"); f != nullptr)
    tag.format = f->as_string();
  if (const json::Value* p = root.find("full_precision_matrix_mult"); p != nullptr) {
    tag.full_precision = p->as_bool();
  }
  if (const json::Value* c = root.find("convrot"); c != nullptr) {
    tag.convrot = c->as_bool();
  }
  if (const json::Value* g = root.find("convrot_groupsize"); g != nullptr) {
    const int64_t value = g->as_int();
    if (value <= 0 || value > 256) {
      throw std::runtime_error("transformer: '" + name +
                               ".comfy_quant' has invalid convrot_groupsize " +
                               std::to_string(value));
    }
    tag.convrot_group = static_cast<int>(value);
  }

  // Anything else is a layout this port has not been shown, and guessing at one
  // yields finite plausible output rather than a failure.
  if (tag.format != "nvfp4" && tag.format != "float8_e4m3fn" && tag.format != "int8_tensorwise") {
    throw std::runtime_error("transformer: '" + name + "' declares quant format '" + tag.format +
                             "', which this port does not implement");
  }
  return tag;
}

}
