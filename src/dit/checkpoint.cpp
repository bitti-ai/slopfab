#include "vidfab/dit/checkpoint.h"

#include <stdexcept>
#include <string>

namespace vidfab::dit {
namespace {

bool has(const SafeTensors& st, const char* name) { return st.find(name) != nullptr; }

}  // namespace

TransformerArchitecture detect_transformer_architecture(const SafeTensors& checkpoint) {
  // The released pruned checkpoints replace the timestep MLP with a lookup
  // table and contract every AdaLN projection to rank eight.
  if (has(checkpoint, "adaln_t_table") &&
      has(checkpoint, "blocks.0.adaln_proj.linear.weight")) {
    return TransformerArchitecture::kPrunedTable;
  }

  // The Ref2VA file retains the timestep MLP and full AdaLN projections. Its
  // Quantization is deliberately not part of architecture detection: both the
  // NF4 release and the FP8 release use this full-AdaLN Ref2VA architecture.
  if (has(checkpoint, "time_embedder.proj_in.weight") &&
      has(checkpoint, "time_embedder.proj_out.weight") &&
      has(checkpoint, "blocks.0.adaln_proj.linear.weight")) {
    return TransformerArchitecture::kRef2VAFullAdaLN;
  }
  return TransformerArchitecture::kUnknown;
}

TransformerQuantization detect_transformer_quantization(const SafeTensors& checkpoint) {
  const TensorView* weight = checkpoint.find("blocks.0.attn.qkv_proj.weight");
  if (weight == nullptr) return TransformerQuantization::kUnknown;
  if (has(checkpoint,
          "blocks.0.attn.qkv_proj.weight.quant_state.bitsandbytes__nf4")) {
    return TransformerQuantization::kBitsAndBytesNF4;
  }
  if (weight->dtype == DType::kF8E4M3) return TransformerQuantization::kFloat8;
  const TensorView* scale = checkpoint.find("blocks.0.attn.qkv_proj.weight_scale");
  if (weight->dtype == DType::kU8 && scale != nullptr && scale->dtype == DType::kF8E4M3) {
    return TransformerQuantization::kNativeNVFP4;
  }
  return TransformerQuantization::kUnknown;
}

const char* transformer_architecture_name(TransformerArchitecture architecture) {
  switch (architecture) {
    case TransformerArchitecture::kPrunedTable:
      return "pruned AdaLN-table transformer";
    case TransformerArchitecture::kRef2VAFullAdaLN:
      return "full-AdaLN Ref2VA transformer";
    case TransformerArchitecture::kUnknown:
      return "unknown transformer";
  }
  return "unknown transformer";
}

const char* transformer_quantization_name(TransformerQuantization quantization) {
  switch (quantization) {
    case TransformerQuantization::kFloat8: return "float8";
    case TransformerQuantization::kNativeNVFP4: return "native NVFP4";
    case TransformerQuantization::kBitsAndBytesNF4: return "bitsandbytes NF4";
    case TransformerQuantization::kUnknown: return "unknown";
  }
  return "unknown";
}

void require_ref2va_transformer(const SafeTensors& checkpoint, size_t reference_count) {
  if (reference_count == 0) return;
  const TransformerArchitecture architecture = detect_transformer_architecture(checkpoint);
  if (architecture == TransformerArchitecture::kRef2VAFullAdaLN) return;
  throw std::runtime_error(
      "reference-image conditioning requires a Ref2VA transformer, but '" +
      checkpoint.path() + "' is a " + transformer_architecture_name(architecture));
}

}  // namespace vidfab::dit
