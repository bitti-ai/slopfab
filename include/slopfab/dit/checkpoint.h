#pragma once

#include <cstddef>

#include "slopfab/safetensors.h"

namespace slopfab::dit {

// Transformer families have the same top-level projection names, but their
// timestep/AdaLN and quantised-weight layouts are not interchangeable.
enum class TransformerArchitecture {
  kUnknown,
  kPrunedTable,
  kRef2VAPrunedTable,
  kRef2VAFullAdaLN,
};

constexpr bool is_pruned_table_architecture(TransformerArchitecture architecture) {
  return architecture == TransformerArchitecture::kPrunedTable ||
         architecture == TransformerArchitecture::kRef2VAPrunedTable;
}

enum class TransformerQuantization {
  kUnknown,
  kFloat8,
  kInt8ConvRot,
  kNativeNVFP4,
  kBitsAndBytesNF4,
};

TransformerArchitecture detect_transformer_architecture(const SafeTensors& checkpoint);
TransformerQuantization detect_transformer_quantization(const SafeTensors& checkpoint);
const char* transformer_architecture_name(TransformerArchitecture architecture);
const char* transformer_quantization_name(TransformerQuantization quantization);

// Ref2VA inputs must never be passed through an FL2VA/T2VA transformer. This
// is intentionally a header-only inspection and does not touch tensor data.
void require_ref2va_transformer(const SafeTensors& checkpoint, size_t reference_count);

}  // namespace slopfab::dit
