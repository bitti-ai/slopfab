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
  kViggleAnimatePrunedTable,
};

constexpr bool is_pruned_table_architecture(TransformerArchitecture architecture) {
  return architecture == TransformerArchitecture::kPrunedTable ||
         architecture == TransformerArchitecture::kRef2VAPrunedTable ||
         architecture == TransformerArchitecture::kViggleAnimatePrunedTable;
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

// Missing metadata retains the historical contiguous [Q; K; V] layout.
// An explicit interleaved layout is [head, Q/K/V, head_channel].
bool transformer_qkv_is_interleaved(const SafeTensors& checkpoint);
void validate_interleaved_qkv(const TensorView& tensor, int head_dim);
// Copies a contiguous-layout row slice from an interleaved weight or row-scale
// tensor without changing its dtype. Packed/nested quantization is unsupported.
std::vector<uint8_t> deinterleave_qkv_rows(const TensorView& tensor, int head_dim,
                                        size_t row_offset = 0, size_t row_count = 0);

// Ref2VA inputs must never be passed through an FL2VA/T2VA transformer. This
// is intentionally a header-only inspection and does not touch tensor data.
void require_ref2va_transformer(const SafeTensors& checkpoint, size_t reference_count);

}  // namespace slopfab::dit
