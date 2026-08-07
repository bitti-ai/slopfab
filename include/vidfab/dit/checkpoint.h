#pragma once

#include <cstddef>

#include "vidfab/safetensors.h"

namespace vidfab::dit {

// Transformer families have the same top-level projection names, but their
// timestep/AdaLN and quantised-weight layouts are not interchangeable.
enum class TransformerCheckpointKind {
  kUnknown,
  kPrunedTable,
  kRef2VABitsAndBytesNF4,
};

TransformerCheckpointKind detect_transformer_checkpoint(const SafeTensors& checkpoint);
const char* transformer_checkpoint_kind_name(TransformerCheckpointKind kind);

// Ref2VA inputs must never be passed through an FL2VA/T2VA transformer. This
// is intentionally a header-only inspection and does not touch tensor data.
void require_ref2va_transformer(const SafeTensors& checkpoint, size_t reference_count);

}  // namespace vidfab::dit
