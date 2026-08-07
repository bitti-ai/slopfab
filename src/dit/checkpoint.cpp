#include "vidfab/dit/checkpoint.h"

#include <stdexcept>
#include <string>

namespace vidfab::dit {
namespace {

bool has(const SafeTensors& st, const char* name) { return st.find(name) != nullptr; }

}  // namespace

TransformerCheckpointKind detect_transformer_checkpoint(const SafeTensors& checkpoint) {
  // The released pruned checkpoints replace the timestep MLP with a lookup
  // table and contract every AdaLN projection to rank eight.
  if (has(checkpoint, "adaln_t_table") &&
      has(checkpoint, "blocks.0.adaln_proj.linear.weight")) {
    return TransformerCheckpointKind::kPrunedTable;
  }

  // The Ref2VA file retains the timestep MLP and full AdaLN projections. Its
  // bitsandbytes NF4 weights are unambiguously identified by the quant-state
  // tensor; a flattened U8 weight alone is not sufficient evidence.
  if (has(checkpoint, "time_embedder.proj_in.weight") &&
      has(checkpoint, "time_embedder.proj_out.weight") &&
      has(checkpoint,
          "blocks.0.adaln_proj.linear.weight.quant_state.bitsandbytes__nf4") &&
      has(checkpoint, "blocks.0.adaln_proj.linear.weight.absmax")) {
    return TransformerCheckpointKind::kRef2VABitsAndBytesNF4;
  }
  return TransformerCheckpointKind::kUnknown;
}

const char* transformer_checkpoint_kind_name(TransformerCheckpointKind kind) {
  switch (kind) {
    case TransformerCheckpointKind::kPrunedTable:
      return "pruned AdaLN-table transformer";
    case TransformerCheckpointKind::kRef2VABitsAndBytesNF4:
      return "Ref2VA bitsandbytes NF4 transformer";
    case TransformerCheckpointKind::kUnknown:
      return "unknown transformer";
  }
  return "unknown transformer";
}

void require_ref2va_transformer(const SafeTensors& checkpoint, size_t reference_count) {
  if (reference_count == 0) return;
  const TransformerCheckpointKind kind = detect_transformer_checkpoint(checkpoint);
  if (kind == TransformerCheckpointKind::kRef2VABitsAndBytesNF4) return;
  throw std::runtime_error(
      "reference-image conditioning requires a Ref2VA transformer, but '" +
      checkpoint.path() + "' is a " + transformer_checkpoint_kind_name(kind));
}

}  // namespace vidfab::dit
