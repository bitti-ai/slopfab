#include "slopfab/dit/checkpoint.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <cstring>
#include <stdexcept>
#include <string>
#include <set>
#include "slopfab/json.h"
#include "slopfab/dit/adaln.h"

namespace slopfab::dit {
namespace {

bool has(const SafeTensors& st, const char* name) {
  return st.find(name) != nullptr;
}

bool path_marks_ref2va(const SafeTensors& st) {
  std::string name = std::filesystem::path(st.path()).filename().string();
  std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return name.find("ref2va") != std::string::npos;
}

bool marks_viggle(const SafeTensors& st) {
  const auto source = st.metadata().find("source");
  if (source != st.metadata().end())
    return source->second == "Viggle/Viggle-Animate";
  std::string name = std::filesystem::path(st.path()).filename().string();
  std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return name.find("viggle-animate") != std::string::npos;
}

bool marks_fasth3_v2(const SafeTensors& st) {
  // V1 and V2 share tensor layouts but have different trained schedules.
  // Converted files often retain only {format: pt}, so recognize the release
  // filename as well as an explicit source/model_id in renamed archives.
  std::string identity = std::filesystem::path(st.path()).filename().string();
  for (const char* key : {"source", "model_id"}) {
    const auto it = st.metadata().find(key);
    if (it != st.metadata().end())
      identity += " " + it->second;
  }
  std::transform(identity.begin(), identity.end(), identity.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  identity.erase(std::remove_if(identity.begin(), identity.end(),
                                [](char c) {
                                  return c == '_' || c == '-';
                                }),
                 identity.end());
  return identity.find("fasth38stepv2") != std::string::npos;
}

} // namespace

static TransformerArchitecture infer_legacy_architecture(const SafeTensors& checkpoint) {
  // The released pruned checkpoints replace the timestep MLP with a lookup
  // table and contract every AdaLN projection to rank eight.
  if (has(checkpoint, "adaln_t_table") && has(checkpoint, "blocks.0.adaln_proj.linear.weight")) {
    if (has(checkpoint, "blocks.0.attn.to_gate_compress.weight"))
      return marks_fasth3_v2(checkpoint) ? TransformerArchitecture::kFastH3V2PrunedTable
                                         : TransformerArchitecture::kUnknown;
    if (marks_viggle(checkpoint))
      return TransformerArchitecture::kViggleAnimatePrunedTable;
    // The released pruned Ref2VA FP8 file has the exact same tensor names,
    // shapes, dtypes and empty metadata as the FL2VA file. The distribution
    // filename is therefore the only identity signal available in the file.
    if (path_marks_ref2va(checkpoint))
      return TransformerArchitecture::kRef2VAPrunedTable;
    return TransformerArchitecture::kPrunedTable;
  }

  // The unpruned Ref2VA file retains the timestep MLP and full AdaLN
  // projections. Quantization is deliberately not part of architecture
  // detection; the released pruned FP8 variant is handled above.
  if (has(checkpoint, "time_embedder.proj_in.weight") &&
      has(checkpoint, "time_embedder.proj_out.weight") &&
      has(checkpoint, "blocks.0.adaln_proj.linear.weight")) {
    return TransformerArchitecture::kRef2VAFullAdaLN;
  }
  return TransformerArchitecture::kUnknown;
}

void validate_adaln_table_config(int rank, int rows) {
  if (rank != AdaLNTable::kRank || rows != AdaLNTable::kRows)
    throw std::runtime_error("transformer: table modulation requires AdaLN rank 8 and 1025 rows");
}

ModelDescriptor resolve_model_descriptor(const SafeTensors& checkpoint) {
  ModelDescriptor model;
  model.compatibility_architecture = infer_legacy_architecture(checkpoint);
  model.quantization = detect_transformer_quantization(checkpoint);
  const auto legacy = model.compatibility_architecture;
  model.modulation = legacy == TransformerArchitecture::kRef2VAFullAdaLN
                         ? ModulationImplementation::kTimestepMlp
                         : ModulationImplementation::kTable;
  model.supports_references = legacy == TransformerArchitecture::kRef2VAFullAdaLN ||
                              legacy == TransformerArchitecture::kRef2VAPrunedTable ||
                              legacy == TransformerArchitecture::kViggleAnimatePrunedTable;
  model.compressed_attention = has(checkpoint, "blocks.0.attn.to_gate_compress.weight");
  const auto layout = checkpoint.metadata().find("qkv_layout");
  if (layout != checkpoint.metadata().end()) {
    if (layout->second != "contiguous" && layout->second != "interleaved")
      throw std::runtime_error("transformer: unsupported qkv_layout '" + layout->second + "'");
    model.qkv_interleaved = layout->second == "interleaved";
  }
  const auto metadata = checkpoint.metadata().find("slopfab.model");
  if (metadata == checkpoint.metadata().end())
    return model;
  const auto root = json::parse(metadata->second);
  if (!root.is_object())
    throw std::runtime_error("slopfab.model: expected an object");
  const std::set<std::string> keys = {
      "version",    "family",         "modulation",   "supports_references", "compressed_attention",
      "qkv_layout", "legacy_profile", "adaln_grid_id"};
  for (const auto& item : root.as_object())
    if (!keys.count(item.first))
      throw std::runtime_error("slopfab.model: unknown field '" + item.first + "'");
  auto required = [&](const char* key) -> const json::Value& {
    const auto* value = root.find(key);
    if (!value)
      throw std::runtime_error(std::string("slopfab.model: missing '") + key + "'");
    return *value;
  };
  if (required("version").as_number() != 1)
    throw std::runtime_error("slopfab.model: unsupported version");
  if (required("family").as_string() != "h3")
    throw std::runtime_error("slopfab.model: unsupported transformer family");
  const std::string modulation = required("modulation").as_string();
  if (modulation != "table" && modulation != "timestep_mlp")
    throw std::runtime_error("slopfab.model: unsupported modulation implementation");
  model.modulation = modulation == "table" ? ModulationImplementation::kTable
                                           : ModulationImplementation::kTimestepMlp;
  model.supports_references = required("supports_references").as_bool();
  model.compressed_attention = required("compressed_attention").as_bool();
  const std::string qkv = required("qkv_layout").as_string();
  if (qkv != "contiguous" && qkv != "interleaved")
    throw std::runtime_error("slopfab.model: unsupported qkv_layout");
  model.qkv_interleaved = qkv == "interleaved";
  if (layout != checkpoint.metadata().end() && layout->second != qkv)
    throw std::runtime_error("slopfab.model: qkv_layout conflicts with legacy metadata");
  const bool table = model.modulation == ModulationImplementation::kTable;
  if (!has(checkpoint, "blocks.0.adaln_proj.linear.weight") ||
      (table && (!has(checkpoint, "adaln_t_table") ||
                 checkpoint.at("adaln_t_table").shape !=
                     std::vector<int64_t>{AdaLNTable::kRows, AdaLNTable::kRank})) ||
      (!table && (!has(checkpoint, "time_embedder.proj_in.weight") ||
                  !has(checkpoint, "time_embedder.proj_out.weight"))))
    throw std::runtime_error("slopfab.model: modulation tensor contract does not match metadata");
  if (model.compressed_attention != has(checkpoint, "blocks.0.attn.to_gate_compress.weight"))
    throw std::runtime_error("slopfab.model: compressed attention requires matching gate tensors");
  if (!table && (!model.supports_references || model.compressed_attention))
    throw std::runtime_error("slopfab.model: unsupported timestep MLP capability combination");
  model.compatibility_architecture = !table ? TransformerArchitecture::kRef2VAFullAdaLN
                                     : model.supports_references
                                         ? TransformerArchitecture::kRef2VAPrunedTable
                                         : TransformerArchitecture::kPrunedTable;
  if (const auto* profile = root.find("legacy_profile")) {
    const auto& name = profile->as_string();
    if (name == "animate" && table && model.supports_references && !model.compressed_attention)
      model.compatibility_architecture = TransformerArchitecture::kViggleAnimatePrunedTable;
    else if (name == "fast_h3_v2" && table && !model.supports_references &&
             model.compressed_attention)
      model.compatibility_architecture = TransformerArchitecture::kFastH3V2PrunedTable;
    else if (name != "none")
      throw std::runtime_error("slopfab.model: incompatible legacy_profile");
  }
  if (const auto* grid = root.find("adaln_grid_id")) {
    model.adaln_grid_id = grid->as_string();
    if (model.adaln_grid_id.empty() || !table)
      throw std::runtime_error("slopfab.model: adaln_grid_id requires a nonempty table identity");
  }
  model.explicit_metadata = true;
  model.origin = "slopfab.model v1 metadata";
  return model;
}

TransformerArchitecture detect_transformer_architecture(const SafeTensors& checkpoint) {
  return resolve_model_descriptor(checkpoint).compatibility_architecture;
}

TransformerQuantization detect_transformer_quantization(const SafeTensors& checkpoint) {
  const TensorView* weight = checkpoint.find("blocks.0.attn.qkv_proj.weight");
  if (weight == nullptr)
    return TransformerQuantization::kUnknown;
  if (has(checkpoint, "blocks.0.attn.qkv_proj.weight.quant_state.bitsandbytes__nf4")) {
    return TransformerQuantization::kBitsAndBytesNF4;
  }
  if (weight->dtype == DType::kF8E4M3)
    return TransformerQuantization::kFloat8;
  const TensorView* scale = checkpoint.find("blocks.0.attn.qkv_proj.weight_scale");
  if (weight->dtype == DType::kI8 && weight->shape.size() == 2 && scale != nullptr &&
      scale->dtype == DType::kF32 && scale->shape.size() == 2 &&
      scale->shape[0] == weight->shape[0] && scale->shape[1] == 1) {
    return TransformerQuantization::kInt8ConvRot;
  }
  if (weight->dtype == DType::kU8 && scale != nullptr && scale->dtype == DType::kF8E4M3) {
    return TransformerQuantization::kNativeNVFP4;
  }
  return TransformerQuantization::kUnknown;
}

const char* transformer_architecture_name(TransformerArchitecture architecture) {
  switch (architecture) {
  case TransformerArchitecture::kPrunedTable:
    return "pruned AdaLN-table transformer";
  case TransformerArchitecture::kFastH3V2PrunedTable:
    return "FastH3 V2 (VSA-H3, 8-step) transformer";
  case TransformerArchitecture::kRef2VAPrunedTable:
    return "pruned AdaLN-table Ref2VA transformer";
  case TransformerArchitecture::kRef2VAFullAdaLN:
    return "full-AdaLN Ref2VA transformer";
  case TransformerArchitecture::kViggleAnimatePrunedTable:
    return "Viggle-Animate pruned AdaLN-table Ref2VA transformer";
  case TransformerArchitecture::kUnknown:
    return "unknown transformer";
  }
  return "unknown transformer";
}

const char* transformer_quantization_name(TransformerQuantization quantization) {
  switch (quantization) {
  case TransformerQuantization::kFloat8:
    return "float8";
  case TransformerQuantization::kInt8ConvRot:
    return "int8 ConvRot";
  case TransformerQuantization::kNativeNVFP4:
    return "native NVFP4";
  case TransformerQuantization::kBitsAndBytesNF4:
    return "bitsandbytes NF4";
  case TransformerQuantization::kUnknown:
    return "unknown";
  }
  return "unknown";
}

bool transformer_qkv_is_interleaved(const SafeTensors& checkpoint) {
  return resolve_model_descriptor(checkpoint).qkv_interleaved;
}

void validate_interleaved_qkv(const TensorView& tensor, int head_dim) {
  if (head_dim <= 0 || tensor.shape.size() != 2 || tensor.shape[0] <= 0 || tensor.shape[1] <= 0 ||
      tensor.shape[0] % (3LL * head_dim) != 0 ||
      (tensor.dtype != DType::kI8 && tensor.dtype != DType::kF32 && tensor.dtype != DType::kF16 &&
       tensor.dtype != DType::kBF16 && tensor.dtype != DType::kF8E4M3))
    throw std::runtime_error("transformer: unsupported interleaved QKV tensor '" + tensor.name +
                             "'");
}

std::vector<uint8_t> deinterleave_qkv_rows(const TensorView& tensor, int head_dim,
                                           size_t row_offset, size_t row_count) {
  validate_interleaved_qkv(tensor, head_dim);
  const size_t rows = static_cast<size_t>(tensor.shape[0]);
  if (row_count == 0)
    row_count = rows;
  if (row_offset > rows || row_count > rows - row_offset || tensor.nbytes % rows)
    throw std::runtime_error("transformer: invalid interleaved QKV row slice");
  const size_t row_bytes = tensor.nbytes / rows;
  const size_t inner = rows / 3;
  std::vector<uint8_t> result(row_count * row_bytes);
  for (size_t i = 0; i < row_count; ++i) {
    const size_t row = row_offset + i;
    const size_t channel = row % inner;
    const size_t source = (channel / head_dim * 3 + row / inner) * head_dim + channel % head_dim;
    std::memcpy(result.data() + i * row_bytes,
                static_cast<const uint8_t*>(tensor.data) + source * row_bytes, row_bytes);
  }
  return result;
}

void require_ref2va_transformer(const SafeTensors& checkpoint, size_t reference_count) {
  if (reference_count == 0)
    return;
  const auto model = resolve_model_descriptor(checkpoint);
  if (model.supports_references)
    return;
  const auto architecture = model.compatibility_architecture;
  throw std::runtime_error("reference conditioning requires a Ref2VA transformer, but '" +
                           checkpoint.path() + "' is a " +
                           transformer_architecture_name(architecture));
}

} // namespace slopfab::dit
