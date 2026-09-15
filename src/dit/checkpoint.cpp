#include "slopfab/dit/checkpoint.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <cstring>
#include <stdexcept>
#include <string>

namespace slopfab::dit {
namespace {

bool has(const SafeTensors& st, const char* name) { return st.find(name) != nullptr; }

bool path_marks_ref2va(const SafeTensors& st) {
  std::string name = std::filesystem::path(st.path()).filename().string();
  std::transform(name.begin(), name.end(), name.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return name.find("ref2va") != std::string::npos;
}

bool marks_viggle(const SafeTensors& st) {
  const auto source = st.metadata().find("source");
  if (source != st.metadata().end()) return source->second == "Viggle/Viggle-Animate";
  std::string name = std::filesystem::path(st.path()).filename().string();
  std::transform(name.begin(), name.end(), name.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return name.find("viggle-animate") != std::string::npos;
}

}  // namespace

TransformerArchitecture detect_transformer_architecture(const SafeTensors& checkpoint) {
  // The released pruned checkpoints replace the timestep MLP with a lookup
  // table and contract every AdaLN projection to rank eight.
  if (has(checkpoint, "adaln_t_table") &&
      has(checkpoint, "blocks.0.adaln_proj.linear.weight")) {
    if (marks_viggle(checkpoint)) return TransformerArchitecture::kViggleAnimatePrunedTable;
    // The released pruned Ref2VA FP8 file has the exact same tensor names,
    // shapes, dtypes and empty metadata as the FL2VA file. The distribution
    // filename is therefore the only identity signal available in the file.
    if (path_marks_ref2va(checkpoint)) return TransformerArchitecture::kRef2VAPrunedTable;
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

TransformerQuantization detect_transformer_quantization(const SafeTensors& checkpoint) {
  const TensorView* weight = checkpoint.find("blocks.0.attn.qkv_proj.weight");
  if (weight == nullptr) return TransformerQuantization::kUnknown;
  if (has(checkpoint,
          "blocks.0.attn.qkv_proj.weight.quant_state.bitsandbytes__nf4")) {
    return TransformerQuantization::kBitsAndBytesNF4;
  }
  if (weight->dtype == DType::kF8E4M3) return TransformerQuantization::kFloat8;
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
    case TransformerQuantization::kFloat8: return "float8";
    case TransformerQuantization::kInt8ConvRot: return "int8 ConvRot";
    case TransformerQuantization::kNativeNVFP4: return "native NVFP4";
    case TransformerQuantization::kBitsAndBytesNF4: return "bitsandbytes NF4";
    case TransformerQuantization::kUnknown: return "unknown";
  }
  return "unknown";
}

bool transformer_qkv_is_interleaved(const SafeTensors& checkpoint) {
  const auto it = checkpoint.metadata().find("qkv_layout");
  if (it == checkpoint.metadata().end() || it->second == "contiguous") return false;
  if (it->second == "interleaved") return true;
  throw std::runtime_error("transformer: unsupported qkv_layout '" + it->second + "'");
}

void validate_interleaved_qkv(const TensorView& tensor, int head_dim) {
  if (head_dim <= 0 || tensor.shape.size() != 2 || tensor.shape[0] <= 0 ||
      tensor.shape[1] <= 0 || tensor.shape[0] % (3LL * head_dim) != 0 ||
      (tensor.dtype != DType::kI8 && tensor.dtype != DType::kF32 &&
       tensor.dtype != DType::kF16 && tensor.dtype != DType::kBF16 &&
       tensor.dtype != DType::kF8E4M3))
    throw std::runtime_error("transformer: unsupported interleaved QKV tensor '" + tensor.name + "'");
}

std::vector<uint8_t> deinterleave_qkv_rows(const TensorView& tensor, int head_dim,
                                        size_t row_offset, size_t row_count) {
  validate_interleaved_qkv(tensor, head_dim);
  const size_t rows = static_cast<size_t>(tensor.shape[0]);
  if (row_count == 0) row_count = rows;
  if (row_offset > rows || row_count > rows - row_offset || tensor.nbytes % rows)
    throw std::runtime_error("transformer: invalid interleaved QKV row slice");
  const size_t row_bytes = tensor.nbytes / rows;
  const size_t inner = rows / 3;
  std::vector<uint8_t> result(row_count * row_bytes);
  for (size_t i = 0; i < row_count; ++i) {
    const size_t row = row_offset + i;
    const size_t channel = row % inner;
    const size_t source = (channel / head_dim * 3 + row / inner) * head_dim +
                          channel % head_dim;
    std::memcpy(result.data() + i * row_bytes,
                static_cast<const uint8_t*>(tensor.data) + source * row_bytes, row_bytes);
  }
  return result;
}

void require_ref2va_transformer(const SafeTensors& checkpoint, size_t reference_count) {
  if (reference_count == 0) return;
  const TransformerArchitecture architecture = detect_transformer_architecture(checkpoint);
  if (architecture == TransformerArchitecture::kRef2VAPrunedTable ||
      architecture == TransformerArchitecture::kViggleAnimatePrunedTable ||
      architecture == TransformerArchitecture::kRef2VAFullAdaLN) return;
  throw std::runtime_error(
      "reference conditioning requires a Ref2VA transformer, but '" +
      checkpoint.path() + "' is a " + transformer_architecture_name(architecture));
}

}  // namespace slopfab::dit
