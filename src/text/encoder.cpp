// Host-side half of the Qwen3-VL conditioner: checkpoint validation, the layer
// blob layout, the RoPE tables and the embedding gather. Nothing here touches
// CUDA, which is what lets it be read and tested without a device — and what
// lets `include/vidfab/text/encoder.h` stay free of CUDA types for host
// translation units. The forward pass lives in src/cuda/encoder_kernels.cu.
//
// The validation is deliberately unforgiving. Every silent failure in
// docs/text_encoder_spec.md section 9 is a wrong *number*, not a wrong shape,
// so shapes are the only thing a loader can check — and a checkpoint that
// differs in any of them is not the one the spec describes.

#include "vidfab/text/encoder.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace vidfab::text {
namespace {

constexpr int kConvRotGroup = 256;

// Mirrors vidfab::cuda::kNVFP4BlockSize. Duplicated rather than included
// because this file is compiled by the host compiler and linear.cuh drags in
// cuBLAS; `encoder_kernels.cu` static_asserts the two agree.
constexpr int64_t kNVFP4Block = 16;
// Two E2M1 values per stored byte.
constexpr int64_t kNVFP4Pack = 2;

// Every allocation in the blob starts on a 256-byte boundary. The int8 weights
// are already multiples of 256 bytes, but the F32 scales and BF16 norms are
// not, and cuBLAS wants its operands aligned.
size_t align_up(size_t n) { return (n + 255) / 256 * 256; }

std::string layer_prefix(int layer) { return "model.layers." + std::to_string(layer) + "."; }

const char* kQuantSuffixes[7] = {
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj",    "mlp.up_proj",      "mlp.down_proj",
};

// Removes ASCII whitespace and trailing NULs so the payload can be matched with
// plain substring tests. The file is 72 bytes of JSON in a U8 tensor and the
// project carries no JSON dependency into this layer.
std::string squeeze(const void* data, size_t n) {
  const char* p = static_cast<const char*>(data);
  std::string out;
  out.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    const char c = p[i];
    if (c == '\0') break;
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r') continue;
    out.push_back(c);
  }
  return out;
}

void require(bool ok, const std::string& message) {
  if (!ok) throw std::runtime_error("text encoder: " + message);
}

std::string shape_string(const std::vector<int64_t>& shape) {
  std::string s = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) s += ", ";
    s += std::to_string(shape[i]);
  }
  return s + "]";
}

void check_tensor(const SafeTensors& checkpoint, const std::string& name, DType dtype, int64_t dim0,
                  int64_t dim1) {
  const TensorView* view = checkpoint.find(name);
  require(view != nullptr, name + " is missing");
  require(view->dtype == dtype, name + " has dtype " + dtype_name(view->dtype) + ", expected " +
                                    dtype_name(dtype));
  const size_t rank = dim1 == 0 ? 1u : 2u;
  require(view->shape.size() == rank,
          name + " has shape " + shape_string(view->shape) + ", expected rank " +
              std::to_string(rank));
  require(view->shape[0] == dim0, name + " has shape " + shape_string(view->shape) +
                                      ", expected first dim " + std::to_string(dim0));
  if (dim1 != 0) {
    require(view->shape[1] == dim1, name + " has shape " + shape_string(view->shape) +
                                        ", expected second dim " + std::to_string(dim1));
  }
}

}  // namespace

WeightFormat detect_weight_format(const SafeTensors& checkpoint) {
  // Layer 0's q_proj descriptor decides for the whole file. `validate_checkpoint`
  // then re-reads all 350 and insists they agree, so a mixed-format checkpoint
  // is rejected rather than half-read off this one sample.
  const std::string name = "model.layers.0.self_attn.q_proj.comfy_quant";
  const TensorView* view = checkpoint.find(name);
  require(view != nullptr,
          name + " is missing; this file does not look like either shipped Qwen3-VL build");
  const std::string payload = squeeze(view->data, view->nbytes);
  if (payload.find("\"format\":\"nvfp4\"") != std::string::npos) return WeightFormat::kNVFP4Awq;
  if (payload.find("\"format\":\"int8_tensorwise\"") != std::string::npos) {
    return WeightFormat::kI8ConvRot;
  }
  throw std::runtime_error("text encoder: " + name + " declares an unknown format: " + payload);
}

namespace {

// Throws on kAuto rather than picking a default. Every caller either has a file
// to detect from or is a test that knows what it is building, and silently
// choosing one of two incompatible layouts is the sort of thing that shows up
// as wrong numbers 50 layers later.
WeightFormat resolved(const EncoderConfig& c) {
  require(c.format != WeightFormat::kAuto,
          "EncoderConfig::format is kAuto here; resolve it with detect_weight_format first");
  return c.format;
}

constexpr TensorSpec kAbsent = {nullptr, DType::kUnknown, 0, 0};

}  // namespace

TensorSpec layer_tensor_spec(const EncoderConfig& c, LayerTensor which) {
  const int64_t hidden = c.hidden_size;
  const int64_t q_width = static_cast<int64_t>(c.num_attention_heads) * c.head_dim;   // 8192
  const int64_t kv_width = static_cast<int64_t>(c.num_key_value_heads) * c.head_dim;  // 1024
  const int64_t inner = c.intermediate_size;

  if (resolved(c) == WeightFormat::kNVFP4Awq) {
    // Two E2M1 nibbles per byte, so the contracted dimension is halved on disk;
    // one e4m3 scale per 16 contracted elements, so it is divided by 16. The
    // scale tensor's declared shape is [out, in/16] but its *bytes* are in a
    // 128x4 tile swizzle — see temp_launch_dequant_nvfp4_swizzled in
    // src/cuda/encoder_kernels.cu. The shape check here is still worth making:
    // it is the byte count that the swizzle assumes.
    const int64_t half = kNVFP4Pack;
    const int64_t blk = kNVFP4Block;
    switch (which) {
      case LayerTensor::kQWeight:
        return {"self_attn.q_proj.weight", DType::kU8, q_width, hidden / half};
      case LayerTensor::kQScale:
        return {"self_attn.q_proj.weight_scale", DType::kF8E4M3, q_width, hidden / blk};
      case LayerTensor::kKWeight:
        return {"self_attn.k_proj.weight", DType::kU8, kv_width, hidden / half};
      case LayerTensor::kKScale:
        return {"self_attn.k_proj.weight_scale", DType::kF8E4M3, kv_width, hidden / blk};
      case LayerTensor::kVWeight:
        return {"self_attn.v_proj.weight", DType::kU8, kv_width, hidden / half};
      case LayerTensor::kVScale:
        return {"self_attn.v_proj.weight_scale", DType::kF8E4M3, kv_width, hidden / blk};
      case LayerTensor::kOWeight:
        return {"self_attn.o_proj.weight", DType::kU8, hidden, q_width / half};
      case LayerTensor::kOScale:
        return {"self_attn.o_proj.weight_scale", DType::kF8E4M3, hidden, q_width / blk};
      // Present on o_proj and down_proj only. The other five had it folded into
      // the preceding norm by the quantiser, which is why this checkpoint's
      // input_layernorm and post_attention_layernorm differ from the int8
      // build's while its q_norm and k_norm are bitwise identical to them.
      case LayerTensor::kOPreQuantScale:
        return {"self_attn.o_proj.pre_quant_scale", DType::kBF16, q_width, 0};
      case LayerTensor::kGateWeight:
        return {"mlp.gate_proj.weight", DType::kU8, inner, hidden / half};
      case LayerTensor::kGateScale:
        return {"mlp.gate_proj.weight_scale", DType::kF8E4M3, inner, hidden / blk};
      case LayerTensor::kUpWeight:
        return {"mlp.up_proj.weight", DType::kU8, inner, hidden / half};
      case LayerTensor::kUpScale:
        return {"mlp.up_proj.weight_scale", DType::kF8E4M3, inner, hidden / blk};
      case LayerTensor::kDownWeight:
        return {"mlp.down_proj.weight", DType::kU8, hidden, inner / half};
      case LayerTensor::kDownScale:
        return {"mlp.down_proj.weight_scale", DType::kF8E4M3, hidden, inner / blk};
      case LayerTensor::kDownPreQuantScale:
        return {"mlp.down_proj.pre_quant_scale", DType::kBF16, inner, 0};
      case LayerTensor::kInputLayerNorm:
        return {"input_layernorm.weight", DType::kBF16, hidden, 0};
      case LayerTensor::kPostAttentionLayerNorm:
        return {"post_attention_layernorm.weight", DType::kBF16, hidden, 0};
      case LayerTensor::kQNorm:
        return {"self_attn.q_norm.weight", DType::kBF16, c.head_dim, 0};
      case LayerTensor::kKNorm:
        return {"self_attn.k_norm.weight", DType::kBF16, c.head_dim, 0};
      case LayerTensor::kCount:
        break;
    }
    throw std::runtime_error("text encoder: layer_tensor_spec: bad LayerTensor");
  }

  switch (which) {
    // The int8 build has no AWQ activation scaling: its smoothing is the
    // ConvRot rotation, which is on the weight and needs no stored vector.
    case LayerTensor::kOPreQuantScale:
    case LayerTensor::kDownPreQuantScale:
      return kAbsent;
    // Every linear is stored PyTorch-style [out_features, in_features]; there
    // are no transposes anywhere in this checkpoint (spec section 8.1). The
    // scales are [out, 1] F32 — per output channel, despite the format tag
    // reading "int8_tensorwise" (spec section 5.1).
    case LayerTensor::kQWeight:
      return {"self_attn.q_proj.weight", DType::kI8, q_width, hidden};
    case LayerTensor::kQScale:
      return {"self_attn.q_proj.weight_scale", DType::kF32, q_width, 1};
    case LayerTensor::kKWeight:
      return {"self_attn.k_proj.weight", DType::kI8, kv_width, hidden};
    case LayerTensor::kKScale:
      return {"self_attn.k_proj.weight_scale", DType::kF32, kv_width, 1};
    case LayerTensor::kVWeight:
      return {"self_attn.v_proj.weight", DType::kI8, kv_width, hidden};
    case LayerTensor::kVScale:
      return {"self_attn.v_proj.weight_scale", DType::kF32, kv_width, 1};
    case LayerTensor::kOWeight:
      return {"self_attn.o_proj.weight", DType::kI8, hidden, q_width};
    case LayerTensor::kOScale:
      return {"self_attn.o_proj.weight_scale", DType::kF32, hidden, 1};
    case LayerTensor::kGateWeight:
      return {"mlp.gate_proj.weight", DType::kI8, inner, hidden};
    case LayerTensor::kGateScale:
      return {"mlp.gate_proj.weight_scale", DType::kF32, inner, 1};
    case LayerTensor::kUpWeight:
      return {"mlp.up_proj.weight", DType::kI8, inner, hidden};
    case LayerTensor::kUpScale:
      return {"mlp.up_proj.weight_scale", DType::kF32, inner, 1};
    case LayerTensor::kDownWeight:
      return {"mlp.down_proj.weight", DType::kI8, hidden, inner};
    case LayerTensor::kDownScale:
      return {"mlp.down_proj.weight_scale", DType::kF32, hidden, 1};
    case LayerTensor::kInputLayerNorm:
      return {"input_layernorm.weight", DType::kBF16, hidden, 0};
    case LayerTensor::kPostAttentionLayerNorm:
      return {"post_attention_layernorm.weight", DType::kBF16, hidden, 0};
    // q_norm and k_norm are [head_dim], one vector shared by all 64 (resp. 8)
    // heads and applied over the 128-wide head axis only — not over 8192 or
    // 1024 (spec section 4.1).
    case LayerTensor::kQNorm:
      return {"self_attn.q_norm.weight", DType::kBF16, c.head_dim, 0};
    case LayerTensor::kKNorm:
      return {"self_attn.k_norm.weight", DType::kBF16, c.head_dim, 0};
    case LayerTensor::kCount:
      break;
  }
  throw std::runtime_error("text encoder: layer_tensor_spec: bad LayerTensor");
}

LayerLayout make_layer_layout(const EncoderConfig& config) {
  LayerLayout layout;
  size_t cursor = 0;
  for (int i = 0; i < kLayerTensorCount; ++i) {
    const TensorSpec spec = layer_tensor_spec(config, static_cast<LayerTensor>(i));
    if (!spec.present()) {
      // Zero bytes at the current cursor. The offset is still well defined so
      // that a caller which indexes it without checking gets a pointer into its
      // own blob rather than out of bounds — but `bytes` is what says whether
      // there is anything there.
      layout.offset[i] = cursor;
      layout.bytes[i] = 0;
      continue;
    }
    const int64_t elements = spec.dim1 == 0 ? spec.dim0 : spec.dim0 * spec.dim1;
    const size_t bytes = static_cast<size_t>(elements) * dtype_size(spec.dtype);
    layout.offset[i] = cursor;
    layout.bytes[i] = bytes;
    cursor = align_up(cursor + bytes);
  }
  layout.total_bytes = cursor;
  return layout;
}

LayerGlobalScales read_global_scales(const SafeTensors& checkpoint, const EncoderConfig& config,
                                     int layer) {
  LayerGlobalScales out;
  if (resolved(config) != WeightFormat::kNVFP4Awq) return out;
  const std::string prefix = layer_prefix(layer);
  for (int i = 0; i < 7; ++i) {
    const std::string name = prefix + kQuantSuffixes[i] + ".weight_scale_2";
    const TensorView& view = checkpoint.at(name);
    require(view.dtype == DType::kF32 && view.nbytes == sizeof(float),
            name + " is not a single F32; weight_scale_2 is one scalar per tensor");
    std::memcpy(&out.value[i], view.data, sizeof(float));
    require(std::isfinite(out.value[i]) && out.value[i] > 0.0f,
            name + " is not a positive finite float");
  }
  return out;
}

void validate_checkpoint(const SafeTensors& checkpoint, const EncoderConfig& base_config) {
  EncoderConfig config = base_config;
  const WeightFormat detected = detect_weight_format(checkpoint);
  if (config.format == WeightFormat::kAuto) {
    config.format = detected;
  } else {
    require(config.format == detected,
            "the checkpoint declares the other weight format; detection is per file and there is "
            "no flag to override it");
  }
  const bool nvfp4 = config.format == WeightFormat::kNVFP4Awq;

  require(config.num_layers > 0, "num_layers must be positive");
  require(config.head_dim % 2 == 0, "head_dim must be even");
  require(config.num_attention_heads % config.num_key_value_heads == 0,
          "num_attention_heads must be a multiple of num_key_value_heads");
  if (nvfp4) {
    // Every contraction width must be a whole number of 16-wide blocks, and the
    // 128x4 scale swizzle needs whole tiles in both axes. Both hold for all
    // seven linears of this checkpoint with nothing left over, which is why the
    // loader can refuse a padded layout it has never seen rather than guess one.
    require(config.hidden_size % (kNVFP4Block * 4) == 0 &&
                config.intermediate_size % (kNVFP4Block * 4) == 0 &&
                (config.num_attention_heads * config.head_dim) % (kNVFP4Block * 4) == 0,
            "every nvfp4 contraction width must be a multiple of 64 (16 per block, 4 blocks per "
            "scale tile)");
    require(config.hidden_size % 128 == 0 && config.intermediate_size % 128 == 0 &&
                (config.num_attention_heads * config.head_dim) % 128 == 0 &&
                (config.num_key_value_heads * config.head_dim) % 128 == 0,
            "every nvfp4 out_features must be a multiple of 128, the scale swizzle's tile height");
  } else {
    // Every in_features here is 5120, 8192 or 25600, all divisible by 256, so
    // every quantised weight in this checkpoint is rotated and there is no
    // skip-when-not-divisible case (spec section 5.2).
    require(config.hidden_size % kConvRotGroup == 0 &&
                config.intermediate_size % kConvRotGroup == 0 &&
                (config.num_attention_heads * config.head_dim) % kConvRotGroup == 0,
            "every ConvRot contraction width must be a multiple of 256");
  }

  // The embedding table's storage does not follow the linears': the nvfp4 build
  // ships it as I8 with a per-row F32 scale, the int8+ConvRot build as BF16.
  if (nvfp4) {
    check_tensor(checkpoint, "model.embed_tokens.weight", DType::kI8, config.vocab_size,
                 config.hidden_size);
    check_tensor(checkpoint, "model.embed_tokens.weight_scale", DType::kF32, config.vocab_size, 1);
  } else {
    check_tensor(checkpoint, "model.embed_tokens.weight", DType::kBF16, config.vocab_size,
                 config.hidden_size);
  }

  // The absence of these two is what makes "the raw output of the last layer
  // present" the right answer, and it is worth failing loudly if a future
  // checkpoint reintroduces them: `hidden_states[N]` in HF is the *post-norm*
  // value, which is exactly the conditioning encoders.py:142-149 rejects
  // (spec section 1.4).
  require(checkpoint.find("model.norm.weight") == nullptr,
          "model.norm.weight is present. This checkpoint is not the 50-layer truncation this "
          "port implements; taking the last layer's raw output would no longer match "
          "hidden_states[50]");
  require(checkpoint.find("lm_head.weight") == nullptr,
          "lm_head.weight is present; the conditioner path never runs a head");

  size_t layer_tensors = 0;
  size_t visual_tensors = 0;
  size_t other_tensors = 0;
  for (const auto& entry : checkpoint.tensors()) {
    const std::string& name = entry.first;
    if (name.rfind("model.layers.", 0) == 0) {
      ++layer_tensors;
    } else if (name.rfind("visual.", 0) == 0) {
      ++visual_tensors;
    } else {
      ++other_tensors;
    }
  }

  // int8:  18 loaded + 7 comfy_quant                                     = 25.
  // nvfp4: 4 norms + 7 x (weight, weight_scale, weight_scale_2, comfy_quant)
  //        + 2 pre_quant_scale                                            = 34.
  // (spec section 8.3)
  const size_t per_layer = nvfp4 ? 34u : 25u;
  const size_t expected_layer_tensors = static_cast<size_t>(config.num_layers) * per_layer;
  require(layer_tensors == expected_layer_tensors,
          "found " + std::to_string(layer_tensors) + " model.layers.* tensors, expected " +
              std::to_string(expected_layer_tensors) + " (" + std::to_string(per_layer) +
              " per layer x " + std::to_string(config.num_layers) + ")");
  // The nvfp4 build's embedding brings a weight_scale and a comfy_quant with it.
  const size_t expected_other = nvfp4 ? 3u : 1u;
  require(other_tensors == expected_other,
          "found " + std::to_string(other_tensors) +
              " tensors outside model.layers.* and visual.*, expected " +
              std::to_string(expected_other));
  (void)visual_tensors;  // present in the file, never loaded (spec section 8.4)

  for (int layer = 0; layer < config.num_layers; ++layer) {
    const std::string prefix = layer_prefix(layer);
    for (int i = 0; i < kLayerTensorCount; ++i) {
      const TensorSpec spec = layer_tensor_spec(config, static_cast<LayerTensor>(i));
      if (!spec.present()) {
        // Positively assert absence. A pre_quant_scale appearing in a file we
        // decided was the unrotated-free int8 build would mean the smoothing is
        // somewhere this port is not applying it, which is silent.
        continue;
      }
      check_tensor(checkpoint, prefix + spec.suffix, spec.dtype, spec.dim0, spec.dim1);
    }
    if (!nvfp4) {
      for (const char* linear : {"self_attn.o_proj", "mlp.down_proj"}) {
        const std::string name = prefix + linear + ".pre_quant_scale";
        require(checkpoint.find(name) == nullptr,
                name + " is present in a checkpoint that declares int8_tensorwise. This port "
                       "applies no activation scaling on that path, so the smoothing would be "
                       "silently dropped");
      }
    }
    for (const char* linear : kQuantSuffixes) {
      const std::string name = prefix + linear + ".comfy_quant";
      const TensorView* view = checkpoint.find(name);
      require(view != nullptr, name + " is missing");
      const std::string payload = squeeze(view->data, view->nbytes);
      if (nvfp4) {
        require(payload.find("\"format\":\"nvfp4\"") != std::string::npos,
                name + " declares an unexpected format: " + payload);
        // Every quantised linear of this build is checkpoint-declared full
        // precision, so none of them may ever take a native fp4 GEMM. This is
        // the file saying so; nothing may infer it from which scales exist.
        require(payload.find("\"full_precision_matrix_mult\":true") != std::string::npos,
                name + " does not declare full_precision_matrix_mult; this port dequantises every "
                       "text-encoder linear and would be ignoring the file's own instruction: " +
                    payload);
        require(payload.find("\"convrot\":true") == std::string::npos,
                name + " declares convrot on an nvfp4 tensor: " + payload);
      } else {
        // A wrong Hadamard or a skipped activation rotation is silent — it
        // produces well-scaled noise (spec section 9, items 1 and 2) — so the
        // one thing the file does tell us about the rotation is checked.
        require(payload.find("\"convrot\":true") != std::string::npos,
                name + " does not declare convrot; this port only implements the rotated form");
        require(payload.find("\"convrot_groupsize\":256") != std::string::npos,
                name + " declares a ConvRot group size other than 256: " + payload);
        require(payload.find("\"format\":\"int8_tensorwise\"") != std::string::npos,
                name + " declares an unexpected format: " + payload);
      }
    }
  }
}

void pack_layer(const SafeTensors& checkpoint, const EncoderConfig& config, int layer,
                const LayerLayout& layout, uint8_t* dst) {
  require(layer >= 0 && layer < config.num_layers,
          "pack_layer: layer " + std::to_string(layer) + " out of range");
  const std::string prefix = layer_prefix(layer);
  for (int i = 0; i < kLayerTensorCount; ++i) {
    const TensorSpec spec = layer_tensor_spec(config, static_cast<LayerTensor>(i));
    if (!spec.present()) continue;
    const std::string name = prefix + spec.suffix;
    const TensorView& view = checkpoint.at(name);
    require(view.nbytes == layout.bytes[i],
            name + " is " + std::to_string(view.nbytes) + " bytes, layout expects " +
                std::to_string(layout.bytes[i]));
    std::memcpy(dst + layout.offset[i], view.data, view.nbytes);
  }
}

void upload_layer_direct(const SafeTensors& checkpoint, const EncoderConfig& config, int layer,
                         const LayerLayout& layout, uint8_t* dst, void* stream) {
  require(layer >= 0 && layer < config.num_layers,
          "upload_layer_direct: layer " + std::to_string(layer) + " out of range");
  const std::string prefix = layer_prefix(layer);
  for (int i = 0; i < kLayerTensorCount; ++i) {
    const TensorSpec spec = layer_tensor_spec(config, static_cast<LayerTensor>(i));
    if (!spec.present()) continue;
    const std::string name = prefix + spec.suffix;
    const TensorView& view = checkpoint.at(name);
    require(view.nbytes == layout.bytes[i],
            name + " is " + std::to_string(view.nbytes) + " bytes, layout expects " +
                std::to_string(layout.bytes[i]));
    const cudaError_t status =
        cudaMemcpyAsync(dst + layout.offset[i], view.data, view.nbytes, cudaMemcpyHostToDevice,
                        static_cast<cudaStream_t>(stream));
    if (status != cudaSuccess) {
      throw std::runtime_error("text encoder: uploading " + name + " failed: " +
                               cudaGetErrorString(status));
    }
  }
}

std::vector<float> rope_inv_freq(int head_dim, float theta) {
  require(head_dim > 0 && head_dim % 2 == 0, "rope_inv_freq: head_dim must be positive and even");
  require(theta > 0.0f, "rope_inv_freq: theta must be positive");
  const int half = head_dim / 2;
  std::vector<float> inv(static_cast<size_t>(half));
  for (int j = 0; j < half; ++j) {
    // The reference evaluates `base ** (arange(0, dim, 2) / dim)` in fp32.
    // Doing it in fp64 and rounding differs by at most 1.19e-7 relative, far
    // below the bf16 activation noise, and is marginally more accurate
    // (spec section 6.4).
    const double exponent = -static_cast<double>(2 * j) / static_cast<double>(head_dim);
    inv[static_cast<size_t>(j)] =
        static_cast<float>(std::pow(static_cast<double>(theta), exponent));
  }
  return inv;
}

void build_rope_tables(int num_tokens, const std::vector<float>& inv_freq, std::vector<float>& cos,
                       std::vector<float>& sin) {
  require(num_tokens > 0, "build_rope_tables: num_tokens must be positive");
  require(!inv_freq.empty(), "build_rope_tables: inv_freq is empty");
  const int half = static_cast<int>(inv_freq.size());
  const int head_dim = half * 2;
  const size_t n = static_cast<size_t>(num_tokens) * head_dim;
  cos.assign(n, 0.0f);
  sin.assign(n, 0.0f);

  for (int s = 0; s < num_tokens; ++s) {
    const float pos = static_cast<float>(s);
    float* cos_row = cos.data() + static_cast<size_t>(s) * head_dim;
    float* sin_row = sin.data() + static_cast<size_t>(s) * head_dim;
    for (int j = 0; j < half; ++j) {
      // fp32 throughout: the reference computes the angles inside an explicit
      // `maybe_autocast(enabled=False)` block, and `(float)s` is exact for any
      // reachable L (spec section 6.1).
      const float angle = pos * inv_freq[static_cast<size_t>(j)];
      const float c = std::cos(angle);
      const float sn = std::sin(angle);
      // The half period is duplicated, not interleaved. That duplication is
      // what makes the GPT-NeoX pairing of j with j+half correct, and it is
      // the reference's own `cat((freqs, freqs), -1)` (spec section 2.4).
      cos_row[j] = c;
      cos_row[j + half] = c;
      sin_row[j] = sn;
      sin_row[j + half] = sn;
    }
  }
}

void gather_embedding_rows(const TensorView& embed, const TensorView* weight_scale,
                           const std::vector<int32_t>& ids, std::vector<uint16_t>& out) {
  require(embed.dtype == DType::kBF16 || embed.dtype == DType::kI8,
          "embedding table has dtype " + std::string(dtype_name(embed.dtype)) +
              ", expected BF16 or I8");
  require(embed.shape.size() == 2, "embedding table is not 2-D");
  const int64_t vocab = embed.shape[0];
  const int64_t hidden = embed.shape[1];
  const bool quantised = embed.dtype == DType::kI8;
  require(quantised == (weight_scale != nullptr),
          quantised ? "an I8 embedding table needs its per-row weight_scale"
                    : "a BF16 embedding table takes no weight_scale");

  const float* scale = nullptr;
  if (quantised) {
    require(weight_scale->dtype == DType::kF32,
            "embedding weight_scale has dtype " + std::string(dtype_name(weight_scale->dtype)) +
                ", expected F32");
    require(weight_scale->numel() == vocab,
            "embedding weight_scale has " + std::to_string(weight_scale->numel()) +
                " elements, expected one per row of " + std::to_string(vocab));
    scale = static_cast<const float*>(weight_scale->data);
  }

  out.assign(ids.size() * static_cast<size_t>(hidden), 0);

  for (size_t i = 0; i < ids.size(); ++i) {
    const int32_t id = ids[i];
    require(id >= 0 && id < vocab, "token id " + std::to_string(id) + " at position " +
                                       std::to_string(i) + " is outside the vocabulary of " +
                                       std::to_string(vocab));
    uint16_t* dst = out.data() + i * static_cast<size_t>(hidden);
    const size_t row = static_cast<size_t>(id) * static_cast<size_t>(hidden);
    if (!quantised) {
      std::memcpy(dst, static_cast<const uint16_t*>(embed.data) + row,
                  static_cast<size_t>(hidden) * sizeof(uint16_t));
      continue;
    }
    // The scale multiplies. Verified against the other build's bf16 table, where
    // dividing is wrong by seven orders of magnitude — but note that both
    // directions are finite and neither changes the shape, so nothing but the
    // comparison would have told us.
    const int8_t* src = static_cast<const int8_t*>(embed.data) + row;
    const float s = scale[static_cast<size_t>(id)];
    for (int64_t j = 0; j < hidden; ++j) {
      dst[j] = f32_to_bf16(static_cast<float>(src[j]) * s);
    }
  }
}

}  // namespace vidfab::text
