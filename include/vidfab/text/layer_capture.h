#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace vidfab::text {

#pragma pack(push, 1)
struct QwenLayerCaptureHeader {
  char magic[8] = {'V','F','Q','W','E','N','L','1'};
  uint32_t version = 1;
  uint32_t sequence = 0;
  uint32_t hidden = 0;
  uint32_t query_heads = 0;
  uint32_t kv_heads = 0;
  uint32_t head_dim = 0;
  uint32_t intermediate = 0;
  std::array<uint8_t, 32> checkpoint_sha256{};
  std::array<uint8_t, 32> tokenizer_sha256{};
  uint64_t input_fnv64 = 0;
  uint64_t rope_fnv64 = 0;
  std::array<uint64_t, 11> boundary_fnv64{};
};
#pragma pack(pop)

static_assert(sizeof(QwenLayerCaptureHeader) == 204,
              "Qwen capture header is a durable packed ABI");

struct QwenLayerCapture {
  QwenLayerCaptureHeader header;
  std::vector<int32_t> token_ids;
  std::vector<uint16_t> input_bf16;
  std::vector<float> cosine;
  std::vector<float> sine;
};

QwenLayerCapture read_qwen_layer_capture(const std::string& path);
void write_qwen_layer_capture(const std::string& path,
                              const QwenLayerCapture& capture);

}  // namespace vidfab::text
