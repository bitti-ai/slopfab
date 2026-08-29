#include "vidfab/text/layer_capture.h"

#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace vidfab::text {
namespace {

uint64_t checked_product(uint64_t left, uint64_t right) {
  if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left)
    throw std::runtime_error("Qwen capture: size overflow");
  return left * right;
}

void validate(const QwenLayerCapture& capture) {
  const auto& h = capture.header;
  const char expected[8] = {'V','F','Q','W','E','N','L','1'};
  if (std::memcmp(h.magic, expected, sizeof(expected)) != 0 || h.version != 1 ||
      h.sequence == 0 || h.hidden == 0 || h.query_heads == 0 ||
      h.kv_heads == 0 || h.head_dim == 0 || h.intermediate == 0 ||
      capture.token_ids.size() != h.sequence ||
      capture.input_bf16.size() != checked_product(h.sequence, h.hidden) ||
      capture.cosine.size() != checked_product(h.sequence, h.head_dim) ||
      capture.sine.size() != capture.cosine.size()) {
    throw std::runtime_error("Qwen capture: invalid header or payload shape");
  }
}

template <typename T>
void read_vector(std::ifstream& input, std::vector<T>& values, size_t count) {
  values.resize(count);
  input.read(reinterpret_cast<char*>(values.data()),
             static_cast<std::streamsize>(values.size() * sizeof(T)));
  if (!input) throw std::runtime_error("Qwen capture: truncated payload");
}

template <typename T>
void write_vector(std::ofstream& output, const std::vector<T>& values) {
  output.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(T)));
}

}  // namespace

QwenLayerCapture read_qwen_layer_capture(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("Qwen capture: cannot open " + path);
  QwenLayerCapture result;
  input.read(reinterpret_cast<char*>(&result.header), sizeof(result.header));
  if (!input) throw std::runtime_error("Qwen capture: truncated header");
  const uint64_t token_count = result.header.sequence;
  const uint64_t input_count = checked_product(
      result.header.sequence, result.header.hidden);
  const uint64_t rope_count = checked_product(
      result.header.sequence, result.header.head_dim);
  if (token_count > SIZE_MAX || input_count > SIZE_MAX || rope_count > SIZE_MAX)
    throw std::runtime_error("Qwen capture: payload exceeds host address space");
  read_vector(input, result.token_ids, static_cast<size_t>(token_count));
  read_vector(input, result.input_bf16, static_cast<size_t>(input_count));
  read_vector(input, result.cosine, static_cast<size_t>(rope_count));
  read_vector(input, result.sine, static_cast<size_t>(rope_count));
  char trailing = 0;
  if (input.read(&trailing, 1))
    throw std::runtime_error("Qwen capture: trailing payload bytes");
  validate(result);
  return result;
}

void write_qwen_layer_capture(const std::string& path,
                              const QwenLayerCapture& capture) {
  validate(capture);
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) throw std::runtime_error("Qwen capture: cannot write " + path);
  output.write(reinterpret_cast<const char*>(&capture.header),
               sizeof(capture.header));
  write_vector(output, capture.token_ids);
  write_vector(output, capture.input_bf16);
  write_vector(output, capture.cosine);
  write_vector(output, capture.sine);
  if (!output) throw std::runtime_error("Qwen capture: write failed");
}

}  // namespace vidfab::text
