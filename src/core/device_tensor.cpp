#include "vidfab/device_tensor.h"

#include <limits>

namespace vidfab {
namespace {

uint64_t scalar_bytes(ScalarType type) {
  switch (type) {
    case ScalarType::kFloat32:
      return 4;
  }
  throw std::invalid_argument("tensor: unsupported scalar type");
}

uint64_t checked_multiply(uint64_t a, uint64_t b) {
  if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
    throw std::overflow_error("tensor: shape byte size overflow");
  }
  return a * b;
}

}  // namespace

TensorLayout TensorLayout::contiguous(const uint64_t* extents, uint32_t count) {
  if (count > 6 || (count != 0 && extents == nullptr)) {
    throw std::invalid_argument("tensor: rank must be in [0,6]");
  }
  TensorLayout result;
  result.rank = count;
  uint64_t stride = 1;
  for (uint32_t axis = count; axis-- > 0;) {
    if (extents[axis] == 0) throw std::invalid_argument("tensor: extents must be positive");
    result.extent[axis] = extents[axis];
    result.stride[axis] = stride;
    stride = checked_multiply(stride, extents[axis]);
  }
  return result;
}

uint64_t TensorLayout::elements() const {
  if (rank > 6) throw std::invalid_argument("tensor: invalid rank");
  uint64_t count = 1;
  for (uint32_t axis = 0; axis < rank; ++axis) {
    if (extent[axis] == 0) throw std::invalid_argument("tensor: extents must be positive");
    count = checked_multiply(count, extent[axis]);
  }
  return count;
}

uint64_t TensorLayout::bytes(ScalarType type) const {
  return checked_multiply(elements(), scalar_bytes(type));
}

bool TensorLayout::is_contiguous() const {
  if (rank > 6) return false;
  uint64_t expected = 1;
  for (uint32_t axis = rank; axis-- > 0;) {
    if (extent[axis] == 0 || stride[axis] != expected) return false;
    if (extent[axis] != 0 && expected > std::numeric_limits<uint64_t>::max() / extent[axis]) {
      return false;
    }
    expected *= extent[axis];
  }
  return true;
}

}  // namespace vidfab
