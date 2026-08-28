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

uint64_t TensorLayout::storage_bytes(ScalarType type) const {
  if (rank > 6) throw std::invalid_argument("tensor: invalid rank");
  uint64_t largest = 0;
  for (uint32_t axis = 0; axis < rank; ++axis) {
    if (extent[axis] == 0) throw std::invalid_argument("tensor: extents must be positive");
    const uint64_t contribution = checked_multiply(extent[axis] - 1, stride[axis]);
    if (largest > std::numeric_limits<uint64_t>::max() - contribution) {
      throw std::overflow_error("tensor: strided storage size overflow");
    }
    largest += contribution;
  }
  if (largest == std::numeric_limits<uint64_t>::max()) {
    throw std::overflow_error("tensor: strided storage size overflow");
  }
  return checked_multiply(largest + 1, scalar_bytes(type));
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

DeviceTensorView DeviceTensorView::slice(uint64_t offset,
                                         const TensorLayout& slice_layout,
                                         uint64_t alignment) const {
  if (context == 0 || resource == 0 || alignment == 0 ||
      (alignment & (alignment - 1)) != 0) {
    throw std::invalid_argument("tensor: invalid device view slice");
  }
  const uint64_t bytes = slice_layout.storage_bytes(type);
  if (byte_offset > std::numeric_limits<uint64_t>::max() - offset) {
    throw std::overflow_error("tensor: device view offset overflow");
  }
  if (offset > byte_size || bytes > byte_size - offset) {
    throw std::out_of_range("tensor: device view slice exceeds allocation");
  }
  const uint64_t absolute = byte_offset + offset;
  if (absolute > std::numeric_limits<uint64_t>::max() - bytes) {
    throw std::overflow_error("tensor: device view end overflow");
  }
  if ((absolute & (alignment - 1)) != 0) {
    throw std::invalid_argument("tensor: device view slice is misaligned");
  }
  DeviceTensorView result = *this;
  result.layout = slice_layout;
  result.byte_offset = absolute;
  result.byte_size = bytes;
  return result;
}

}  // namespace vidfab
