#include "harness.h"

#include <cstdint>
#include <limits>
#include <stdexcept>

#include "slopfab/device_tensor.h"

SLOPFAB_TEST(device_tensor_layout_contract) {
  const uint64_t shape[] = {2, 3, 5};
  const slopfab::TensorLayout layout = slopfab::TensorLayout::contiguous(shape, 3);
  CHECK(layout.rank == 3);
  CHECK(layout.elements() == 30);
  CHECK(layout.bytes(slopfab::ScalarType::kFloat32) == 120);
  CHECK(layout.bytes(slopfab::ScalarType::kUInt8) == 30);
  CHECK(layout.bytes(slopfab::ScalarType::kInt8) == 30);
  CHECK(layout.storage_bytes(slopfab::ScalarType::kFloat32) == 120);
  CHECK(layout.stride[0] == 15);
  CHECK(layout.stride[1] == 5);
  CHECK(layout.stride[2] == 1);
  CHECK(layout.is_contiguous());

  const uint64_t huge[] = {std::numeric_limits<uint64_t>::max(), 2};
  bool overflow_rejected = false;
  try {
    (void)slopfab::TensorLayout::contiguous(huge, 2);
  } catch (const std::overflow_error&) {
    overflow_rejected = true;
  }
  CHECK(overflow_rejected);

  slopfab::DeviceTensorView view;
  view.backend = slopfab::DeviceBackend::kVulkan;
  view.context = 17;
  view.resource = 23;
  view.byte_offset = 64;
  view.byte_size = 256;
  const uint64_t sub_extent = 16;
  const auto sub_layout = slopfab::TensorLayout::contiguous(&sub_extent, 1);
  const slopfab::DeviceTensorView slice = view.slice(32, sub_layout, 16);
  CHECK(slice.context == view.context);
  CHECK(slice.resource == view.resource);
  CHECK(slice.byte_offset == 96);
  CHECK(slice.byte_size == 64);

  bool alignment_rejected = false;
  try {
    (void)view.slice(1, sub_layout, 4);
  } catch (const std::invalid_argument&) {
    alignment_rejected = true;
  }
  CHECK(alignment_rejected);
  for (uint64_t offset = 1; offset < 4; ++offset) {
    bool scalar_alignment_rejected = false;
    try {
      (void)view.slice(offset, sub_layout);
    } catch (const std::invalid_argument&) {
      scalar_alignment_rejected = true;
    }
    CHECK(scalar_alignment_rejected);
  }

  slopfab::DeviceTensorView corrupt = view;
  corrupt.byte_offset = std::numeric_limits<uint64_t>::max() - 31;
  bool end_overflow_rejected = false;
  try {
    (void)corrupt.slice(0, sub_layout, 1);
  } catch (const std::overflow_error&) {
    end_overflow_rejected = true;
  }
  CHECK(end_overflow_rejected);

  slopfab::TensorLayout strided = sub_layout;
  strided.stride[0] = 2;
  const auto strided_slice = view.slice(0, strided, 4);
  CHECK(strided_slice.byte_size == 124);
  CHECK(strided_slice.layout.storage_bytes(slopfab::ScalarType::kFloat32) == 124);
}
