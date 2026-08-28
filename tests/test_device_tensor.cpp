#include "harness.h"

#include <cstdint>
#include <limits>
#include <stdexcept>

#include "vidfab/device_tensor.h"

VIDFAB_TEST(device_tensor_layout_contract) {
  const uint64_t shape[] = {2, 3, 5};
  const vidfab::TensorLayout layout = vidfab::TensorLayout::contiguous(shape, 3);
  CHECK(layout.rank == 3);
  CHECK(layout.elements() == 30);
  CHECK(layout.bytes(vidfab::ScalarType::kFloat32) == 120);
  CHECK(layout.stride[0] == 15);
  CHECK(layout.stride[1] == 5);
  CHECK(layout.stride[2] == 1);
  CHECK(layout.is_contiguous());

  const uint64_t huge[] = {std::numeric_limits<uint64_t>::max(), 2};
  bool overflow_rejected = false;
  try {
    (void)vidfab::TensorLayout::contiguous(huge, 2);
  } catch (const std::overflow_error&) {
    overflow_rejected = true;
  }
  CHECK(overflow_rejected);
}
