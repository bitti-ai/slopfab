#include "harness.h"
#include "slopfab/dit/offload.h"

SLOPFAB_TEST(block_offload_budget_includes_transfer_buffers) {
  using slopfab::dit::plan_block_offload;
  auto p = plan_block_offload({100, 100, 100, 100, 100}, 50, 550);
  CHECK(p.count == 0);
  CHECK(p.device_bytes == 550);
  p = plan_block_offload({100, 100, 100, 100, 100}, 50, 450);
  CHECK(p.first == 2);
  CHECK(p.count == 3);
  CHECK(p.device_bytes == 450);
  CHECK(p.host_bytes == 300);
  p = plan_block_offload({20, 60, 30, 100}, 10, 250, 3);
  CHECK(p.first == 1);
  CHECK(p.slot_bytes == 100);
  CHECK(p.device_bytes == 230);
  CHECK(slopfab::test::throws([] {
    plan_block_offload({100, 100}, 50, 249);
  }));
  CHECK(slopfab::test::throws([] {
    plan_block_offload({100}, 0, 1000, 2);
  }));
  p = plan_block_offload({}, 50, 50);
  CHECK(p.count == 0);
  CHECK(p.device_bytes == 50);
}
