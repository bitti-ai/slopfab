#include "detail/nn_kernels_fixture.h"

SLOPFAB_TEST_CATEGORY(workspace_arena, "synthetic") {
  Workspace ws;
  ws.reserve(4096);
  CHECK(ws.capacity() >= 4096);
  void* a = ws.alloc(100);
  void* b = ws.alloc(100);
  // 256-byte grain, so consecutive carvings are cuBLAS-aligned.
  CHECK(reinterpret_cast<uintptr_t>(b) - reinterpret_cast<uintptr_t>(a) == 256);
  CHECK(reinterpret_cast<uintptr_t>(a) % 256 == 0);
  {
    Workspace::Scope scope(ws);
    ws.alloc(1024);
    CHECK(ws.used() > 512);
  }
  CHECK(ws.used() == 356);
  bool threw = false;
  try {
    ws.alloc(1u << 30);
  } catch (const std::exception&) {
    threw = true;
  }
  CHECK(threw);
  ws.clear();
  CHECK(ws.used() == 0);
}

SLOPFAB_TEST_CATEGORY(workspace_resize_releases_stage_reservation, "synthetic") {
  Workspace ws;
  ws.reserve(1 << 20);
  ws.alloc(8192);
  ws.resize(513);
  CHECK(ws.capacity() == 768);
  CHECK(ws.used() == 0);
  auto* data = ws.alloc_n<float>(128);
  SLOPFAB_CUDA_CHECK(cudaMemset(data, 0, 128 * sizeof(float)));
  std::vector<float> host(128, 1.0f);
  SLOPFAB_CUDA_CHECK(cudaMemcpy(host.data(), data, 128 * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK(host == std::vector<float>(128, 0.0f));
  ws.resize(0);
  CHECK(ws.capacity() == 0);
  CHECK(ws.used() == 0);
}

SLOPFAB_TEST_CATEGORY(workspace_reserve_below_capacity_preserves_carved_pointers, "synthetic") {
  Workspace ws;
  ws.reserve(1 << 20);
  const size_t capacity = ws.capacity();
  CHECK(capacity >= (1u << 20));

  void* a = ws.alloc(4096);
  void* b = ws.alloc(4096);
  const size_t used = ws.used();
  CHECK(a != nullptr && b != nullptr && a != b);
  CHECK(reinterpret_cast<uintptr_t>(a) % 256 == 0);
  CHECK(reinterpret_cast<uintptr_t>(b) % 256 == 0);

  // Exactly the shape of the inner reserve: a smaller request against an arena
  // that is already big enough.
  ws.reserve(capacity / 2);
  CHECK_MSG(ws.capacity() == capacity, "reserve below capacity reallocated: %zu -> %zu", capacity,
            ws.capacity());
  CHECK_MSG(ws.used() == used, "reserve below capacity moved the cursor: %zu -> %zu", used,
            ws.used());
  void* c = ws.alloc(4096);
  CHECK_MSG(c != a && c != b, "reserve below capacity handed back a live pointer");

  // And the case the encoder's sizing exists to prevent, so the test states
  // what "too small a reserve" would actually have done.
  ws.reserve(capacity * 2);
  CHECK(ws.capacity() >= capacity * 2);
  CHECK_MSG(ws.used() == 0, "a growing reserve must reset the cursor, got %zu", ws.used());

  // An over-carve is a loud throw, not a silent overrun -- the other half of
  // why an under-sized reserve cannot corrupt results quietly.
  Workspace tight;
  tight.reserve(1024);
  bool threw = false;
  try {
    tight.alloc(1 << 20);
  } catch (const std::exception&) {
    threw = true;
  }
  CHECK_MSG(threw, "Workspace::alloc past the end must throw");
}
