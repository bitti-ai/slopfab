#pragma once

namespace slopfab::test {

// Test-only thread-local guard. The executable's global allocation operators
// throw while this object is alive, allowing a production record path to prove
// that all host-side command metadata was reserved during construction.
class HostAllocationGuard {
public:
  HostAllocationGuard() noexcept;
  ~HostAllocationGuard();
  HostAllocationGuard(const HostAllocationGuard&) = delete;
  HostAllocationGuard& operator=(const HostAllocationGuard&) = delete;
};

} // namespace slopfab::test
