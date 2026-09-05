// A float buffer that does not zero itself on resize.
//
// The decoded video is gigabytes at the heavy config, and the de-normalise
// pass that produces it writes every element before anything reads it. A
// std::vector value-initialises, so sizing one costs a full-width memset of
// values nothing ever looks at, on top of the page faults the first real write
// would take anyway. That is pure bandwidth, and at these sizes it is a
// visible fraction of the decode. (The decoder's internal `assembled` buffer
// has the same problem and solves it with a bare new float[], which it can
// because it never leaves the function; this type exists for the output, which
// is public and has to stay a container.)
//
// The allocator below default-initialises instead, which for float means "do
// nothing". It is deliberately narrow: `construct` with arguments still
// forwards normally, so only the no-argument case — exactly the resize path —
// changes behaviour. Reading an element before writing it is undefined, which
// is the trade being made; the use site fills its buffer completely.
#pragma once

#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace slopfab {

template <typename T, typename Base = std::allocator<T>>
struct DefaultInitAllocator : Base {
  using Base::Base;

  template <typename U>
  struct rebind {
    using other =
        DefaultInitAllocator<U,
                             typename std::allocator_traits<Base>::template rebind_alloc<U>>;
  };

  DefaultInitAllocator() = default;
  template <typename U, typename V>
  DefaultInitAllocator(const DefaultInitAllocator<U, V>& other) : Base(other) {}

  // The whole point: `new (p) U` rather than `new (p) U()`.
  template <typename U>
  void construct(U* p) noexcept(std::is_nothrow_default_constructible<U>::value) {
    ::new (static_cast<void*>(p)) U;
  }

  template <typename U, typename... Args>
  void construct(U* p, Args&&... args) {
    std::allocator_traits<Base>::construct(static_cast<Base&>(*this), p,
                                           std::forward<Args>(args)...);
  }
};

// Planar float pixels. Distinct from std::vector<float> only in that resizing
// it does not zero, so it is used for the buffers big enough for that to
// matter and passed through to the writers unchanged.
using PixelBuffer = std::vector<float, DefaultInitAllocator<float>>;

}  // namespace slopfab
