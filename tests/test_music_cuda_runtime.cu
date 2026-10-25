#include "harness.h"
#include "../src/moss/runtime.cuh"
#include "slopfab/safetensors_write.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <numeric>
#include <thread>

namespace {
using namespace slopfab;
using namespace slopfab::moss;

bool have_cuda() {
  if (cuda::device_count() > 0)
    return true;
  SKIP_UNSUPPORTED_HARDWARE("CUDA required");
  return false;
}

__global__ void fill_values(uint32_t* p, size_t count) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    p[i] = uint32_t(i);
}

__global__ void increment_values(uint32_t* p, size_t count) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    ++p[i];
}
}

SLOPFAB_TEST_CATEGORY(music_cuda_pool_reuse_and_scope_lifetime, "synthetic") {
  if (!have_cuda())
    return;
  CHECK(!temporary_buffer_pool(TemporaryBufferKind::kActivation));
  TemporaryBuffer<uint32_t> survivor;
  std::weak_ptr<cuda::ReferenceBufferPool> weak;
  constexpr size_t count = 1 << 18;
  {
    TemporaryBufferScope scope;
    weak = scope.activations();
    uint32_t* pointer;
    {
      TemporaryBuffer<uint32_t> first(count);
      pointer = first.get();
      fill_values<<<int((count + 255) / 256), 256>>>(pointer, count);
      SLOPFAB_CUDA_CHECK(cudaGetLastError());
    }
    survivor = TemporaryBuffer<uint32_t>(count);
    CHECK(survivor.get() == pointer);
    CHECK(scope.activations()->allocations() == 1);
    increment_values<<<int((count + 255) / 256), 256>>>(survivor.get(), count);
    SLOPFAB_CUDA_CHECK(cudaGetLastError());
    TemporaryBuffer<uint32_t> concurrent(count);
    CHECK(concurrent.get() != survivor.get());
    CHECK(scope.activations()->allocations() == 2);
    TemporaryBuffer<uint32_t> scratch(count, TemporaryBufferKind::kScratch);
    CHECK(scratch.get() != survivor.get());
    CHECK(scope.temporaries()->allocations() == 1);
    bool other_thread_has_pool = true;
    std::thread thread([&] {
      other_thread_has_pool = bool(temporary_buffer_pool(TemporaryBufferKind::kActivation));
    });
    thread.join();
    CHECK(!other_thread_has_pool);
    try {
      TemporaryBufferScope nested;
      CHECK(temporary_buffer_pool(TemporaryBufferKind::kActivation) != scope.activations());
      throw std::runtime_error("exercise unwind");
    } catch (const std::runtime_error&) {
    }
    CHECK(temporary_buffer_pool(TemporaryBufferKind::kActivation) == scope.activations());
  }
  CHECK(!temporary_buffer_pool(TemporaryBufferKind::kActivation));
  CHECK(!weak.expired());
  std::vector<uint32_t> actual(count);
  SLOPFAB_CUDA_CHECK(
      cudaMemcpy(actual.data(), survivor.get(), survivor.nbytes(), cudaMemcpyDeviceToHost));
  std::vector<uint32_t> expected(count);
  std::iota(expected.begin(), expected.end(), uint32_t(1));
  CHECK(actual == expected);
  TemporaryBuffer<uint32_t> moved(std::move(survivor));
  CHECK(survivor.empty());
  CHECK(moved.size() == count);
  {
    TemporaryBufferScope other;
    TemporaryBuffer<uint32_t> target(count);
    target = std::move(moved);
    CHECK(moved.empty());
    CHECK(!weak.expired());
  }
  CHECK(weak.expired());
}

SLOPFAB_TEST_CATEGORY(music_cuda_pool_preserves_projection_and_attention, "synthetic") {
  if (!have_cuda())
    return;
  auto path = std::filesystem::temp_directory_path() /
              ("slopfab_music_cuda_pool_" +
               std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
               ".safetensors");

  struct Cleanup {
    std::filesystem::path path;

    ~Cleanup() {
      std::error_code error;
      std::filesystem::remove(path, error);
    }
  } cleanup{path};

  const int heads = 4, kv_heads = 2, dim = 128, rows = 3, keys = 317;
  auto values = [](size_t size, int modulus) {
    std::vector<float> v(size);
    for (size_t i = 0; i < size; ++i)
      v[i] = float(int(i % modulus) - modulus / 2) / 32;
    return v;
  };
  write_safetensors(
      path.string(),
      {{"proj.weight", {heads * dim, heads * dim}, values(size_t(heads * dim) * heads * dim, 13)},
       {"proj.bias", {heads * dim}, values(heads * dim, 7)}});
  Runtime rt(path.string());
  rt.cache_dense_weights = true;
  auto q = from_host(values(rows * heads * dim, 17), rows, heads * dim);
  auto k = from_host(values(keys * kv_heads * dim, 11), keys, kv_heads * dim);
  auto v = from_host(values(keys * kv_heads * dim, 19), keys, kv_heads * dim);
  const auto projection = to_host(rt.project(q, "proj"));
  const auto bf_attention = to_host(rt.attend(q, k, v, heads, kv_heads));
  const auto fp_attention = to_host(rt.attend(q, k, v, heads, kv_heads, false, true));
  Tensor survivor;
  std::weak_ptr<cuda::ReferenceBufferPool> weak;
  {
    TemporaryBufferScope scope;
    weak = scope.activations();
    // Repeat with asynchronous intermediates to expose premature reuse.
    for (int repeat = 0; repeat < 4; ++repeat) {
      CHECK(to_host(rt.project(q, "proj")) == projection);
      CHECK(to_host(rt.attend(q, k, v, heads, kv_heads)) == bf_attention);
      CHECK(to_host(rt.attend(q, k, v, heads, kv_heads, false, true)) == fp_attention);
      survivor = activate(slice(rt.project(q, "proj"), 0, dim));
    }
  }
  CHECK(!weak.expired());
  CHECK(to_host(survivor) == to_host(activate(slice(rt.project(q, "proj"), 0, dim))));
  survivor = Tensor();
  CHECK(weak.expired());
}

SLOPFAB_TEST_CATEGORY(music_cuda_batched_decode_attention, "synthetic") {
  if (!have_cuda())
    return;
  auto path = std::filesystem::temp_directory_path() /
              ("slopfab_music_decode_" +
               std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
               ".safetensors");

  struct Cleanup {
    std::filesystem::path path;

    ~Cleanup() {
      std::error_code error;
      std::filesystem::remove(path, error);
    }
  } cleanup{path};

  write_safetensors(path.string(), {{"unused", {1}, {0}}});
  Runtime rt(path.string());
  constexpr int dim = 128, kv_heads = 8;
  const auto bf = [](float x) {
    return __bfloat162float(__float2bfloat16(x));
  };
  for (int heads : {16, 32})
    for (int keys : {17, 317, 1025}) {
      auto qvalues = test::make_data(heads * dim, 17, .5f);
      auto kvalues = test::make_data((keys + 7) * kv_heads * dim, 29, .5f);
      auto vvalues = test::make_data((keys + 7) * kv_heads * dim, 43, .5f);
      for (auto* values : {&qvalues, &kvalues, &vvalues})
        for (auto& value : *values)
          value = bf(value);
      // Poison unused capacity: neither path may attend past the live prefix.
      std::fill(kvalues.begin() + size_t(keys) * kv_heads * dim, kvalues.end(), 100.f);
      std::fill(vvalues.begin() + size_t(keys) * kv_heads * dim, vvalues.end(), 100.f);
      auto q = from_host(qvalues, 1, heads * dim);
      auto k = from_host(kvalues, keys + 7, kv_heads * dim);
      auto v = from_host(vvalues, keys + 7, kv_heads * dim);
      k.rows = v.rows = keys;
      const auto original = to_host(rt.attend(q, k, v, heads, kv_heads));
      std::vector<float> expected(heads * dim);
      for (int head = 0; head < heads; ++head) {
        int kh = head / (heads / kv_heads);
        std::vector<double> probabilities(keys);
        double sum = 0;
        for (int row = 0; row < keys; ++row) {
          double score = 0;
          for (int c = 0; c < dim; ++c)
            score +=
                double(qvalues[head * dim + c]) * kvalues[(size_t(row) * kv_heads + kh) * dim + c];
          sum += (probabilities[row] = std::exp(score / std::sqrt(double(dim))));
        }
        for (auto& p : probabilities)
          p = bf(float(p / sum));
        for (int c = 0; c < dim; ++c) {
          double output = 0;
          for (int row = 0; row < keys; ++row)
            output += probabilities[row] * vvalues[(size_t(row) * kv_heads + kh) * dim + c];
          expected[head * dim + c] = bf(float(output));
        }
      }
      TemporaryBufferScope scope;
      std::vector<float> previous;
      for (int repeat = 0; repeat < 4; ++repeat) {
        auto actual = to_host(rt.attend(q, k, v, heads, kv_heads));
        if (repeat)
          CHECK(actual == previous);
        else {
          double error = 0, energy = 0;
          float maximum = 0;
          int different = 0;
          for (size_t i = 0; i < actual.size(); ++i) {
            double delta = double(actual[i]) - original[i];
            error += delta * delta;
            energy += double(original[i]) * original[i];
            maximum = std::max(maximum, float(std::abs(delta)));
            different += actual[i] != original[i];
          }
          std::printf("decode H=%d KV=%d K=%d: different=%d max_abs=%.8g relative_rms=%.8g\n",
                      heads, kv_heads, keys, different, maximum, std::sqrt(error / energy));
          CHECK_CLOSE_REL(original, actual, .001, .01, "batched vs per-head decode");
          CHECK_CLOSE_REL(expected, actual, .001, .01, "batched vs CPU decode");
          CHECK(std::sqrt(error / energy) < .005);
        }
        previous = std::move(actual);
      }
    }
}
