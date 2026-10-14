#include "detail/veda_fixture.h"
#include "slopfab/cuda/veda_attention.cuh"
#include "slopfab/cuda/device.h"

SLOPFAB_TEST(veda_cuda_matches_reference) {
  using namespace slopfab;
  cuda::Stream stream;
  for (int dim : {64, 128})
    for (const char* budget : {"2.3", "0%"}) {
      test::VedaFixture fixture(dim, budget, "cuda", dim == 128);
      const size_t n = fixture.q.size();
      cuda::DeviceBuffer<__nv_bfloat16> q(n), k(n), v(n), out(n);
      q.copy_from_host(reinterpret_cast<const __nv_bfloat16*>(fixture.q.data()), n, stream.get());
      k.copy_from_host(reinterpret_cast<const __nv_bfloat16*>(fixture.k.data()), n, stream.get());
      v.copy_from_host(reinterpret_cast<const __nv_bfloat16*>(fixture.v.data()), n, stream.get());
      cuda::VedaAttention attention(fixture.request);
      attention.prepare(stream.get());
      CHECK(attention.device_bytes() > 0);
      for (int layer : {0, 1}) {
        attention.forward(stream.get(), layer, q.get(), k.get(), v.get(), out.get());
        std::vector<__nv_bfloat16> got(n);
        out.copy_to_host(got.data(), n, stream.get());
        SLOPFAB_CUDA_CHECK(cudaStreamSynchronize(stream.get()));
        std::vector<float> values(n);
        for (size_t i = 0; i < n; ++i)
          values[i] = __bfloat162float(got[i]);
        CHECK_CLOSE(fixture.reference(layer), values, .004, "CUDA VEDA vs FP64 reference");
      }
    }
}
