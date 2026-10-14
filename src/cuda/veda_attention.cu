#include "slopfab/cuda/veda_attention.cuh"
#include "slopfab/cuda/device.h"
#include "slopfab/cuda/cublas_dispatch.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <math_constants.h>

namespace slopfab::cuda {
namespace {
void blas_check(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("VEDA: cuBLAS error " + std::to_string(int(status)));
}

__global__ void pool(const __nv_bfloat16* q, const __nv_bfloat16* k, const int32_t* rows,
                     const int32_t* sizes, float* features, int n, int heads, int dim, int head) {
  const int tile = blockIdx.x, d = threadIdx.x;
  if (d >= dim)
    return;
  float sq = 0, sk = 0, maxq = -CUDART_INF_F, maxk = -CUDART_INF_F;
  float minq = CUDART_INF_F, mink = CUDART_INF_F;
  const int count = sizes[tile];
  for (int r = 0; r < count; ++r) {
    const size_t index = (size_t(rows[tile * 128 + r]) * heads + head) * dim + d;
    const float a = __bfloat162float(q[index]), b = __bfloat162float(k[index]);
    sq += a;
    sk += b;
    maxq = fmaxf(maxq, a);
    maxk = fmaxf(maxk, b);
    minq = fminf(minq, a);
    mink = fminf(mink, b);
  }
  const size_t dst = size_t(tile) * 3 * dim + d, stride = size_t(n) * 3 * dim;
  features[dst] = sq / count;
  features[dst + dim] = maxq;
  features[dst + 2 * dim] = minq;
  features[stride + dst] = sk / count;
  features[stride + dst + dim] = maxk;
  features[stride + dst + 2 * dim] = mink;
}

__global__ void widen(const __nv_bfloat16* q, const __nv_bfloat16* k, float* out, int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    out[i] = __bfloat162float(q[i]);
    out[count + i] = __bfloat162float(k[i]);
  }
}

__global__ void residual(const float* features, float* projected, int n, int dim) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n * dim) {
    const int src = i / dim * 3 * dim + i % dim;
    projected[i] += features[src];
    projected[n * dim + i] += features[n * 3 * dim + src];
  }
}

__global__ void route(const float* scores, uint8_t* mask, const int32_t* keeps, int n, int video,
                      int padded) {
  const int row = blockIdx.x, tid = threadIdx.x;
  uint8_t* dest = mask + size_t(row) * n;
  for (int key = tid; key < n; key += blockDim.x)
    dest[key] = row >= video || key >= video;
  if (row >= video)
    return;
  extern __shared__ char storage[];
  auto* values = reinterpret_cast<float*>(storage);
  auto* ids = reinterpret_cast<int*>(values + padded);
  for (int i = tid; i < padded; i += blockDim.x) {
    values[i] = i == row    ? CUDART_INF_F
                : i < video ? scores[size_t(row) * video + i]
                            : -CUDART_INF_F;
    ids[i] = i;
  }
  __syncthreads();
  for (int width = 2; width <= padded; width *= 2)
    for (int step = width / 2; step; step /= 2) {
      for (int i = tid; i < padded; i += blockDim.x) {
        const int j = i ^ step;
        if (j <= i)
          continue;
        const bool before = values[i] > values[j] || (values[i] == values[j] && ids[i] < ids[j]);
        if (before != ((i & width) == 0)) {
          const float value = values[i];
          values[i] = values[j];
          values[j] = value;
          const int id = ids[i];
          ids[i] = ids[j];
          ids[j] = id;
        }
      }
      __syncthreads();
    }
  for (int i = tid; i < keeps[row]; i += blockDim.x)
    dest[ids[i]] = 1;
}
} // namespace

struct VedaAttention::Impl {
  std::shared_ptr<dit::VedaRequest> request;

  struct Maps {
    DeviceBuffer<int32_t> rows, sizes, keeps;
  };

  std::vector<Maps> maps;
  std::vector<std::vector<uint16_t>> host_q, host_k;
  DeviceBuffer<__nv_bfloat16> proj_q, proj_k;
  DeviceBuffer<float> weights, features, projected, scores;
  DeviceBuffer<uint8_t> mask;
  cublasHandle_t blas = nullptr;
  int max_tiles = 0, max_video = 0;
  bool ready = false;

  explicit Impl(std::shared_ptr<dit::VedaRequest> r) : request(std::move(r)) {
    if (!request)
      throw std::invalid_argument("VEDA: missing request");
    for (const auto& t : request->tilings) {
      max_tiles = std::max(max_tiles, int(t.sizes.size()));
      max_video = std::max(max_video, t.video_tiles);
    }
  }

  ~Impl() {
    if (blas)
      cublas_destroy(blas);
  }
};

VedaAttention::VedaAttention(std::shared_ptr<dit::VedaRequest> request)
    : impl_(new Impl(std::move(request))) {
}

VedaAttention::~VedaAttention() = default;

size_t VedaAttention::device_bytes() const {
  const auto& s = *impl_;
  const auto& b = s.request->bundle;
  size_t bytes = size_t(2) * b.heads * 3 * b.head_dim * b.head_dim * 2;
  bytes += size_t(6) * b.head_dim * b.head_dim * 4;
  bytes += size_t(8) * s.max_video * b.head_dim * 4 + size_t(s.max_video) * s.max_video * 4;
  bytes += size_t(s.max_tiles) * s.max_tiles;
  for (const auto& t : s.request->tilings)
    bytes += (t.rows.size() + t.sizes.size() + t.video_tiles) * 4;
  return bytes;
}

void VedaAttention::prepare(cudaStream_t stream) {
  auto& s = *impl_;
  if (s.ready)
    return;
  if (current_device_compute_capability() < 80)
    throw std::runtime_error("VEDA requires CUDA SM80 or newer");
  const auto& b = s.request->bundle;
  s.maps.resize(s.request->tilings.size());
  for (size_t i = 0; i < s.maps.size(); ++i) {
    auto& m = s.maps[i];
    const auto& t = s.request->tilings[i];
    m.rows.allocate(t.rows.size());
    m.sizes.allocate(t.sizes.size());
    m.keeps.allocate(t.video_tiles);
    m.rows.copy_from_host(t.rows.data(), t.rows.size(), stream);
    m.sizes.copy_from_host(t.sizes.data(), t.sizes.size(), stream);
    m.keeps.copy_from_host(s.request->keeps[i].data(), t.video_tiles, stream);
  }
  const size_t layer = size_t(b.heads) * 3 * b.head_dim * b.head_dim;
  s.proj_q.allocate(layer);
  s.proj_k.allocate(layer);
  s.weights.allocate(size_t(6) * b.head_dim * b.head_dim);
  s.features.allocate(size_t(6) * s.max_video * b.head_dim);
  s.projected.allocate(size_t(2) * s.max_video * b.head_dim);
  s.scores.allocate(size_t(s.max_video) * s.max_video);
  s.mask.allocate(size_t(s.max_tiles) * s.max_tiles);
  s.host_q.clear();
  s.host_k.clear();
  for (int l = 0; l < b.layers; ++l) {
    s.host_q.push_back(b.projection(l, true));
    s.host_k.push_back(b.projection(l, false));
  }
  blas_check(cublas_create(&s.blas));
  blas_check(cublas_set_math_mode(s.blas, CUBLAS_PEDANTIC_MATH));
  s.ready = true;
}

void VedaAttention::forward(cudaStream_t stream, int layer, const __nv_bfloat16* q,
                            const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* out) {
  auto& s = *impl_;
  const auto& b = s.request->bundle;
  if (!s.ready || layer < 0 || layer >= b.layers || !q || !k || !v || !out || q == k || q == v ||
      k == v || out == q || out == k || out == v)
    throw std::invalid_argument("VEDA: unprepared attention or invalid layer/tensors");
  s.proj_q.copy_from_host(reinterpret_cast<const __nv_bfloat16*>(s.host_q[layer].data()),
                          s.proj_q.size(), stream);
  s.proj_k.copy_from_host(reinterpret_cast<const __nv_bfloat16*>(s.host_k[layer].data()),
                          s.proj_k.size(), stream);
  blas_check(cublas_set_stream(s.blas, stream));
  const int dim = b.head_dim, per_head = 3 * dim * dim;
  const float one = 1, zero = 0, scale = 1 / std::sqrt(float(dim));
  for (int head = 0; head < b.heads; ++head) {
    const int index = s.request->plan.head_shape[layer][head];
    const auto& t = s.request->tilings[index];
    const auto& m = s.maps[index];
    const int n = t.video_tiles, tiles = int(t.sizes.size());
    pool<<<n, 128, 0, stream>>>(q, k, m.rows.get(), m.sizes.get(), s.features.get(), n, b.heads,
                                dim, head);
    widen<<<(per_head + 255) / 256, 256, 0, stream>>>(s.proj_q.get() + size_t(head) * per_head,
                                                      s.proj_k.get() + size_t(head) * per_head,
                                                      s.weights.get(), per_head);
    for (int part = 0; part < 2; ++part)
      blas_check(cublas_sgemm(s.blas, CUBLAS_OP_N, CUBLAS_OP_N, dim, n, 3 * dim, &one,
                              s.weights.get() + part * per_head, dim,
                              s.features.get() + size_t(part) * n * 3 * dim, 3 * dim, &zero,
                              s.projected.get() + size_t(part) * n * dim, dim));
    residual<<<(n * dim + 255) / 256, 256, 0, stream>>>(s.features.get(), s.projected.get(), n,
                                                        dim);
    blas_check(cublas_sgemm(s.blas, CUBLAS_OP_T, CUBLAS_OP_N, n, n, dim, &scale,
                            s.projected.get() + size_t(n) * dim, dim, s.projected.get(), dim, &zero,
                            s.scores.get(), n));
    int padded = 1;
    while (padded < n)
      padded *= 2;
    route<<<tiles, 256, padded * 8, stream>>>(s.scores.get(), s.mask.get(), m.keeps.get(), tiles, n,
                                              padded);
    veda_sparse_forward(stream, q, k, v, out, b.heads, dim, head, m.rows.get(), m.sizes.get(),
                        s.mask.get(), tiles);
  }
  SLOPFAB_CUDA_CHECK(cudaGetLastError());
}
} // namespace slopfab::cuda
