// Experimental aligned exact SM120 mainloop. The production Sol kernel remains
// the fallback. One thread owns one output column and keeps all 64 row outputs
// in registers; the score arena and ping-pong K/V stages are the only large
// shared allocations.
#include "vidfab/cuda/sol_attention.cuh"

#include <cuda.h>
#include <cfloat>
#include <cmath>
#include <mma.h>

#include "vidfab/cuda/device.h"

namespace vidfab::cuda {
namespace {
constexpr int B = 64, D = 128, Threads = 128;
constexpr int MaxBlocks = 1024;
constexpr size_t SmemBytes = 2 * B * D * sizeof(__nv_bfloat16) +
                             B * B * (sizeof(float) + sizeof(__nv_bfloat16));

struct TmaState { uint64_t token[2]; };

__device__ uint64_t issue(const CUtensorMap* map, void* dst, int h, int row,
                          uint64_t* barrier) {
  uint32_t bar = uint32_t(__cvta_generic_to_shared(barrier));
  const uint32_t ptr = uint32_t(__cvta_generic_to_shared(dst));
  uint64_t state;
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n"
               "fence.proxy.async.shared::cta;\n"
               "mbarrier.arrive.expect_tx.shared::cta.b64 %1, [%0], %2;\n"
               "cp.async.bulk.tensor.3d.shared::cta.global.tile."
               "mbarrier::complete_tx::bytes [%3], [%4, {%5, %6, %7}], [%0];"
               : "+r"(bar), "=l"(state)
               : "r"(B * D * int(sizeof(__nv_bfloat16))), "r"(ptr), "l"(map),
                 "r"(0), "r"(h), "r"(row) : "memory");
  return state;
}

__device__ void wait(uint64_t* barrier, uint64_t state) {
  const uint32_t bar = uint32_t(__cvta_generic_to_shared(barrier));
  uint32_t ready;
  do {
    asm volatile("{ .reg .pred p; mbarrier.test_wait.shared::cta.b64 p, [%1], %2; "
                 "selp.b32 %0, 1, 0, p; }"
                 : "=r"(ready) : "r"(bar), "l"(state) : "memory");
  } while (!ready);
}

__global__ __launch_bounds__(Threads, 1) void exact_pipeline(
    const __grid_constant__ CUtensorMap qmap,
    const __grid_constant__ CUtensorMap kmap,
    const __grid_constant__ CUtensorMap vmap, __nv_bfloat16* out,
    const __nv_bfloat16* km, const __nv_bfloat16* vsm,
    const float* tau,
    int seq, int heads, int prefix, float scale,
    unsigned long long* route_counts) {
  const int qb = blockIdx.x, h = blockIdx.y, t = threadIdx.x;
  const int qlo = qb * B, qn=min(B,seq-qlo), blocks = (seq+B-1)/B, warp = t >> 5;
  extern __shared__ __align__(128) unsigned char raw[];
  auto* q = reinterpret_cast<__nv_bfloat16*>(raw);
  auto* score = reinterpret_cast<float*>(raw);
  auto* prob = reinterpret_cast<__nv_bfloat16*>(score + B * B);
  auto* kv = prob + B * B;
  __shared__ alignas(8) uint64_t qbar, bars[2];
  __shared__ uint64_t states[2];
  __shared__ float old_m[B], denom[B], ratio[B], new_m[B];
  __shared__ float qmean[D];
  __shared__ uint16_t exact_ids[MaxBlocks], approx_ids[MaxBlocks];
  __shared__ int exact_count, approx_count;
  using namespace nvcuda;

  if (t == 0) {
    states[0] = issue(&qmap, q, h, qlo, &qbar);
    wait(&qbar, states[0]);
    exact_count=0; approx_count=0;
  }
  wmma::fragment<wmma::accumulator,16,16,16,float> o[8];
  wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> qr[8];
#pragma unroll
  for (int n = 0; n < 8; ++n) wmma::fill_fragment(o[n], 0.0f);
  if (t < B) { old_m[t] = -FLT_MAX; denom[t] = 0.0f; }
  __syncthreads();

  if (t < D) {
    float x=0; for(int r=0;r<qn;++r) x+=__bfloat162float(q[r*D+t]);
    qmean[t]=x/float(qn);
  }
  __syncthreads();
#pragma unroll
  for(int d=0;d<D;d+=16) wmma::load_matrix_sync(qr[d/16],q+warp*16*D+d,D);
  __syncthreads();
  for(int kb=t;kb<blocks;kb+=Threads) {
    float proxy=0;
    for(int d=0;d<D;++d)
      proxy += qmean[d]*__bfloat162float(km[(size_t(kb)*heads+h)*D+d]);
    const bool take=qlo<prefix || kb*B<prefix || abs(qb-kb)<=1 ||
                    proxy*scale>tau[size_t(qb)*heads+h];
    const int slot=take?atomicAdd(&exact_count,1):atomicAdd(&approx_count,1);
    (take?exact_ids:approx_ids)[slot]=kb;
    if(route_counts) atomicAdd(route_counts+(take?0:1),1ull);
  }
  __syncthreads();

  // Compact rejected blocks into 64-wide proxy/PV batches. K centroids and
  // pooled V sums are gathered coalesced, then both products use warp MMA.
  for(int base=0;base<approx_count;base+=B) {
    const int count=min(B,approx_count-base);
    for(int x=t;x<count*D;x+=Threads) {
      const int col=x/D,d=x%D,kb=approx_ids[base+col];
      kv[col*D+d]=km[(size_t(kb)*heads+h)*D+d];
    }
    __syncthreads();
    for(int kr=0;kr<B;kr+=16) {
      wmma::fragment<wmma::accumulator,16,16,16,float> c;
      wmma::fill_fragment(c,0.0f);
#pragma unroll
      for(int d=0;d<D;d+=16) {
        wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major> b;
        wmma::load_matrix_sync(b,kv+kr*D+d,D);
        wmma::mma_sync(c,qr[d/16],b,c);
      }
      for(unsigned i=0;i<c.num_elements;++i)c.x[i]*=scale;
      wmma::store_matrix_sync(score+warp*16*B+kr,c,B,wmma::mem_row_major);
    }
    __syncthreads();
    if(t<qn) {
      float bm=-FLT_MAX;
      for(int j=0;j<count;++j)bm=fmaxf(bm,score[t*B+j]);
      const float nm=fmaxf(old_m[t],bm),rs=expf(old_m[t]-nm);
      float sum=0;
      for(int j=0;j<count;++j) {
        const int kb=approx_ids[base+j],kn=min(B,seq-kb*B);
        sum+=kn*expf(score[t*B+j]-nm);
      }
      ratio[t]=rs;new_m[t]=nm;denom[t]=denom[t]*rs+sum;old_m[t]=nm;
    }
    __syncthreads();
    for(int p=t;p<B*B;p+=Threads)
      prob[p]=p/B<qn && p%B<count?
        __float2bfloat16_rn(expf(score[p]-new_m[p/B])):__float2bfloat16_rn(0);
    for(int x=t;x<count*D;x+=Threads) {
      const int row=x/D,d=x%D,kb=approx_ids[base+row];
      kv[row*D+d]=vsm[(size_t(kb)*heads+h)*D+d];
    }
    __syncthreads();
    for(int n=0;n<8;++n) {
      const int qr0=warp*16,d=n*16;
      auto& c=o[n];
      for(unsigned i=0;i<c.num_elements;++i) {
        const int row=qr0+(t&31)/4+int((i/2)%2)*8;
        if(row<qn)c.x[i]*=ratio[row];
      }
#pragma unroll
      for(int j=0;j<B;j+=16) {
        wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major>a;
        wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::row_major>b;
        wmma::load_matrix_sync(a,prob+qr0*B+j,B);
        wmma::load_matrix_sync(b,kv+j*D+d,D);
        wmma::mma_sync(c,a,b,c);
      }
    }
    __syncthreads();
  }

  if(t==0 && exact_count)
    states[0]=issue(&kmap,kv,h,exact_ids[0]*B,&bars[0]);
  __syncthreads();

  for (int ordinal = 0; ordinal < exact_count; ++ordinal) {
    const int kb=exact_ids[ordinal], stage = ordinal & 1;
    const int kn=min(B,seq-kb*B);
    if (t == 0) {
      wait(&bars[stage], states[stage]);
      if (ordinal + 1 < exact_count)
        states[stage ^ 1] = issue(&kmap, kv + (stage ^ 1) * B * D, h,
                                  exact_ids[ordinal+1] * B, &bars[stage ^ 1]);
    }
    __syncthreads();
    for (int kr = 0; kr < B; kr += 16) {
      wmma::fragment<wmma::accumulator,16,16,16,float> c;
      wmma::fill_fragment(c, 0.0f);
#pragma unroll
      for (int d = 0; d < D; d += 16) {
        wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major> b;
        wmma::load_matrix_sync(b, kv + stage * B * D + kr * D + d, D);
        wmma::mma_sync(c, qr[d/16], b, c);
      }
      for (unsigned i=0;i<c.num_elements;++i) c.x[i] *= scale;
      wmma::store_matrix_sync(score + warp * 16 * B + kr, c, B, wmma::mem_row_major);
    }
    __syncthreads();
    for(int p=t;p<B*B;p+=Threads)
      if(p/B>=qn || p%B>=kn) score[p]=-FLT_MAX;
    __syncthreads();
    if (t < qn) {
      float bm = -FLT_MAX;
      for (int j=0;j<B;++j) bm=fmaxf(bm,score[t*B+j]);
      const float nm=fmaxf(old_m[t],bm), rs=expf(old_m[t]-nm);
      float sum=0; for(int j=0;j<B;++j) sum += expf(score[t*B+j]-nm);
      ratio[t]=rs; new_m[t]=nm; denom[t]=denom[t]*rs+sum; old_m[t]=nm;
    }
    __syncthreads();
    for (int p=t;p<B*B;p+=Threads)
      prob[p]=__float2bfloat16_rn(expf(score[p]-new_m[p/B]));
    __syncthreads();
    if (t == 0) {
      states[stage] = issue(&vmap, kv + stage * B * D, h, kb * B, &bars[stage]);
      wait(&bars[stage], states[stage]);
    }
    __syncthreads();
    for (int n=0;n<8;++n) {
      const int qr0=warp*16, d=n*16;
      auto& c=o[n];
#pragma unroll
      for(unsigned i=0;i<c.num_elements;++i) {
        const int row=qr0+(t&31)/4+int((i/2)%2)*8;
        if(row<qn) c.x[i] *= ratio[row];
      }
#pragma unroll
      for(int j=0;j<B;j+=16) {
        wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> a;
        wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::row_major> b;
        wmma::load_matrix_sync(a,prob+qr0*B+j,B);
        wmma::load_matrix_sync(b,kv+stage*B*D+j*D+d,D);
        wmma::mma_sync(c,a,b,c);
      }
    }
    __syncthreads();
    // The next K was issued before QK and occupies the alternate stage.
  }
  const int lane=t&31;
  for(int n=0;n<8;++n) {
    const int qr0=warp*16, d=n*16;
#pragma unroll
    for(unsigned i=0;i<o[n].num_elements;++i) {
      const int row=qr0+lane/4+int((i/2)%2)*8;
      const int col=d+(lane%4)*2+int(i%2)+int(i/4)*8;
      if(row<qn)
        out[size_t(qlo+row)*heads*D+h*D+col]=
            __float2bfloat16_rn(o[n].x[i]/denom[row]);
    }
  }
}

CUtensorMap map_for(const __nv_bfloat16* p, const AttentionConfig& c) {
  CUtensorMap m{};
  const cuuint64_t dims[3]={D,cuuint64_t(c.num_heads),cuuint64_t(c.seq_len)};
  const cuuint64_t strides[2]={D*sizeof(__nv_bfloat16),
                              cuuint64_t(c.num_heads)*D*sizeof(__nv_bfloat16)};
  const cuuint32_t box[3]={D,1,B}, elem[3]={1,1,1};
  const CUresult status=cuTensorMapEncodeTiled(&m,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,3,
      const_cast<__nv_bfloat16*>(p),dims,strides,box,elem,
      CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (status != CUDA_SUCCESS) return CUtensorMap{};
  return m;
}
}  // namespace

bool sol_pipeline_forward(cudaStream_t stream, const __nv_bfloat16* q,
                          const __nv_bfloat16* k, const __nv_bfloat16* v,
                          const __nv_bfloat16* km,
                          const __nv_bfloat16* vsm,
                          const float* tau, __nv_bfloat16* out,
                          const AttentionConfig& c) {
  if (c.head_dim != D || (c.seq_len+B-1)/B > MaxBlocks) return false;
  const auto q_map=map_for(q,c), k_map=map_for(k,c), v_map=map_for(v,c);
  VIDFAB_CUDA_CHECK(cudaFuncSetAttribute(exact_pipeline,
      cudaFuncAttributeMaxDynamicSharedMemorySize,int(SmemBytes)));
  exact_pipeline<<<dim3((c.seq_len+B-1)/B,c.num_heads),Threads,SmemBytes,stream>>>(
      q_map,k_map,v_map,out,km,vsm,tau,c.seq_len,c.num_heads,c.exact_prefix,
      c.effective_scale(),c.sol_route_counts);
  VIDFAB_CUDA_CHECK(cudaGetLastError());
  return true;
}
}  // namespace vidfab::cuda
