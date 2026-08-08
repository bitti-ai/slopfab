// Experimental aligned exact SM120 mainloop. The production Sol kernel remains
// the fallback. One thread owns one output column and keeps all 64 row outputs
// in registers; the score arena and reusable K/V stage are the only large
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
constexpr size_t SmemBytes = B * D * sizeof(__nv_bfloat16) +
                             B * B * sizeof(float) + 4 * 16 * 16 * sizeof(__nv_bfloat16);

struct TmaState { uint64_t token[2]; };

__device__ __forceinline__ float fast_exp(float x) {
  float y;
  x *= 1.4426950408889634f;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

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
    const __nv_bfloat16* km, const __nv_bfloat16* vm, const float* vs,
    const float* tau,
    int seq, int heads, int prefix, float scale,
    unsigned long long* route_counts) {
  const int qb = blockIdx.x, h = blockIdx.y, t = threadIdx.x;
  const int qlo = qb * B, qn=min(B,seq-qlo), blocks = (seq+B-1)/B, warp = t >> 5;
  extern __shared__ __align__(128) unsigned char raw[];
  auto* q = reinterpret_cast<__nv_bfloat16*>(raw);
  auto* score = reinterpret_cast<float*>(raw);
  auto* prob_scratch = reinterpret_cast<__nv_bfloat16*>(score + B * B);
  auto* kv = prob_scratch + 4 * 16 * 16;
  __shared__ alignas(8) uint64_t qbar, bars[1];
  __shared__ uint64_t states[1];
  __shared__ float old_m[B], denom[B], ratio[B];
  __shared__ uint16_t route_ids[MaxBlocks];
  __shared__ int exact_count, approx_count;
  using namespace nvcuda;

  if (t == 0) {
    states[0] = issue(&qmap, q, h, qlo, &qbar);
    wait(&qbar, states[0]);
    exact_count=0; approx_count=0;
  }
  // Keep these as named fragments. Indexing an array of WMMA fragments makes
  // nvcc materialize it in a 256-byte local stack frame on sm_120a.
  wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major>
      qr0,qr1,qr2,qr3,qr4,qr5,qr6,qr7;
  if (t < B) { old_m[t] = -FLT_MAX; denom[t] = 0.0f; }
  __syncthreads();

  wmma::load_matrix_sync(qr0,q+warp*16*D+  0,D);
  wmma::load_matrix_sync(qr1,q+warp*16*D+ 16,D);
  wmma::load_matrix_sync(qr2,q+warp*16*D+ 32,D);
  wmma::load_matrix_sync(qr3,q+warp*16*D+ 48,D);
  wmma::load_matrix_sync(qr4,q+warp*16*D+ 64,D);
  wmma::load_matrix_sync(qr5,q+warp*16*D+ 80,D);
  wmma::load_matrix_sync(qr6,q+warp*16*D+ 96,D);
  wmma::load_matrix_sync(qr7,q+warp*16*D+112,D);
  __syncthreads();
  float qavg=0; for(int r=0;r<qn;++r) qavg+=__bfloat162float(q[r*D+t]);
  __syncthreads();
  score[t]=qavg/float(qn);
  __syncthreads();
  for(int kb=t;kb<blocks;kb+=Threads) {
    float proxy=0;
    for(int d=0;d<D;++d)
      proxy += score[d]*__bfloat162float(km[(size_t(kb)*heads+h)*D+d]);
    const bool take=qlo<prefix || kb*B<prefix || abs(qb-kb)<=1 ||
                    proxy*scale>tau[size_t(qb)*heads+h];
    const int slot=take?atomicAdd(&exact_count,1):atomicAdd(&approx_count,1);
    route_ids[take?slot:MaxBlocks-1-slot]=uint16_t(kb);
    if(route_counts) atomicAdd(route_counts+(take?0:1),1ull);
  }
  __syncthreads();

  // Compact rejected blocks into 64-wide proxy/PV batches. K centroids and
  // pooled V sums are gathered coalesced, then both products use warp MMA.
  for(int base=0;base<approx_count;base+=B) {
    const int count=min(B,approx_count-base);
    for(int x=t;x<count*D;x+=Threads) {
      const int col=x/D,d=x%D,kb=route_ids[MaxBlocks-1-(base+col)];
      kv[col*D+d]=km[(size_t(kb)*heads+h)*D+d];
    }
    __syncthreads();
    for(int kr=0;kr<B;kr+=16) {
      wmma::fragment<wmma::accumulator,16,16,16,float> c;
      wmma::fill_fragment(c,0.0f);
#define VIDFAB_QK_STEP(QR, OFF) do {                                      \
        wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,             \
                       wmma::col_major> b;                                 \
        wmma::load_matrix_sync(b,kv+kr*D+(OFF),D);                         \
        wmma::mma_sync(c,(QR),b,c);                                        \
      } while (false)
      VIDFAB_QK_STEP(qr0,  0); VIDFAB_QK_STEP(qr1, 16);
      VIDFAB_QK_STEP(qr2, 32); VIDFAB_QK_STEP(qr3, 48);
      VIDFAB_QK_STEP(qr4, 64); VIDFAB_QK_STEP(qr5, 80);
      VIDFAB_QK_STEP(qr6, 96); VIDFAB_QK_STEP(qr7,112);
      for(unsigned i=0;i<c.num_elements;++i)c.x[i]*=scale;
      wmma::store_matrix_sync(score+warp*16*B+kr,c,B,wmma::mem_row_major);
    }
    __syncthreads();
    if(t<qn) {
      float bm=-FLT_MAX;
      for(int j=0;j<count;++j)bm=fmaxf(bm,score[t*B+j]);
      const float nm=fmaxf(old_m[t],bm),rs=fast_exp(old_m[t]-nm);
      float sum=0;
      for(int j=0;j<count;++j) {
        const int kb=route_ids[MaxBlocks-1-(base+j)],kn=min(B,seq-kb*B);
        const float p=fast_exp(score[t*B+j]-nm);
        score[t*B+j]=p;
        sum+=kn*p;
      }
      ratio[t]=denom[t]*rs;denom[t]=ratio[t]+sum;old_m[t]=nm;
    }
    __syncthreads();
    for(int x=t;x<B*D;x+=Threads) {
      const int col=x/D,d=x%D,kb=route_ids[MaxBlocks-1-(base+col)];
      kv[col*D+d]=col<count?vm[(size_t(kb)*heads+h)*D+d]:__float2bfloat16_rn(0.0f);
    }
    __syncthreads();
    for(int n=0;n<8;++n) {
      wmma::fragment<wmma::accumulator,16,16,16,float> add;
      wmma::fill_fragment(add,0.0f);
      for(int j=0;j<count;j+=16) {
        auto* ps=prob_scratch+warp*256;
        for(int p=(t&31);p<256;p+=32) {
          const int row=warp*16+p/16,col=j+p%16;
          float weighted=0;
          if(row<qn&&col<count) {
            const int kb=route_ids[MaxBlocks-1-(base+col)];
            weighted=score[row*B+col]*min(B,seq-kb*B);
          }
          ps[p]=__float2bfloat16_rn(weighted);
        }
        __syncwarp();
        wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> a;
        wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::row_major> b;
        wmma::load_matrix_sync(a,ps,16);
        wmma::load_matrix_sync(b,kv+j*D+n*16,D);
        wmma::mma_sync(add,a,b,add);
      }
      const int lane=t&31,qr0=warp*16,d=n*16;
      for(unsigned i=0;i<add.num_elements;++i) {
        const int row=qr0+lane/4+int((i/2)%2)*8;
        const int col=d+(lane%4)*2+int(i%2)+int(i/4)*8;
        if(row<qn) {
          const size_t pos=size_t(qlo+row)*heads*D+h*D+col;
          const float prev=base?__bfloat162float(out[pos]):0.0f;
          out[pos]=__float2bfloat16_rn((prev*ratio[row]+add.x[i])/denom[row]);
        }
      }
    }
    __syncthreads();
  }

  // Begin the persistent output-fragment lifetime only after approximation.
  wmma::fragment<wmma::accumulator,16,16,16,float>
      o0,o1,o2,o3,o4,o5,o6,o7;
#define VIDFAB_INIT_O(C, OFF) do {                                         \
    auto& frag=(C); const int d=(OFF),lane=t&31,qr0=warp*16;                \
    for(unsigned i=0;i<frag.num_elements;++i) {                             \
      const int row=qr0+lane/4+int((i/2)%2)*8;                              \
      const int col=d+(lane%4)*2+int(i%2)+int(i/4)*8;                      \
      frag.x[i]=(row<qn&&approx_count)?                                    \
          __bfloat162float(out[size_t(qlo+row)*heads*D+h*D+col])*denom[row]:0.0f; \
    }                                                                      \
  } while(false)
  VIDFAB_INIT_O(o0,0); VIDFAB_INIT_O(o1,16); VIDFAB_INIT_O(o2,32); VIDFAB_INIT_O(o3,48);
  VIDFAB_INIT_O(o4,64); VIDFAB_INIT_O(o5,80); VIDFAB_INIT_O(o6,96); VIDFAB_INIT_O(o7,112);
#undef VIDFAB_INIT_O

  for (int ordinal = 0; ordinal < exact_count; ++ordinal) {
    const int kb=route_ids[ordinal], stage = 0;
    const int kn=min(B,seq-kb*B);
    if (t == 0) {
      states[0]=issue(&kmap,kv,h,kb*B,&bars[0]);
      wait(&bars[0],states[0]);
    }
    __syncthreads();
    for (int kr = 0; kr < B; kr += 16) {
      wmma::fragment<wmma::accumulator,16,16,16,float> c;
      wmma::fill_fragment(c, 0.0f);
      VIDFAB_QK_STEP(qr0,  0); VIDFAB_QK_STEP(qr1, 16);
      VIDFAB_QK_STEP(qr2, 32); VIDFAB_QK_STEP(qr3, 48);
      VIDFAB_QK_STEP(qr4, 64); VIDFAB_QK_STEP(qr5, 80);
      VIDFAB_QK_STEP(qr6, 96); VIDFAB_QK_STEP(qr7,112);
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
      const float nm=fmaxf(old_m[t],bm), rs=fast_exp(old_m[t]-nm);
      float sum=0; for(int j=0;j<B;++j) sum += fast_exp(score[t*B+j]-nm);
      ratio[t]=rs; denom[t]=denom[t]*rs+sum; old_m[t]=nm;
    }
    __syncthreads();
    if (t == 0) {
      states[stage] = issue(&vmap, kv + stage * B * D, h, kb * B, &bars[stage]);
      wait(&bars[stage], states[stage]);
    }
    __syncthreads();
#define VIDFAB_SCALE_O(C) do {                                             \
      const int qr0=warp*16; auto& c=(C);                                  \
      for(unsigned i=0;i<c.num_elements;++i) {                              \
        const int row=qr0+(t&31)/4+int((i/2)%2)*8;                          \
        if(row<qn) c.x[i] *= ratio[row];                                    \
      }                                                                    \
    } while(false)
    VIDFAB_SCALE_O(o0); VIDFAB_SCALE_O(o1); VIDFAB_SCALE_O(o2); VIDFAB_SCALE_O(o3);
    VIDFAB_SCALE_O(o4); VIDFAB_SCALE_O(o5); VIDFAB_SCALE_O(o6); VIDFAB_SCALE_O(o7);
#undef VIDFAB_SCALE_O
    for(int j=0;j<B;j+=16) {
      auto* ps=prob_scratch+warp*256;
      for(int p=(t&31);p<256;p+=32) {
        const int row=warp*16+p/16,col=j+p%16;
        ps[p]=(row<qn && col<kn)?__float2bfloat16_rn(
          fast_exp(score[row*B+col]-old_m[row])):__float2bfloat16_rn(0);
      }
      __syncwarp();
      wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> a;
      wmma::load_matrix_sync(a,ps,16);
#define VIDFAB_PV_STEP(C, OFF) do {                                        \
        const int d=(OFF);                                                  \
        wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,               \
                       wmma::row_major> b;                                   \
        wmma::load_matrix_sync(b,kv+stage*B*D+j*D+d,D);                     \
        wmma::mma_sync((C),a,b,(C));                                        \
      } while(false)
      VIDFAB_PV_STEP(o0,  0); VIDFAB_PV_STEP(o1, 16);
      VIDFAB_PV_STEP(o2, 32); VIDFAB_PV_STEP(o3, 48);
      VIDFAB_PV_STEP(o4, 64); VIDFAB_PV_STEP(o5, 80);
      VIDFAB_PV_STEP(o6, 96); VIDFAB_PV_STEP(o7,112);
#undef VIDFAB_PV_STEP
      __syncwarp();
    }
    __syncthreads();
  }
  const int lane=t&31;
#define VIDFAB_STORE_O(C, OFF) do {                                        \
    const int qr0=warp*16, d=(OFF); auto& frag=(C);                         \
    for(unsigned i=0;i<frag.num_elements;++i) {                             \
      const int row=qr0+lane/4+int((i/2)%2)*8;                              \
      const int col=d+(lane%4)*2+int(i%2)+int(i/4)*8;                      \
      if(row<qn)                                                            \
        out[size_t(qlo+row)*heads*D+h*D+col]=                              \
            __float2bfloat16_rn(frag.x[i]/denom[row]);                     \
    }                                                                      \
  } while(false)
  VIDFAB_STORE_O(o0,  0); VIDFAB_STORE_O(o1, 16);
  VIDFAB_STORE_O(o2, 32); VIDFAB_STORE_O(o3, 48);
  VIDFAB_STORE_O(o4, 64); VIDFAB_STORE_O(o5, 80);
  VIDFAB_STORE_O(o6, 96); VIDFAB_STORE_O(o7,112);
#undef VIDFAB_STORE_O
#undef VIDFAB_QK_STEP
}

bool map_for(const __nv_bfloat16* p, const AttentionConfig& c, CUtensorMap* m) {
  *m=CUtensorMap{};
  const cuuint64_t dims[3]={D,cuuint64_t(c.num_heads),cuuint64_t(c.seq_len)};
  const cuuint64_t strides[2]={D*sizeof(__nv_bfloat16),
                              cuuint64_t(c.num_heads)*D*sizeof(__nv_bfloat16)};
  const cuuint32_t box[3]={D,1,B}, elem[3]={1,1,1};
  const CUresult status=cuTensorMapEncodeTiled(m,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,3,
      const_cast<__nv_bfloat16*>(p),dims,strides,box,elem,
      CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return status==CUDA_SUCCESS;
}
}  // namespace

bool sol_pipeline_forward(cudaStream_t stream, const __nv_bfloat16* q,
                          const __nv_bfloat16* k, const __nv_bfloat16* v,
                          const __nv_bfloat16* km, const __nv_bfloat16* vm,
                          const float* vs,
                          const float* tau, __nv_bfloat16* out,
                          const AttentionConfig& c) {
  if (c.head_dim != D || (c.seq_len+B-1)/B > MaxBlocks) return false;
  CUtensorMap q_map{},k_map{},v_map{};
  // Misaligned or otherwise unsupported K/V layouts are valid inputs for the
  // scalar Sol kernel. Never launch TMA with a zero/invalid descriptor.
  if(!map_for(q,c,&q_map)||!map_for(k,c,&k_map)||!map_for(v,c,&v_map)) return false;
  VIDFAB_CUDA_CHECK(cudaFuncSetAttribute(exact_pipeline,
      cudaFuncAttributeMaxDynamicSharedMemorySize,int(SmemBytes)));
  exact_pipeline<<<dim3((c.seq_len+B-1)/B,c.num_heads),Threads,SmemBytes,stream>>>(
      q_map,k_map,v_map,out,km,vm,vs,tau,c.seq_len,c.num_heads,c.exact_prefix,
      c.effective_scale(),c.sol_route_counts);
  VIDFAB_CUDA_CHECK(cudaGetLastError());
  return true;
}
}  // namespace vidfab::cuda
