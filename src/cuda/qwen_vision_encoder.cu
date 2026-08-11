#include "vidfab/text/qwen_vision.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <unordered_map>

#include "vidfab/cuda/attention.cuh"
#include "vidfab/cuda/device.h"
#include "vidfab/cuda/linear.cuh"
#include "vidfab/cuda/qwen_vision.cuh"

namespace vidfab::text {
using cuda::DeviceBuffer;
namespace {
constexpr int kHidden = 1152, kHeads = 16, kHeadDim = 72, kIntermediate = 4304;
constexpr int kMergedDim = 4608, kOutDim = 5120, kPatchDim = 1536, kQkvDim = 3456;

cuda::QuantWeight dense(const DeviceBuffer<uint16_t>& w, int out, int in,
                        const DeviceBuffer<uint16_t>* b = nullptr) {
  cuda::QuantWeight q; q.format = cuda::QuantFormat::kBF16; q.data = w.get();
  q.out_features = out; q.in_features = in;
  if (b) { q.bias = b->get(); q.bias_format = cuda::QuantFormat::kBF16; }
  return q;
}

// Every activation one image needs, in one place so the sizing pass and the
// carving pass cannot disagree about what exists.
struct VisionScratch {
  __nv_bfloat16 *pixels, *x, *normed, *qkv, *q, *k, *v, *branch, *mlp;
  __nv_bfloat16 *merger_normed, *merged, *merger_hidden, *out, *deep[3];
  int32_t* pos_index;
  float *rope_cos, *rope_sin;
};

// A bump allocator that either measures or carves, so `carve_vision` is written
// once and run twice. `Workspace::alloc` rounds to 256, so the measuring pass
// has to round identically or the reserve comes up short by up to 255 bytes a
// buffer.
struct Carver {
  cuda::Workspace* ws = nullptr;  // null while measuring
  size_t bytes = 0;
  template <typename T>
  T* take(size_t n) {
    if (ws != nullptr) return ws->alloc_n<T>(n);
    bytes += (n * sizeof(T) + 255u) & ~size_t(255u);
    return nullptr;
  }
};

VisionScratch carve_vision(Carver& c, int rows, int groups, size_t pos_count, size_t rope_count) {
  const size_t r = static_cast<size_t>(rows), g = static_cast<size_t>(groups);
  VisionScratch s{};
  s.pixels = c.take<__nv_bfloat16>(r * kPatchDim);
  s.x = c.take<__nv_bfloat16>(r * kHidden);
  s.normed = c.take<__nv_bfloat16>(r * kHidden);
  s.qkv = c.take<__nv_bfloat16>(r * kQkvDim);
  s.q = c.take<__nv_bfloat16>(r * kHidden);
  s.k = c.take<__nv_bfloat16>(r * kHidden);
  s.v = c.take<__nv_bfloat16>(r * kHidden);
  s.branch = c.take<__nv_bfloat16>(r * kHidden);
  s.mlp = c.take<__nv_bfloat16>(r * kIntermediate);
  s.merger_normed = c.take<__nv_bfloat16>(r * kHidden);
  s.merged = c.take<__nv_bfloat16>(g * kMergedDim);
  s.merger_hidden = c.take<__nv_bfloat16>(g * kMergedDim);
  s.out = c.take<__nv_bfloat16>(g * kOutDim);
  for (int i = 0; i < 3; ++i) s.deep[i] = c.take<__nv_bfloat16>(g * kOutDim);
  s.pos_index = c.take<int32_t>(pos_count);
  s.rope_cos = c.take<float>(rope_count);
  s.rope_sin = c.take<float>(rope_count);
  return s;
}
}

struct QwenVisionEncoder::Impl {
  bool loaded = false;
  std::string prefix;
  std::unordered_map<std::string, DeviceBuffer<uint16_t>> tensors;
  cublasHandle_t handle = nullptr; cudaStream_t stream = nullptr;
  cuda::LinearRunner linear; cuda::Workspace ws;

  DeviceBuffer<uint16_t>& at(const std::string& suffix) { return tensors.at(prefix + suffix); }
  void open() { if (handle) return; if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("Qwen vision: cublasCreate failed");
    VIDFAB_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)); linear.init(handle, stream); }
  void close() { if (stream) cudaStreamDestroy(stream); if (handle) cublasDestroy(handle);
    stream = nullptr; handle = nullptr; }
  cuda::QwenVisionBlockWeights block(int i) {
    const std::string p = "blocks." + std::to_string(i) + ".";
    cuda::QwenVisionBlockWeights b;
    b.norm1_weight = reinterpret_cast<__nv_bfloat16*>(at(p+"norm1.weight").get());
    b.norm1_bias = reinterpret_cast<__nv_bfloat16*>(at(p+"norm1.bias").get());
    b.norm2_weight = reinterpret_cast<__nv_bfloat16*>(at(p+"norm2.weight").get());
    b.norm2_bias = reinterpret_cast<__nv_bfloat16*>(at(p+"norm2.bias").get());
    b.qkv=dense(at(p+"attn.qkv.weight"),3456,1152,&at(p+"attn.qkv.bias"));
    b.attention_out=dense(at(p+"attn.proj.weight"),1152,1152,&at(p+"attn.proj.bias"));
    b.mlp_fc1=dense(at(p+"mlp.linear_fc1.weight"),4304,1152,&at(p+"mlp.linear_fc1.bias"));
    b.mlp_fc2=dense(at(p+"mlp.linear_fc2.weight"),1152,4304,&at(p+"mlp.linear_fc2.bias"));
    return b;
  }
  cuda::QwenVisionMergerWeights merger(const std::string& p, bool main) {
    cuda::QwenVisionMergerWeights m;
    m.norm_weight=reinterpret_cast<__nv_bfloat16*>(at(p+"norm.weight").get());
    m.norm_bias=reinterpret_cast<__nv_bfloat16*>(at(p+"norm.bias").get());
    m.fc1=dense(at(p+"linear_fc1.weight"),4608,4608,&at(p+"linear_fc1.bias"));
    m.fc2=dense(at(p+"linear_fc2.weight"),5120,4608,&at(p+"linear_fc2.bias"));
    m.norm_before_merge=main; return m;
  }
};

QwenVisionEncoder::QwenVisionEncoder():impl_(new Impl){}
QwenVisionEncoder::~QwenVisionEncoder(){ unload(); impl_->close(); }
void QwenVisionEncoder::unload(){ if(impl_->stream) cudaStreamSynchronize(impl_->stream);
  impl_->tensors.clear(); impl_->ws=cuda::Workspace(); impl_->loaded=false; }
void QwenVisionEncoder::load(const SafeTensors& st){ unload(); auto c=load_qwen3vl_vision_checkpoint(st);
  auto& s=*impl_; s.prefix=c.prefix; s.open();
  for(const auto& kv:st.tensors()) if(kv.first.rfind(s.prefix,0)==0){
    const TensorView& t=kv.second; DeviceBuffer<uint16_t> d(t.nbytes/2);
    d.copy_from_host(static_cast<const uint16_t*>(t.data),t.nbytes/2,s.stream);
    s.tensors.emplace(kv.first,std::move(d)); }
  VIDFAB_CUDA_CHECK(cudaStreamSynchronize(s.stream)); s.loaded=true; }

QwenVisionEmbedding QwenVisionEncoder::encode(const std::vector<QwenPixelValues>& images){
  auto& s=*impl_; if(!s.loaded) throw std::runtime_error("Qwen vision: load first");
  QwenVisionEmbedding result;
  std::vector<cuda::QwenVisionBlockWeights> blocks; for(int i=0;i<27;++i) blocks.push_back(s.block(i));
  auto main=s.merger("merger.",true); cuda::QwenVisionMergerWeights deep[3];
  for(int i=0;i<3;++i) deep[i]=s.merger("deepstack_merger_list."+std::to_string(i)+".",false);
  // Fifteen DeviceBuffers used to be constructed and destroyed per image, and a
  // cudaFree synchronises the whole device (see the note on vit_decoder.cu's
  // scratch). They come out of the arena instead. The arena keeps its high-water
  // mark for the life of the encoder, which is bounded: `unload()` -- called
  // right after `encode` in encoder_kernels.cu -- replaces it with a fresh one.
  //
  // The reserve covers the activations *plus* the attention scratch, because
  // qwen_vision_attention reserves the arena itself further down the call and a
  // grow there would free everything carved here. Reserving the sum up front
  // makes that inner reserve a no-op, which is the whole contract.
  for(const auto& image:images){
    const int rows=static_cast<int>(image.grid.patch_count()), groups=rows/4;
    if(image.rows.size()!=static_cast<size_t>(rows)*kPatchDim) throw std::runtime_error("Qwen vision: pixel row mismatch");
    std::vector<uint16_t> hp(image.rows.size()); for(size_t i=0;i<hp.size();++i) hp[i]=f32_to_bf16(image.rows[i]);
    auto positions=qwen3vl_vision_positions(image.grid); std::vector<float> hc,hs;
    qwen3vl_vision_rope_tables(positions,hc,hs);

    cuda::AttentionConfig acfg; acfg.seq_len=rows; acfg.num_heads=kHeads; acfg.head_dim=kHeadDim;
    const size_t attn_bytes=cuda::attention_workspace_bytes(acfg,cuda::attention_preferred_backend(acfg));
    Carver measure; carve_vision(measure,rows,groups,positions.learned.size(),hc.size());
    s.ws.reserve(measure.bytes+attn_bytes);
    const size_t reserved=s.ws.capacity();
    cuda::Workspace::Scope scope(s.ws);
    Carver carver; carver.ws=&s.ws;
    const VisionScratch b=carve_vision(carver,rows,groups,positions.learned.size(),hc.size());

    VIDFAB_CUDA_CHECK(cudaMemcpyAsync(b.pixels,hp.data(),hp.size()*sizeof(uint16_t),cudaMemcpyHostToDevice,s.stream));
    VIDFAB_CUDA_CHECK(cudaMemcpyAsync(b.pos_index,positions.learned.data(),positions.learned.size()*sizeof(int32_t),cudaMemcpyHostToDevice,s.stream));
    VIDFAB_CUDA_CHECK(cudaMemcpyAsync(b.rope_cos,hc.data(),hc.size()*sizeof(float),cudaMemcpyHostToDevice,s.stream));
    VIDFAB_CUDA_CHECK(cudaMemcpyAsync(b.rope_sin,hs.data(),hs.size()*sizeof(float),cudaMemcpyHostToDevice,s.stream));
    auto patch=dense(s.at("patch_embed.proj.weight"),kHidden,kPatchDim,&s.at("patch_embed.proj.bias"));
    cuda::qwen_vision_patch_embed(s.linear,patch,b.pixels,reinterpret_cast<__nv_bfloat16*>(s.at("pos_embed.weight").get()),b.pos_index,b.x,rows,s.ws,s.stream);
    cuda::QwenVisionBlockScratch bs{b.normed,b.qkv,b.q,b.k,b.v,b.branch,b.mlp};
    __nv_bfloat16* dp[3]; for(int i=0;i<3;++i)dp[i]=b.deep[i];
    cuda::qwen_vision_tower_forward(s.handle,s.stream,s.linear,blocks.data(),main,deep,b.rope_cos,b.rope_sin,b.x,rows,bs,b.merger_normed,b.merged,b.merger_hidden,b.out,dp,s.ws);
    const size_t out_n=static_cast<size_t>(groups)*kOutDim;
    const size_t old=result.main.size(); result.main.resize(old+out_n);
    VIDFAB_CUDA_CHECK(cudaMemcpyAsync(result.main.data()+old,b.out,out_n*sizeof(uint16_t),cudaMemcpyDeviceToHost,s.stream));
    for(int i=0;i<3;++i){size_t o=result.deepstack[i].size();result.deepstack[i].resize(o+out_n);
      VIDFAB_CUDA_CHECK(cudaMemcpyAsync(result.deepstack[i].data()+o,b.deep[i],out_n*sizeof(uint16_t),cudaMemcpyDeviceToHost,s.stream));}
    result.tokens+=groups;
    VIDFAB_CUDA_CHECK(cudaStreamSynchronize(s.stream));
    // If the inner reserve had grown the arena it would have freed everything
    // carved above and reset the cursor, and the results just copied out would
    // have come from a dangling pointer -- plausible numbers, no crash. The
    // reserve is sized so that cannot happen; this says so out loud rather than
    // leaving it to a comment.
    if(s.ws.capacity()!=reserved) throw std::runtime_error(
        "Qwen vision: attention grew the arena under the carved activations");
  } return result;
}
} // namespace vidfab::text
