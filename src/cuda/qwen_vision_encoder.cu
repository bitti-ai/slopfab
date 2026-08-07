#include "vidfab/text/qwen_vision.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <unordered_map>

#include "vidfab/cuda/device.h"
#include "vidfab/cuda/linear.cuh"
#include "vidfab/cuda/qwen_vision.cuh"

namespace vidfab::text {
using cuda::DeviceBuffer;
namespace {
cuda::QuantWeight dense(const DeviceBuffer<uint16_t>& w, int out, int in,
                        const DeviceBuffer<uint16_t>* b = nullptr) {
  cuda::QuantWeight q; q.format = cuda::QuantFormat::kBF16; q.data = w.get();
  q.out_features = out; q.in_features = in;
  if (b) { q.bias = b->get(); q.bias_format = cuda::QuantFormat::kBF16; }
  return q;
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
  for(const auto& image:images){
    const int rows=static_cast<int>(image.grid.patch_count()), groups=rows/4;
    if(image.rows.size()!=static_cast<size_t>(rows)*1536) throw std::runtime_error("Qwen vision: pixel row mismatch");
    std::vector<uint16_t> hp(image.rows.size()); for(size_t i=0;i<hp.size();++i) hp[i]=f32_to_bf16(image.rows[i]);
    auto positions=qwen3vl_vision_positions(image.grid); std::vector<float> hc,hs;
    qwen3vl_vision_rope_tables(positions,hc,hs);
    DeviceBuffer<uint16_t> pixels(hp.size()),x(static_cast<size_t>(rows)*1152),normed(static_cast<size_t>(rows)*1152),qkv(static_cast<size_t>(rows)*3456),q(static_cast<size_t>(rows)*1152),k(q.size()),v(q.size()),branch(q.size()),mlp(static_cast<size_t>(rows)*4304);
    DeviceBuffer<int32_t> pi(positions.learned.size()); DeviceBuffer<float> dc(hc.size()),ds(hs.size());
    DeviceBuffer<uint16_t> mn(static_cast<size_t>(rows)*1152),merged(static_cast<size_t>(groups)*4608),mh(merged.size()),out(static_cast<size_t>(groups)*5120),dout[3]={DeviceBuffer<uint16_t>(out.size()),DeviceBuffer<uint16_t>(out.size()),DeviceBuffer<uint16_t>(out.size())};
    pixels.copy_from_host(hp.data(),hp.size(),s.stream); pi.copy_from_host(positions.learned.data(),positions.learned.size(),s.stream); dc.copy_from_host(hc.data(),hc.size(),s.stream); ds.copy_from_host(hs.data(),hs.size(),s.stream);
    auto patch=dense(s.at("patch_embed.proj.weight"),1152,1536,&s.at("patch_embed.proj.bias"));
    cuda::qwen_vision_patch_embed(s.linear,patch,reinterpret_cast<__nv_bfloat16*>(pixels.get()),reinterpret_cast<__nv_bfloat16*>(s.at("pos_embed.weight").get()),pi.get(),reinterpret_cast<__nv_bfloat16*>(x.get()),rows,s.ws,s.stream);
    cuda::QwenVisionBlockScratch bs{reinterpret_cast<__nv_bfloat16*>(normed.get()),reinterpret_cast<__nv_bfloat16*>(qkv.get()),reinterpret_cast<__nv_bfloat16*>(q.get()),reinterpret_cast<__nv_bfloat16*>(k.get()),reinterpret_cast<__nv_bfloat16*>(v.get()),reinterpret_cast<__nv_bfloat16*>(branch.get()),reinterpret_cast<__nv_bfloat16*>(mlp.get())};
    __nv_bfloat16* dp[3]; for(int i=0;i<3;++i)dp[i]=reinterpret_cast<__nv_bfloat16*>(dout[i].get());
    cuda::qwen_vision_tower_forward(s.handle,s.stream,s.linear,blocks.data(),main,deep,dc.get(),ds.get(),reinterpret_cast<__nv_bfloat16*>(x.get()),rows,bs,reinterpret_cast<__nv_bfloat16*>(mn.get()),reinterpret_cast<__nv_bfloat16*>(merged.get()),reinterpret_cast<__nv_bfloat16*>(mh.get()),reinterpret_cast<__nv_bfloat16*>(out.get()),dp,s.ws);
    const size_t old=result.main.size(); result.main.resize(old+out.size()); out.copy_to_host(result.main.data()+old,out.size(),s.stream);
    for(int i=0;i<3;++i){size_t o=result.deepstack[i].size();result.deepstack[i].resize(o+dout[i].size());dout[i].copy_to_host(result.deepstack[i].data()+o,dout[i].size(),s.stream);} result.tokens+=groups;
    VIDFAB_CUDA_CHECK(cudaStreamSynchronize(s.stream));
  } return result;
}
} // namespace vidfab::text
