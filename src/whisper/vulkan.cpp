#include "internal.h"
#include "../music/vulkan_runtime.h"
#include <algorithm>
#include <tuple>

namespace slopfab::whisper {
namespace {
namespace vk=slopfab::vulkan;
vk::DeviceTensor& tensor(const Tensor& t){return *std::static_pointer_cast<vk::DeviceTensor>(t.data);}
class Vulkan final:public Backend {
  music_gpu::Engine engine;
  std::map<std::pair<int,int>,vk::DenseGemmPlan> plans;
  Tensor make(int r,int c){return {r,c,std::make_shared<vk::DeviceTensor>(engine.allocate(r,c,ScalarType::kFloat32))};}
  void op(int code,const Tensor& x,const Tensor& w,const Tensor& b,Tensor& y,int n,int cols,
          int a=0,int c=0,int d=0,int e=0) {
    engine.op(tensor(x),tensor(w),tensor(b),tensor(y),
      {uint32_t(code),uint32_t(n),uint32_t(cols),uint32_t(a),uint32_t(c),uint32_t(d),uint32_t(e)},
      code==100 || code==105?uint32_t(n):uint32_t((n+255)/256));
  }
public:
  Tensor upload(const std::vector<float>& v,int r,int c)override {
    return {r,c,std::make_shared<vk::DeviceTensor>(engine.upload(v,r,c,true))};
  }
  std::vector<float> download(const Tensor& t)override {
    engine.flush();std::vector<float> v(size_t(t.rows)*t.cols);
    engine.context.download(tensor(t),v.data(),v.size());return v;
  }
  Tensor linear(const Tensor& x,const Tensor& w,const Tensor* b)override {
    auto y=make(x.rows,w.rows);auto key=std::make_pair(w.rows,w.cols);
    auto found=plans.find(key);
    if(found==plans.end()) {
      DenseGemmPlanDesc d;d.max_rows=3000;d.out_features=w.rows;d.in_features=w.cols;d.mode=DenseGemmMode::kFloat32;
      found=plans.emplace(key,vk::DenseGemmPlan::create(engine.context,d)).first;
    }
    found->second.record(engine.batch(),tensor(x),tensor(w),tensor(y),x.rows);
    if(b){auto z=make(y.rows,y.cols);op(107,y,*b,*b,z,y.rows*y.cols,y.cols);return z;}
    return y;
  }
  Tensor norm(const Tensor& x,const Tensor& w,const Tensor& b)override {
    auto y=make(x.rows,x.cols);op(100,x,w,b,y,x.rows,x.cols);return y;
  }
  Tensor gelu(const Tensor& x)override {
    auto y=make(x.rows,x.cols);op(101,x,x,x,y,x.rows*x.cols,x.cols);return y;
  }
  Tensor add(const Tensor& a,const Tensor& b)override {
    auto y=make(a.rows,a.cols);engine.batch().add(tensor(a),tensor(b),tensor(y));return y;
  }
  Tensor rows(const Tensor& x,int first,int count)override {
    auto y=make(count,x.cols);engine.batch().copy_rows(tensor(x),tensor(y),first,0,count);return y;
  }
  Tensor concat(const Tensor& a,const Tensor& b)override {
    if(!a.data)return b;
    auto y=make(a.rows+b.rows,a.cols);engine.batch().copy_rows(tensor(a),tensor(y),0,0,a.rows);
    engine.batch().copy_rows(tensor(b),tensor(y),0,a.rows,b.rows);return y;
  }
  Tensor im2col(const Tensor& x,int stride)override {
    auto y=make((x.rows+stride-1)/stride,x.cols*3);op(102,x,x,x,y,y.rows*y.cols,x.cols,x.rows,stride);return y;
  }
  Tensor attention(const Tensor& q,const Tensor& k,const Tensor& v,int heads,bool causal,std::vector<float>* probabilities)override {
    auto out=make(q.rows,q.cols);int dim=q.cols/heads;
    if(probabilities)probabilities->resize(size_t(heads)*q.rows*k.rows);
    // Bound score buffers independently of model/context size.
    constexpr int tile=32;
    for(int first=0;first<q.rows;first+=tile) {
      int count=std::min(tile,q.rows-first);
      auto scores=make(heads*count,k.rows),probs=make(heads*count,k.rows);
      op(103,q,k,k,scores,scores.rows*scores.cols,dim,k.rows,count,first,heads);
      op(105,scores,scores,scores,probs,scores.rows,k.rows,causal,count,first);
      if(probabilities) {
        auto values=download(probs);
        for(int h=0;h<heads;++h)std::copy_n(values.data()+size_t(h)*count*k.rows,size_t(count)*k.rows,
          probabilities->data()+(size_t(h)*q.rows+first)*k.rows);
      }
      op(104,probs,v,v,out,heads*count*dim,dim,k.rows,count,first,heads);
    }
    return out;
  }
};
}
std::unique_ptr<Backend> vulkan_backend(){return std::make_unique<Vulkan>();}
}
