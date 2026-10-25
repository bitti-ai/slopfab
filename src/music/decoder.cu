#include "runtime.cuh"
#include "slopfab/cuda/audio_vae_kernels.cuh"
#include "slopfab/tensor_convert.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>

namespace slopfab::music {
namespace {
struct AudioTensor {
  int channels,length;
  cuda::DeviceBuffer<float> data;
  AudioTensor(int c,int t):channels(c),length(t),data(size_t(c)*t){}
};
class Decoder {
public:
  Runtime& rt;
  std::map<std::string,cuda::DeviceBuffer<float>> weights;
  explicit Decoder(Runtime& runtime):rt(runtime){}
  const float* weight(const std::string& p) {
    auto it=weights.find(p);if(it!=weights.end())return it->second.get();
    auto v=to_f32(rt.checkpoint.at(p+".weight_v"));
    auto g=to_f32(rt.checkpoint.at(p+".weight_g"));
    if(v.size()%g.size())throw std::invalid_argument("music VAE weight normalization shape");
    size_t width=v.size()/g.size();
    for(size_t r=0;r<g.size();++r) {
      double sum=0;for(size_t c=0;c<width;++c){double a=v[r*width+c];sum+=a*a;}
      float scale=g[r]/float(std::sqrt(sum));
      for(size_t c=0;c<width;++c)v[r*width+c]*=scale;
    }
    cuda::DeviceBuffer<float> dev(v.size());
    SLOPFAB_CUDA_CHECK(cudaMemcpy(dev.get(),v.data(),dev.nbytes(),cudaMemcpyHostToDevice));
    return weights.emplace(p,std::move(dev)).first->second.get();
  }
  AudioTensor conv(const AudioTensor& x,const std::string& p,int dilation=1,int stride=1) {
    const auto& shape=rt.checkpoint.at(p+".weight_v").shape;
    const int kernel=int(shape.at(2));const bool transpose=stride>1;
    const int channels=int(shape[transpose?1:0]);
    if(shape[transpose?0:1]!=x.channels)throw std::invalid_argument("music VAE convolution shape: "+p);
    int pad=transpose?(stride+1)/2:dilation*(kernel-1)/2;
    int length=transpose?(x.length-1)*stride-2*pad+kernel:x.length;
    AudioTensor y(channels,length);
    auto bias=rt.checkpoint.find(p+".bias")?rt.float_weight(p+".bias"):nullptr;
    if(transpose)cuda::launch_conv_transpose1d(x.data.get(),weight(p),bias,y.data.get(),1,x.channels,channels,x.length,length,kernel,stride,pad,nullptr);
    else cuda::launch_conv1d(x.data.get(),weight(p),bias,y.data.get(),1,x.channels,channels,x.length,length,kernel,pad,dilation,nullptr);
    return y;
  }
  void snake(AudioTensor& x,const std::string& p) {
    cuda::launch_snake_beta(x.data.get(),rt.float_weight(p+".alpha"),rt.float_weight(p+".beta"),1,x.channels,x.length,nullptr);
  }
  AudioTensor residual(AudioTensor x,const std::string& p,int dilation) {
    AudioTensor h(x.channels,x.length);
    SLOPFAB_CUDA_CHECK(cudaMemcpy(h.data.get(),x.data.get(),x.data.nbytes(),cudaMemcpyDeviceToDevice));
    snake(h,p+"0");h=conv(h,p+"1",dilation);snake(h,p+"2");h=conv(h,p+"3");
    cuda::launch_add_inplace(x.data.get(),h.data.get(),x.data.size(),nullptr);return x;
  }
};
}
std::vector<float> decode_yue(Runtime& rt,const std::vector<float>& latent,Hook hook) {
  if(latent.empty()||latent.size()%64)throw std::invalid_argument("YuE2 latent shape");
  int frames=int(latent.size()/64),total=frames*1920-64;
  std::vector<float> output(size_t(total)*2);Decoder decoder(rt);
  constexpr int core=256,halo=16,strides[]={6,5,4,4,2,2};
  const std::string base="vae.decoder.layers.";
  for(int start=0;start<frames;start+=core) {
    if(hook)hook();
    int end=std::min(frames,start+core),left=std::max(0,start-halo),right=std::min(frames,end+halo),count=right-left;
    std::vector<float> planar(size_t(count)*64);
    for(int t=0;t<count;++t)for(int c=0;c<64;++c)planar[size_t(c)*count+t]=latent[size_t(t+left)*64+c];
    AudioTensor x(64,count);SLOPFAB_CUDA_CHECK(cudaMemcpy(x.data.get(),planar.data(),x.data.nbytes(),cudaMemcpyHostToDevice));
    x=decoder.conv(x,base+"0");
    for(int i=0;i<6;++i) {
      if(hook)hook();auto p=base+std::to_string(i+1)+".layers.";
      decoder.snake(x,p+"0");x=decoder.conv(x,p+"1",1,strides[i]);
      for(int j=0,d=1;j<3;++j,d*=3)x=decoder.residual(std::move(x),p+std::to_string(j+2)+".layers.",d);
    }
    decoder.snake(x,base+"7");x=decoder.conv(x,base+"8");
    int begin=start*1920,finish=std::min(end*1920,total),offset=(start-left)*1920;
    if(x.channels!=2||offset+finish-begin>x.length)throw std::runtime_error("YuE2 VAE tile extent mismatch");
    planar.resize(x.data.size());SLOPFAB_CUDA_CHECK(cudaMemcpy(planar.data(),x.data.get(),x.data.nbytes(),cudaMemcpyDeviceToHost));
    for(int t=begin;t<finish;++t)for(int c=0;c<2;++c)output[size_t(t)*2+c]=planar[size_t(c)*x.length+offset+t-begin];
  }
  return output;
}
}
