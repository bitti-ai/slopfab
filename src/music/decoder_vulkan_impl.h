// Included inside the shared codec's private namespace.
namespace vk = slopfab::vulkan;
struct AudioTensor {
  int channels,length;
  std::shared_ptr<music_gpu::Engine> engine;
  mutable vk::DeviceTensor data;
  AudioTensor(Runtime& rt,int c,int t):channels(c),length(t),engine(rt.engine) {
    uint64_t shape[]={1,uint64_t(c),uint64_t(t)};
    data=engine->batch().allocate(TensorLayout::contiguous(shape,3));
  }
};
class Decoder {
public:
  Runtime& rt;
  std::map<std::string,vk::DeviceTensor> weights;
  explicit Decoder(Runtime& r):rt(r) {}
  vk::DeviceTensor& upload(const std::string& key,const std::vector<float>& v,std::vector<uint64_t> shape) {
    rt.engine->flush();auto dev=rt.engine->context.allocate(TensorLayout::contiguous(shape.data(),uint32_t(shape.size())));
    rt.engine->context.upload_transient(dev,v.data(),v.size());
    return weights.emplace(key,std::move(dev)).first->second;
  }
  vk::DeviceTensor& parameter(const std::string& name) {
    auto it=weights.find(name);if(it!=weights.end())return it->second;
    auto v=to_f32(rt.checkpoint.at(name));return upload(name,v,{v.size()});
  }
  vk::DeviceTensor& weight(const std::string& p) {
    auto it=weights.find(p);if(it!=weights.end())return it->second;
    bool plain=rt.checkpoint.find(p+".weight")!=nullptr;
    auto& view=rt.checkpoint.at(p+(plain?".weight":".weight_v"));auto v=to_f32(view);
    if(!plain) {
      auto g=to_f32(rt.checkpoint.at(p+".weight_g"));size_t width=v.size()/g.size();
      if(v.size()%g.size())throw std::invalid_argument("music VAE weight normalization shape");
      for(size_t r=0;r<g.size();++r) {
        double sum=0;for(size_t c=0;c<width;++c){double a=v[r*width+c];sum+=a*a;}
        float scale=g[r]/float(std::sqrt(sum));for(size_t c=0;c<width;++c)v[r*width+c]*=scale;
      }
    }
    return upload(p,v,std::vector<uint64_t>(view.shape.begin(),view.shape.end()));
  }
  AudioTensor conv(const AudioTensor& x,const std::string& p,int dilation=1,int stride=1) {
    const auto& shape=rt.checkpoint.at(p+(rt.checkpoint.find(p+".weight")?".weight":".weight_v")).shape;
    int kernel=int(shape.at(2)),channels=int(shape[stride>1?1:0]);
    if(shape[stride>1?0:1]!=x.channels)throw std::invalid_argument("music VAE convolution shape: "+p);
    int pad=stride>1?(stride+1)/2:dilation*(kernel-1)/2;
    int length=stride>1?(x.length-1)*stride-2*pad+kernel:x.length;
    AudioTensor y(rt,channels,length);auto& w=weight(p);
    auto* bias=rt.checkpoint.find(p+".bias")?&parameter(p+".bias"):nullptr;
    if(stride>1) {
      vae::AudioConvTranspose1DDesc d{1,uint32_t(x.channels),uint32_t(channels),uint32_t(x.length),uint32_t(length),uint32_t(kernel),uint32_t(stride),uint32_t(pad)};
      rt.engine->batch().audio_conv_transpose1d(x.data,w,bias,y.data,d);
    } else {
      vae::AudioConv1DDesc d{1,uint32_t(x.channels),uint32_t(channels),uint32_t(x.length),uint32_t(length),uint32_t(kernel),uint32_t(pad),uint32_t(dilation)};
      rt.engine->batch().audio_conv1d(x.data,w,bias,y.data,d);
    }
    return y;
  }
  void snake(AudioTensor& x,const std::string& p) {
    auto& a=parameter(p+".alpha");
    if(rt.checkpoint.find(p+".beta")) {
      auto& b=parameter(p+".beta");rt.engine->batch().audio_snake_beta_inplace(x.data,a,b,1,x.channels,x.length);
    } else {
      uint32_t n=x.channels*x.length;rt.engine->op(x.data,a,a,x.data,{15,n,uint32_t(x.length)},(n+255)/256);
    }
  }
  AudioTensor residual(AudioTensor x,const std::string& p,int dilation) {
    AudioTensor h(rt,x.channels,x.length);rt.engine->batch().copy(x.data,h.data);
    snake(h,p+"0");h=conv(h,p+"1",dilation);snake(h,p+"2");h=conv(h,p+"3");
    rt.engine->batch().audio_add_inplace(x.data,h.data);return x;
  }
};
void audio_upload(AudioTensor& x,const std::vector<float>& v) {
  x.engine->flush();x.engine->context.upload_transient(x.data,v.data(),v.size());
}
std::vector<float> audio_download(AudioTensor& x) {
  x.engine->flush();std::vector<float> v(size_t(x.channels)*x.length);x.engine->context.download(x.data,v.data(),v.size());return v;
}
void audio_tanh(AudioTensor& x) {
  uint32_t n=x.channels*x.length;x.engine->op(x.data,x.data,x.data,x.data,{16,n},(n+255)/256);
}
