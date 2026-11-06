#define SLOPFAB_IMAGE_VULKAN 1
#include "harness.h"
#include "../src/fibo/runtime.cuh"
#include "../src/seedvr2/vulkan_runtime.h"
#include "slopfab/safetensors_write.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>

namespace {
using namespace slopfab;
namespace mv = moss_vulkan;
namespace fv = fibo_vulkan;
namespace sv = seedvr2_vulkan;
using music_gpu::round_bf;
struct Fixture {
  std::filesystem::path path = std::filesystem::temp_directory_path() /
      ("slopfab-image-vk-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".safetensors");
  ~Fixture() { std::error_code ec; std::filesystem::remove(path,ec); }
};
bool available() {
  return vulkan::Instance::available() && !vulkan::Instance::create().enumerate_devices().empty();
}
std::vector<float> data(size_t n,int seed=1,float scale=.5f) {
  auto v=test::make_data(n,seed,scale);
  for(auto& a:v)a=round_bf(a);
  return v;
}
}

SLOPFAB_TEST(image_vulkan_normalization_and_edit_layouts) {
  if(!available()){SKIP_UNSUPPORTED_HARDWARE("Vulkan unavailable");return;}
  Fixture f;
  auto w=data(128,3),bias=data(128,7);
  write_safetensors(f.path.string(),{{"n.weight",{128},w},{"n.bias",{128},bias}});
  mv::Runtime rt(f.path.string());
  auto input=data(3*128); auto x=mv::from_host(input,3,128);
  for(bool offset:{false,true}) {
    auto actual=mv::to_host(rt.norm(x,"n.weight",128,offset));
    std::vector<float> expected(input.size());
    for(int row=0;row<3;++row){double sum=0;
      for(int c=0;c<128;++c)sum+=double(input[row*128+c])*input[row*128+c];
      float inv=float(1/std::sqrt(sum/128+1e-6));
      for(int c=0;c<128;++c){float v=input[row*128+c]*inv;
        expected[row*128+c]=offset?round_bf(v*(1+w[c])):round_bf(round_bf(v)*w[c]);
      }
    }
    CHECK_CLOSE(expected,actual,.016,"MOSS RMS rounding");
  }
  auto actual=mv::to_host(fv::layer_norm(rt,x,"n",false,1e-5f));
  std::vector<float> expected(input.size());
  for(int row=0;row<3;++row){double mean=0,var=0;
    for(int c=0;c<128;++c)mean+=input[row*128+c]/128.0;
    for(int c=0;c<128;++c)var+=std::pow(input[row*128+c]-mean,2)/128.0;
    for(int c=0;c<128;++c)expected[row*128+c]=round_bf(float((input[row*128+c]-mean)/std::sqrt(var+1e-5))*w[c]+bias[c]);
  }
  CHECK_CLOSE(expected,actual,.008,"Fibo affine layer normalization");
  auto merged=fv::reshape(x,1,384);
  CHECK_CLOSE(input,mv::to_host(merged),0,"Fibo contiguous vision merge");
  auto target=mv::from_host(std::vector<float>(5*128,-1),5,128);
  fv::scatter(target,x,{4,0,2},false);
  actual=mv::to_host(target);
  for(int row=0;row<3;++row)for(int c=0;c<128;++c)
    CHECK((actual[std::vector<int>{4,0,2}[row]*128+c]==input[row*128+c]));
  CHECK(actual[128]==-1&&actual[384]==-1);
  auto gate=mv::from_host(w,1,128),v=mv::from_host(bias,1,128);
  actual=mv::to_host(mv::activate(v,&gate));
  for(int c=0;c<128;++c)CHECK_NEAR(actual[c],round_bf(bias[c]*round_bf(w[c]/(1+std::exp(-w[c])))),.004);
  actual=mv::to_host(fv::gelu(x,false));
  for(size_t i=0;i<input.size();++i)
    CHECK_NEAR(actual[i],round_bf(.5f*input[i]*(1+std::erf(input[i]/std::sqrt(2.f)))),.004);
}

SLOPFAB_TEST(moss_vulkan_recurrent_delta) {
  if(!available()){SKIP_UNSUPPORTED_HARDWARE("Vulkan unavailable");return;}
  Fixture f;write_safetensors(f.path.string(),{{"dummy",{1},{0}}});
  mv::Runtime rt(f.path.string());
  constexpr int rows=4;
  auto qkv=data(rows*8192,18);
  std::vector<float> params(rows*64),expected(rows*4096);
  for(int row=0;row<rows;++row)for(int head=0;head<32;++head){
    params[(row*32+head)*2]=.5f+head*.01f;
    params[(row*32+head)*2+1]=round_bf(.4f+row*.1f);
  }
  auto x=mv::from_host(qkv,rows,8192);
  auto p=rt.engine->upload(params,rows,64,true);
  mv::Tensor y(rows,4096);
  krea_vulkan::dispatch(*rt.engine,x.data,p,p,y.data,{mv::RecurrentDelta,rows},32);
  for(int head=0;head<32;++head){
    std::vector<float> state(128*128,0);
    for(int row=0;row<rows;++row){float q[128],k[128];double qs=1e-6,ks=1e-6;
      for(int c=0;c<128;++c){q[c]=qkv[row*8192+(head/2)*128+c];k[c]=qkv[row*8192+2048+(head/2)*128+c];qs+=q[c]*q[c];ks+=k[c]*k[c];}
      for(int c=0;c<128;++c){q[c]/=float(std::sqrt(qs)*std::sqrt(128.));k[c]/=float(std::sqrt(ks));}
      for(int v=0;v<128;++v){float memory=0;
        for(int c=0;c<128;++c){state[v*128+c]*=params[(row*32+head)*2];memory+=state[v*128+c]*k[c];}
        float delta=(qkv[row*8192+4096+head*128+v]-memory)*params[(row*32+head)*2+1],out=0;
        for(int c=0;c<128;++c){state[v*128+c]+=k[c]*delta;out+=state[v*128+c]*q[c];}
        expected[row*4096+head*128+v]=round_bf(out);
      }
    }
  }
  CHECK_CLOSE(expected,mv::to_host(y),.0005,"MOSS recurrent state and grouped heads");
}

SLOPFAB_TEST(seedvr2_vulkan_temporal_vae) {
  if(!available()){SKIP_UNSUPPORTED_HARDWARE("Vulkan unavailable");return;}
  Fixture f;
  auto cw=data(4*2*27,71,.125f),cb=data(4,4),nw=data(32,5),nb=data(32,8);
  write_safetensors(f.path.string(),{{"conv.weight",{4,2,3,3,3},cw},{"conv.bias",{4},cb},
                     {"norm.weight",{32},nw},{"norm.bias",{32},nb}});
  sv::Runtime rt(f.path.string());rt.clear(rt.checkpoint);
  auto input=data(5*4*6*2,17);
  auto x=rt.upload(input,5,4,6,2);
  for(bool down:{false,true})for(int ts:{1,2}){
    auto out=rt.conv(x,"conv",down,ts);
    std::vector<float> expected(out.size());
    for(int t=0;t<out.t;++t)for(int h=0;h<out.h;++h)for(int w=0;w<out.w;++w)for(int co=0;co<4;++co){
      float sum=0;
      for(int ci=0;ci<2;++ci)for(int kz=0;kz<3;++kz)for(int ky=0;ky<3;++ky)for(int kx=0;kx<3;++kx){
        int z=std::max(0,t*ts+kz-2),yy=h*(down?2:1)+ky-(down?0:1),xx=w*(down?2:1)+kx-(down?0:1);
        if(z<5&&yy>=0&&yy<4&&xx>=0&&xx<6)
          sum+=input[((z*4+yy)*6+xx)*2+ci]*cw[(((co*2+ci)*3+kz)*3+ky)*3+kx];
      }
      expected[((t*out.h+h)*out.w+w)*4+co]=round_bf(round_bf(sum)+cb[co]);
    }
    CHECK_CLOSE(expected,rt.download(out),.008,"SeedVR2 causal temporal convolution and spatial padding");
  }
  auto v=data(2*3*5*32,25);
  auto norm=rt.groupnorm(rt.upload(v,2,3,5,32),"norm",true);
  std::vector<float> expected(v.size());
  for(int frame=0;frame<2;++frame)for(int c=0;c<32;++c){double mean=0,var=0;
    for(int row=0;row<15;++row)mean+=v[(frame*15+row)*32+c]/15.0;
    for(int row=0;row<15;++row)var+=std::pow(v[(frame*15+row)*32+c]-mean,2)/15.0;
    for(int row=0;row<15;++row){int i=(frame*15+row)*32+c;
      float a=round_bf(float((v[i]-mean)/std::sqrt(var+1e-6))*nw[c]+nb[c]);
      expected[i]=round_bf(a/(1+std::exp(-a)));
    }
  }
  CHECK_CLOSE(expected,rt.download(norm),.008,"SeedVR2 per-frame group norm and activation");
  for(int tr:{1,2}){
    v=data(3*2*3*8*tr,52);auto out=rt.upsample(rt.upload(v,3,2,3,8*tr),tr);
    auto actual=rt.download(out);
    for(int z=0;z<out.t;++z)for(int yy=0;yy<out.h;++yy)for(int xx=0;xx<out.w;++xx)for(int c=0;c<2;++c){
      int time=z?z+tr-1:0,ch=(((yy%2)*2+xx%2)*tr+time%tr)*2+c;
      CHECK(actual[((z*out.h+yy)*out.w+xx)*2+c]==v[((time/tr*2+yy/2)*3+xx/2)*8*tr+ch]);
    }
  }
}

SLOPFAB_TEST(seedvr2_vulkan_window_attention) {
  if(!available()){SKIP_UNSUPPORTED_HARDWARE("Vulkan unavailable");return;}
  Fixture f;
  write_safetensors(f.path.string(),{{"a.norm_q.vid.weight",{128},std::vector<float>(128,1)},
    {"a.norm_k.vid.weight",{128},std::vector<float>(128,1)},
    {"a.norm_q.txt.weight",{128},std::vector<float>(128,1)},
    {"a.norm_k.txt.weight",{128},std::vector<float>(128,1)},
    {"a.rope.rope.freqs",{21},std::vector<float>(21,.125f)}});
  sv::Runtime rt(f.path.string());rt.clear(rt.checkpoint);
  // Zero Q/K makes the expected result the window's exact V average.
  constexpr int d=256,nt=3,nv=12;
  std::vector<float> video(nv*3*d,0),text(nt*3*d,0);
  for(int r=0;r<nv;++r)for(int c=0;c<d;++c)video[r*3*d+2*d+c]=float(r+c%11);
  for(int r=0;r<nt;++r)for(int c=0;c<d;++c)text[r*3*d+2*d+c]=float(20+r+c%11);
  std::vector<seedvr2::Window> windows{{0,1,0,2,0,1},{0,1,0,2,1,3},{1,2,0,2,0,3}};
  auto vqkv=rt.upload(video,2,2,3,3*d),tqkv=rt.upload(text,1,1,nt,3*d);
  sv::Tensor vo(2,2,3,d),to(1,1,nt,d);
  rt.window_attention(vqkv,tqkv,vo,to,"a.","vid","txt",windows);
  std::vector<float> ev(nv*d),et(nt*d,0);
  for(auto win:windows){int count=(win.t1-win.t0)*(win.y1-win.y0)*(win.x1-win.x0)+nt;
    for(int c=0;c<d;++c){float sum=0;
      for(int z=win.t0;z<win.t1;++z)for(int y=win.y0;y<win.y1;++y)for(int x=win.x0;x<win.x1;++x)sum+=video[((z*2+y)*3+x)*3*d+2*d+c];
      for(int r=0;r<nt;++r)sum+=text[r*3*d+2*d+c];
      float avg=round_bf(sum/count);
      for(int z=win.t0;z<win.t1;++z)for(int y=win.y0;y<win.y1;++y)for(int x=win.x0;x<win.x1;++x)ev[((z*2+y)*3+x)*d+c]=avg;
      for(int r=0;r<nt;++r)et[r*d+c]+=avg;
    }
  }
  for(auto& a:et)a=round_bf(a/windows.size());
  CHECK_CLOSE(ev,rt.download(vo),.125,"SeedVR2 window video scatter");
  CHECK_CLOSE(et,rt.download(to),.125,"SeedVR2 FP32 window text pooling");
}
