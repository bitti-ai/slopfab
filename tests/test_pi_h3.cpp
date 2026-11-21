#include "harness.h"
#include "slopfab/dit/pi_h3.h"
#include "slopfab/pipeline.h"
#include "slopfab/generate.h"
#include "slopfab/vae/sol_h3.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace {
template<class F> bool fails(F f) { try { f(); } catch (const std::exception&) { return true; } return false; }
}

SLOPFAB_TEST(pi_h3_trained_grids) {
  using namespace slopfab;
  const auto two=dit::pi_h3_base_sigmas(2),four=dit::pi_h3_base_sigmas(4);
  CHECK(two.size()==3 && four.size()==5);
  CHECK_NEAR(two[0],.93,1e-7); CHECK_NEAR(two[1],.34875,1e-7); CHECK(two.back()==0);
  CHECK_NEAR(four[0],.96,1e-7); CHECK_NEAR(four[1],.6933333333,1e-7);
  CHECK_NEAR(four[2],.4266666667,1e-7); CHECK_NEAR(four[3],.16,1e-7); CHECK(four.back()==0);
  CHECK(fails([] { dit::pi_h3_base_sigmas(3); }));
  CHECK(dit::pi_h3_steps("weights/pi_h3_2step_dwd_adapter_int8_convrot.safetensors")==2);
  CHECK(dit::pi_h3_steps("weights/pi_h3_4step_dwd_adapter_int8_convrot.safetensors")==4);
  CHECK(dit::pi_h3_steps("2step/renamed.safetensors")==0);
  sampler::FlowScheduler schedule(12); schedule.set_base_sigmas(two);
  CHECK_NEAR(schedule.sigmas()[0],12*.93/(1+11*.93),1e-7);
  CHECK_NEAR(schedule.sigmas()[1],12*.34875/(1+11*.34875),1e-7);
  CHECK_NEAR(schedule.timesteps().back(),1-schedule.sigmas()[1],1e-7);
  GeneratePlan plan; plan.pi_h3_steps=2;
  CHECK(fails([&] { validate_sampling_sampler(plan,sampler::SamplerKind::kRenoise); }));
  GenerateRequest req; RunOptions opts; opts.save_latents_path="unusable.safetensors";
  CHECK(fails([&] { validate_generation_options(req,plan,opts); }));
}

SLOPFAB_TEST(pi_h3_convrot_inverse_matches_tensor_product) {
  // Independent dense Kronecker reference, including signs and 1/sqrt(256).
  const int h[4][4]={{1,1,1,-1},{1,1,-1,1},{1,-1,1,1},{-1,1,1,1}};
  std::vector<float> input(512),actual;
  for (size_t i=0;i<input.size();++i) input[i]=float(int(i%17)-8)*.03125f;
  actual=input; slopfab::dit::pi_h3_unrotate(actual.data(),actual.size());
  for (int b=0;b<2;++b) for (int i=0;i<256;++i) {
    double expected=0;
    for (int j=0;j<256;++j) {
      int sign=1;
      for (int digit=0;digit<4;++digit) sign*=h[(i>>(digit*2))&3][(j>>(digit*2))&3];
      expected+=sign*input[b*256+j]/16;
    }
    CHECK_NEAR(actual[b*256+i],expected,1e-6);
  }
  slopfab::dit::pi_h3_unrotate(actual.data(),actual.size());
  for (size_t i=0;i<input.size();++i) CHECK_NEAR(actual[i],input[i],1e-6);
  CHECK(fails([&] { slopfab::dit::pi_h3_unrotate(actual.data(),255); }));
}

SLOPFAB_TEST(pi_h3_feature_expansion_and_first_frame_pixels) {
  using namespace slopfab;
  constexpr int T=7,H=2,W=3,N=T*H*W;
  std::vector<float> input(N*8192),expanded(N*8192,-1);
  for (int r=0;r<N;++r) for (int sub=0;sub<4;++sub)
    std::fill_n(input.data()+r*8192+sub*2048,2048,float(r*4+sub));
  // Split in the middle of a frame to check chunk offsets/reordering.
  dit::pi_h3_expand_rows(input.data(),expanded.data(),0,11,H,W);
  dit::pi_h3_expand_rows(input.data()+11*8192,expanded.data(),11,N-11,H,W);
  for (int t=0;t<T;++t) for (int y=0;y<H*2;++y) for (int x=0;x<W*2;++x) {
    const auto row=(t*H*2+y)*W*2+x;
    const float expected=float(((t*H+y/2)*W+x/2)*4+(y%2)*2+x%2);
    CHECK(expanded[row*2048]==expected); CHECK(expanded[row*2048+2047]==expected);
  }
  vae::DecodedVideo out;
  out.frames=vae::sol_h3_frames(T); out.height=H*16; out.width=W*16;
  out.data.resize(size_t(3)*out.frames*out.height*out.width,-1);
  std::vector<float> regular(N*3072,0),first(N*768,1);
  dit::pi_h3_pixels(regular.data(),0,N,H,W,false,out);
  dit::pi_h3_pixels(first.data(),0,N,H,W,true,out);
  const float means[]={.485f,.456f,.406f},stds[]={.229f,.224f,.225f};
  for (int c=0;c<3;++c) for (int f=0;f<out.frames;++f) {
    const float expected=means[c]+((f==0||f==17)?stds[c]:0);
    for (int p=0;p<out.height*out.width;++p)
      CHECK_NEAR(out.data[(size_t(c)*out.frames+f)*out.height*out.width+p],expected,1e-6);
  }
  std::fill(first.begin(),first.end(),100);
  dit::pi_h3_pixels(first.data(),0,N,H,W,true,out);
  CHECK(out.data[0]==1);
}

SLOPFAB_TEST(pi_h3_output_norm_uses_scale_then_shift_and_activated_table_code) {
  using namespace slopfab;
  constexpr int D=5376;
  std::vector<float> weight(2*D*8,0),bias(2*D,0),hidden(2*D,7);
  for (int c=0;c<D;++c) {
    weight[c*8]=2; weight[(D+c)*8+1]=3;
    bias[c]=.25f; bias[D+c]=-.5f;
    hidden[D+c]=c%2 ? 3.f : -1.f;
  }
  auto st=SafeTensors::borrowed({
      {"dwd_norm_out.linear.weight",{"dwd_norm_out.linear.weight",DType::kF32,{2*D,8},weight.data(),weight.size()*4}},
      {"dwd_norm_out.linear.bias",{"dwd_norm_out.linear.bias",DType::kF32,{2*D},bias.data(),bias.size()*4}}}, {}, "norm");
  const auto out=dit::pi_h3_modulate(st,hidden.data(),2,{.5f,.25f,0,0,0,0,0,0});
  for (int c=0;c<D;++c) {
    CHECK_NEAR(out[c],.25f,1e-6);
    CHECK_NEAR(out[D+c],(c%2?2.f:-2.f)/std::sqrt(4.f+1e-5f)*2.25f+.25f,1e-6);
  }
}

SLOPFAB_TEST(pi_h3_local_adapter_contract) {
  using namespace slopfab;
  const char* adapter_path=std::getenv("SLOPFAB_PI_H3_ADAPTER");
  const char* base_path=std::getenv("SLOPFAB_PI_H3_BASE");
  if (!adapter_path || !base_path) { SKIP_OPT_IN("set SLOPFAB_PI_H3_ADAPTER and SLOPFAB_PI_H3_BASE"); return; }
  SafeTensors adapter,base;
  adapter.open(adapter_path); base.open(base_path);
  CHECK(dit::is_pi_h3(adapter)); dit::validate_pi_h3(adapter);
  auto overlay=dit::pi_h3_overlay(base,adapter);
  CHECK(overlay.mapping_base()==nullptr && overlay.is_open());
  CHECK(overlay.at("blocks.46.mlp.fc1.weight").data==adapter.at("blocks.46.mlp.fc1.weight").data);
  CHECK(overlay.at("blocks.45.mlp.fc1.weight").data==base.at("blocks.45.mlp.fc1.weight").data);
  CHECK(!overlay.find("dwd_norm_out.linear.weight"));
  CHECK(!overlay.find("blocks.46.mlp.fc1.weight_scale_2"));
  CHECK(overlay.at("blocks.46.mlp.fc1.weight").dtype==DType::kI8);
  GenerateRequest req; req.transformer_path=base_path; req.loras={{adapter_path,1}};
  const auto plan=resolve_plan(req);
  CHECK(plan.pi_h3_steps==dit::pi_h3_steps(adapter_path));
  CHECK(plan.num_model_evaluations()==plan.pi_h3_steps);
  CHECK(plan.video_sigmas.front()<1 && plan.audio_sigmas.front()<1);
  LoraAdapters loaded; loaded.load(req.loras,base);
  CHECK(bool(loaded.pi_h3()) && loaded.projection_count()==138);
  CHECK(loaded.find("blocks.45.mlp.fc1") && !loaded.find("blocks.46.mlp.fc1"));
  CHECK(loaded.find("blocks.0.attn.out_proj")->front().rank==64);
  req.num_inference_steps=3;
  CHECK(fails([&] { resolve_plan(req); }));
  req.num_inference_steps=0; req.loras[0].strength=.5f;
  CHECK(fails([&] { resolve_plan(req); }));
  req.loras[0].strength=1; req.sampling.video_sigma_shift=3;
  CHECK(fails([&] { resolve_plan(req); }));
  req.sampling.video_sigma_shift.reset(); req.skip_every=2;
  CHECK(fails([&] { resolve_plan(req); }));
  auto views=adapter.tensors(); views["unknown.weight"]=views.begin()->second;
  auto bad=SafeTensors::borrowed(std::move(views),adapter.metadata(),"bad");
  CHECK(fails([&] { dit::validate_pi_h3(bad); }));
}
