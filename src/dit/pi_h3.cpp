// pi-H3: Lakonik/ComfyUI-piFlow bcb6ffb8ff6191115e5803c4ddddef6a73b0b8a8.
#include "slopfab/dit/pi_h3.h"
#include "slopfab/dit/checkpoint.h"
#include "slopfab/tensor_convert.h"
#include "slopfab/json.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <set>
#include <stdexcept>

namespace slopfab::dit {
namespace {
bool starts(const std::string& s, const std::string& p) { return s.compare(0, p.size(), p) == 0; }
bool floating(DType t) { return t == DType::kF16 || t == DType::kBF16 || t == DType::kF32; }
void quant(const SafeTensors& st, const std::string& p) {
  const auto& tag = st.at(p + ".comfy_quant");
  if (tag.dtype != DType::kU8) throw std::runtime_error("pi-H3: invalid quantization tag " + p);
  const auto j = json::parse(std::string(static_cast<const char*>(tag.data), tag.nbytes));
  const auto* f = j.find("format"), *r = j.find("convrot"), *g = j.find("convrot_groupsize");
  if (!f || f->as_string() != "int8_tensorwise" || !r || !r->as_bool() ||
      !g || g->as_number() != 256)
    throw std::runtime_error("pi-H3: expected INT8 ConvRot group 256 for " + p);
}
} // namespace

bool is_pi_h3(const SafeTensors& st) {
  return st.find("dwd_norm_out.linear.weight") || st.find("dwd_pixel_refiner.feature_expand.weight");
}
void validate_pi_h3(const SafeTensors& st) {
  std::set<std::string> expected;
  auto shape = [&](const std::string& name, std::vector<int64_t> dims, bool fp = true) {
    const auto& v = st.at(name);
    if (v.shape != dims || (fp && !floating(v.dtype)))
      throw std::runtime_error("pi-H3: invalid tensor " + name);
    expected.insert(name);
  };
  auto linear = [&](const std::string& p, int out, int in, bool bias, bool int8) {
    shape(p + ".weight", {out, in}, !int8);
    if (bias) shape(p + ".bias", {out});
    if (int8) {
      if (st.at(p + ".weight").dtype != DType::kI8)
        throw std::runtime_error("pi-H3: expected INT8 " + p);
      shape(p + ".weight_scale", {out, 1});
      quant(st, p); expected.insert(p + ".comfy_quant");
    }
  };
  for (int i = 0; i < 50; ++i) {
    const auto p = "blocks." + std::to_string(i) + ".";
    linear(p + "adaln_proj.linear", 96768, 8, i >= 46, false);
    for (const auto& m : std::vector<std::pair<std::string, std::pair<int,int>>>{
        {"attn.out_proj", {5376,7168}}, {"mlp.fc1", {28672,5376}}, {"mlp.fc2", {5376,14336}}}) {
      if (i >= 46) linear(p + m.first, m.second.first, m.second.second, false, true);
      else {
        shape(p + m.first + ".lora_A.weight", {64, m.second.second});
        shape(p + m.first + ".lora_B.weight", {m.second.first, 64});
      }
    }
  }
  shape("final_layer.norm.weight", {5376});
  linear("final_layer.adaln_proj.linear", 10752, 8, true, false);
  linear("final_layer.video_out", 96, 5376, true, false);
  linear("final_layer.audio_out", 32, 5376, true, false);
  linear("dwd_norm_out.linear", 10752, 8, true, false);
  const std::string p = "dwd_pixel_refiner.";
  linear(p + "feature_expand", 8192, 5376, true, false);
  shape(p + "register_tokens", {1,4,2048});
  shape(p + "norm_out.weight", {2048}); shape(p + "norm_out.bias", {2048});
  linear(p + "proj_out", 3072, 2048, true, false);
  linear(p + "first_frame_proj_out", 768, 2048, true, false);
  for (int i = 0; i < 8; ++i) {
    const auto b = p + "blocks." + std::to_string(i) + ".";
    for (const auto* s : {"norm1.weight", "norm2.weight", "scale1", "scale2"}) shape(b + s, {2048});
    linear(b + "attn.to_qkv", 6144, 2048, true, true);
    linear(b + "attn.to_out", 2048, 2048, true, true);
    linear(b + "ff.w1", 16384, 2048, true, true);
    linear(b + "ff.w2", 2048, 8192, true, true);
  }
  for (const auto& [name, view] : st.tensors())
    if (!expected.count(name)) throw std::runtime_error("pi-H3: unsupported tensor " + name);
  const auto it = st.metadata().find("policy_config");
  if (it == st.metadata().end()) throw std::runtime_error("pi-H3: missing DX policy_config");
  const auto policy = json::parse(it->second);
  for (const char* modality : {"image", "audio"}) {
    const auto* m = policy.find(modality);
    if (!m || !m->find("mode") || !m->find("type") ||
        m->find("mode")->as_string() != "grid" || m->find("type")->as_string() != "DX")
      throw std::runtime_error("pi-H3: only grid DX policies are supported");
  }
}
void validate_pi_h3_base(const SafeTensors& st) {
  const auto m = resolve_model_descriptor(st);
  if (m.compatibility_architecture != TransformerArchitecture::kPrunedTable ||
      m.family != "h3" || m.compressed_attention || m.pdd_steps != 1 ||
      st.at("adaln_t_table").shape != std::vector<int64_t>{1025,8})
    throw std::runtime_error("pi-H3 requires a standard rank-8 pruned FL2VA H3 base");
}
int pi_h3_steps(const std::string& path) {
  const auto name = std::filesystem::path(path).filename().string();
  const bool two = name.find("2step") != std::string::npos, four = name.find("4step") != std::string::npos;
  return two == four ? 0 : two ? 2 : 4;
}
std::vector<float> pi_h3_base_sigmas(int steps) {
  if (steps != 2 && steps != 4) throw std::invalid_argument("pi-H3 requires its trained 2 or 4 steps");
  const double start = steps == 2 ? .93 : .96, end = start * .6 / (steps - 1 + .6);
  std::vector<float> out;
  for (int i = 0; i < steps; ++i) out.push_back(float(start + (end - start) * i / (steps - 1)));
  out.push_back(0); return out;
}
SafeTensors pi_h3_overlay(const SafeTensors& base, const SafeTensors& adapter) {
  validate_pi_h3_base(base); validate_pi_h3(adapter);
  std::map<std::string, TensorView> views;
  for (const auto& [name, v] : base.tensors()) {
    auto key = name;
    const std::string prefix = "model.diffusion_model.";
    if (starts(key, prefix)) key.erase(0, prefix.size());
    auto view = v; view.name = key; views.emplace(key, std::move(view));
  }
  // Remove every old quantizer companion when replacing a matrix, including
  // packed NVFP4/W4A8 representations. Never retain a stale scale/input scale.
  for (const auto& [name, v] : adapter.tensors()) {
    if (starts(name, "dwd_") || name.find(".lora_") != std::string::npos) continue;
    if (name.size() >= 7 && name.substr(name.size() - 7) == ".weight") {
      const auto p = name.substr(0, name.size() - 6);
      for (auto it = views.begin(); it != views.end();)
        if (starts(it->first, p) && it->first != p + "bias") it = views.erase(it);
        else ++it;
    }
  }
  for (const auto& [name, v] : adapter.tensors())
    if (!starts(name, "dwd_") && name.find(".lora_") == std::string::npos) views[name] = v;
  return SafeTensors::borrowed(std::move(views), base.metadata(), base.path());
}
void pi_h3_unrotate(float* v, size_t count) {
  if (!v || count % 256) throw std::invalid_argument("pi-H3: invalid ConvRot span");
  for (size_t b = 0; b < count; b += 256) {
    for (size_t stride = 1; stride < 256; stride *= 4)
      for (size_t group = 0; group < 256; group += 4 * stride)
        for (size_t j = 0; j < stride; ++j) {
          auto* p = v + b + group + j;
          const float a=p[0], c=p[2*stride], d=p[3*stride], bb=p[stride];
          p[0]=a+bb+c-d; p[stride]=a+bb-c+d;
          p[2*stride]=a-bb+c+d; p[3*stride]=-a+bb+c+d;
        }
    for (size_t j=0;j<256;++j) v[b+j] *= .0625f;
  }
}
std::vector<float> pi_h3_weight(const SafeTensors& st, const std::string& name) {
  const auto& v = st.at(name);
  if (floating(v.dtype)) return to_f32(v);
  if (v.dtype != DType::kI8 || v.shape.size() != 2 || v.shape[1] % 256)
    throw std::runtime_error("pi-H3: unsupported pixel weight " + name);
  const auto p = name.substr(0, name.size()-7);
  quant(st,p);
  const auto scales = to_f32(st.at(p + ".weight_scale"));
  if (scales.size() != size_t(v.shape[0])) throw std::runtime_error("pi-H3: invalid scales");
  std::vector<float> out(v.numel());
  for (size_t row=0;row<scales.size();++row) {
    if (!std::isfinite(scales[row]) || scales[row] <= 0) throw std::runtime_error("pi-H3: invalid scale");
    for (int64_t c=0;c<v.shape[1];++c) {
      const size_t i=row*v.shape[1]+c;
      out[i]=static_cast<const int8_t*>(v.data)[i]*scales[row];
    }
  }
  pi_h3_unrotate(out.data(),out.size()); return out;
}
std::vector<float> pi_h3_modulate(const SafeTensors& st, const float* x, size_t rows,
                                 const std::array<float,8>& code) {
  constexpr int D=5376;
  auto mod=to_f32(st.at("dwd_norm_out.linear.bias"));
  const auto w=to_f32(st.at("dwd_norm_out.linear.weight"));
  for (int i=0;i<2*D;++i) for (int j=0;j<8;++j) mod[i]+=w[i*8+j]*code[j];
  std::vector<float> out(rows*D);
  for (size_t r=0;r<rows;++r) {
    double mean=0,ss=0;
    for (int c=0;c<D;++c) mean+=x[r*D+c]; mean/=D;
    for (int c=0;c<D;++c) { const double v=x[r*D+c]-mean; ss+=v*v; }
    const float inv=float(1/std::sqrt(ss/D+1e-5));
    for (int c=0;c<D;++c) out[r*D+c]=(x[r*D+c]-float(mean))*inv*(1+mod[c])+mod[D+c];
  }
  return out;
}
void pi_h3_expand_rows(const float* expanded, float* rows, int begin, int count, int H, int W) {
  for (int r=0;r<count;++r) {
    const int row=begin+r,t=row/(H*W),y=(row/W)%H,x=row%W;
    for (int dy=0;dy<2;++dy) for (int dx=0;dx<2;++dx) for (int c=0;c<2048;++c)
      rows[((size_t(t)*H*2+y*2+dy)*W*2+x*2+dx)*2048+c]=
        bf16_to_f32(f32_to_bf16(expanded[size_t(r)*8192+(dy*2+dx)*2048+c]));
  }
}
void pi_h3_pixels(const float* patches, int begin, int count, int H, int W,
                  bool first_frame, vae::DecodedVideo& out) {
  const int phases=first_frame?1:4,dim=phases*768;
  const float mean[]={.485f,.456f,.406f},stddev[]={.229f,.224f,.225f};
  for (int r=0;r<count;++r) {
    const int row=begin+r,t=row/(H*W),y=(row/W)%H,x=row%W;
    if ((t%5==0)!=first_frame) continue;
    for (int c=0;c<3;++c) for (int phase=0;phase<phases;++phase)
      for (int py=0;py<16;++py) for (int px=0;px<16;++px) {
        const int frame=t*4-3*(t/5+1)+(first_frame?3:phase);
        const auto dst=((size_t(c)*out.frames+frame)*out.height+y*16+py)*out.width+x*16+px;
        const auto src=size_t(r)*dim+((c*phases+phase)*16+py)*16+px;
        if (!std::isfinite(patches[src])) throw std::runtime_error("pi-H3: non-finite pixel prediction");
        out.data[dst]=std::clamp(patches[src]*stddev[c]+mean[c],0.f,1.f);
      }
  }
}
} // namespace slopfab::dit
