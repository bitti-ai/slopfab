// SeedVR2 architecture port based on ByteDance-Seed/SeedVR (Apache-2.0).
// See third_party/seedvr2/NOTICE for pinned reference provenance.
#include "runtime.cuh"
#include "slopfab/tensor_convert.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <cstdlib>
#include <filesystem>
#include "slopfab/safetensors_write.h"

namespace slopfab::seedvr2 {
namespace {
void expect(const SafeTensors& f, const std::string& name, std::vector<int64_t> shape) {
  if (f.at(name).shape != shape)
    throw std::runtime_error("SeedVR2: incompatible tensor " + name);
}

std::vector<std::pair<int, int>> tiles(int extent, int tile) {
  if (!tile || extent <= tile)
    return {{0, extent}};
  std::vector<std::pair<int, int>> out;
  for (int p = 0;; p += tile - 64) {
    if (p + tile >= extent) {
      out.emplace_back(extent - tile, extent);
      break;
    }
    out.emplace_back(p, p + tile);
  }
  return out;
}

float feather(int p, int lo, int hi, int extent, int overlap) {
  float a = 1;
  if (lo > 0)
    a = std::min(a, float(p - lo + 1) / overlap);
  if (hi < extent)
    a = std::min(a, float(hi - p) / overlap);
  return a;
}
}

struct Restorer::Impl {
  Options o;
  SafeTensors dit, vae;
  std::unique_ptr<Runtime> rt;
  Restorer* owner;
  std::string capture_dir;

  Impl(const Options& options, Restorer* parent) : o(options), owner(parent) {
    validate(o);
    if (const char* capture = std::getenv("SLOPFAB_SEEDVR2_CAPTURE_DIR")) {
      capture_dir = capture;
      std::filesystem::create_directories(std::filesystem::u8path(capture_dir));
    }
    dit.open(o.transformer);
    vae.open(o.vae);
    expect(dit, "vid_in.proj.weight", {2560, 132});
    expect(dit, "vid_out.proj.weight", {64, 2560});
    expect(dit, "emb_in.proj_out.weight", {15360, 2560});
    expect(dit, "vid_out_norm.weight", {2560});
    expect(dit, "txt_in.weight", {2560, 5120});
    expect(dit, "vid_out_ada.out_shift", {2560});
    expect(dit, "vid_out_ada.out_scale", {2560});
    for (int b = 0; b < 32; ++b) {
      const auto p = "blocks." + std::to_string(b) + ".";
      for (const auto& branch : (b < 10 ? std::vector<std::string>{"vid", "txt"} : std::vector<std::string>{"all"})) {
        expect(dit, p + "attn.proj_qkv." + branch + ".weight", {7680, 2560});
        expect(dit, p + "attn.proj_out." + branch + ".weight", {2560, 2560});
        expect(dit, p + "attn.proj_out." + branch + ".bias", {2560});
        expect(dit, p + "attn.norm_q." + branch + ".weight", {128});
        expect(dit, p + "attn.norm_k." + branch + ".weight", {128});
        for (const auto& layer : {"attn", "mlp"})
          for (const auto& mode : {"shift", "scale", "gate"})
            expect(dit, p + "ada." + branch + "." + layer + "_" + mode, {2560});
        expect(dit, p + "mlp." + branch + ".proj_in.weight", {6912, 2560});
        expect(dit, p + "mlp." + branch + ".proj_in_gate.weight", {6912, 2560});
        expect(dit, p + "mlp." + branch + ".proj_out.weight", {2560, 6912});
      }
      expect(dit, p + "attn.rope.rope.freqs", {21});
    }
    const auto& text = dit.at("positive_conditioning");
    if (text.shape.size() != 2 || text.shape[0] < 1 || text.shape[0] > 512 || text.shape[1] != 5120)
      throw std::runtime_error("SeedVR2: missing or incompatible embedded positive conditioning");
    for (const auto& item : dit.tensors()) {
      if (item.first.find(".comfy_quant") != std::string::npos)
        throw std::runtime_error(
            "SeedVR2: use the Comfy-Org 3B FP16 or FP8 checkpoint; packed quantization is unsupported");
    }
    expect(vae, "encoder.conv_in.weight", {128, 3, 3, 3, 3});
    expect(vae, "decoder.conv_in.weight", {512, 16, 3, 3, 3});
    cuda::set_device(o.device);
    rt = std::make_unique<Runtime>();
  }

  void capture(const std::string& name, const std::vector<float>& data, std::vector<int64_t> shape) {
    if (!capture_dir.empty())
      write_safetensors((std::filesystem::u8path(capture_dir) / (name + ".safetensors")).u8string(),
                        {{name, std::move(shape), data}});
  }
  void capture(const std::string& name, const Tensor& tensor) {
    if (!capture_dir.empty()) capture(name, rt->download(tensor), {tensor.t, tensor.h, tensor.w, tensor.c});
  }

  void progress(const std::string& message) {
    if (owner->cancelled && owner->cancelled())
      throw std::runtime_error("SeedVR2: cancelled");
    if (owner->progress)
      owner->progress(message);
  }

  Tensor resnet(Tensor x, const std::string& p) {
    auto h = rt->groupnorm(x, p + ".norm1");
    h = rt->conv(h, p + ".conv1");
    h = rt->groupnorm(h, p + ".norm2");
    h = rt->conv(h, p + ".conv2");
    if (vae.find(p + ".conv_shortcut.weight"))
      x = rt->conv(x, p + ".conv_shortcut");
    rt->add(h, x);
    return h;
  }

  Tensor mid(Tensor x, const std::string& p) {
    x = resnet(std::move(x), p + ".resnets.0");
    const std::string a = p + ".attentions.0";
    auto norm = rt->groupnorm(x, a + ".group_norm", false);
    // Per-frame spatial attention, never across time.
    for (int t = 0; t < x.t; ++t) {
      Tensor frame(1, x.h, x.w, x.c);
      SLOPFAB_CUDA_CHECK(cudaMemcpy(frame.data.get(), norm.data.get() + size_t(t) * frame.size(),
                                    frame.size() * sizeof(BFloat), cudaMemcpyDeviceToDevice));
      auto q = rt->linear(frame, a + ".to_q"), k = rt->linear(frame, a + ".to_k"),
           v = rt->linear(frame, a + ".to_v");
      auto out = rt->attention(q, k, v, 1, 512);
      out = rt->linear(out, a + ".to_out.0");
      SLOPFAB_CUDA_CHECK(cudaMemcpy(norm.data.get() + size_t(t) * frame.size(), out.data.get(),
                                    frame.size() * sizeof(BFloat), cudaMemcpyDeviceToDevice));
    }
    rt->add(x, norm);
    norm = Tensor();
    return resnet(std::move(x), p + ".resnets.1");
  }

  Tensor encode(Tensor x) {
    x = rt->conv(x, "encoder.conv_in");
    for (int b = 0; b < 4; ++b) {
      for (int j = 0; j < 2; ++j)
        x = resnet(std::move(x),
                   "encoder.down_blocks." + std::to_string(b) + ".resnets." + std::to_string(j));
      if (b < 3)
        x = rt->conv(x, "encoder.down_blocks." + std::to_string(b) + ".downsamplers.0.conv", true,
                     b ? 2 : 1);
    }
    x = mid(std::move(x), "encoder.mid_block");
    x = rt->groupnorm(x, "encoder.conv_norm_out");
    return rt->conv(x, "encoder.conv_out");
  }

  Tensor decode(Tensor x) {
    x = rt->conv(x, "decoder.conv_in");
    x = mid(std::move(x), "decoder.mid_block");
    for (int b = 0; b < 4; ++b) {
      for (int j = 0; j < 3; ++j)
        x = resnet(std::move(x),
                   "decoder.up_blocks." + std::to_string(b) + ".resnets." + std::to_string(j));
      if (b < 3) {
        auto p = "decoder.up_blocks." + std::to_string(b) + ".upsamplers.0";
        x = rt->conv(x, p + ".upscale_conv");
        x = rt->upsample(x, b < 2 ? 2 : 1);
        x = rt->conv(x, p + ".conv");
      }
    }
    x = rt->groupnorm(x, "decoder.conv_norm_out");
    return rt->conv(x, "decoder.conv_out");
  }

  std::vector<float> encode_tiled(const std::vector<Frame>& frames, int h, int w) {
    rt->clear(vae);
    const int t = int(frames.size()), lt = (t - 1) / 4 + 1, lh = h / 8, lw = w / 8;
    std::vector<float> moments(size_t(lt) * lh * lw * 32, 0), coverage(size_t(lh) * lw, 0);
    auto ys = tiles(h, o.vae_tile), xs = tiles(w, o.vae_tile);
    int tile = 0;
    for (auto y : ys)
      for (auto x : xs) {
        progress("VAE encode tile " + std::to_string(++tile) + "/" +
                 std::to_string(ys.size() * xs.size()));
        int th = y.second - y.first, tw = x.second - x.first;
        std::vector<float> pixels(size_t(t) * th * tw * 3);
        for (int z = 0; z < t; ++z)
          for (int yy = 0; yy < th; ++yy)
            for (int xx = 0; xx < tw; ++xx)
              for (int c = 0; c < 3; ++c) {
                int sy = std::min(y.first + yy, o.height - 1),
                    sx = std::min(x.first + xx, o.width - 1);
                pixels[((size_t(z) * th + yy) * tw + xx) * 3 + c] =
                    2 * frames[z][(size_t(sy) * o.width + sx) * 3 + c] - 1;
              }
      auto encoded = encode(rt->upload(pixels, t, th, tw, 3));
      auto values = rt->download(encoded);
      if (tile == 1) {
        capture("vae_input", pixels, {t, th, tw, 3});
        capture("vae_moments", values, {encoded.t, encoded.h, encoded.w, encoded.c});
      }
        for (int yy = y.first / 8; yy < y.second / 8; ++yy)
          for (int xx = x.first / 8; xx < x.second / 8; ++xx) {
            float weight = feather(yy, y.first / 8, y.second / 8, lh, 8) *
                           feather(xx, x.first / 8, x.second / 8, lw, 8);
            coverage[size_t(yy) * lw + xx] += weight;
            for (int z = 0; z < lt; ++z)
              for (int c = 0; c < 32; ++c)
                moments[((size_t(z) * lh + yy) * lw + xx) * 32 + c] +=
                    weight * values[((size_t(z) * (th / 8) + yy - y.first / 8) * (tw / 8) + xx -
                                     x.first / 8) *
                                        32 +
                                    c];
          }
      }
    for (size_t i = 0; i < moments.size(); ++i)
      moments[i] /= coverage[(i / 32) % (size_t(lh) * lw)];
    return moments;
  }

  std::vector<Frame> decode_tiled(const std::vector<float>& latent, int t, int h, int w) {
    rt->clear(vae);
    const int lt = (t - 1) / 4 + 1, lh = h / 8, lw = w / 8;
    std::vector<Frame> result(t, Frame(size_t(o.height) * o.width * 3, 0));
    std::vector<float> coverage(size_t(o.height) * o.width, 0);
    auto ys = tiles(h, o.vae_tile), xs = tiles(w, o.vae_tile);
    int tile = 0;
    for (auto y : ys)
      for (auto x : xs) {
        progress("VAE decode tile " + std::to_string(++tile) + "/" +
                 std::to_string(ys.size() * xs.size()));
        int th = (y.second - y.first) / 8, tw = (x.second - x.first) / 8;
        std::vector<float> values(size_t(lt) * th * tw * 16);
        for (int z = 0; z < lt; ++z)
          for (int yy = 0; yy < th; ++yy)
            for (int xx = 0; xx < tw; ++xx)
              for (int c = 0; c < 16; ++c)
                values[((size_t(z) * th + yy) * tw + xx) * 16 + c] =
                    latent[((size_t(z) * lh + yy + y.first / 8) * lw + xx + x.first / 8) * 16 + c] /
                    0.9152f;
      auto decoded = decode(rt->upload(values, lt, th, tw, 16));
      auto pixels = rt->download(decoded);
      if (tile == 1) {
        capture("vae_latent", values, {lt, th, tw, 16});
        capture("vae_decoded", pixels, {decoded.t, decoded.h, decoded.w, decoded.c});
      }
        for (int yy = y.first; yy < std::min(y.second, o.height); ++yy)
          for (int xx = x.first; xx < std::min(x.second, o.width); ++xx) {
            float weight =
                feather(yy, y.first, y.second, h, 64) * feather(xx, x.first, x.second, w, 64);
            size_t pos = size_t(yy) * o.width + xx;
            coverage[pos] += weight;
            for (int z = 0; z < t; ++z)
              for (int c = 0; c < 3; ++c)
                result[z][pos * 3 + c] +=
                    weight *
                    pixels[((size_t(z) * th * 8 + yy - y.first) * tw * 8 + xx - x.first) * 3 + c];
          }
      }
    for (auto& frame : result)
      for (size_t i = 0; i < frame.size(); ++i)
        frame[i] = frame[i] / coverage[i / 3] * 0.5f + 0.5f;
    return result;
  }

  std::vector<float> denoise(const std::vector<float>& moments, int t, int h, int w,
                             uint64_t first) {
    rt->clear(dit);
    const int n = t * h * w, ph = h / 2, pw = w / 2;
    std::mt19937_64 rng(o.seed + first);
    std::normal_distribution<float> gaussian;
    std::vector<float> noise(size_t(n) * 16), cond(size_t(n) * 16);
    for (int i = 0; i < n; ++i)
      for (int c = 0; c < 16; ++c) {
        float logvar = std::clamp(moments[size_t(i) * 32 + c + 16], -30.0f, 20.0f);
        cond[size_t(i) * 16 + c] =
            (moments[size_t(i) * 32 + c] + std::exp(0.5f * logvar) * gaussian(rng)) * 0.9152f;
      }
    for (auto& v : noise)
      v = gaussian(rng);
    std::vector<float> patches(size_t(t) * ph * pw * 132);
    for (int z = 0; z < t; ++z)
      for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
          size_t in = (size_t(z) * h + y) * w + x,
                 out = (((size_t(z) * ph + y / 2) * pw + x / 2) * 4 + (y % 2) * 2 + x % 2) * 33;
          for (int c = 0; c < 16; ++c) {
            patches[out + c] = noise[in * 16 + c];
            patches[out + 16 + c] = cond[in * 16 + c];
          }
          patches[out + 32] = 1;
        }
    capture("dit_patches", patches, {t, ph, pw, 132});
    Tensor video = rt->linear(rt->upload(patches, t, ph, pw, 132), "vid_in.proj");
    auto& txt = dit.at("positive_conditioning");
    Tensor text = rt->linear(rt->upload(to_f32(txt), 1, 1, int(txt.shape[0]), 5120), "txt_in");
    std::vector<float> sinusoid(256);
    for (int i = 0; i < 128; ++i) {
      float phase = 1000.0f * std::exp(-std::log(10000.0f) * i / 128);
      sinusoid[i] = std::sin(phase);
      sinusoid[i + 128] = std::cos(phase);
    }
    Tensor emb = rt->linear(rt->upload(sinusoid, 1, 1, 1, 256), "emb_in.proj_in");
    rt->activation(emb);
    emb = rt->linear(emb, "emb_in.proj_hid");
    rt->activation(emb);
    emb = rt->linear(emb, "emb_in.proj_out");
    capture("dit_embedding", emb);
    capture("dit_video_in", video);
    capture("dit_text_in", text);
    const auto regular = attention_windows(t, ph, pw, false),
               shifted = attention_windows(t, ph, pw, true);
    for (int b = 0; b < 32; ++b) {
      progress("DiT block " + std::to_string(b + 1) + "/32");
      rt->clear(dit); // Bound weight residency to one block, also on smaller GPUs.
      std::string p = "blocks." + std::to_string(b) + ".", vb = b < 10 ? "vid" : "all",
                  tb = b < 10 ? "txt" : "all";
      auto vn = rt->rms(video), tn = rt->rms(text);
      rt->modulate(vn, emb, p + "ada." + vb + ".attn", 0, false);
      // The final block's Ada MMModule is video-only for attention as well as
      // the MLP. Its text K/V use unmodulated RMS-normalized text.
      if (b != 31) rt->modulate(tn, emb, p + "ada." + tb + ".attn", 0, false);
      auto vqkv = rt->linear(vn, p + "attn.proj_qkv." + vb),
           tqkv = rt->linear(tn, p + "attn.proj_qkv." + tb);
      rt->window_attention(vqkv, tqkv, vn, tn, p + "attn.", vb, tb, b % 2 ? shifted : regular);
      vqkv = Tensor();
      tqkv = Tensor();
      vn = rt->linear(vn, p + "attn.proj_out." + vb);
      tn = rt->linear(tn, p + "attn.proj_out." + tb);
      rt->modulate(vn, emb, p + "ada." + vb + ".attn", 0, true);
      if (b != 31) rt->modulate(tn, emb, p + "ada." + tb + ".attn", 0, true);
      rt->add(video, vn);
      rt->add(text, tn);
      vn = Tensor();
      tn = Tensor();
      auto mlp = [&](Tensor& x, const std::string& branch) {
        auto norm = rt->rms(x);
        rt->modulate(norm, emb, p + "ada." + branch + ".mlp", 1, false);
        auto hidden = rt->linear(norm, p + "mlp." + branch + ".proj_in");
        auto gate = rt->linear(norm, p + "mlp." + branch + ".proj_in_gate");
        norm = Tensor();
        rt->activation(hidden, &gate);
        gate = Tensor();
        hidden = rt->linear(hidden, p + "mlp." + branch + ".proj_out");
        rt->modulate(hidden, emb, p + "ada." + branch + ".mlp", 1, true);
        rt->add(x, hidden);
      };
      mlp(video, vb);
      if (b != 31)
        mlp(text, tb);
      capture("dit_block_" + std::to_string(b), video);
      capture("dit_text_" + std::to_string(b), text);
    }
    rt->clear(dit);
    video = rt->rms(video, "vid_out_norm.weight");
    // The upstream cache aliases output modulation with the first (attention)
    // slice. Re-slicing emb as a one-layer AdaSingle would have the wrong width.
    rt->modulate(video, emb, "vid_out_ada.out", 0, false);
    video = rt->linear(video, "vid_out.proj");
    auto prediction = rt->download(video);
    capture("dit_prediction", prediction, {t, ph, pw, 64});
    for (int z = 0; z < t; ++z)
      for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
          for (int c = 0; c < 16; ++c) {
            size_t out = ((size_t(z) * h + y) * w + x) * 16 + c;
            size_t in =
                (((size_t(z) * ph + y / 2) * pw + x / 2) * 4 + (y % 2) * 2 + x % 2) * 16 + c;
            noise[out] -= prediction[in]; // One Euler step, sigma 1 -> 0, CFG 1.
          }
    return noise;
  }
};

Restorer::Restorer(const Options& o) : impl_(std::make_unique<Impl>(o, this)) {
}

Restorer::~Restorer() = default;

std::vector<Frame> Restorer::restore(const std::vector<Frame>& frames, uint64_t first) {
  auto& p = *impl_;
  const auto& o = p.o;
  if (frames.empty() || frames.size() > size_t(o.segment_frames) || (frames.size() - 1) % 4)
    throw std::invalid_argument("SeedVR2: restoration needs 4n+1 frames within segment_frames");
  for (const auto& f : frames) {
    if (f.size() != size_t(o.width) * o.height * 3)
      throw std::invalid_argument("SeedVR2: frame size mismatch");
    for (float v : f)
      if (!std::isfinite(v) || v < 0 || v > 1)
        throw std::invalid_argument("SeedVR2: input pixels must be finite and in [0,1]");
  }
  cuda::set_device(o.device);
  int h = (o.height + 15) / 16 * 16, w = (o.width + 15) / 16 * 16, t = int(frames.size());
  auto moments = p.encode_tiled(frames, h, w);
  auto latent = p.denoise(moments, (t - 1) / 4 + 1, h / 8, w / 8, first);
  moments.clear();
  moments.shrink_to_fit();
  auto result = p.decode_tiled(latent, t, h, w);
  p.rt->weights.clear();
  p.rt->scratch.resize(0);
  for (size_t f = 0; f < result.size(); ++f) {
    auto& out = result[f];
    if (o.color_match)
      for (int c = 0; c < 3; ++c) {
        double a = 0, b = 0, aa = 0, bb = 0;
        size_t n = out.size() / 3;
        for (size_t i = c; i < out.size(); i += 3) {
          a += out[i];
          b += frames[f][i];
        }
        a /= n;
        b /= n;
        for (size_t i = c; i < out.size(); i += 3) {
          aa += (out[i] - a) * (out[i] - a);
          bb += (frames[f][i] - b) * (frames[f][i] - b);
        }
        double scale = std::sqrt((bb / n + 1e-6) / (aa / n + 1e-6));
        for (size_t i = c; i < out.size(); i += 3)
          out[i] = float((out[i] - a) * scale + b);
      }
    for (auto& v : out) {
      if (!std::isfinite(v))
        throw std::runtime_error("SeedVR2: nonfinite restored pixel");
      v = std::clamp(v, 0.0f, 1.0f);
    }
  }
  return result;
}
} // namespace slopfab::seedvr2
