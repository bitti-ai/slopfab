#include "runtime.cuh"
#include "slopfab/int8_weight.h"
#include "slopfab/tensor_convert.h"
#include "slopfab/text/tokenizer.h"
#include "slopfab/json.h"
#include "slopfab/audio/wav.h"
#include "slopfab/safetensors_write.h"
#include "slopfab/sampler/noise.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <numeric>
#include <random>
#include <stdexcept>

namespace slopfab::SLOPFAB_MUSIC_NS {
namespace {
float bf(float v) {
  return gpu_bf(v);
}

struct Cache {
  Tensor k, v;
};

using Caches = std::vector<Cache>;
const std::string ar = "text_encoders.model.", nar = "model.diffusion_model.";

Tensor row_range(const Tensor& x, int start, int count) {
  if (start < 0 || count < 1 || start + count > x.rows)
    throw std::invalid_argument("music row range");
  Tensor out(count, x.cols);
  gpu_copy_rows(x, out, start, 0, count);
  return out;
}

void append(Tensor& dest, Tensor source, int capacity) {
  if (!dest.rows) {
    dest = Tensor(capacity, source.cols);
    dest.rows = 0;
  }
  if (dest.rows + source.rows > capacity)
    throw std::invalid_argument("YuE2 context capacity exceeded");
  gpu_copy_rows(source, dest, 0, dest.rows, source.rows);
  dest.rows += source.rows;
}

Tensor layers(Runtime& rt, Tensor x, const std::string& prefix, int offset, Caches& cache,
              int capacity, bool flow, Hook hook, const std::string& trace = "") {
  auto dump = [&](const Tensor& t, const std::string& key) {
    if (!trace.empty())
      write_safetensors(trace + "/" + (flow ? "nar-" : "ar-") + key + ".safetensors",
                        {{"output", {t.rows, t.cols}, moss::to_host(t)}});
  };
  dump(x, "input");
  auto rotary = rope(x.rows, offset);
  if (cache.empty())
    cache.resize(28);
  for (int i = 0; i < 28; ++i) {
    if (hook)
      hook();
    const auto p = prefix + "layers." + std::to_string(i) + ".";
    auto qkv = rt.project(yue_norm(rt, x, p + "input_layernorm.weight"), p + "self_attn.qkv_proj");
    if (i == 0 || i == 11)
      dump(qkv, "qkv" + std::to_string(i));
    auto q = yue_norm(rt, moss::slice(qkv, 0, 2048), p + "self_attn.q_norm.weight", 128);
    auto k = yue_norm(rt, moss::slice(qkv, 2048, 1024), p + "self_attn.k_norm.weight", 128);
    auto v = moss::slice(qkv, 3072, 1024);
    if (i == 11)
      dump(q, "qnorm11");
    moss::rotary(q, 16, rotary, 128);
    moss::rotary(k, 8, rotary, 128);
    if (i == 0 || i == 11) {
      dump(q, "q" + std::to_string(i));
      dump(k, "k" + std::to_string(i));
    }
    Tensor a;
    if (flow) {
      auto keys = moss::concatenate(cache[i].k, k), values = moss::concatenate(cache[i].v, v);
      a = rt.attend(q, keys, values, 16, 8);
    } else {
      append(cache[i].k, std::move(k), capacity);
      append(cache[i].v, std::move(v), capacity);
      a = offset == 0 && x.rows > 1 ? gpu_causal_attention(rt, q, cache[i].k, cache[i].v)
                                    : rt.attend(q, cache[i].k, cache[i].v, 16, 8);
    }
    moss::residual(x, rt.project(a, p + "self_attn.o_proj"));
    if (i == 0 || i == 11) {
      dump(a, "attn" + std::to_string(i));
      dump(x, "residual" + std::to_string(i));
    }
    auto gu =
        rt.project(yue_norm(rt, x, p + "post_attention_layernorm.weight"), p + "mlp.gate_up_proj");
    auto gate = moss::slice(gu, 0, 6144), up = moss::slice(gu, 6144, 6144);
    moss::residual(x, rt.project(moss::activate(up, &gate), p + "mlp.down_proj"));
    dump(x, "layer" + std::to_string(i));
  }
  return x;
}

Tensor ar_forward(Runtime& rt, const std::vector<int32_t>& ids, Caches& cache, int capacity,
                  Hook hook, const std::string& trace = "") {
  int offset = cache.empty() ? 0 : cache.front().k.rows;
  if (ids.empty() || offset + ids.size() > context)
    throw std::invalid_argument("YuE2 prompt/generation exceeds 24576-token context");
  return layers(rt, embedding(rt, ar + "embed_tokens.weight", ids), ar, offset, cache, capacity,
                false, hook, trace);
}

std::vector<float> logits(Runtime& rt, const Tensor& hidden) {
  return moss::to_host(rt.project(
      yue_norm(rt, row_range(hidden, hidden.rows - 1, 1), ar + "norm.weight"), ar + "lm_head"));
}

int sample(std::vector<float> values, const std::vector<int32_t>& generated, bool abc,
           const MusicSettings& s, std::mt19937_64& rng) {
  const float temperature = abc ? .7f : s.temperature,
              penalty = abc ? 1.005f : s.repetition_penalty;
  const int window = abc ? 100 : 50, end = abc ? abc_end : music_end, low = abc ? 0 : codec_offset;
  const int high = abc ? eod : codec_offset + codec_size, minimum = abc ? 32 : 200;
  const bool legacy = !abc && s.planning == "off";
  auto round = [&](float v) {
    return legacy ? bf(v) : v;
  };
  std::map<int, int> counts;
  for (int i = std::max(0, int(generated.size()) - window); i < int(generated.size()); ++i)
    ++counts[generated[i]];
  for (auto [id, n] : counts) {
    float factor = round(std::pow(penalty, float(n)));
    values[id] = round(values[id] < 0 ? values[id] * factor : values[id] / factor);
  }
  std::vector<std::pair<float, int>> candidates;
  for (int id = low; id < high; ++id)
    candidates.emplace_back(values[id], id);
  if (int(generated.size()) >= minimum)
    candidates.emplace_back(values[end], end);
  for (const auto& v : candidates)
    if (!std::isfinite(v.first))
      throw std::runtime_error("non-finite YuE2 logits");
  std::sort(candidates.begin(), candidates.end(), [](auto a, auto b) {
    return a.first > b.first;
  });
  if (temperature == 0)
    return candidates.front().second;
  for (auto& v : candidates)
    v.first = round(v.first / temperature);
  const int topk = abc ? 30 : s.top_k;
  if (topk > 0 && candidates.size() > size_t(topk)) {
    float threshold = candidates[topk - 1].first;
    auto it = std::find_if(candidates.begin() + topk, candidates.end(), [&](auto v) {
      return v.first < threshold;
    });
    candidates.erase(it, candidates.end());
  }
  std::vector<double> weights(candidates.size());
  double sum = 0;
  for (size_t i = 0; i < weights.size(); ++i)
    sum += (weights[i] = std::exp(double(candidates[i].first - candidates[0].first)));
  double cumulative = 0;
  const float topp = abc ? .9f : s.top_p;
  for (size_t i = 0; i < weights.size(); ++i) {
    if (cumulative > topp && i >= size_t(legacy ? 3 : 1)) {
      candidates.resize(i);
      weights.resize(i);
      break;
    }
    cumulative += weights[i] / sum;
  }
  return candidates[std::discrete_distribution<size_t>(weights.begin(), weights.end())(rng)].second;
}

std::vector<int32_t> generate_tokens(Runtime& rt, const std::vector<int32_t>& prefix,
                                     const std::vector<int32_t>& negative, bool abc, int maximum,
                                     const MusicSettings& s, uint64_t seed, Hook hook) {
#ifndef SLOPFAB_MUSIC_VULKAN
  moss::TemporaryBufferScope buffers;
#endif
  if (prefix.size() + maximum > context || negative.size() + maximum > context)
    throw std::invalid_argument("YuE2 prompt plus requested token budget exceeds context");
  Caches cache, uncond;
  auto h = ar_forward(rt, prefix, cache, int(prefix.size()) + maximum, hook);
  Tensor uh;
  const bool cfg = !abc && s.semantic_guidance != 1;
  if (cfg)
    uh = ar_forward(rt, negative, uncond, int(negative.size()) + maximum, hook);
  std::mt19937_64 rng(seed);
  std::vector<int32_t> out;
  for (int i = 0; i < maximum; ++i) {
    if (hook)
      hook();
    auto scores = logits(rt, h);
    if (cfg) {
      auto u = logits(rt, uh);
      for (size_t j = 0; j < scores.size(); ++j)
        scores[j] = bf(u[j] + bf(bf(scores[j] - u[j]) * s.semantic_guidance));
    }
    int id = sample(std::move(scores), out, abc, s, rng);
    if (id == (abc ? abc_end : music_end))
      break;
    out.push_back(id);
    if (i + 1 < maximum) {
      h = ar_forward(rt, {id}, cache, int(prefix.size()) + maximum, hook);
      if (cfg)
        uh = ar_forward(rt, {id}, uncond, int(negative.size()) + maximum, hook);
    }
  }
  return out;
}

Tensor velocity(Runtime& rt, const std::vector<float>& state, float time, Caches& cache, Hook hook,
                const std::string& trace = "") {
  int frames = int(state.size() / 64), count = frames + 2;
  std::vector<float> padded(size_t(count) * 64, 0);
  std::copy(state.begin(), state.end(), padded.begin() + 64);
  auto x = rt.project(moss::from_host(padded, count, 64), nar + "vae2llm");
  const float raw =
      bf(std::clamp(std::log(std::max(time, 1e-20f) / std::max(1 - time, 1e-20f)), -20.f, 20.f));
  const float shifted = bf(1 / (1 + std::exp(-raw)));
  std::vector<float> sinusoid(256);
  for (int i = 0; i < 128; ++i) {
    float a = shifted * std::exp(-std::log(10000.f) * i / 128);
    sinusoid[i] = std::cos(a);
    sinusoid[i + 128] = std::sin(a);
  }
  auto t = rt.project(
      moss::activate(rt.project(moss::from_host(sinusoid, 1, 256), nar + "time_embedder.mlp.0")),
      nar + "time_embedder.mlp.2");
  gpu_condition(rt, x, t, nar + "latent_pos_embed.pe");
  auto h =
      layers(rt, std::move(x), nar + "model.", cache.front().k.rows, cache, 0, true, hook, trace);
  return row_range(rt.project(yue_norm(rt, h, nar + "model.norm.weight"), nar + "llm2vae"), 1,
                   frames);
}

struct Cancelled {};
}

Tensor yue_norm(Runtime& rt, const Tensor& x, const std::string& name, int width) {
  if (!width)
    width = x.cols;
  if (x.cols % width || rt.checkpoint.at(name).numel() != width)
    throw std::invalid_argument("YuE2 norm shape: " + name);
  return gpu_yue_norm(rt, x, name, width);
}

Tensor embedding(Runtime& rt, const std::string& name, const std::vector<int32_t>& ids) {
  const auto& w = rt.checkpoint.at(name);
  const int width = int(w.shape.at(1));
  Int8WeightState q;
  if (w.dtype == DType::kI8)
    q = read_int8_weight(rt.checkpoint, name, "music embedding");
  std::vector<float> values(ids.size() * width);
  for (size_t r = 0; r < ids.size(); ++r) {
    if (ids[r] < 0 || ids[r] >= w.shape[0])
      throw std::invalid_argument("music token outside vocabulary");
    if (w.dtype != DType::kI8) {
      auto row = w;
      row.shape = {width};
      row.nbytes = size_t(width) * dtype_size(w.dtype);
      row.data = static_cast<const uint8_t*>(w.data) + size_t(ids[r]) * row.nbytes;
      auto decoded = to_f32(row);
      std::copy(decoded.begin(), decoded.end(), values.begin() + r * width);
      continue;
    }
    const int group = q.rotation_group;
    for (int c = 0; c < width; c += group) {
      float* v = values.data() + r * width + c;
      for (int i = 0; i < group; ++i)
        v[i] = q.codes[size_t(ids[r]) * width + c + i];
      for (int stride = 1; stride < group; stride *= 4)
        for (int base = 0; base < group; base += 4 * stride)
          for (int i = 0; i < stride; ++i) {
            int j = base + i;
            float a = v[j], b = v[j + stride], cc = v[j + 2 * stride], d = v[j + 3 * stride];
            v[j] = a + b + cc - d;
            v[j + stride] = a + b - cc + d;
            v[j + 2 * stride] = a - b + cc + d;
            v[j + 3 * stride] = -a + b + cc + d;
          }
      for (int i = 0; i < group; ++i)
        v[i] *= q.scales[ids[r]] / std::sqrt(float(group));
    }
  }
  return moss::from_host(values, int(ids.size()), width);
}

Tensor rope(int count, int offset, int dim, float theta) {
  std::vector<float> values(size_t(count) * dim);
  for (int r = 0; r < count; ++r)
    for (int j = 0; j < dim / 2; ++j) {
      float a = (r + offset) * std::pow(theta, -2.f * j / dim);
      values[size_t(r) * dim + j] = std::sin(a);
      values[size_t(r) * dim + dim / 2 + j] = std::cos(a);
    }
  return moss::from_host(values, count, dim);
}

Tensor probe_yue(Runtime& rt, const std::vector<int32_t>& ids, const std::vector<float>& state,
                 const std::string& trace) {
  Caches cache;
  auto h = ar_forward(rt, ids, cache, int(ids.size()), {}, trace);
  if (!state.empty())
    return velocity(rt, state, .5f, cache, {}, trace);
  return rt.project(yue_norm(rt, h, ar + "norm.weight"), ar + "lm_head");
}

RunResult run_yue(const GenerateRequest& request, const GeneratePlan& plan,
                  const RunOptions& options) {
  RunResult result;
  using Clock = std::chrono::steady_clock;
  auto notify = [&](RunStage stage, int step = -1, int steps = 0) {
    if (options.on_progress && !options.on_progress(stage, step, steps, options.hook_userdata))
      throw Cancelled{};
  };
  auto elapsed = [](auto start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
  };
  try {
    validate_generation_options(request, plan, options);
    notify(RunStage::kStarting);
    Runtime rt(request.transformer_path);
    rt.cache_dense_weights = true;
    text::Tokenizer tokenizer;
    load_yue_tokenizer(tokenizer, rt.checkpoint);
    auto s = request.music;
    if (!s.top_k)
      s.top_k = 100;
    if (s.semantic_guidance == 0)
      s.semantic_guidance = s.planning == "off" ? 1.01f : 1.f;
    const auto instruction = yue_instruction(s.planning);
    auto prefix = tokenizer.encode(
        instruction + "\n[Tags]\n" + request.prompt + "\n[Lyrics]\n" + s.lyrics + "\n", false);
    prefix.insert(prefix.begin(), eod);
    prefix.push_back(abc_start);
    notify(RunStage::kConditioning);
    auto start = Clock::now();
    Hook conditioning = [&] {
      notify(RunStage::kConditioning);
    };
    auto abc = tokenizer.encode(s.abc, false);
    if (s.planning != "off" && s.abc.empty())
      abc = generate_tokens(rt, prefix, {}, true, s.abc_max_tokens, s, request.seed, conditioning);
    if (options.verbose && !abc.empty())
      std::printf("YuE2 ABC\n%s\n", tokenizer.decode(abc).c_str());
    prefix.insert(prefix.end(), abc.begin(), abc.end());
    prefix.push_back(abc_end);
    prefix.push_back(music_start);
    auto negative = tokenizer.encode(instruction, false);
    negative.insert(negative.begin(), eod);
    if (s.planning != "off") {
      negative.push_back(abc_start);
      negative.insert(negative.end(), abc.begin(), abc.end());
      negative.push_back(abc_end);
    }
    negative.push_back(music_start);
    auto codes = generate_tokens(rt, prefix, negative, false, plan.music_frames, s, request.seed,
                                 conditioning);
    if (codes.empty())
      throw std::runtime_error("YuE2 produced no audio tokens");
    result.conditioner_executed = true;
    result.seconds_conditioning = elapsed(start);
    if (options.verbose)
      std::printf("YuE2 generated %zu semantic frames\n", codes.size());
    const int chunk = std::min((context - int(prefix.size()) - 3) / 2, context - 2);
    if (chunk < 1)
      throw std::invalid_argument("YuE2 prompt leaves no room for audio flow");
    std::vector<float> noise(codes.size() * 64), latent;
    latent.reserve(noise.size());
    sampler::fill_normal(request.seed, sampler::NoiseStream::kAudioLatents, noise.data(),
                         noise.size());
    start = Clock::now();
    for (int first = 0; first < int(codes.size()); first += chunk) {
#ifndef SLOPFAB_MUSIC_VULKAN
      moss::TemporaryBufferScope buffers;
#endif
      const int count = std::min(chunk, int(codes.size()) - first);
      auto tokens = prefix;
      tokens.insert(tokens.end(), codes.begin() + first, codes.begin() + first + count);
      tokens.push_back(music_end);
      Caches cache;
      ar_forward(rt, tokens, cache, int(tokens.size()), conditioning);
      std::vector<float> state(noise.begin() + size_t(first) * 64,
                               noise.begin() + size_t(first + count) * 64);
      for (auto& v : state)
        v = bf(v);
      const float dt = 1.f / plan.num_inference_steps;
      for (int step = 0; step < plan.num_inference_steps; ++step) {
        Hook progress = [&] {
          notify(RunStage::kDenoising, step, plan.num_inference_steps);
        };
        progress();
        auto v = moss::to_host(velocity(rt, state, 1 - step * dt, cache, progress));
        auto mid = state;
        for (size_t i = 0; i < state.size(); ++i)
          mid[i] = bf(state[i] - bf(v[i] * (dt / 2)));
        v = moss::to_host(velocity(rt, mid, 1 - (step + .5f) * dt, cache, progress));
        for (size_t i = 0; i < state.size(); ++i) {
          state[i] = bf(state[i] - bf(v[i] * dt));
          if (!std::isfinite(state[i]))
            throw std::runtime_error("non-finite YuE2 flow state");
        }
        ++result.steps_computed;
        if (options.verbose)
          std::printf("YuE2 flow %d/%d\n", step + 1, plan.num_inference_steps);
      }
      latent.insert(latent.end(), state.begin(), state.end());
    }
    result.seconds_denoise = result.seconds_denoise_loop = elapsed(start);
    rt.release_prefix("text_encoders.");
    rt.release_prefix("model.diffusion_model.");
    if (!options.dump_latents_path.empty()) {
      auto parent = std::filesystem::path(options.dump_latents_path).parent_path();
      if (!parent.empty())
        std::filesystem::create_directories(parent);
      write_safetensors(options.dump_latents_path,
                        {{"audio_latents", {int64_t(codes.size()), 64}, latent},
                         {"semantic_tokens",
                          {int64_t(codes.size())},
                          std::vector<float>(codes.begin(), codes.end())}},
                        {{"slopfab.model_family", "yue2"}});
    }
    notify(RunStage::kAudioDecode);
    start = Clock::now();
    auto audio = decode_yue(rt, latent, [&] {
      notify(RunStage::kAudioDecode);
    });
    result.seconds_audio_decode = elapsed(start);
    for (float v : audio)
      if (!std::isfinite(v))
        throw std::runtime_error("non-finite YuE2 audio");
    notify(RunStage::kDelivering);
    RunSamples samples;
    samples.channels = 0;
    samples.audio_channels = 2;
    samples.audio_sample_rate = 48000;
    samples.audio = &audio;
    if (!options.on_samples || !options.on_samples(samples, options.hook_userdata)) {
      if (std::filesystem::path(request.out_path).extension() != ".wav")
        throw std::invalid_argument("music file output requires .wav");
      auto parent = std::filesystem::path(request.out_path).parent_path();
      if (!parent.empty())
        std::filesystem::create_directories(parent);
      audio::write_wav(request.out_path, audio, 2, 48000);
      result.outputs.push_back(request.out_path);
    }
    notify(RunStage::kFinished);
    result.ok = true;
  } catch (const Cancelled&) {
    result.cancelled = true;
    result.message = "music generation cancelled";
  } catch (const std::exception& e) {
    result.message = e.what();
  }
  return result;
}
}
