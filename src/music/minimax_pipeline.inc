#include "runtime.cuh"
#include "slopfab/tensor_convert.h"
#include "slopfab/audio/wav.h"
#include "slopfab/sampler/noise.h"
#include "slopfab/safetensors_write.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <random>
#include <stdexcept>

namespace slopfab::SLOPFAB_MUSIC_NS {
namespace {
float bf(float x) {
  return gpu_bf(x);
}

Tensor rows(const Tensor& x, int first, int count) {
  if (first < 0 || count < 1 || first + count > x.rows)
    throw std::invalid_argument("music row slice");
  Tensor out(count, x.cols);
  gpu_copy_rows(x, out, first, 0, count);
  return out;
}

void append(Tensor& dest, Tensor source, int capacity) {
  if (!dest.rows) {
    dest = Tensor(capacity, source.cols);
    dest.rows = 0;
  }
  if (dest.rows + source.rows > capacity)
    throw std::invalid_argument("MiniMax Music3 KV capacity exceeded");
  gpu_copy_rows(source, dest, 0, dest.rows, source.rows);
  dest.rows += source.rows;
}

struct Cache {
  Tensor k, v;
};

using Caches = std::vector<Cache>;

Tensor global(Runtime& rt, Tensor x, Caches& cache, int capacity, Hook hook) {
  int offset = cache.empty() ? 0 : cache.front().k.rows;
  if (cache.empty())
    cache.resize(36);
  auto pos = rope(x.rows, offset);
  for (int i = 0; i < 36; ++i) {
    if (hook)
      hook();
    auto p = "model.layers." + std::to_string(i) + ".";
    auto qkv = rt.project(rt.norm(x, p + "input_layernorm.weight"), p + "self_attn.qkv_proj");
    auto q = rt.norm(moss::slice(qkv, 0, 4096), p + "self_attn.q_norm.weight", 128);
    auto k = rt.norm(moss::slice(qkv, 4096, 1024), p + "self_attn.k_norm.weight", 128);
    auto v = moss::slice(qkv, 5120, 1024);
    moss::rotary(q, 32, pos, 128);
    moss::rotary(k, 8, pos, 128);
    append(cache[i].k, std::move(k), capacity);
    append(cache[i].v, std::move(v), capacity);
    auto a = rt.attend(q, cache[i].k, cache[i].v, 32, 8, offset == 0 && x.rows > 1);
    moss::residual(x, rt.project(a, p + "self_attn.o_proj"));
    auto gu = rt.project(rt.norm(x, p + "post_attention_layernorm.weight"), p + "mlp.gate_up_proj");
    auto gate = moss::slice(gu, 0, 12288), up = moss::slice(gu, 12288, 12288);
    moss::residual(x, rt.project(moss::activate(up, &gate), p + "mlp.down_proj"));
  }
  return rt.norm(rows(x, x.rows - 1, 1), "model.norm.weight");
}

Tensor depth(Runtime& rt, const Tensor& sequence, Hook hook) {
  const std::string p = "model.audio_decoder.";
  std::vector<int32_t> ids(sequence.rows);
  for (int i = 0; i < sequence.rows; ++i)
    ids[i] = i;
  auto x = embedding(rt, p + "pos_embedding.weight", ids);
  moss::residual(x, sequence);
  for (int i = 0; i < 4; ++i) {
    if (hook)
      hook();
    auto b = p + "layers." + std::to_string(i) + ".";
    auto qkv = rt.project(rt.norm(x, b + "input_layernorm.weight"), b + "self_attn.qkv_proj");
    auto q = moss::slice(qkv, 0, 4096), k = moss::slice(qkv, 4096, 4096),
         v = moss::slice(qkv, 8192, 4096);
    moss::residual(x, rt.project(rt.attend(q, k, v, 16, 16, true), b + "self_attn.o_proj"));
    auto gu = rt.project(rt.norm(x, b + "post_attention_layernorm.weight"), b + "mlp.gate_up_proj");
    auto gate = moss::slice(gu, 0, 6144), up = moss::slice(gu, 6144, 6144);
    moss::residual(x, rt.project(moss::activate(up, &gate), b + "mlp.down_proj"));
  }
  return rt.norm(rows(x, x.rows - 1, 1), p + "norm.weight");
}

int choose(const std::vector<float>& c, const std::vector<float>& u, float guidance, int topk,
           bool global_head, bool pruned, std::mt19937_64& rng) {
  std::vector<std::pair<float, int>> scores;
  if (global_head) {
    scores.emplace_back(c[pruned ? 0 : 151670], pruned ? 0 : 151670);
    for (int i = 0; i < 16384; ++i)
      scores.emplace_back(c[(pruned ? 1 : 151675) + i], (pruned ? 1 : 151675) + i);
    std::sort(scores.begin(), scores.end(), [](auto a, auto b) {
      return a.first > b.first;
    });
    const float threshold = scores[topk - 1].first;
    scores.erase(std::find_if(scores.begin() + topk, scores.end(),
                              [&](auto v) {
                                return v.first < threshold;
                              }),
                 scores.end());
  } else
    for (int i = 0; i < int(c.size()); ++i)
      scores.emplace_back(c[i], i);
  for (auto& item : scores) {
    item.first = u[item.second] + (c[item.second] - u[item.second]) * guidance;
    if (!std::isfinite(item.first))
      throw std::runtime_error("non-finite MiniMax Music3 logits");
  }
  std::sort(scores.begin(), scores.end(), [](auto a, auto b) {
    return a.first > b.first;
  });
  if (scores.size() > size_t(topk)) {
    float threshold = scores[topk - 1].first;
    scores.erase(std::find_if(scores.begin() + topk, scores.end(),
                              [&](auto v) {
                                return v.first < threshold;
                              }),
                 scores.end());
  }
  std::vector<double> weights;
  for (auto item : scores)
    weights.push_back(std::exp(double(item.first - scores[0].first)));
  return scores[std::discrete_distribution<size_t>(weights.begin(), weights.end())(rng)].second;
}

void parent(const std::string& path) {
  auto p = std::filesystem::path(path).parent_path();
  if (!p.empty())
    std::filesystem::create_directories(p);
}

struct Cancelled {};
}

std::vector<float> minimax_hidden(Runtime& rt, const std::vector<int32_t>& ids,
                                  const MusicSettings& settings, int maximum, uint64_t seed,
                                  Hook hook, const std::string& trace) {
  if (ids.empty() || ids.size() > 5000 || ids.size() + maximum + 1 > 10240)
    throw std::invalid_argument("MiniMax Music3 prompt plus audio exceeds the 10240-token context");
  const bool pruned = rt.checkpoint.find("model.embed_tokens_prefill.weight") != nullptr;
  const bool pruned_head = rt.checkpoint.find("model.lm_head_pruned.weight") != nullptr;
  const std::string embed =
      pruned ? "model.embed_tokens_prefill.weight" : "model.embed_tokens.weight";
  const std::string head = pruned_head ? "model.lm_head_pruned" : "model.lm_head";
  auto negative = ids;
  std::fill(negative.begin() + 1, negative.end() - 2, 151654);
  int capacity = int(ids.size()) + maximum + 1;
  Caches conditional, unconditional;
  auto h = global(rt, embedding(rt, embed, ids), conditional, capacity, hook);
  auto uh = global(rt, embedding(rt, embed, negative), unconditional, capacity, hook);
  const float cfg = settings.semantic_guidance ? settings.semantic_guidance : 1.5f;
  const int topk = settings.top_k ? settings.top_k : 50;
  std::mt19937_64 rng(seed ^ 0xa0761d6478bd642full);
  std::vector<float> output;
  std::vector<float> trace_codes, trace_hidden;
  output.reserve(size_t(maximum) * 32768);
  const std::string d = "model.audio_decoder.";
  for (int frame = 0; frame <= maximum; ++frame) {
    if (hook)
      hook();
    int code = choose(moss::to_host(rt.project(h, head)), moss::to_host(rt.project(uh, head)), cfg,
                      topk, true, pruned_head, rng);
    if (code == (pruned_head ? 0 : 151670))
      break;
    int c0 = code - (pruned_head ? 1 : 151675);
    auto cembed =
        embedding(rt, pruned ? "model.embed_tokens_audio.weight" : "model.embed_tokens.weight",
                  {pruned ? c0 : c0 + 151675});
    auto projected = rt.project(cembed, d + "projection");
    auto seq = moss::concatenate(rt.project(h, d + "projection"), projected);
    auto useq = moss::concatenate(rt.project(uh, d + "projection"), projected);
    std::vector<float> hidden = moss::to_host(h);
    std::vector<int32_t> extra;
    for (int i = 0; i < 7; ++i) {
      auto local = depth(rt, seq, hook), ulocal = depth(rt, useq, hook);
      auto values = moss::to_host(local);
      hidden.insert(hidden.end(), values.begin(), values.end());
      auto p = d + "audio_heads." + std::to_string(i);
      int residual = choose(moss::to_host(rt.project(local, p)),
                            moss::to_host(rt.project(ulocal, p)), cfg, topk, false, false, rng);
      extra.push_back(residual + i * 1024);
      if (i < 6) {
        auto next = rt.project(embedding(rt, "model.audio_extra_embedding.weight", {extra.back()}),
                               d + "projection");
        seq = moss::concatenate(seq, next);
        useq = moss::concatenate(useq, next);
      }
    }
    // The first prediction primes the acoustic recurrence and is not a frame.
    if (!trace.empty()) {
      trace_codes.push_back(float(c0));
      for (int i = 0; i < 7; ++i)
        trace_codes.push_back(float(extra[i] - i * 1024));
      trace_hidden.insert(trace_hidden.end(), hidden.begin(), hidden.end());
    }
    if (frame > 0)
      output.insert(output.end(), hidden.begin(), hidden.end());
    if (frame == maximum)
      break;
    auto cvalues = moss::to_host(cembed),
         extra_values = moss::to_host(embedding(rt, "model.audio_extra_embedding.weight", extra));
    for (int c = 0; c < 4096; ++c) {
      float sum = 0;
      for (int i = 0; i < 7; ++i)
        sum += extra_values[size_t(i) * 4096 + c];
      cvalues[c] = bf(bf(cvalues[c] + bf(sum)) * float(1 / std::sqrt(8.)));
    }
    h = global(rt, moss::from_host(cvalues, 1, 4096), conditional, capacity, hook);
    uh = global(rt, moss::from_host(cvalues, 1, 4096), unconditional, capacity, hook);
  }
  if (output.empty())
    throw std::runtime_error("MiniMax Music3 generated zero audio frames");
  if (!trace.empty()) {
    parent(trace);
    write_safetensors(
        trace, {{"codes", {int64_t(trace_codes.size() / 8), 8}, trace_codes},
                {"hidden", {int64_t(trace_hidden.size() / 32768), 32768}, trace_hidden},
                {"tokens", {int64_t(ids.size())}, std::vector<float>(ids.begin(), ids.end())}});
  }
  return output;
}

Tensor minimax_condition(Runtime& rt, const std::vector<float>& hidden) {
  if (hidden.empty() || hidden.size() % 32768)
    throw std::invalid_argument("MiniMax Music3 conditioning must have width 32768");
  int frames = int(hidden.size() / 32768), length = minimax_latent_length(frames);
  auto logits = to_f32(rt.checkpoint.at("cond_layer_logits"));
  float sum = 0, maximum = *std::max_element(logits.begin(), logits.end());
  for (auto& v : logits)
    sum += (v = std::exp(bf(v) - bf(maximum)));
  for (auto& v : logits)
    v = bf(v / sum);
  float scale = bf(to_f32(rt.checkpoint.at("cond_layer_scale"))[0]);
  std::vector<float> fused(size_t(frames) * 4096);
  for (int r = 0; r < frames; ++r)
    for (int c = 0; c < 4096; ++c) {
      float v = 0;
      for (int l = 0; l < 8; ++l)
        v += bf(hidden[size_t(r) * 32768 + l * 4096 + c]) * logits[l];
      fused[size_t(r) * 4096 + c] = bf(bf(v) * scale);
    }
  std::vector<float> columns(size_t(frames) * 12288, 0);
  for (int r = 0; r < frames; ++r)
    for (int c = 0; c < 4096; ++c)
      for (int k = 0; k < 3; ++k)
        if (r + k >= 1 && r + k - 1 < frames)
          columns[size_t(r) * 12288 + c * 3 + k] = fused[size_t(r + k - 1) * 4096 + c];
  // cuDNN Conv1d adds the bias after the BF16 convolution result is rounded.
  auto projected = moss::to_host(
      rt.project(moss::from_host(columns, frames, 12288), "latent_conditioners.0", false));
  std::vector<float> aligned(size_t(length) * 2048);
  for (int r = 0; r < length; ++r)
    std::copy_n(projected.data() + size_t(int64_t(r) * frames / length) * 2048, 2048,
                aligned.data() + size_t(r) * 2048);
  return moss::from_host(aligned, length, 2048);
}

Tensor minimax_prediction(Runtime& rt, const Tensor& state, const Tensor& condition, float time,
                          Hook hook, const std::string& trace) {
  auto dump = [&](const Tensor& x, const std::string& key) {
    if (!trace.empty())
      write_safetensors(trace + "/flow-" + key + ".safetensors",
                        {{"output", {x.rows, x.cols}, moss::to_host(x)}});
  };
  if (state.cols != 128 || condition.cols != 2048 || condition.rows != state.rows)
    throw std::invalid_argument("MiniMax Music3 flow shape mismatch");
  const std::string p = "diffusion_transformer.", t = p + "transformer.";
  Tensor full(state.rows, 2304);
  gpu_zero(full);
  gpu_copy_columns(state, full, 0);
  gpu_copy_columns(condition, full, 256);
  moss::residual(full, rt.project(full, p + "preprocess_conv"));
  dump(full, "preprocess");
  auto frequencies = to_f32(rt.checkpoint.at(p + "timestep_features.weight"));
  std::vector<float> features(256);
  for (int i = 0; i < 128; ++i) {
    // Fourier phases are computed in FP32; only sin/cos outputs are narrowed.
    float a = (2 * 3.14159265358979323846f * time) * frequencies[i];
    features[i] = std::cos(a);
    features[i + 128] = std::sin(a);
  }
  auto temb = rt.project(
      moss::activate(rt.project(moss::from_host(features, 1, 256), p + "to_timestep_embed.0")),
      p + "to_timestep_embed.2");
  auto x = moss::concatenate(temb, rt.project(full, t + "project_in"));
  dump(x, "input");
  const auto inv = to_f32(rt.checkpoint.at(t + "rotary_pos_emb.inv_freq"));
  // BF16 cannot distinguish adjacent positions beyond 256. Evaluate phases
  // and trig functions in FP32 before narrowing their bounded outputs.
  auto phases = minimax_rotary_table(x.rows, inv);
  auto rotation = moss::from_host(phases, x.rows, 32);
  for (int i = 0; i < 36; ++i) {
    if (hook)
      hook();
    auto b = t + "layers." + std::to_string(i) + ".";
    auto normalized = gpu_ln(rt, x, b + "pre_norm");
    if (i == 0)
      dump(normalized, "norm0");
    auto qkv = rt.project(normalized, b + "self_attn.to_qkv");
    if (i == 0)
      dump(qkv, "qkv0");
    auto q = moss::slice(qkv, 0, 2048), k = moss::slice(qkv, 2048, 2048),
         v = moss::slice(qkv, 4096, 2048);
    moss::rotary(q, 32, rotation, 32);
    moss::rotary(k, 32, rotation, 32);
    auto attn = rt.attend(q, k, v, 32, 32, false, true);
    if (i == 0) {
      dump(q, "q0");
      dump(attn, "attn0");
    }
    moss::residual(x, rt.project(attn, b + "self_attn.to_out"));
    auto vg = rt.project(gpu_ln(rt, x, b + "ff_norm"), b + "ff.ff.0.proj");
    auto value = moss::slice(vg, 0, 8192), gate = moss::slice(vg, 8192, 8192);
    moss::residual(x, rt.project(moss::activate(value, &gate), b + "ff.ff.2"));
    dump(x, "layer" + std::to_string(i));
  }
  auto out = rt.project(rows(x, 1, state.rows), t + "project_out");
  moss::residual(out, rt.project(out, p + "postprocess_conv"));
  return out;
}

Tensor probe_minimax_global(Runtime& rt, const std::vector<int32_t>& ids) {
  Caches cache;
  return global(rt,
                embedding(rt,
                          rt.checkpoint.find("model.embed_tokens_prefill.weight")
                              ? "model.embed_tokens_prefill.weight"
                              : "model.embed_tokens.weight",
                          ids),
                cache, int(ids.size()), {});
}

Tensor probe_minimax_depth(Runtime& rt, const Tensor& sequence) {
  return depth(rt, sequence, {});
}

RunResult run_minimax_from_hidden(const GenerateRequest& r, const GeneratePlan& plan,
                                  const RunOptions& options, std::vector<float> hidden) {
  RunResult result;
  using Clock = std::chrono::steady_clock;
  auto notify = [&](RunStage stage, int step = -1, int total = 0) {
    if (options.on_progress && !options.on_progress(stage, step, total, options.hook_userdata))
      throw Cancelled{};
  };
  auto elapsed = [](auto start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
  };
  try {
    validate_generation_options(r, plan, options);
    notify(RunStage::kStarting);
    auto start = Clock::now();
    if (hidden.empty()) {
      Runtime ar(r.text_encoder_path);
      ar.cache_dense_weights = true;
      auto tokens = minimax_prompt(ar.checkpoint, r.prompt, r.music.lyrics);
      hidden = minimax_hidden(ar, tokens, r.music, plan.music_frames, r.seed, [&] {
        notify(RunStage::kConditioning);
      });
    }
    result.conditioner_executed = true;
    result.seconds_conditioning = elapsed(start);
    int frames = int(hidden.size() / 32768);
    if (options.verbose)
      std::printf("MiniMax Music3 generated %d acoustic frames\n", frames);
    std::vector<TensorWrite> chunks;
    std::vector<float> previous, previous_condition;
    start = Clock::now();
    {
      Runtime flow(r.transformer_path);
      flow.cache_dense_weights = true;
      std::vector<int> starts = {0};
      if (frames > 200) {
        starts.clear();
        for (int i = 0; i < frames - 100; i += 100)
          starts.push_back(i);
      }
      for (size_t chunk = 0; chunk < starts.size(); ++chunk) {
        int first = starts[chunk], count = std::min(200, frames - first);
        std::vector<float> window(hidden.begin() + size_t(first) * 32768,
                                  hidden.begin() + size_t(first + count) * 32768);
        auto cond = minimax_condition(flow, window);
        int length = cond.rows;
        int overlap = std::min(int(previous.size() / 128), length);
        if (overlap) {
          auto prior = moss::from_host(
              std::vector<float>(previous_condition.begin(),
                                 previous_condition.begin() + size_t(overlap) * 2048),
              overlap, 2048);
          gpu_copy_rows(prior, cond, 0, 0, overlap);
        }
        Tensor zero(length, 2048);
        gpu_zero(zero);
        std::vector<float> state(size_t(length) * 128);
        sampler::fill_normal(r.seed + chunk, sampler::NoiseStream::kAudioLatents, state.data(),
                             state.size());
        for (auto& v : state)
          v = bf(v);
        auto noise = state;
        float dt = 1.f / plan.num_inference_steps,
              cfg = r.music.flow_guidance ? r.music.flow_guidance : 1.7f;
        for (int step = 0; step < plan.num_inference_steps; ++step) {
          float time = step * dt;
          for (int i = 0; i < overlap * 128; ++i)
            state[i] = bf(bf((1 - time) * noise[i]) + bf(time * previous[i]));
          Hook progress = [&] {
            notify(RunStage::kDenoising, int(chunk) * plan.num_inference_steps + step,
                   int(starts.size()) * plan.num_inference_steps);
          };
          progress();
          auto input = moss::from_host(state, length, 128);
          auto c = moss::to_host(minimax_prediction(flow, input, cond, time, progress));
          auto u = moss::to_host(minimax_prediction(flow, input, zero, time, progress));
          for (size_t i = 0; i < state.size(); ++i) {
            float velocity = bf(u[i] + bf(bf(c[i] - u[i]) * cfg));
            state[i] = bf(state[i] + dt * velocity);
            if (!std::isfinite(state[i]))
              throw std::runtime_error("non-finite MiniMax Music3 flow state");
          }
          ++result.steps_computed;
          if (options.verbose)
            std::printf("MiniMax Music3 window %zu/%zu step %d/%d\n", chunk + 1, starts.size(),
                        step + 1, plan.num_inference_steps);
        }
        std::copy_n(previous.begin(), size_t(overlap) * 128, state.begin());
        int lo = std::max(0, length - 344), hi = std::max(lo, length - 172);
        previous.assign(state.begin() + size_t(lo) * 128, state.begin() + size_t(hi) * 128);
        auto condition = moss::to_host(cond);
        previous_condition.assign(condition.begin() + size_t(lo) * 2048,
                                  condition.begin() + size_t(hi) * 2048);
        chunks.push_back({"audio_chunk_" + std::to_string(chunk), {length, 128}, std::move(state)});
      }
    }
    result.seconds_denoise = result.seconds_denoise_loop = elapsed(start);
    if (!options.dump_latents_path.empty()) {
      parent(options.dump_latents_path);
      write_safetensors(options.dump_latents_path, chunks,
                        {{"slopfab.model_family", "minimax_music3"}});
    }
    notify(RunStage::kAudioDecode);
    start = Clock::now();
    std::vector<float> audio;
    {
      Runtime decoder(r.audio_vae_path);
      for (size_t i = 0; i < chunks.size(); ++i) {
        auto wave = decode_minimax(decoder, chunks[i].data, [&] {
          notify(RunStage::kAudioDecode);
        });
        size_t left = i ? 86 * 512 * 2 : 0, right = i + 1 == chunks.size() ? 0 : 258 * 512 * 2;
        if (left + right >= wave.size())
          throw std::runtime_error("MiniMax Music3 waveform overlap exceeds chunk");
        audio.insert(audio.end(), wave.begin() + left, wave.end() - right);
      }
    }
    result.seconds_audio_decode = elapsed(start);
    for (float v : audio)
      if (!std::isfinite(v))
        throw std::runtime_error("non-finite MiniMax Music3 audio");
    notify(RunStage::kDelivering);
    RunSamples samples;
    samples.channels = 0;
    samples.audio = &audio;
    samples.audio_channels = 2;
    samples.audio_sample_rate = 44100;
    if (!options.on_samples || !options.on_samples(samples, options.hook_userdata)) {
      if (std::filesystem::path(r.out_path).extension() != ".wav")
        throw std::invalid_argument("music output requires .wav");
      parent(r.out_path);
      audio::write_wav(r.out_path, audio, 2, 44100);
      result.outputs.push_back(r.out_path);
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

RunResult run_minimax(const GenerateRequest& r, const GeneratePlan& plan,
                      const RunOptions& options) {
  return run_minimax_from_hidden(r, plan, options, {});
}
}
