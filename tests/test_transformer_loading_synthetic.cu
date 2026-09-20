#include "detail/transformer_fixture.h"

SLOPFAB_TEST_CATEGORY(transformer_offloaded_blocks_match_resident, "synthetic") {
  auto cfg = tiny_config(); cfg.num_layers = 6;
  const auto weights = build_synthetic(cfg);
  const std::string path = write_synthetic(weights);
  const auto adapter_path = std::filesystem::temp_directory_path() / "slopfab_streamed_lora.safetensors";
  std::vector<slopfab::TensorWrite> adapters;
  for (int block = 0; block < cfg.num_layers; ++block) {
    for (const char* suffix : {"attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"}) {
      const std::string name = "blocks." + std::to_string(block) + "." + suffix;
      const auto& shape = weights.at(name + ".weight").shape;
      adapters.push_back({name + ".lora_A.weight", {4, shape[1]},
          make_data(size_t(4 * shape[1]), 7300 + block, .05f)});
      adapters.push_back({name + ".lora_B.weight", {shape[0], 4},
          make_data(size_t(4 * shape[0]), 7400 + block, .05f)});
    }
  }
  slopfab::write_safetensors(adapter_path.string(), adapters);
  auto run = [&](int offload, bool cache) {
    slopfab::SafeTensors st; st.open(path);
    slopfab::LoraAdapters loras; loras.load({{adapter_path.string(), 1}}, st);
    Transformer model;
    slopfab::dit::TransformerLoadOptions load;
    load.offload_blocks = offload; load.headroom_bytes = 0;
    model.load(st, cfg, &loras, load);
    CHECK(model.offloaded_blocks() == size_t(offload));
    CHECK((model.offloaded_host_bytes() > 0) == (offload > 0));
    // The streamer owns its bytes after both source objects are released.
    st.close(); loras = slopfab::LoraAdapters();
    model.set_row_chunk(128);
    if (cache) {
      slopfab::dit::BlockCacheConfig bc; bc.span = 3; bc.interval = 2; bc.warmup = 2;
      model.set_block_cache(bc, 6);
    }
    std::vector<float> result;
    for (int text : {259, 131}) {
      const auto c = make_case(cfg, text, .31f);
      model.prepare_text(c.prompt.data(), c.layout.num_text);
      model.prepare_sequence(c.layout, c.idx, c.pos);
      for (int step = 0; step < 6; ++step) {
        model.set_denoise_step(step);
        std::vector<float> video(c.video_rows.size()), audio(c.audio_rows.size());
        model.forward(c.video_rows.data(), c.audio_rows.data(), c.rt, video.data(), audio.data());
        result.insert(result.end(), video.begin(), video.end());
        result.insert(result.end(), audio.begin(), audio.end());
      }
    }
    CHECK(all_finite(result));
    if (cache) CHECK(model.block_cache_reused() > 0);
    const size_t bytes = model.weight_bytes();
    model.unload();
    CHECK(model.weight_bytes() == 0);
    CHECK(model.offloaded_blocks() == 0);
    CHECK(model.offloaded_host_bytes() == 0);
    return std::make_pair(result, bytes);
  };
  for (bool cache : {false, true}) {
    const auto resident = run(0, cache);
    for (int offload : {1, 5, 6}) {
      const auto streamed = run(offload, cache);
      CHECK_CLOSE(resident.first, streamed.first, 0.0, "streamed blocks/LoRA preserve repeated and cached forwards");
      if (offload > 2) CHECK(streamed.second < resident.second);
    }
  }
  std::filesystem::remove(path); std::filesystem::remove(adapter_path);
}

SLOPFAB_TEST_CATEGORY(transformer_wrapped_checkpoint_matches_unwrapped, "synthetic") {
  const TransformerConfig cfg = tiny_config();
  const Tensors tensors = build_synthetic(cfg);
  const Case c = make_case(cfg, 5, 0.31f);
  auto evaluate = [&](const Tensors& weights, bool interleaved = false) {
    const std::string path = write_synthetic(weights, interleaved
        ? std::map<std::string, std::string>{{"source", "Viggle/Viggle-Animate"},
                                           {"qkv_layout", "interleaved"}}
        : std::map<std::string, std::string>{});
    slopfab::SafeTensors st;
    st.open(path);
    Transformer model;
    model.load(st, cfg);
    model.prepare_text(c.prompt.data(), c.layout.num_text);
    model.prepare_sequence(c.layout, c.idx, c.pos);
    std::vector<float> result(c.video_rows.size() + c.audio_rows.size());
    model.forward(c.video_rows.data(), c.audio_rows.data(), c.rt,
                  result.data(), result.data() + c.video_rows.size());
    CHECK(all_finite(result));
    return result;
  };
  const auto expected = evaluate(tensors);
  Tensors wrapped;
  for (const auto& kv : tensors) {
    auto tensor = kv.second;
    tensor.name = "model.diffusion_model." + tensor.name;
    wrapped.emplace(tensor.name, std::move(tensor));
  }
  CHECK(expected == evaluate(wrapped));
  Tensors viggle = tensors;
  for (auto& kv : viggle) {
    auto& tensor = kv.second;
    if (tensor.name.find(".attn.qkv_proj.weight") == std::string::npos ||
        tensor.shape.size() != 2) continue;
    const auto original = tensor.data;
    const int cols = int(tensor.shape[1]);
    size_t dst = 0;
    for (int head = 0; head < cfg.num_attention_heads; ++head)
      for (int part = 0; part < 3; ++part)
        for (int channel = 0; channel < cfg.attention_head_dim; ++channel)
          for (int col = 0; col < cols; ++col)
            tensor.data[dst++] = original[size_t(part * cfg.inner_dim() +
                head * cfg.attention_head_dim + channel) * cols + col];
  }
  CHECK(expected == evaluate(viggle, true));
}

SLOPFAB_TEST_CATEGORY(transformer_viggle_lora_targets_reach_forward, "synthetic") {
  const auto cfg = tiny_config();
  const auto tensors = build_synthetic(cfg);
  const auto c = make_case(cfg, 259, .31f);
  const auto adapter_path = std::filesystem::temp_directory_path() / "slopfab_viggle_cuda_lora.safetensors";
  auto evaluate = [&](const Tensors& weights, const std::vector<slopfab::TensorWrite>& adapter,
                      bool compact = true) {
    slopfab::SafeTensors base;
    base.open(write_synthetic(weights));
    slopfab::LoraAdapters loras;
    if (!adapter.empty()) {
      slopfab::write_safetensors(adapter_path.string(), adapter);
      loras.load({{adapter_path.string(), 1}}, base);
    }
    Transformer model;
    model.load(base, cfg, &loras);
    model.set_row_chunk(128);
    model.set_query_chunking(compact);
    model.prepare_text(c.prompt.data(), c.layout.num_text);
    model.prepare_sequence(c.layout, c.idx, c.pos);
    std::vector<float> result(c.video_rows.size() + c.audio_rows.size());
    model.forward(c.video_rows.data(), c.audio_rows.data(), c.rt,
                  result.data(), result.data() + c.video_rows.size());
    CHECK(all_finite(result));
    return result;
  };
  const auto baseline = evaluate(tensors, {});
  struct Target { const char* source; const char* native; int part; };
  for (const Target& target : std::vector<Target>{
      {"proj_in", "video_patch_proj", -1}, {"proj_out", "final_layer.video_out", -1},
      {"transformer_blocks.0.attn.to_q", "blocks.0.attn.qkv_proj", 0},
      {"transformer_blocks.0.attn.to_k", "blocks.0.attn.qkv_proj", 1},
      {"transformer_blocks.0.attn.to_v", "blocks.0.attn.qkv_proj", 2},
      {"transformer_blocks.0.attn.to_out.0", "blocks.0.attn.out_proj", -1},
      {"transformer_blocks.0.ff.net.0.proj", "blocks.0.mlp.fc1", -1},
      {"transformer_blocks.0.ff.net.2", "blocks.0.mlp.fc2", -1},
      {"blocks.0.adaln_proj.linear", "blocks.0.adaln_proj.linear", -1},
      {"final_layer.adaln_proj.linear", "final_layer.adaln_proj.linear", -1}}) {
    const auto& weight = tensors.at(std::string(target.native) + ".weight");
    const int in = int(weight.shape[1]), out = int(weight.shape[0]) / (target.part < 0 ? 1 : 3);
    constexpr int rank = 16;
    const auto a = make_data(size_t(rank) * in, 893, .25f);
    const auto b = make_data(size_t(out) * rank, 981, .25f);
    const auto actual = evaluate(tensors, {
        {std::string(target.source) + ".lora_A.weight", {rank, in}, a},
        {std::string(target.source) + ".lora_B.weight", {out, rank}, b}});
    CHECK_CLOSE(actual, evaluate(tensors, {
        {std::string(target.source) + ".lora_A.weight", {rank, in}, a},
        {std::string(target.source) + ".lora_B.weight", {out, rank}, b}}, false),
        0.0, "compact attention preserves LoRA contributions");
    CHECK_MSG(actual != baseline, "%s update did not reach forward", target.source);
    if (std::string(target.source).compare(0, 5, "proj_") == 0 ||
        std::string(target.source).find("adaln_proj") != std::string::npos) {
      auto merged = tensors;
      auto& w = merged.at(std::string(target.native) + ".weight").data;
      for (int row = 0; row < out; ++row) for (int col = 0; col < in; ++col) {
        double delta = 0;
        for (int r = 0; r < rank; ++r) delta += double(b[row * rank + r]) * a[r * in + col];
        w[row * in + col] = float(double(w[row * in + col]) + delta);
      }
      CHECK(actual == evaluate(merged, {}));
    } else {
      auto native_b = b;
      if (target.part >= 0) {
        native_b.assign(size_t(out) * 3 * rank, 0);
        std::copy(b.begin(), b.end(), native_b.begin() + size_t(target.part) * out * rank);
      } else if (std::string(target.native) == "blocks.0.mlp.fc1") {
        std::rotate(native_b.begin(), native_b.begin() + native_b.size() / 2, native_b.end());
      }
      CHECK(actual == evaluate(tensors, {
          {std::string(target.native) + ".lora_A.weight", {rank, in}, a},
          {std::string(target.native) + ".lora_B.weight", {weight.shape[0], rank}, native_b}}));
    }
  }
  std::filesystem::remove(adapter_path);
}

SLOPFAB_TEST_CATEGORY(transformer_load_rejects_bad_shapes, "synthetic") {
  const TransformerConfig cfg = tiny_config();
  Tensors tensors = build_synthetic(cfg);
  // A qkv_proj that is 2*inner rows instead of 3*inner is exactly what a port
  // that forgot `v` would produce, and it must not load.
  tensors["blocks.0.attn.qkv_proj.weight"].shape = {2 * cfg.inner_dim(), cfg.hidden_size};
  tensors["blocks.0.attn.qkv_proj.weight"].data.resize(
      static_cast<size_t>(2 * cfg.inner_dim()) * cfg.hidden_size);
  const std::string path = write_synthetic(tensors);

  slopfab::SafeTensors st;
  st.open(path);
  Transformer model;
  bool threw = false;
  std::string message;
  try {
    model.load(st, cfg);
  } catch (const std::exception& e) {
    threw = true;
    message = e.what();
  }
  CHECK(threw);
  CHECK_MSG(message.find("blocks.0.attn.qkv_proj.weight") != std::string::npos,
            "load error does not name the offending tensor: %s", message.c_str());

  st.close();
  std::error_code ec;
  std::filesystem::remove(path, ec);
}
