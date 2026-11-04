#include "internal.h"
#include "slopfab/tensor_convert.h"
#include <algorithm>
#include <filesystem>
#include <stdexcept>

namespace slopfab::whisper {
Config inspect(const SafeTensors& checkpoint, const std::string& requested) {
  const auto& conv = checkpoint.at("model.encoder.conv1.weight");
  const auto& embed = checkpoint.at("model.decoder.embed_tokens.weight");
  if (conv.shape.size() != 3 || embed.shape.size() != 2)
    throw std::invalid_argument("Whisper: invalid model dimensions");
  int enc = 0, dec = 0;
  while (checkpoint.find("model.encoder.layers." + std::to_string(enc) + ".fc1.weight"))
    ++enc;
  while (checkpoint.find("model.decoder.layers." + std::to_string(dec) + ".fc1.weight"))
    ++dec;
  std::string variant = requested == "turbo" ? "large-v3-turbo" : requested;
  std::vector<Config> candidates;
  for (const auto& entry : assets().find("profiles")->as_object()) {
    auto integer = [&](const char* name) {
      return int(entry.second.find(name)->as_number());
    };
    if (variant != "auto" && variant != entry.first)
      continue;
    Config c;
    c.name = entry.first;
    c.width = integer("width");
    c.heads = integer("heads");
    c.encoder_layers = integer("encoder_layers");
    c.decoder_layers = integer("decoder_layers");
    c.mels = integer("mels");
    c.vocab = integer("vocab");
    c.multilingual = c.vocab != 51864;
    if (conv.shape != std::vector<int64_t>{c.width, c.mels, 3} ||
        embed.shape != std::vector<int64_t>{c.vocab, c.width} || enc != c.encoder_layers ||
        dec != c.decoder_layers)
      continue;
    for (const auto& v : entry.second.find("alignment_heads")->as_array())
      c.alignment_heads.emplace_back(int(v.as_array()[0].as_number()),
                                     int(v.as_array()[1].as_number()));
    candidates.push_back(c);
  }
  if (candidates.empty())
    throw std::invalid_argument("Whisper: weights do not match a supported standard profile");
  if (candidates.size() > 1)
    throw std::invalid_argument(
        "Whisper: large-v1/v2 have identical shapes; select --variant large-v1 or large-v2");
  auto c = candidates.front();
  auto tensor = [&](const std::string& name, std::vector<int64_t> shape) {
    const auto& t = checkpoint.at(name);
    if (t.shape != shape ||
        (t.dtype != DType::kF16 && t.dtype != DType::kBF16 && t.dtype != DType::kF32))
      throw std::invalid_argument("Whisper: unsupported tensor shape/dtype: " + name);
  };
  auto norm = [&](const std::string& p) {
    tensor(p + ".weight", {c.width});
    tensor(p + ".bias", {c.width});
  };
  auto linear = [&](const std::string& p, int out, int in, bool bias = true) {
    tensor(p + ".weight", {out, in});
    if (bias)
      tensor(p + ".bias", {out});
  };
  tensor("model.encoder.conv1.weight", {c.width, c.mels, 3});
  tensor("model.encoder.conv1.bias", {c.width});
  tensor("model.encoder.conv2.weight", {c.width, c.width, 3});
  tensor("model.encoder.conv2.bias", {c.width});
  tensor("model.encoder.embed_positions.weight", {1500, c.width});
  tensor("model.decoder.embed_positions.weight", {448, c.width});
  tensor("model.decoder.embed_tokens.weight", {c.vocab, c.width});
  if (checkpoint.find("proj_out.weight"))
    tensor("proj_out.weight", {c.vocab, c.width});
  for (bool decoder : {false, true}) {
    std::string root = decoder ? "model.decoder." : "model.encoder.";
    norm(root + "layer_norm");
    for (int i = 0; i < (decoder ? c.decoder_layers : c.encoder_layers); ++i) {
      std::string p = root + "layers." + std::to_string(i) + ".";
      norm(p + "self_attn_layer_norm");
      norm(p + "final_layer_norm");
      linear(p + "fc1", 4 * c.width, c.width);
      linear(p + "fc2", c.width, 4 * c.width);
      for (const std::string& att : decoder ? std::vector<std::string>{"self_attn", "encoder_attn"}
                                            : std::vector<std::string>{"self_attn"}) {
        if (att == "encoder_attn")
          norm(p + "encoder_attn_layer_norm");
        for (const char* q : {"q_proj", "k_proj", "v_proj", "out_proj"})
          linear(p + att + "." + q, c.width, c.width, std::string(q) != "k_proj");
      }
    }
  }
  return c;
}

Model::Model(const Options& o) {
  auto path = std::filesystem::u8path(o.model);
  if (std::filesystem::is_directory(path))
    path /= "model.safetensors";
  checkpoint.open(path.u8string());
  config = inspect(checkpoint, o.variant);
  backend_name = o.backend;
  if (backend_name == "auto") {
#if SLOPFAB_WITH_CUDA
    backend_name = "cuda";
#elif SLOPFAB_WITH_VULKAN
    backend_name = "vulkan";
#else
    backend_name = "cpu";
#endif
  }
  if (backend_name == "cpu")
    backend = cpu_backend();
#if SLOPFAB_WITH_CUDA
  if (backend_name == "cuda")
    backend = cuda_backend();
#endif
#if SLOPFAB_WITH_VULKAN
  if (backend_name == "vulkan")
    backend = vulkan_backend();
#endif
  if (!backend)
    throw std::invalid_argument("Whisper: requested backend is unavailable: " + backend_name);
  check = [](const std::string&) {
  };
}

const Tensor& Model::weight(const std::string& name) {
  auto found = weights.find(name);
  if (found != weights.end())
    return found->second;
  const auto& t = checkpoint.at(name);
  int rows = t.shape.size() == 1 ? 1 : int(t.shape[0]);
  auto value = backend->upload(to_f32(t), rows, int(t.numel() / rows));
  return weights.emplace(name, std::move(value)).first->second;
}

Tensor Model::projection(const Tensor& x, const std::string& p) {
  return backend->linear(x, weight(p + ".weight"),
                         checkpoint.find(p + ".bias") ? &weight(p + ".bias") : nullptr);
}

Tensor Model::normalization(const Tensor& x, const std::string& p) {
  return backend->norm(x, weight(p + ".weight"), weight(p + ".bias"));
}

void Model::encode(const std::vector<float>& mel) {
  if (mel.size() != size_t(3000 * config.mels))
    throw std::invalid_argument("Whisper: expected 30-second mel window");
  reset();
  cross_cache.clear();
  auto x = backend->upload(mel, 3000, config.mels);
  x = backend->gelu(projection(backend->im2col(x, 1), "model.encoder.conv1"));
  x = backend->gelu(projection(backend->im2col(x, 2), "model.encoder.conv2"));
  x = backend->add(x, weight("model.encoder.embed_positions.weight"));
  for (int i = 0; i < config.encoder_layers; ++i) {
    check("encoder");
    auto p = "model.encoder.layers." + std::to_string(i) + ".";
    auto n = normalization(x, p + "self_attn_layer_norm");
    auto q = projection(n, p + "self_attn.q_proj"), k = projection(n, p + "self_attn.k_proj"),
         v = projection(n, p + "self_attn.v_proj");
    x = backend->add(
        x, projection(backend->attention(q, k, v, config.heads, false), p + "self_attn.out_proj"));
    n = normalization(x, p + "final_layer_norm");
    x = backend->add(x, projection(backend->gelu(projection(n, p + "fc1")), p + "fc2"));
  }
  audio = normalization(x, "model.encoder.layer_norm");
}

void Model::reset() {
  self_cache.assign(config.decoder_layers, {});
}

std::vector<float> Model::decode(const std::vector<int>& ids,
                                 std::vector<std::vector<float>>* alignment) {
  if (ids.empty() || !audio.data)
    throw std::invalid_argument("Whisper: empty decoder input/audio");
  int offset = self_cache[0].k.rows;
  if (offset + int(ids.size()) > 448)
    throw std::invalid_argument("Whisper: decoder context exceeded");
  const auto& t = checkpoint.at("model.decoder.embed_tokens.weight");
  std::vector<float> embeddings;
  for (int id : ids) {
    if (id < 0 || id >= config.vocab)
      throw std::invalid_argument("Whisper: token out of range");
    TensorView row = t;
    row.shape = {config.width};
    row.nbytes = t.nbytes / config.vocab;
    row.data = static_cast<const uint8_t*>(t.data) + size_t(id) * row.nbytes;
    auto v = to_f32(row);
    embeddings.insert(embeddings.end(), v.begin(), v.end());
  }
  auto x = backend->add(
      backend->upload(embeddings, int(ids.size()), config.width),
      backend->rows(weight("model.decoder.embed_positions.weight"), offset, int(ids.size())));
  if (cross_cache.empty())
    cross_cache.resize(config.decoder_layers);
  if (alignment)
    alignment->clear();
  for (int i = 0; i < config.decoder_layers; ++i) {
    check("decoder");
    auto p = "model.decoder.layers." + std::to_string(i) + ".";
    auto n = normalization(x, p + "self_attn_layer_norm");
    auto q = projection(n, p + "self_attn.q_proj");
    auto& cache = self_cache[i];
    cache.k = backend->concat(cache.k, projection(n, p + "self_attn.k_proj"));
    cache.v = backend->concat(cache.v, projection(n, p + "self_attn.v_proj"));
    // Prefill is causal; later calls consume one new token attending all cached keys.
    if (offset && ids.size() != 1)
      throw std::invalid_argument("Whisper: cached decode requires one token");
    x = backend->add(x,
                     projection(backend->attention(q, cache.k, cache.v, config.heads, offset == 0),
                                p + "self_attn.out_proj"));
    n = normalization(x, p + "encoder_attn_layer_norm");
    q = projection(n, p + "encoder_attn.q_proj");
    auto& cross = cross_cache[i];
    if (!cross.k.data) {
      cross.k = projection(audio, p + "encoder_attn.k_proj");
      cross.v = projection(audio, p + "encoder_attn.v_proj");
    }
    std::vector<float> probabilities;
    bool capture = alignment && std::any_of(config.alignment_heads.begin(),
                                            config.alignment_heads.end(), [&](auto h) {
                                              return h.first == i;
                                            });
    auto a = backend->attention(q, cross.k, cross.v, config.heads, false,
                                capture ? &probabilities : nullptr);
    if (capture)
      for (auto [layer, head] : config.alignment_heads)
        if (layer == i) {
          size_t count = size_t(q.rows) * cross.k.rows;
          alignment->emplace_back(probabilities.begin() + head * count,
                                  probabilities.begin() + (head + 1) * count);
        }
    x = backend->add(x, projection(a, p + "encoder_attn.out_proj"));
    n = normalization(x, p + "final_layer_norm");
    x = backend->add(x, projection(backend->gelu(projection(n, p + "fc1")), p + "fc2"));
  }
  x = backend->rows(normalization(x, "model.decoder.layer_norm"), x.rows - 1, 1);
  // Tile the vocabulary head below portable Vulkan storage-buffer limits.
  const std::string name =
      checkpoint.find("proj_out.weight") ? "proj_out.weight" : "model.decoder.embed_tokens.weight";
  const auto& source = checkpoint.at(name);
  std::vector<float> logits;
  for (int first = 0; first < config.vocab; first += 4096) {
    int count = std::min(4096, config.vocab - first);
    const std::string key = name + ":" + std::to_string(first);
    auto found = weights.find(key);
    if (found == weights.end()) {
      auto view = source;
      size_t row_bytes = source.nbytes / config.vocab;
      view.data = static_cast<const uint8_t*>(source.data) + size_t(first) * row_bytes;
      view.shape = {count, config.width};
      view.nbytes = size_t(count) * row_bytes;
      found = weights.emplace(key, backend->upload(to_f32(view), count, config.width)).first;
    }
    auto part = backend->download(backend->linear(x, found->second));
    logits.insert(logits.end(), part.begin(), part.end());
  }
  return logits;
}
}
