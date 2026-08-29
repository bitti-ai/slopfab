// Host-side orchestration of the MiniMax H3 audio VAE decoder.
//
// The graph is a straight chain — no tiling, no chunking, no cross-fading. A
// ten-second clip is 405 latents in and 324,000 samples out, and the widest
// intermediate is 256 channels by 10,125 samples, so the whole decode fits in
// about 160 MiB of activations no matter which stage is running. See
// docs/audio_vae_spec.md, §2 for the graph and §13 for these sizes.
//
// Two structural facts worth keeping in view while reading this:
//
//   * The encoder half of the checkpoint (encoder.*, pre_block.*, mean_proj.*,
//     logs_proj.* — 136 tensors, 281 MiB, including the whole 8-head causal
//     attention projection) is NOT part of decode and is never uploaded.
//     DacAudioVAE.decode is two lines: dec_in_proj, then BigVGAN.
//   * The two stereo channels are two batch items through one mono decoder, so
//     everything below carries a batch dimension of 2 and only the very last
//     step interleaves.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vidfab/cuda/audio_vae_kernels.cuh"
#include "vidfab/cuda/device.h"
#include "vidfab/tensor_convert.h"
#include "vidfab/vae/audio_decoder.h"
#include "vidfab/vae/audio_primitives.h"

namespace vidfab::vae {
namespace {

using ::vidfab::cuda::DeviceBuffer;
using ::vidfab::cuda::Stream;

constexpr int kStereo = 2;
constexpr int kResblockDilations[3] = {1, 3, 5};

// get_padding(kernel_size, dilation) — dac_utils.py:11-12. SAME padding for the
// odd kernels this model uses.
int get_padding(int kernel, int dilation) { return (kernel * dilation - dilation) / 2; }

// A slice of the single weight allocation. Offsets are in floats.
struct Slice {
  size_t offset = 0;
  size_t count = 0;
};

struct ConvSpec {
  Slice weight;
  Slice bias;  // count == 0 when the conv ships without one (conv_post)
  int out_channels = 0;
  int in_channels = 0;
  int kernel = 0;
};

// One Activation1d: the anti-aliased SnakeBeta wrapper.
struct ActSpec {
  Slice log_alpha;
  Slice log_beta;
  Slice up_filter;
  Slice down_filter;
  int channels = 0;
};

struct AmpBlockSpec {
  int channels = 0;
  int kernel = 0;
  ConvSpec convs1[3];
  ConvSpec convs2[3];
  ActSpec acts[6];
};

struct StageSpec {
  ConvSpec up;  // ConvTranspose1d, weight is [Cin, Cout, K]
  int rate = 0;
  int kernel = 0;
  AmpBlockSpec blocks[3];
};

}  // namespace

struct AudioDecoder::Impl {
  AudioVAEConfig config;
  Stream stream;

  DeviceBuffer<float> weights;
  size_t weight_floats = 0;
  std::vector<float> latents_mean;
  std::vector<float> latents_std;

  ConvSpec dec_in_proj;
  ConvSpec conv_pre;
  std::vector<StageSpec> stages;
  ActSpec activation_post;
  ConvSpec conv_post;

  // Activation pool. Six equal buffers plus one double-width scratch for the
  // 2x anti-alias intermediate; sized once from the widest (channels * length)
  // product the graph reaches.
  DeviceBuffer<float> pool[6];
  DeviceBuffer<float> aa_scratch;
  size_t pool_floats = 0;

  const float* w(const Slice& s) const { return weights.get() + s.offset; }

  void ensure_pool(size_t floats) {
    if (floats <= pool_floats) return;
    for (auto& buf : pool) buf.allocate(floats);
    aa_scratch.allocate(2 * floats);
    pool_floats = floats;
  }

  void run_conv(const ConvSpec& conv, const float* x, float* y, int batch, int len_in, int len_out,
                int pad, int dilation) const {
    cuda::launch_conv1d(x, w(conv.weight), conv.bias.count != 0 ? w(conv.bias) : nullptr, y, batch,
                        conv.in_channels, conv.out_channels, len_in, len_out, conv.kernel, pad,
                        dilation, stream.get());
  }

  // Activation1d: 2x upsample -> SnakeBeta -> 2x lowpass decimation. Length in
  // equals length out; the two halves are separate kernels because the
  // intermediate is twice the size of everything else in flight.
  void run_activation(const ActSpec& act, const float* x, float* y, int batch, int len,
                      float* scratch) const {
    cuda::launch_aa_upsample_snake(x, w(act.up_filter), w(act.log_alpha), w(act.log_beta), scratch,
                                   batch, act.channels, len, stream.get());
    cuda::launch_aa_downsample(scratch, w(act.down_filter), y, batch, act.channels,
                               cuda::kAudioAARatio * len, len, stream.get());
  }

  // AMPBlock1.forward (dac_bigvgan.py:97-106). Note the activation pairing: the
  // j-th dilation uses activations[2j] and activations[2j+1], because the
  // reference slices the list as [::2] and [1::2]. Reading them as "the first
  // three go before convs1" produces a plausible waveform and is wrong.
  void run_amp_block(const AmpBlockSpec& block, float* x, int batch, int len, float* t1, float* t2,
                     float* scratch) const {
    const size_t elems = static_cast<size_t>(batch) * block.channels * len;
    for (int j = 0; j < 3; ++j) {
      const int dilation = kResblockDilations[j];
      run_activation(block.acts[2 * j], x, t1, batch, len, scratch);
      run_conv(block.convs1[j], t1, t2, batch, len, len, get_padding(block.kernel, dilation),
               dilation);
      run_activation(block.acts[2 * j + 1], t2, t1, batch, len, scratch);
      // convs2 is always dilation 1, whatever convs1 used (dac_bigvgan.py:66-81).
      run_conv(block.convs2[j], t1, t2, batch, len, len, get_padding(block.kernel, 1), 1);
      cuda::launch_add_inplace(x, t2, elems, stream.get());
    }
  }
};

namespace {

// Staging accumulator: copies each fp32 tensor into one contiguous host vector
// and records its offset, so the whole decode path becomes a single cudaMalloc
// plus a single upload. 779 tensors, 247.64 MiB.
class Staging {
 public:
  explicit Staging(const SafeTensors& checkpoint) : ckpt_(checkpoint) {}

  Slice add(const std::string& name, std::initializer_list<int64_t> expect) {
    const TensorView& t = ckpt_.at(name);
    if (t.dtype != DType::kF32 && t.dtype != DType::kF16 && t.dtype != DType::kBF16)
      throw std::runtime_error("audio vae: " + name + " is not a floating tensor");
    if (t.shape.size() != expect.size()) {
      throw std::runtime_error("audio vae: " + name + " has rank " +
                               std::to_string(t.shape.size()) + ", expected " +
                               std::to_string(expect.size()));
    }
    size_t i = 0;
    for (int64_t want : expect) {
      if (t.shape[i] != want) {
        throw std::runtime_error("audio vae: " + name + " dim " + std::to_string(i) + " is " +
                                 std::to_string(t.shape[i]) + ", expected " +
                                 std::to_string(want));
      }
      ++i;
    }
    const size_t count = static_cast<size_t>(t.numel());
    const Slice slice{data_.size(), count};
    const std::vector<float> values = to_f32(t);
    data_.insert(data_.end(), values.begin(), values.end());
    ++tensors_;
    return slice;
  }

  Slice add_conv(const std::string& name, std::initializer_list<int64_t> expect,
                 uint32_t bias_channels) {
    const std::vector<int64_t> shape(expect);
    AudioConvWeights loaded =
        load_audio_conv_weights(ckpt_, name, shape, bias_channels, false);
    const Slice slice{data_.size(), loaded.weight.size()};
    data_.insert(data_.end(), loaded.weight.begin(), loaded.weight.end());
    tensors_ += loaded.folded_weight_norm ? 2 : 1;
    return slice;
  }

  const std::vector<float>& data() const { return data_; }
  size_t tensors() const { return tensors_; }

 private:
  const SafeTensors& ckpt_;
  std::vector<float> data_;
  size_t tensors_ = 0;
};

std::string join(const std::string& prefix, int index, const std::string& suffix) {
  return prefix + std::to_string(index) + suffix;
}

}  // namespace

AudioDecoder::AudioDecoder() : impl_(std::make_unique<Impl>()) {}
AudioDecoder::~AudioDecoder() = default;

void AudioDecoder::load(const SafeTensors& checkpoint, const AudioVAEConfig& config) {
  Impl& im = *impl_;
  // See the note on `SafeTensors::prefetch`: this loader consumes the whole
  // file, so it asks for it up front rather than one page fault at a time.
  checkpoint.prefetch();
  im.config = config;

  if (config.decoder_rates.size() != config.decoder_kernel_sizes.size()) {
    throw std::runtime_error("audio vae: decoder_rates and decoder_kernel_sizes differ in length");
  }
  // The reference pairs rate u with kernel 2u for even u and 2u-1 for odd u
  // (dac_audio_vae.py:178-179). Checking it here rather than trusting the
  // config is cheap and catches a `kernel = 2*rate` "simplification".
  for (size_t i = 0; i < config.decoder_rates.size(); ++i) {
    const int u = config.decoder_rates[i];
    const int expect = 2 * u - (u % 2);
    if (config.decoder_kernel_sizes[i] != expect) {
      throw std::runtime_error("audio vae: decoder stage " + std::to_string(i) + " pairs rate " +
                               std::to_string(u) + " with kernel " +
                               std::to_string(config.decoder_kernel_sizes[i]) + ", expected " +
                               std::to_string(expect));
    }
  }

  const int zc = config.latent_channels;
  const int ld = config.latent_dim;
  const int dd = config.decoder_dim;

  Staging st(checkpoint);

  im.dec_in_proj = ConvSpec{st.add_conv("dec_in_proj", {ld, zc, 1}, ld),
                            st.add("dec_in_proj.bias", {ld}), ld, zc, 1};
  im.conv_pre = ConvSpec{st.add_conv("decoder.conv_pre", {dd, ld, 7}, dd),
                         st.add("decoder.conv_pre.bias", {dd}), dd, ld, 7};

  const int num_stages = static_cast<int>(config.decoder_rates.size());
  const int num_kernels = static_cast<int>(config.resblock_kernel_sizes.size());
  // AMPBlock1 is fixed at three dilations and BigVGAN at three resblock kernels
  // (dac_audio_vae.py:181-182); the per-stage arrays are sized for that.
  if (num_kernels != 3) {
    throw std::runtime_error("audio vae: expected 3 resblock kernel sizes, got " +
                             std::to_string(num_kernels));
  }
  im.stages.clear();
  im.stages.reserve(static_cast<size_t>(num_stages));

  int ch = dd;
  for (int i = 0; i < num_stages; ++i) {
    StageSpec stage;
    stage.rate = config.decoder_rates[static_cast<size_t>(i)];
    stage.kernel = config.decoder_kernel_sizes[static_cast<size_t>(i)];
    const int out_ch = ch / 2;

    // ConvTranspose1d weights are [Cin, Cout, K] — input channels first, the
    // opposite of Conv1d. See docs/audio_vae_spec.md §5.
    const std::string up = "decoder.ups." + std::to_string(i) + ".0.";
    stage.up = ConvSpec{st.add_conv(up.substr(0, up.size() - 1),
                                    {ch, out_ch, stage.kernel}, out_ch),
                        st.add(up + "bias", {out_ch}), out_ch, ch, stage.kernel};

    for (int j = 0; j < num_kernels; ++j) {
      const int rb = i * num_kernels + j;
      const std::string p = "decoder.resblocks." + std::to_string(rb) + ".";
      AmpBlockSpec& block = stage.blocks[j];
      block.channels = out_ch;
      block.kernel = config.resblock_kernel_sizes[static_cast<size_t>(j)];
      for (int d = 0; d < 3; ++d) {
        block.convs1[d] =
            ConvSpec{st.add_conv(join(p + "convs1.", d, ""),
                                 {out_ch, out_ch, block.kernel}, out_ch),
                     st.add(join(p + "convs1.", d, ".bias"), {out_ch}), out_ch, out_ch,
                     block.kernel};
        block.convs2[d] =
            ConvSpec{st.add_conv(join(p + "convs2.", d, ""),
                                 {out_ch, out_ch, block.kernel}, out_ch),
                     st.add(join(p + "convs2.", d, ".bias"), {out_ch}), out_ch, out_ch,
                     block.kernel};
      }
      for (int a = 0; a < 6; ++a) {
        const std::string ap = join(p + "activations.", a, ".");
        block.acts[a] = ActSpec{st.add(ap + "act.alpha", {out_ch}),
                                st.add(ap + "act.beta", {out_ch}),
                                st.add(ap + "upsample.filter", {1, 1, cuda::kAudioAAKernel}),
                                st.add(ap + "downsample.lowpass.filter",
                                       {1, 1, cuda::kAudioAAKernel}),
                                out_ch};
      }
    }
    im.stages.push_back(stage);
    ch = out_ch;
  }

  im.activation_post =
      ActSpec{st.add("decoder.activation_post.act.alpha", {ch}),
              st.add("decoder.activation_post.act.beta", {ch}),
              st.add("decoder.activation_post.upsample.filter", {1, 1, cuda::kAudioAAKernel}),
              st.add("decoder.activation_post.downsample.lowpass.filter",
                     {1, 1, cuda::kAudioAAKernel}),
              ch};

  // use_bias_at_final is false for this checkpoint, so conv_post has no bias
  // tensor at all (dac_audio_vae.py:184). Requiring one would fail the load.
  if (checkpoint.find("decoder.conv_post.bias") != nullptr) {
    throw std::runtime_error(
        "audio vae: decoder.conv_post.bias is present, but the 32 kHz config sets "
        "use_bias_at_final=false");
  }
  im.conv_post = ConvSpec{st.add_conv("decoder.conv_post", {1, ch, 7}, 0),
                          Slice{}, 1, ch, 7};

  // Latent statistics ship as tensors as well as in the metadata JSON; prefer
  // the tensors so no JSON has to be parsed on the decode path.
  if (const TensorView* mean = checkpoint.find("latents_mean")) {
    im.latents_mean = to_f32(*mean);
    im.latents_std = to_f32(checkpoint.at("latents_std"));
  } else {
    im.latents_mean = {
        -.0202116875f, .3876466480f, -.0439827980f, -.2859151494f, .0817968621f,
        -.3578264135f, .0406238100f, -.0155253450f, -.2233624817f, .1821006843f,
        .2941778784f, -.0790116760f, -.0568150728f, -.3699028222f, -.3161631559f,
        .5905951377f, -.0521395681f, .0136731603f, -.0369164786f, .0973266065f,
        -.3394662329f, -.3068567754f, -.2450459891f, -.0346985245f, .0286803218f,
        -.2121777927f, -.1678263170f, .3221287889f, -.1223055852f, .4356604928f,
        -.0502599202f, .3979258376f};
    im.latents_std = {
        1.6895524230f, 2.7626372722f, 1.7945344281f, 1.6801681847f, 1.6390226547f,
        2.7788298349f, 1.7659090096f, 1.6199757612f, 2.6336525640f, 1.8539356673f,
        2.5056497897f, 1.8110192379f, 1.9579657791f, 1.6685498244f, 1.4922469314f,
        3.2986701981f, 1.9491804497f, 1.8720003270f, 1.8334080103f, 1.6488070417f,
        1.6176957696f, 1.9131449235f, 1.5695245398f, 1.6943659940f, 1.8318420763f,
        1.5540637422f, 1.9344930329f, 1.5991982161f, 1.7180459898f, 1.6307219191f,
        1.8661226051f, 1.5613768203f};
  }
  if (im.latents_mean.size() != static_cast<size_t>(zc) ||
      im.latents_std.size() != static_cast<size_t>(zc)) {
    throw std::runtime_error("audio vae: latents_mean/latents_std are not [" +
                             std::to_string(zc) + "]");
  }

  im.weight_floats = st.data().size();
  im.weights.allocate(im.weight_floats);
  im.weights.copy_from_host(st.data().data(), im.weight_floats, im.stream.get());
  im.stream.synchronize();
}

const AudioVAEConfig& AudioDecoder::config() const { return impl_->config; }

size_t AudioDecoder::weight_bytes() const { return impl_->weight_floats * sizeof(float); }

void AudioDecoder::unload() {
  Impl& im = *impl_;
  im.weights.reset();
  im.weight_floats = 0;
  im.stages.clear();
  im.latents_mean.clear();
  im.latents_std.clear();
  for (auto& buf : im.pool) buf.reset();
  im.aa_scratch.reset();
  im.pool_floats = 0;
}

const std::vector<float>& AudioDecoder::latents_mean() const { return impl_->latents_mean; }
const std::vector<float>& AudioDecoder::latents_std() const { return impl_->latents_std; }

DecodedAudio AudioDecoder::decode(const float* latents, int num_latents,
                                  AudioDecodeTrace* trace) {
  Impl& im = *impl_;
  if (im.weight_floats == 0) throw std::runtime_error("audio vae: decode before load");
  if (latents == nullptr || num_latents <= 0) {
    throw std::runtime_error("audio vae: decode needs at least one latent");
  }

  const AudioVAEConfig& cfg = im.config;
  // The decoder itself is mono; the two stereo channels ride through it as two
  // batch items (decoders.py:130), so the batch dimension IS the channel count.
  const int batch = cfg.output_channels;
  if (batch != kStereo) {
    throw std::runtime_error("audio vae: decode expects [2, 32, A], got output_channels " +
                             std::to_string(batch));
  }
  const int zc = cfg.latent_channels;
  const int total_upsample = cfg.total_upsample();

  // Peak (channels * length) over the whole graph. It is flat from stage 1
  // onward — every upsample halves the channels and doubles or quintuples the
  // length — but computing it rather than assuming keeps the pool honest if the
  // rate table ever changes.
  size_t widest = static_cast<size_t>(cfg.latent_dim) * num_latents;
  {
    size_t len = static_cast<size_t>(num_latents);
    size_t ch = static_cast<size_t>(cfg.decoder_dim);
    widest = std::max(widest, ch * len);
    for (const StageSpec& stage : im.stages) {
      len *= static_cast<size_t>(stage.rate);
      ch /= 2;
      widest = std::max(widest, ch * len);
    }
  }
  im.ensure_pool(widest * static_cast<size_t>(batch));

  float* buf_a = im.pool[0].get();
  float* buf_b = im.pool[1].get();
  float* acc = im.pool[2].get();
  float* work = im.pool[3].get();
  float* t1 = im.pool[4].get();
  float* t2 = im.pool[5].get();
  float* scratch = im.aa_scratch.get();
  if (trace != nullptr) {
    trace->boundaries.clear();
    trace->boundaries.reserve(13);
  }
  auto capture = [&](const float* source, size_t count) {
    if (trace == nullptr) return;
    trace->boundaries.emplace_back(count);
    VIDFAB_CUDA_CHECK(cudaMemcpyAsync(trace->boundaries.back().data(), source,
                                      count * sizeof(float),
                                      cudaMemcpyDeviceToHost, im.stream.get()));
  };

  // Upload [2, 32, A] and project it up to the BigVGAN input width.
  const size_t latent_floats = static_cast<size_t>(batch) * zc * num_latents;
  DeviceBuffer<float> z(latent_floats);
  z.copy_from_host(latents, latent_floats, im.stream.get());

  im.run_conv(im.dec_in_proj, z.get(), buf_a, batch, num_latents, num_latents, 0, 1);
  capture(buf_a, static_cast<size_t>(batch) * cfg.latent_dim * num_latents);
  im.run_conv(im.conv_pre, buf_a, buf_b, batch, num_latents, num_latents, 3, 1);
  capture(buf_b, static_cast<size_t>(batch) * cfg.decoder_dim * num_latents);

  float* cur = buf_b;
  float* spare = buf_a;
  int len = num_latents;
  for (const StageSpec& stage : im.stages) {
    const int len_out = len * stage.rate;
    const int pad = (stage.kernel - stage.rate) / 2;
    cuda::launch_conv_transpose1d(cur, im.w(stage.up.weight), im.w(stage.up.bias), spare, batch,
                                  stage.up.in_channels, stage.up.out_channels, len, len_out,
                                  stage.kernel, stage.rate, pad, im.stream.get());
    // `cur` is now free; `spare` holds the upsampled activation.
    std::swap(cur, spare);
    len = len_out;

    const size_t elems = static_cast<size_t>(batch) * stage.up.out_channels * len;
    for (int j = 0; j < 3; ++j) {
      float* dst = (j == 0) ? acc : work;
      VIDFAB_CUDA_CHECK(cudaMemcpyAsync(dst, cur, elems * sizeof(float), cudaMemcpyDeviceToDevice,
                                        im.stream.get()));
      im.run_amp_block(stage.blocks[j], dst, batch, len, t1, t2, scratch);
      if (j != 0) cuda::launch_add_inplace(acc, work, elems, im.stream.get());
    }
    // BigVGAN averages the three resblocks; it does not sum them
    // (dac_bigvgan.py:195). Dropping this is 3x too loud and then clips.
    cuda::launch_scale_inplace(acc, 1.0f / 3.0f, elems, im.stream.get());
    capture(acc, elems);

    // Rotate: the averaged result becomes the next stage's input, and the
    // buffer it displaces becomes the next scratch.
    float* next_spare = cur;
    cur = acc;
    acc = next_spare;
  }

  im.run_activation(im.activation_post, cur, spare, batch, len, scratch);
  capture(spare, static_cast<size_t>(batch) * im.activation_post.channels * len);
  im.run_conv(im.conv_post, spare, cur, batch, len, len, 3, 1);
  capture(cur, static_cast<size_t>(batch) * len);

  const size_t frames = static_cast<size_t>(len);
  const size_t samples = frames * static_cast<size_t>(batch);
  // use_tanh_at_final is false, so the output is bounded by a clamp
  // (dac_bigvgan.py:204), not compressed by a tanh.
  cuda::launch_clamp_inplace(cur, -1.0f, 1.0f, samples, im.stream.get());
  capture(cur, samples);
  cuda::launch_interleave(cur, spare, batch, len, im.stream.get());
  capture(spare, samples);

  DecodedAudio out;
  out.channels = batch;
  out.sample_rate = cfg.sample_rate;
  out.samples.resize(samples);
  VIDFAB_CUDA_CHECK(cudaMemcpyAsync(out.samples.data(), spare, samples * sizeof(float),
                                    cudaMemcpyDeviceToHost, im.stream.get()));
  im.stream.synchronize();
  if (trace != nullptr && trace->boundaries.size() != 13)
    throw std::logic_error("audio vae: diagnostic boundary count drift");

  if (len != num_latents * total_upsample) {
    throw std::runtime_error("audio vae: decoded " + std::to_string(len) + " samples, expected " +
                             std::to_string(num_latents * total_upsample));
  }
  return out;
}

}  // namespace vidfab::vae
