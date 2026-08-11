#include "vidfab/vae/keyframe_encoder.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vidfab/cuda/device.h"
#include "vidfab/cuda/keyframe_encoder.cuh"
#include "vidfab/cuda/linear.cuh"
#include "vidfab/cuda/nf4_weight.cuh"
#include "vidfab/nf4.h"
#include "vidfab/tensor_convert.h"

namespace vidfab::vae {
namespace {

using cuda::DeviceBuffer;

struct ConvWeight {
  cuda::F16Weight weight;
  DeviceBuffer<__half> bias;
};

struct NormWeight {
  DeviceBuffer<__half> weight;
  DeviceBuffer<__half> bias;
};

// One upload pass over the checkpoint. Exists so the ~130 tensors share one
// set of NF4 scratch buffers and one stream synchronise instead of allocating
// five device buffers and synchronising per tensor: `cudaFree` synchronises
// the whole device, so the old shape serialised the entire load against itself.
//
// Everything the device may still be reading — the retained host conversions
// below and the scratch — outlives the loader, and the loader outlives the
// single synchronise at the end of the constructor.
class Loader {
 public:
  Loader(const SafeTensors& checkpoint, cudaStream_t stream)
      : ckpt_(checkpoint), stream_(stream) {}

  Loader(const Loader&) = delete;
  Loader& operator=(const Loader&) = delete;

  DeviceBuffer<__half> half(const std::string& name) {
    const TensorView& view = ckpt_.at(name);
    if (is_nf4_weight(ckpt_, name)) {
      const NF4State state = read_nf4_state(ckpt_, name, "keyframe encoder");
      size_t logical = 1;
      for (int64_t dim : state.shape) logical *= static_cast<size_t>(dim);
      const TensorView& absmax = ckpt_.at(name + ".absmax");
      const TensorView& qmap = ckpt_.at(name + ".quant_map");
      const TensorView& nested_map = ckpt_.at(name + ".nested_quant_map");
      const TensorView& nested_absmax = ckpt_.at(name + ".nested_absmax");
      uint8_t* codes = grow(codes_, view.nbytes);
      uint8_t* scales = grow(scales_, absmax.nbytes);
      float* qm = grow(qmap_, static_cast<size_t>(qmap.numel()));
      float* nm = grow(nested_map_, static_cast<size_t>(nested_map.numel()));
      float* na = grow(nested_absmax_, static_cast<size_t>(nested_absmax.numel()));
      DeviceBuffer<__half> out(logical);
      // Scratch reuse is safe without a synchronise: every copy and the kernel
      // below are on one stream, so the next tensor's copy into the scratch is
      // already ordered after this tensor's dequantisation kernel has read it.
      codes_.copy_from_host(static_cast<const uint8_t*>(view.data), view.nbytes, stream_);
      scales_.copy_from_host(static_cast<const uint8_t*>(absmax.data), absmax.nbytes, stream_);
      qmap_.copy_from_host(static_cast<const float*>(qmap.data),
                           static_cast<size_t>(qmap.numel()), stream_);
      nested_map_.copy_from_host(static_cast<const float*>(nested_map.data),
                                 static_cast<size_t>(nested_map.numel()), stream_);
      nested_absmax_.copy_from_host(static_cast<const float*>(nested_absmax.data),
                                    static_cast<size_t>(nested_absmax.numel()), stream_);
      cuda::launch_dequant_nf4_f16(codes, scales, qm, nm, na, state.block_size,
                                   state.nested_block_size, state.nested_offset, out.get(),
                                   logical, stream_);
      return out;
    }
    DeviceBuffer<__half> out(static_cast<size_t>(view.numel()));
    if (view.dtype == DType::kF16) {
      out.copy_from_host(static_cast<const __half*>(view.data), out.size(), stream_);
    } else {
      // The converted block is retained rather than left on the stack: without
      // the per-tensor synchronise there is no point at which it is known to
      // have been consumed, and these are all one-dimensional affines, so
      // holding every one of them costs a few hundred kilobytes.
      const std::vector<float> f = to_f32(view);
      retained_.emplace_back(f.size());
      std::vector<__half>& h = retained_.back();
      for (size_t i = 0; i < f.size(); ++i) h[i] = __float2half_rn(f[i]);
      out.copy_from_host(h.data(), h.size(), stream_);
    }
    return out;
  }

  ConvWeight conv(const std::string& name) {
    const TensorView& w = ckpt_.at(name + ".weight");
    ConvWeight result;
    size_t elements = static_cast<size_t>(w.numel());
    if (is_nf4_weight(ckpt_, name + ".weight")) {
      elements = 1;
      for (int64_t dim : read_nf4_state(ckpt_, name + ".weight", "keyframe encoder").shape)
        elements *= static_cast<size_t>(dim);
    }
    result.weight.load(ckpt_, name + ".weight", elements, stream_, "keyframe encoder");
    result.bias = half(name + ".bias");
    return result;
  }

  NormWeight norm(const std::string& name) {
    NormWeight result;
    result.weight = half(name + ".weight");
    result.bias = half(name + ".bias");
    return result;
  }

 private:
  // Grows to the high-water mark and never shrinks, so the common case — every
  // affine the same 1024 elements or fewer — allocates once.
  template <typename T>
  static T* grow(DeviceBuffer<T>& buffer, size_t count) {
    if (buffer.size() < count) buffer.allocate(count);
    return buffer.get();
  }

  const SafeTensors& ckpt_;
  cudaStream_t stream_;
  DeviceBuffer<uint8_t> codes_, scales_;
  DeviceBuffer<float> qmap_, nested_map_, nested_absmax_;
  std::vector<std::vector<__half>> retained_;
};

}  // namespace

struct KeyframeEncoder::Impl {
  std::map<std::string, ConvWeight> convs;
  std::map<std::string, NormWeight> norms;
  cuda::Stream stream;
  DeviceBuffer<__half> weight_workspace;
  size_t weight_workspace_elements = 0;

  explicit Impl(const SafeTensors& checkpoint) {
    // See the note on `SafeTensors::prefetch`: issued first because it is
    // asynchronous, so validation below runs while the OS is already reading.
    // The whole file rather than the encoder's extent, mirroring
    // `ViTDecoder::load`: the decoder half of this same video VAE is loaded by
    // the same pipeline, and this is a hint either way.
    checkpoint.prefetch();
    validate_keyframe_encoder_weights(checkpoint);

    // Page-locks the mapping for the whole of the load below. Every weight here
    // is copied straight out of `view.data`, by `F16Weight::load` for the conv
    // kernels and by `Loader` for the affines; from a pageable mapping each of
    // those is a synchronous copy staged through the driver, out of a
    // registered one it is a real DMA. Declared before `load` so that it is
    // destroyed after it, and after the synchronise at the end of this
    // constructor — nothing may still be reading the mapping when it is
    // unregistered.
    const cuda::RegisteredMapping mapping(checkpoint.mapping_base(), checkpoint.file_size());
    if (!mapping.registered()) {
      std::fprintf(stderr,
                   "vidfab: could not page-lock the video vae mapping for the keyframe encoder; "
                   "uploading via the staged path, which is slower\n");
    }
    Loader load(checkpoint, stream.get());

    constexpr int channels[] = {128, 256, 256, 512, 512, 1024};
    constexpr int down[] = {2, 2, 2, 2, 1, 1};
    convs.emplace("encoder.conv_in", load.conv("encoder.conv_in"));
    int previous = 128;
    for (int level = 0; level < 6; ++level) {
      const int output = channels[level];
      for (int block = 0; block < 2; ++block) {
        const int input = block == 0 ? previous : output;
        const std::string p = "encoder.down." + std::to_string(level) + ".block." +
                              std::to_string(block);
        norms.emplace(p + ".norm1", load.norm(p + ".norm1"));
        norms.emplace(p + ".norm2", load.norm(p + ".norm2"));
        convs.emplace(p + ".conv1", load.conv(p + ".conv1"));
        convs.emplace(p + ".conv2", load.conv(p + ".conv2"));
        if (input != output) convs.emplace(p + ".nin_shortcut", load.conv(p + ".nin_shortcut"));
      }
      if (down[level] == 2) {
        const std::string p = "encoder.down." + std::to_string(level) + ".downsample.conv";
        convs.emplace(p, load.conv(p));
      }
      previous = output;
    }
    norms.emplace("encoder.norm_out", load.norm("encoder.norm_out"));
    convs.emplace("encoder.conv_out", load.conv("encoder.conv_out"));
    convs.emplace("quant_conv", load.conv("quant_conv"));
    for (const auto& item : convs)
      weight_workspace_elements = std::max(weight_workspace_elements, item.second.weight.elements());
    weight_workspace.allocate(weight_workspace_elements);
    // The one synchronise for the whole load. Every upload above was enqueued
    // on this stream and nothing has read a result yet.
    stream.synchronize();
  }

  DeviceBuffer<float> conv(const DeviceBuffer<float>& x, const std::string& name, int cin,
                           int cout, int h, int w, int kernel, int stride = 1,
                           bool asymmetric = false) {
    const int oh = stride == 2 ? h / 2 : h;
    const int ow = stride == 2 ? w / 2 : w;
    DeviceBuffer<float> y(static_cast<size_t>(cout) * oh * ow);
    const ConvWeight& cw = convs.at(name);
    const __half* weight = cw.weight.materialize(weight_workspace.get(), weight_workspace_elements,
                                                 stream.get());
    cuda::launch_keyframe_conv3d(x.get(), weight, cw.bias.get(), y.get(), cin, cout, h,
                                 w, kernel, stride, true, asymmetric, stream.get());
    return y;
  }

  DeviceBuffer<float> norm(const DeviceBuffer<float>& x, const std::string& name, int channels,
                           int h, int w) {
    DeviceBuffer<float> y(static_cast<size_t>(channels) * h * w);
    const NormWeight& nw = norms.at(name);
    cuda::launch_keyframe_groupnorm_silu(x.get(), nw.weight.get(), nw.bias.get(), y.get(),
                                         channels, h, w, 32, 1e-6f, stream.get());
    return y;
  }
};

KeyframeEncoder::KeyframeEncoder(const SafeTensors& checkpoint)
    : impl_(std::make_unique<Impl>(checkpoint)) {}
KeyframeEncoder::~KeyframeEncoder() = default;
KeyframeEncoder::KeyframeEncoder(KeyframeEncoder&&) noexcept = default;
KeyframeEncoder& KeyframeEncoder::operator=(KeyframeEncoder&&) noexcept = default;

std::vector<float> KeyframeEncoder::encode_moments(const float* pixels, int height, int width) {
  if (!pixels || height <= 0 || width <= 0 || height % 16 || width % 16)
    throw std::runtime_error("keyframe encoder: dimensions must be positive multiples of 16");
  DeviceBuffer<float> hbuf(static_cast<size_t>(3) * height * width);
  hbuf.copy_from_host(pixels, hbuf.size(), impl_->stream.get());
  hbuf = impl_->conv(hbuf, "encoder.conv_in", 3, 128, height, width, 3);

  constexpr int channels[] = {128, 256, 256, 512, 512, 1024};
  constexpr int down[] = {2, 2, 2, 2, 1, 1};
  int current = 128;
  int h = height, w = width;
  for (int level = 0; level < 6; ++level) {
    const int output = channels[level];
    for (int block = 0; block < 2; ++block) {
      const int input = current;
      const std::string p = "encoder.down." + std::to_string(level) + ".block." +
                            std::to_string(block);
      DeviceBuffer<float> residual;
      if (input != output) residual = impl_->conv(hbuf, p + ".nin_shortcut", input, output, h, w, 1);
      DeviceBuffer<float> tmp = impl_->norm(hbuf, p + ".norm1", input, h, w);
      tmp = impl_->conv(tmp, p + ".conv1", input, output, h, w, 3);
      tmp = impl_->norm(tmp, p + ".norm2", output, h, w);
      tmp = impl_->conv(tmp, p + ".conv2", output, output, h, w, 3);
      DeviceBuffer<float> sum(static_cast<size_t>(output) * h * w);
      const DeviceBuffer<float>& skip = input == output ? hbuf : residual;
      cuda::launch_keyframe_add(skip.get(), tmp.get(), sum.get(), sum.size(), impl_->stream.get());
      hbuf = std::move(sum);
      current = output;
    }
    if (down[level] == 2) {
      const std::string p = "encoder.down." + std::to_string(level) + ".downsample.conv";
      hbuf = impl_->conv(hbuf, p, current, current, h, w, 3, 2, true);
      h /= 2;
      w /= 2;
    }
  }
  hbuf = impl_->norm(hbuf, "encoder.norm_out", 1024, h, w);
  hbuf = impl_->conv(hbuf, "encoder.conv_out", 1024, 48, h, w, 3);
  hbuf = impl_->conv(hbuf, "quant_conv", 48, 48, h, w, 1);
  std::vector<float> moments(hbuf.size());
  hbuf.copy_to_host(moments.data(), moments.size(), impl_->stream.get());
  impl_->stream.synchronize();
  return moments;
}

std::vector<float> KeyframeEncoder::encode_condition_rows(
    const RGBImage& image, const float* normal, const std::vector<float>& latents_mean,
    const std::vector<float>& latents_std) {
  if (!normal) throw std::runtime_error("keyframe encoder: missing posterior normal field");
  const std::vector<float> pixels = prepare_keyframe_pixels(image);
  const std::vector<float> moments = encode_moments(pixels.data(), image.height, image.width);
  const int latent_h = image.height / 16;
  const int latent_w = image.width / 16;
  const std::vector<float> latents = sample_keyframe_latents(
      moments.data(), normal, latent_h, latent_w, latents_mean, latents_std);
  return patchify_keyframe_latents(latents.data(), latent_h, latent_w);
}

std::vector<float> KeyframeEncoder::encode_reference_image(
    const RGBImage& image, const std::vector<float>& latents_mean,
    const std::vector<float>& latents_std) {
  if (image.height <= 0 || image.width <= 0 || image.height % 16 || image.width % 16)
    throw std::runtime_error("keyframe encoder: reference dimensions must be multiples of 16");
  const size_t count = static_cast<size_t>(24) * (image.height / 16) * (image.width / 16);
  const std::vector<float> normal = torch_cpu_normal_seed42(count);
  return encode_condition_rows(image, normal.data(), latents_mean, latents_std);
}

}  // namespace vidfab::vae
