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
  const __half* bias = nullptr;  // into Loader's affine arena
};

struct NormWeight {
  const __half* weight = nullptr;  // into Loader's affine arena
  const __half* bias = nullptr;
};

// Logical element count of an affine (a bias or a norm weight), whether it is
// stored plainly or as bitsandbytes NF4. Used twice: once to size the arena
// before any upload, once to place each tensor in it.
size_t affine_elements(const SafeTensors& ckpt, const std::string& name) {
  if (!is_nf4_weight(ckpt, name)) return static_cast<size_t>(ckpt.at(name).numel());
  size_t logical = 1;
  for (int64_t dim : read_nf4_state(ckpt, name, "keyframe encoder").shape) {
    logical *= static_cast<size_t>(dim);
  }
  return logical;
}

// The tensor names this encoder loads, in load order. Built before anything is
// uploaded so every device allocation can be sized from it up front.
struct WeightPlan {
  std::vector<std::string> convs;  // ".weight" and ".bias" hang off these
  std::vector<std::string> norms;  // ".weight" and ".bias" hang off these
};

WeightPlan plan_weights() {
  constexpr int channels[] = {128, 256, 256, 512, 512, 1024};
  constexpr int down[] = {2, 2, 2, 2, 1, 1};
  WeightPlan plan;
  plan.convs.emplace_back("encoder.conv_in");
  int previous = 128;
  for (int level = 0; level < 6; ++level) {
    const int output = channels[level];
    for (int block = 0; block < 2; ++block) {
      const int input = block == 0 ? previous : output;
      const std::string p =
          "encoder.down." + std::to_string(level) + ".block." + std::to_string(block);
      plan.norms.push_back(p + ".norm1");
      plan.norms.push_back(p + ".norm2");
      plan.convs.push_back(p + ".conv1");
      plan.convs.push_back(p + ".conv2");
      if (input != output) plan.convs.push_back(p + ".nin_shortcut");
    }
    if (down[level] == 2) {
      plan.convs.push_back("encoder.down." + std::to_string(level) + ".downsample.conv");
    }
    previous = output;
  }
  plan.norms.emplace_back("encoder.norm_out");
  plan.convs.emplace_back("encoder.conv_out");
  plan.convs.emplace_back("quant_conv");
  return plan;
}

// One upload pass over the checkpoint.
//
// Every device buffer it needs is sized from the plan and allocated in the
// constructor, so the upload loop itself allocates nothing and frees nothing.
// That matters twice over. It is what lets the whole load run on one stream
// with a single synchronise at the end — `cudaMalloc` and `cudaFree` are
// implicit device-wide sync points, so an allocation per tensor serialises the
// load against itself just as effectively as an explicit synchronise did. And
// it removes the reallocation hazard: growing a scratch buffer on demand means
// `DeviceBuffer::allocate` -> `reset` -> `cudaFree` on a pointer an
// already-enqueued kernel may still be reading, which stream ordering does not
// make safe. That is benign today only because `cudaFree` synchronises the
// device, a property of the legacy allocator; under `cudaFreeAsync` it would be
// a use-after-free no test could catch.
//
// What still allocates per tensor is `F16Weight::load`, once for a dense conv
// weight and five times for an NF4 one — roughly 33 allocations for this graph.
// Those live in `src/cuda/nf4_weight.cu`, which the video VAE and ViT decoders
// share, so removing them is not a change this file can make alone.
class Loader {
 public:
  Loader(const SafeTensors& checkpoint, const WeightPlan& plan, cudaStream_t stream)
      : ckpt_(checkpoint), stream_(stream) {
    // Arena for every affine, plus the high-water mark of each NF4 scratch.
    size_t arena = 0;
    size_t codes = 0, scales = 0, qmap = 0, nested_map = 0, nested_absmax = 0;
    auto account = [&](const std::string& name) {
      arena += align_up(affine_elements(ckpt_, name));
      if (!is_nf4_weight(ckpt_, name)) return;
      codes = std::max(codes, ckpt_.at(name).nbytes);
      scales = std::max(scales, ckpt_.at(name + ".absmax").nbytes);
      qmap = std::max(qmap, static_cast<size_t>(ckpt_.at(name + ".quant_map").numel()));
      nested_map =
          std::max(nested_map, static_cast<size_t>(ckpt_.at(name + ".nested_quant_map").numel()));
      nested_absmax =
          std::max(nested_absmax, static_cast<size_t>(ckpt_.at(name + ".nested_absmax").numel()));
    };
    for (const std::string& name : plan.convs) account(name + ".bias");
    for (const std::string& name : plan.norms) {
      account(name + ".weight");
      account(name + ".bias");
    }
    if (arena != 0) affines_.allocate(arena);
    if (codes != 0) codes_.allocate(codes);
    if (scales != 0) scales_.allocate(scales);
    if (qmap != 0) qmap_.allocate(qmap);
    if (nested_map != 0) nested_map_.allocate(nested_map);
    if (nested_absmax != 0) nested_absmax_.allocate(nested_absmax);
  }

  Loader(const Loader&) = delete;
  Loader& operator=(const Loader&) = delete;

  // Uploads one affine into the arena and returns where it landed. The arena
  // outlives the loader, so the pointer stays valid for the encoder's life.
  const __half* half(const std::string& name) {
    const size_t count = affine_elements(ckpt_, name);
    __half* out = affines_.get() + cursor_;
    cursor_ += align_up(count);
    if (cursor_ > affines_.size()) {
      throw std::runtime_error("keyframe encoder: affine arena overflow at '" + name + "'");
    }

    const TensorView& view = ckpt_.at(name);
    if (is_nf4_weight(ckpt_, name)) {
      const NF4State state = read_nf4_state(ckpt_, name, "keyframe encoder");
      const TensorView& absmax = ckpt_.at(name + ".absmax");
      const TensorView& qmap = ckpt_.at(name + ".quant_map");
      const TensorView& nested_map = ckpt_.at(name + ".nested_quant_map");
      const TensorView& nested_absmax = ckpt_.at(name + ".nested_absmax");
      // Reusing the scratch across tensors needs no synchronise: every copy and
      // every kernel here is on one stream, so the next tensor's copy into the
      // scratch is ordered after this tensor's dequantisation has read it. The
      // scratch is never reallocated — see the constructor for why that is a
      // different question from reuse.
      codes_.copy_from_host(static_cast<const uint8_t*>(view.data), view.nbytes, stream_);
      scales_.copy_from_host(static_cast<const uint8_t*>(absmax.data), absmax.nbytes, stream_);
      qmap_.copy_from_host(static_cast<const float*>(qmap.data),
                           static_cast<size_t>(qmap.numel()), stream_);
      nested_map_.copy_from_host(static_cast<const float*>(nested_map.data),
                                 static_cast<size_t>(nested_map.numel()), stream_);
      nested_absmax_.copy_from_host(static_cast<const float*>(nested_absmax.data),
                                    static_cast<size_t>(nested_absmax.numel()), stream_);
      cuda::launch_dequant_nf4_f16(codes_.get(), scales_.get(), qmap_.get(), nested_map_.get(),
                                   nested_absmax_.get(), state.block_size,
                                   state.nested_block_size, state.nested_offset, out, count,
                                   stream_);
      return out;
    }
    if (view.dtype == DType::kF16) {
      copy_in(out, static_cast<const __half*>(view.data), count);
      return out;
    }
    // `h` is a stack local and the copy below is asynchronous, which is safe
    // and needs no synchronise: a host-to-device `cudaMemcpyAsync` from
    // *pageable* memory stages into a driver buffer before it returns, so the
    // source need not outlive the call. `F16Weight::load` in
    // src/cuda/nf4_weight.cu relies on exactly this for its own conversion
    // buffer. It would not hold for pinned or registered memory — the mapping
    // itself is registered here, which is precisely why *those* copies are real
    // DMAs and why the mapping must outlive the stream.
    const std::vector<float> f = to_f32(view);
    std::vector<__half> h(f.size());
    for (size_t i = 0; i < f.size(); ++i) h[i] = __float2half_rn(f[i]);
    copy_in(out, h.data(), count);
    return out;
  }

  ConvWeight conv(const std::string& name) {
    ConvWeight result;
    size_t elements = static_cast<size_t>(ckpt_.at(name + ".weight").numel());
    if (is_nf4_weight(ckpt_, name + ".weight")) {
      elements = affine_elements(ckpt_, name + ".weight");
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

  // Element count of the largest conv weight, for `materialize`'s workspace.
  static size_t max_conv_elements(const SafeTensors& ckpt, const WeightPlan& plan) {
    size_t most = 0;
    for (const std::string& name : plan.convs) {
      const std::string w = name + ".weight";
      const size_t n = is_nf4_weight(ckpt, w) ? affine_elements(ckpt, w)
                                              : static_cast<size_t>(ckpt.at(w).numel());
      most = std::max(most, n);
    }
    return most;
  }

  // The arena is sized by one walk of the plan and filled by another, in a
  // different order. Summing `align_up` over the same set is order-independent,
  // so the two must land on exactly the same total — not merely fit. Checking
  // equality rather than the bound turns a mis-accounted tensor into a loud
  // failure on the first real load instead of a silently short arena that
  // happens to fit because something else was over-counted.
  //
  // This matters more than usual here: the graph is fixed but there is no
  // video-VAE fixture small enough to commit, so this check and
  // `validate_keyframe_encoder_weights` are what stand in for a unit test.
  void verify_arena_full() const {
    if (cursor_ != affines_.size()) {
      throw std::runtime_error("keyframe encoder: affine arena accounting disagrees — placed " +
                               std::to_string(cursor_) + " of " +
                               std::to_string(affines_.size()) + " elements");
    }
  }

  DeviceBuffer<__half> release_arena() { return std::move(affines_); }

 private:
  // 16-byte slots, so every tensor in the arena starts at an address the
  // vectorised kernels are happy to read from.
  static size_t align_up(size_t elements) { return (elements + 7) / 8 * 8; }

  void copy_in(__half* dst, const __half* src, size_t count) {
    if (count == 0) return;
    VIDFAB_CUDA_CHECK(cudaMemcpyAsync(dst, src, count * sizeof(__half), cudaMemcpyHostToDevice,
                                      stream_));
  }

  const SafeTensors& ckpt_;
  cudaStream_t stream_;
  DeviceBuffer<__half> affines_;
  size_t cursor_ = 0;
  DeviceBuffer<uint8_t> codes_, scales_;
  DeviceBuffer<float> qmap_, nested_map_, nested_absmax_;
};

}  // namespace

struct KeyframeEncoder::Impl {
  std::map<std::string, ConvWeight> convs;
  std::map<std::string, NormWeight> norms;
  cuda::Stream stream;
  // Backs every `ConvWeight::bias` and `NormWeight::weight`/`bias` pointer, so
  // it must outlive them; declared before them would be wrong, but they are
  // plain pointers and own nothing, so only lifetime matters and this member
  // lives exactly as long as they do.
  DeviceBuffer<__half> affines;
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
    const WeightPlan plan = plan_weights();
    weight_workspace_elements = Loader::max_conv_elements(checkpoint, plan);
    weight_workspace.allocate(weight_workspace_elements);
    Loader load(checkpoint, plan, stream.get());

    // Declared last, so it is destroyed first: the stream is drained before
    // `load` frees its scratch and before `mapping` unregisters. That ordering
    // only matters on the failure path — an exception out of the loop below
    // never reaches the explicit synchronise, and unwinding would otherwise run
    // `cudaFree` and `cudaHostUnregister` with copies still in flight. On the
    // success path the explicit synchronise has already drained it and this
    // costs nothing.
    struct DrainOnExit {
      cudaStream_t stream;
      ~DrainOnExit() { cudaStreamSynchronize(stream); }
    } drain{stream.get()};

    // The affine half of this is pure enqueue: no allocation, no free, no
    // synchronise, one stream. The conv half is not, and saying so matters —
    // `F16Weight::load` allocates its own device buffers, so the ~33 conv
    // weights still cost an implicit device-wide sync each. Removing those
    // means giving `F16Weight` an arena, and it lives in a file the video VAE
    // and ViT decoders share.
    for (const std::string& name : plan.norms) norms.emplace(name, load.norm(name));
    for (const std::string& name : plan.convs) convs.emplace(name, load.conv(name));
    load.verify_arena_full();
    affines = load.release_arena();

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
    cuda::launch_keyframe_conv3d(x.get(), weight, cw.bias, y.get(), cin, cout, h,
                                 w, kernel, stride, !asymmetric, asymmetric, stream.get());
    return y;
  }

  DeviceBuffer<float> norm(const DeviceBuffer<float>& x, const std::string& name, int channels,
                           int h, int w) {
    DeviceBuffer<float> y(static_cast<size_t>(channels) * h * w);
    const NormWeight& nw = norms.at(name);
    cuda::launch_keyframe_groupnorm_silu(x.get(), nw.weight, nw.bias, y.get(),
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
