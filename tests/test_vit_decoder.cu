// The video VAE decoder's window forward pass, and specifically the buffers a
// decoded window lands in.
//
// forward_windows used to D2H every window into one reused pinned staging
// buffer and then memcpy it into the caller's slot: 1.2 GiB of host-to-host
// copying per decode, for bytes that were already where they needed to be a
// moment earlier. It now page-locks the caller's slot and has the DMA land
// there directly, keeping the staged path as a fallback.
//
// A page-lock outlives the call, so what these tests actually guard is the
// *lifecycle*: registering a fresh slot, hitting the cached registration on the
// next call, re-registering after an explicit release, and — the one that would
// corrupt memory if it were wrong — dropping the lock *before* a resize
// reallocates the block underneath it. All four must produce byte-identical
// pixels, because none of them changes an arithmetic operation.

#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "harness.h"
#include "slopfab/safetensors.h"
#include "slopfab/safetensors_write.h"
#include "slopfab/vae/vit_decoder.h"

namespace {

// Small enough to build in a test, shaped like the real thing where the kernels
// care: head_dim 64 and rope_dim 48 are the shipped values, because the
// attention backend and the RoPE table layout are chosen from them.
slopfab::vae::ViTConfig tiny_config() {
  slopfab::vae::ViTConfig cfg;
  cfg.num_layers = 1;
  cfg.dim = 128;
  cfg.heads = 2;
  cfg.head_dim = 64;
  cfg.ffn_inner = 64;
  cfg.in_channels = 4;
  cfg.out_channels = 3;
  cfg.patch = 2;
  cfg.patch_t = 2;
  cfg.num_register = 4;
  cfg.num_suffix = 5;
  cfg.rope_dim = 48;
  return cfg;
}

void add(std::vector<slopfab::TensorWrite>* out, const std::string& name,
         std::vector<int64_t> shape, uint32_t seed) {
  size_t n = 1;
  for (int64_t d : shape)
    n *= static_cast<size_t>(d);
  // Small weights: one transformer block at fp16 amplifies, and a decoder that
  // saturates to inf would compare equal to itself for the wrong reason.
  out->push_back({name, std::move(shape), slopfab::test::make_data(n, seed, 0.05f)});
}

std::string write_tiny_checkpoint(const slopfab::vae::ViTConfig& cfg) {
  const int dim = cfg.dim;
  const int ch = cfg.in_channels;
  const int inner = cfg.ffn_inner;
  const int pd = cfg.patch_dim();

  std::vector<slopfab::TensorWrite> t;
  uint32_t seed = 17;
  add(&t, "decoder.x_embedder.weight", {dim, ch}, seed += 7);
  add(&t, "decoder.x_embedder.bias", {dim}, seed += 7);
  add(&t, "decoder.register_tokens", {cfg.num_register, dim}, seed += 7);
  add(&t, "decoder.norm_out.weight", {dim}, seed += 7);
  add(&t, "decoder.norm_out.bias", {dim}, seed += 7);
  add(&t, "decoder.proj_out.weight", {pd, dim}, seed += 7);
  add(&t, "decoder.proj_out.bias", {pd}, seed += 7);
  add(&t, "post_quant_conv.weight", {ch, ch}, seed += 7);
  add(&t, "post_quant_conv.bias", {ch}, seed += 7);
  for (int i = 0; i < cfg.num_layers; ++i) {
    const std::string p = "decoder.transformer_blocks." + std::to_string(i) + ".";
    add(&t, p + "norm1.weight", {dim}, seed += 7);
    add(&t, p + "norm2.weight", {dim}, seed += 7);
    add(&t, p + "scale1", {dim}, seed += 7);
    add(&t, p + "scale2", {dim}, seed += 7);
    add(&t, p + "attn.to_qkv.weight", {3 * dim, dim}, seed += 7);
    add(&t, p + "attn.to_qkv.bias", {3 * dim}, seed += 7);
    add(&t, p + "attn.to_out.weight", {dim, dim}, seed += 7);
    add(&t, p + "attn.to_out.bias", {dim}, seed += 7);
    add(&t, p + "ff.w1.weight", {2 * inner, dim}, seed += 7);
    add(&t, p + "ff.w1.bias", {2 * inner}, seed += 7);
    add(&t, p + "ff.w2.weight", {dim, inner}, seed += 7);
    add(&t, p + "ff.w2.bias", {dim}, seed += 7);
  }

  const std::filesystem::path dir =
      std::filesystem::temp_directory_path() / "slopfab_vit_decoder_test";
  std::filesystem::create_directories(dir);
  const std::string path = (dir / "tiny_vae.safetensors").string();
  slopfab::write_safetensors(path, t);
  return path;
}

size_t window_pixels(const slopfab::vae::ViTConfig& cfg, int T, int H, int W) {
  return static_cast<size_t>(cfg.out_channels) * (T * cfg.patch_t) * (H * cfg.patch) *
         (W * cfg.patch);
}

bool all_finite_and_not_all_zero(const std::vector<float>& v) {
  bool nonzero = false;
  for (float x : v) {
    if (!(x == x) || x > 1e30f || x < -1e30f)
      return false;
    if (x != 0.0f)
      nonzero = true;
  }
  return nonzero;
}

} // namespace

SLOPFAB_TEST(vit_decoder_window_lands_in_the_callers_buffer) {
  // Establish up front whether this machine can page-lock an ordinary heap
  // allocation right now. If it cannot, the decoder takes its staged fallback
  // and everything below still holds — but it is then testing the fallback, so
  // say which one ran rather than reporting a green run that proved less than
  // it looks.
  //
  // Reported, not asserted. Page-locking fails for reasons that are nothing to
  // do with this code — a machine short of lockable pages, or another process
  // holding the GPU — and a test that fails on those is a test that cries wolf.
  {
    std::vector<float> probe(1 << 16);
    const cudaError_t rc =
        cudaHostRegister(probe.data(), probe.size() * sizeof(float), cudaHostRegisterDefault);
    if (rc == cudaSuccess) {
      cudaHostUnregister(probe.data());
      std::printf("  page-locking available: exercising the direct-landing path\n");
    } else {
      std::printf("  cudaHostRegister unavailable (%s): exercising the staged fallback\n",
                  cudaGetErrorName(rc));
    }
    cudaGetLastError();
  }

  const slopfab::vae::ViTConfig cfg = tiny_config();
  const std::string path = write_tiny_checkpoint(cfg);
  slopfab::SafeTensors ckpt;
  ckpt.open(path);

  slopfab::vae::ViTDecoder decoder;
  decoder.load(ckpt, cfg);

  const int batch = 2;
  const int T = 2, H = 2, W = 2;
  const size_t voxels = static_cast<size_t>(T) * H * W;
  const std::vector<float> z =
      slopfab::test::make_data(static_cast<size_t>(batch) * cfg.in_channels * voxels, 991, 1.0f);
  const std::vector<float> z_small = slopfab::test::make_data(
      static_cast<size_t>(batch) * cfg.in_channels * (voxels / 2), 991, 1.0f);
  const size_t slots[2] = {0, 1};
  const size_t pixels = window_pixels(cfg, T, H, W);

  std::vector<std::vector<float>> out(batch);
  std::vector<std::vector<float>> grown(batch);

  // Declared after both buffers so it runs before they are destroyed: a lock
  // must never outlive the memory it covers, including on a throwing path.
  struct Guard {
    slopfab::vae::ViTDecoder* decoder;

    ~Guard() {
      decoder->release_host_registrations();
    }
  } guard{&decoder};

  // (a) fresh, empty slots: the resize allocates and the registration is new.
  decoder.forward_windows(z.data(), batch, T, H, W, out, slots);
  CHECK(out[0].size() == pixels && out[1].size() == pixels);
  CHECK(all_finite_and_not_all_zero(out[0]));
  CHECK(all_finite_and_not_all_zero(out[1]));
  CHECK(out[0] != out[1]); // two documents, two different latents
  const std::vector<float> ref0 = out[0];
  const std::vector<float> ref1 = out[1];

  // (b) the same slots again, unchanged: the cached registration is hit and
  // nothing is re-locked. This is what every chunk after the first does.
  decoder.forward_windows(z.data(), batch, T, H, W, out, slots);
  CHECK_CLOSE(ref0, out[0], 0.0, "cached registration, slot 0");
  CHECK_CLOSE(ref1, out[1], 0.0, "cached registration, slot 1");

  // (c) explicitly released, then used again: the slots are re-registered at
  // the same addresses.
  decoder.release_host_registrations();
  decoder.forward_windows(z.data(), batch, T, H, W, out, slots);
  CHECK_CLOSE(ref0, out[0], 0.0, "re-registration, slot 0");
  CHECK_CLOSE(ref1, out[1], 0.0, "re-registration, slot 1");

  // (d) slots first sized by a smaller window, then grown. The resize
  // reallocates, so the lock on the old block has to be dropped while that
  // block is still alive — get this wrong and the unregister lands on freed
  // memory. Same pixels as (a) once grown.
  decoder.forward_windows(z_small.data(), batch, 1, H, W, grown, slots);
  CHECK(grown[0].size() == window_pixels(cfg, 1, H, W));
  decoder.forward_windows(z.data(), batch, T, H, W, grown, slots);
  CHECK_CLOSE(ref0, grown[0], 0.0, "grown slot 0");
  CHECK_CLOSE(ref1, grown[1], 0.0, "grown slot 1");

  // Nothing above may have left a sticky CUDA error for an unrelated call to
  // inherit — the registration path swallows its own failures on purpose.
  CHECK(cudaGetLastError() == cudaSuccess);

  ckpt.close();
  std::error_code ec;
  std::filesystem::remove(path, ec);
}

SLOPFAB_TEST(vit_decoder_single_window_releases_its_lock) {
  // forward_window moves its buffer into a vector the decoder cannot see, so
  // it has to release the lock itself. If it did not, the lock would outlive
  // `out` here and the unregister at decoder teardown would hit freed memory.
  const slopfab::vae::ViTConfig cfg = tiny_config();
  const std::string path = write_tiny_checkpoint(cfg);
  slopfab::SafeTensors ckpt;
  ckpt.open(path);

  slopfab::vae::ViTDecoder decoder;
  decoder.load(ckpt, cfg);

  const int T = 2, H = 2, W = 2;
  const std::vector<float> z =
      slopfab::test::make_data(static_cast<size_t>(cfg.in_channels) * T * H * W, 991, 1.0f);

  std::vector<float> single;
  decoder.forward_window(z.data(), T, H, W, single);
  CHECK(single.size() == window_pixels(cfg, T, H, W));
  CHECK(all_finite_and_not_all_zero(single));

  // Re-registering the same address must now succeed, which it can only do if
  // the previous lock really was released.
  const cudaError_t rc =
      cudaHostRegister(single.data(), single.size() * sizeof(float), cudaHostRegisterDefault);
  CHECK_MSG(rc != cudaErrorHostMemoryAlreadyRegistered,
            "forward_window left its output buffer page-locked");
  if (rc == cudaSuccess)
    cudaHostUnregister(single.data());
  cudaGetLastError();

  // A batched call against the same latent must agree with the single-window
  // path exactly: the batching only shares the token-wise projections.
  std::vector<std::vector<float>> batched(1);
  const size_t slot = 0;
  decoder.forward_windows(z.data(), 1, T, H, W, batched, &slot);
  CHECK_CLOSE(single, batched[0], 0.0, "single vs batched window");
  decoder.release_host_registrations();

  // The lock must also go when the scope unwinds rather than returns. Every
  // SLOPFAB_CUDA_CHECK inside forward_windows can throw after a slot has been
  // registered — a device out-of-memory is the realistic one — and the buffer
  // would then be destroyed still page-locked. The throw is raised here rather
  // than injected into CUDA because what is under test is the guard, not the
  // failure: the registration is real and the unwind is real.
  {
    std::vector<std::vector<float>> unwound(1);
    try {
      slopfab::vae::ViTDecoder::HostRegistrationScope scope(decoder);
      decoder.forward_windows(z.data(), 1, T, H, W, unwound, &slot);
      throw std::runtime_error("simulated failure after registration");
    } catch (const std::runtime_error&) {
      // The guard has run; `unwound` is still alive and must be unlocked.
    }
    const cudaError_t rc2 = cudaHostRegister(unwound[0].data(), unwound[0].size() * sizeof(float),
                                             cudaHostRegisterDefault);
    CHECK_MSG(rc2 != cudaErrorHostMemoryAlreadyRegistered,
              "an unwind left the output buffer page-locked");
    if (rc2 == cudaSuccess)
      cudaHostUnregister(unwound[0].data());
    cudaGetLastError();
  }

  ckpt.close();
  std::error_code ec;
  std::filesystem::remove(path, ec);
}
