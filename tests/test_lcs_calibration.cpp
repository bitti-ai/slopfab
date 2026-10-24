#include "harness.h"
#include "slopfab/lcs.h"
#include <cstdlib>
#include <filesystem>

SLOPFAB_TEST_CATEGORY(lcs_real_vae_calibration, "integration") {
  const char* path = std::getenv("SLOPFAB_LCS_TEST_VAE");
  if (!path || !*path) {
    SKIP_OPT_IN("Set SLOPFAB_LCS_TEST_VAE to an H3 VAE with encoder weights");
    return;
  }
  const char* backend_env = std::getenv("SLOPFAB_LCS_TEST_BACKEND");
  const bool vk = backend_env && std::string(backend_env) == "vulkan";
  const auto output =
      (std::filesystem::temp_directory_path() /
       (vk ? "slopfab-lcs-vulkan-test.safetensors" : "slopfab-lcs-cuda-test.safetensors"))
          .string();
  int calls = 0;
  CHECK(slopfab::calibrate_lcs(path,
                               vk ? slopfab::DeviceBackend::kVulkan : slopfab::DeviceBackend::kCuda,
                               true, output, 8, 64, [&](int done, int total) {
                                 CHECK(done == calls++ && total == 24);
                                 return true;
                               }));
  const auto data = slopfab::LcsData::load(output);
  CHECK(data->vae_sha256 == slopfab::lcs_vae_fingerprint(path));
  CHECK(data->basis.size() == 288 && data->sharpness.size() == 96);
  CHECK(data->explained_variance > .5f);
  std::printf("  LCS real VAE variance captured: %.3f%%\n", 100 * data->explained_variance);
  std::filesystem::remove(output);
}
