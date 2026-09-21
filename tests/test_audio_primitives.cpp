#include "harness.h"

#include <cmath>
#include <filesystem>
#include <vector>

#include "slopfab/safetensors.h"
#include "slopfab/safetensors_write.h"
#include "slopfab/vae/audio_primitives.h"

SLOPFAB_TEST(audio_primitive_descriptors) {
  using namespace slopfab::vae;
  AudioConv1DDesc conv{2, 3, 5, 17, 17, 11, 25, 5};
  conv.validate();
  CHECK(conv.input_elements() == 102);
  CHECK(conv.output_elements() == 170);
  CHECK(conv.weight_elements() == 165);
  AudioConv1DDesc bad_conv = conv;
  bad_conv.length_out = 16;
  bool bad_conv_rejected = false;
  try {
    bad_conv.validate();
  } catch (const std::invalid_argument&) {
    bad_conv_rejected = true;
  }
  CHECK(bad_conv_rejected);

  AudioConvTranspose1DDesc transpose{2, 7, 4, 13, 65, 9, 5, 2};
  transpose.validate();
  CHECK(transpose.input_elements() == 182);
  CHECK(transpose.output_elements() == 520);
  CHECK(transpose.weight_elements() == 252);
  AudioConvTranspose1DDesc bad_transpose = transpose;
  bad_transpose.padding = 3;
  bool bad_transpose_rejected = false;
  try {
    bad_transpose.validate();
  } catch (const std::invalid_argument&) {
    bad_transpose_rejected = true;
  }
  CHECK(bad_transpose_rejected);
}

SLOPFAB_TEST(audio_typed_weight_loader) {
  using namespace slopfab;
  const auto path =
      std::filesystem::temp_directory_path() / "slopfab_audio_primitive_weights.safetensors";
  std::filesystem::remove(path);
  write_safetensors(path.string(),
                    {
                        {"fold.weight_v", {2, 2, 2}, {1, 2, 3, 4, -1, 2, -3, 4}},
                        {"fold.weight_g", {2, 1, 1}, {2, 3}},
                        {"fold.bias", {2}, {0.25f, -0.5f}},
                        {"transpose.weight", {3, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}},
                        {"transpose.bias", {2}, {0.5f, -0.25f}},
                    });
  SafeTensors checkpoint;
  checkpoint.open(path.string());
  const vae::AudioConvWeights folded =
      vae::load_audio_conv_weights(checkpoint, "fold", {2, 2, 2}, 2, true);
  CHECK(folded.folded_weight_norm);
  CHECK(folded.weight.size() == 8);
  CHECK(folded.bias == std::vector<float>({0.25f, -0.5f}));
  const float norm = static_cast<float>(std::sqrt(30.0));
  const float scale0 = 2.0f / norm;
  const float scale1 = 3.0f / norm;
  CHECK_NEAR(folded.weight[0], 1.0f * scale0, 0.0);
  CHECK_NEAR(folded.weight[3], 4.0f * scale0, 0.0);
  CHECK_NEAR(folded.weight[4], -1.0f * scale1, 0.0);
  CHECK_NEAR(folded.weight[7], 4.0f * scale1, 0.0);

  const vae::AudioConvWeights transpose =
      vae::load_audio_conv_weights(checkpoint, "transpose", {3, 2, 2}, 2, true);
  CHECK(!transpose.folded_weight_norm);
  CHECK(transpose.weight.front() == 1.0f);
  CHECK(transpose.weight.back() == 12.0f);
  CHECK(transpose.bias == std::vector<float>({0.5f, -0.25f}));
  checkpoint.close();
  std::filesystem::remove(path);
}
