#pragma once
#include "slopfab/text/encoder.h"
#include <functional>

namespace slopfab::text {
// NicoLab28 ClipProj safetensors: standardized ridge, residual MLP, or both.
// The projection runs on the host after releasing the student's GPU weights.
class ClipProjection {
public:
  explicit ClipProjection(const std::string& path);
  int input_width() const;
  int output_width() const;
  int tap() const;
  PromptEmbedding apply(const PromptEmbedding& hidden,
                        const std::function<void()>& check_cancel = {}) const;

private:
  SafeTensors file_;
  int input_ = 0, output_ = 0, tap_ = 0;
  std::vector<int> layers_;
  std::vector<float> mean_in_, std_in_, mean_out_, std_out_, sink_;
};

struct ClipStudentConfig {
  std::string prefix;
  int width = 0, layers = 0;
};

ClipStudentConfig validate_clip_student(const SafeTensors&, const ClipProjection&);
std::vector<float> clip_student_embeddings(const SafeTensors&, const ClipStudentConfig&,
                                           const std::vector<int32_t>& ids);
// Dense BF16/FP16/FP32 and Comfy FP8-scaled matrices; used only by the new path
// (existing I8 ConvRot runtimes keep their normal native weight handling).
std::vector<float> clip_dense_values(const SafeTensors&, const std::string& name);
}
