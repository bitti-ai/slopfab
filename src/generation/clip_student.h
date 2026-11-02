#pragma once
#include "slopfab/text/clip_projection.h"

namespace slopfab::text {
PromptEmbedding encode_clip_student_cuda(const std::string&, const ClipProjection&,
                                         const std::vector<int32_t>&, const std::function<void()>&);
PromptEmbedding encode_clip_student_vulkan(const std::string&, const ClipProjection&,
                                           const std::vector<int32_t>&,
                                           const std::function<void()>&);
}
