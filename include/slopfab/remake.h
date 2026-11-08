#pragma once
#include <string>
#include <vector>
#include "slopfab/safetensors.h"
#include "slopfab/sampling_settings.h"

namespace slopfab {
struct GenerateRequest;
// Published merged exports have no metadata; source/mode metadata takes
// precedence over the distribution filename when present.
std::string remake_checkpoint_mode(const SafeTensors& checkpoint);
SamplingSettings remake_sampling(const std::string& mode);
void validate_remake_request(const GenerateRequest& request);
std::vector<std::string> remake_prompt_segments(int frames, size_t images,
                                               const std::string& prompt);
} // namespace slopfab
