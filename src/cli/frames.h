#pragma once
#include "agent.h"
#include <functional>

namespace slopfab::cli {
// All image buffers and decoders belong to the executable, never the C API.
using FrameCheck = std::function<void()>;
void validate_frame_options(const Value& options);
Value extract_frames(const Value& options, const FrameCheck& check);
Value frames_from_rgb(const uint8_t* bytes, size_t size, const Value& options,
                      const FrameCheck& check);
Value frames_from_samples(const RunSamples& samples, video::FrameRate fps, const Value& options,
                          const FrameCheck& check);
int frames_command(const std::vector<std::string>& args);
}
