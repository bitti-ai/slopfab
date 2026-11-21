#pragma once
#include "slopfab/generate.h"
#include <array>

namespace slopfab::generation {
struct PiH3DecodeInput {
  std::shared_ptr<const SafeTensors> adapter;
  std::vector<float> hidden;
  std::array<float, 8> code{};
};
RunResult decode_and_deliver(const GenerateRequest&, const RunOptions&,
                             const std::shared_ptr<const LatentClip>& completed, RunResult,
                             const PiH3DecodeInput* pi_h3 = nullptr);
}
