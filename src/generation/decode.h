#pragma once
#include "slopfab/generate.h"

namespace slopfab::generation {
RunResult decode_and_deliver(const GenerateRequest&, const RunOptions&,
                             const std::shared_ptr<const LatentClip>& completed, RunResult);
}
