// slopfab - MiniMax H3 video generation in C++/CUDA.

#include <algorithm>
#include <cctype>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <exception>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// Included directly rather than picked up from slopfab/generate.h, which is
// behind the CUDA guard below: the flag parsing that names an attention mode
// is not, so a build with SLOPFAB_ENABLE_CUDA=OFF could not see this type at
// all. It is a core header and costs a CPU-only build nothing.
#include "slopfab/attention_mode.h"
#include "slopfab/dtype.h"
#include "slopfab/json.h"
#include "slopfab/dit/checkpoint.h"
#include "slopfab/dit/step_cache.h"
#include "slopfab/pipeline.h"
#include "slopfab/generate.h"
#include "slopfab/safetensors.h"
#include "slopfab/safetensors_write.h"
#include "slopfab/sampler/scheduler.h"
#include "slopfab/tensor_convert.h"
#include "slopfab/text/tokenizer.h"
#include "reference_decode.h"

#include "slopfab/video/y4m.h"
#include "slopfab/video/y4m_compare.h"

#if SLOPFAB_WITH_VULKAN
#include "slopfab/vulkan/runtime.h"
#include "slopfab/vulkan/yuv_converter.h"
#endif

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#include <winhttp.h>
#endif

#if SLOPFAB_WITH_CUDA
#include <chrono>

#include "slopfab/cuda/device.h"
#include "slopfab/cuda/cublas_dispatch.h"
#include "slopfab/cuda/deterministic_attention.cuh"
#include "slopfab/cuda/profile.h"
#include "slopfab/dit/transformer.h"
#include "slopfab/generate.h"
#include "slopfab/vae/vit_decoder.h"
#endif

#include "commands.h"

namespace slopfab::cli {
int cmd_devices() {
#if !SLOPFAB_WITH_CUDA
  std::printf("CUDA inference       unavailable in this build\n");
#else
  const int count = slopfab::cuda::device_count();
  if (count == 0) {
    std::printf("CUDA inference       no visible device\n");
  }
  for (int i = 0; i < count; ++i) {
    const slopfab::cuda::DeviceInfo d = slopfab::cuda::query_device(i);
    std::printf("device %d  %s\n", d.index, d.name.c_str());
    std::printf("  compute capability  %d.%d\n", d.major, d.minor);
    std::printf("  memory              %s free of %s\n", format_bytes(d.free_memory).c_str(),
                format_bytes(d.total_memory).c_str());
    std::printf("  multiprocessors     %d\n", d.multiprocessors);
    std::printf("  shared mem / block  %s\n", format_bytes(d.shared_memory_per_block).c_str());
    std::printf("  numeric support     bf16=%s fp8=%s fp4=%s\n", d.supports_bf16 ? "yes" : "no",
                d.supports_fp8 ? "yes" : "no", d.supports_fp4 ? "yes" : "no");
  }
#endif
#if SLOPFAB_WITH_VULKAN
  std::string diagnostic;
  if (!slopfab::vulkan::Instance::available(&diagnostic)) {
    std::printf("Vulkan output        unavailable: %s\n", diagnostic.c_str());
  } else {
    try {
      auto instance = slopfab::vulkan::Instance::create();
      const auto devices = instance.enumerate_devices();
      if (devices.empty())
        std::printf("Vulkan output        no compute device\n");
      for (size_t i = 0; i < devices.size(); ++i) {
        const auto& d = devices[i].info();
        std::printf("Vulkan output %zu     %s (timeline=%s)\n", i, d.name.c_str(),
                    d.timeline_semaphore ? "yes" : "no");
      }
    } catch (const std::exception& error) {
      std::printf("Vulkan output        unavailable: %s\n", error.what());
    }
  }
#else
  std::printf("Vulkan output        disabled in this build\n");
#endif
  return 0;
}

}
