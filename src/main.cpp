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
#include "cli/reference_decode.h"

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

#include "cli/commands.h"
using namespace slopfab::cli;

int main(int argc, char** argv) {
#if defined(_WIN32)
  SetConsoleOutputCP(CP_UTF8);
  SetConsoleCP(CP_UTF8);
#endif
#if SLOPFAB_WITH_CUDA
  try {
    consume_cuda_version_option(argc, argv);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "slopfab: %s\n", e.what());
    return 1;
  }
#endif

  if (argc < 2) {
    print_usage();
    return 2;
  }

  const std::string_view command = argv[1];

  // `--help` is honoured for every command here rather than inside each one,
  // so a command that takes no arguments at all still answers it. Checked
  // before dispatch, so it never runs the command by accident.
  if (const CommandHelp* c = find_command(command); c != nullptr && wants_help(argc - 2, argv + 2)) {
    return print_command_help(*c);
  }

  try {
    if (command == "generate") return cmd_generate(argc - 2, argv + 2, argv[0]);
    if (command == "prepare-lora") return cmd_prepare_lora(argc - 2, argv + 2);
    if (command == "inspect") return cmd_inspect(argc - 2, argv + 2);
    if (command == "compare") return cmd_compare(argc - 2, argv + 2);
    if (command == "compare-y4m") return cmd_compare_y4m(argc - 2, argv + 2);
    if (command == "devices") return cmd_devices();
    if (command == "tokenize") return cmd_tokenize(argc - 2, argv + 2);
#if SLOPFAB_WITH_CUDA
    if (command == "decode") return cmd_decode(argc - 2, argv + 2);
#endif
    if (command == "version") {
      std::printf("slopfab %s\n", kVersion);
      return 0;
    }
    if (command == "help" || command == "--help" || command == "-h") {
      if (argc > 2) {
        const CommandHelp* c = find_command(argv[2]);
        if (c != nullptr) return print_command_help(*c);
        std::fprintf(stderr, "slopfab: unknown command '%s'\n\n", argv[2]);
        print_usage();
        return 2;
      }
      print_usage();
      return 0;
    }
    std::fprintf(stderr, "slopfab: unknown command '%s'\n\n", argv[1]);
    print_usage();
    return 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "slopfab: %s\n", e.what());
    return 1;
  }
}

