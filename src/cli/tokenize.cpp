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
int cmd_tokenize(int argc, char** argv) {
  if (wants_help(argc, argv))
    return print_command_help(*find_command("tokenize"));

  std::string tok_path;
  std::string text;
  bool show_pieces = false;

  for (int i = 0; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (arg == "--tokenizer" && i + 1 < argc) {
      tok_path = argv[++i];
    } else if (arg == "--pieces") {
      show_pieces = true;
    } else if (!arg.empty() && arg.front() == '-') {
      std::fprintf(stderr, "slopfab: unrecognised option '%s'\n", argv[i]);
      return 2;
    } else {
      if (!text.empty())
        text += " ";
      text += argv[i];
    }
  }
  slopfab::text::Tokenizer tok;
  if (tok_path.empty())
    tok.load_embedded();
  else
    tok.load(tok_path);
  std::printf("vocab      %zu tokens\n", tok.vocab_size());

  if (show_pieces) {
    std::printf("pieces     ");
    for (const std::string& p : tok.pre_tokenize(text))
      std::printf("[%s]", p.c_str());
    std::printf("\n");
  }

  const std::vector<int32_t> ids = tok.encode(text);
  std::printf("ids (%zu)   ", ids.size());
  for (int32_t id : ids)
    std::printf("%d ", id);
  std::printf("\n");
  std::printf("tokens     ");
  for (int32_t id : ids)
    std::printf("[%s]", tok.id_to_token(id).c_str());
  std::printf("\n");

  const std::string round = tok.decode(ids);
  std::printf("decoded    %s\n", round.c_str());
  std::printf("round trip %s\n", round == text ? "OK" : "MISMATCH");
  return round == text ? 0 : 1;
}

// Resolves a request all the way to a plan, then runs it. The stages are added
// one at a time; until they all exist this reports precisely which one is
// missing rather than pretending. `--dry-run` stops after the plan, which
// costs no I/O and is the fastest way to check geometry and schedule.

}
