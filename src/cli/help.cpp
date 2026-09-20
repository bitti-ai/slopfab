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
const CommandHelp kCommands[] = {
    {"prepare-lora", "slopfab prepare-lora --adapter FILE --width N [--download]",
     "prepare an adapter's AdaLN grid outside inference",
     "  --adapter FILE             adapter to update atomically\n"
     "  --width N                  base transformer hidden width\n"
     "  --download                 allow verified legacy grid download\n"},
    {"generate", "slopfab generate --prompt <text> [options]", "text to video and audio",
     "  --prompt <text>              the prompt (MiniMax Context-IR structure)\n"
     "  --prompt-file <file>         read that same prompt from a UTF-8 text file\n"
     "                               instead; a BOM and surrounding blank space are\n"
     "                               stripped. Cannot be combined with --prompt\n"
     "  --reference-image <file>     ordered Ref2VA image; repeat up to 9 times.\n"
     "  --refmod <file>              pre-encoded H3 reference safetensors; repeatable.\n"
     "  --refmod-strength <0..1>     strength of preceding refmod (default 1).\n"
     "  --refmod-copies <1..10>      copies of preceding refmod (default 1).\n"
     "  --reference-video <file>     ingest clip and soundtrack (up to 3).\n"
     "  --reference-audio <file>     ingest standalone audio (up to 3).\n"
     "                               Video/audio generation supports CUDA/Vulkan; file\n"
     "                               ingestion requires ffmpeg and ffprobe.\n"
#if SLOPFAB_WITH_FFMPEG
     "                               Any still or video FFmpeg can decode (a video\n"
     "                               contributes its first frame), plus binary PPM.\n"
#else
     "                               PNG, JPEG, BMP, TIFF, GIF and binary PPM on\n"
     "                               Windows; binary PPM elsewhere (no FFmpeg in this\n"
     "                               build).\n"
#endif
     "                               Requires Ref2VA transformer weights. Files are\n"
     "                               read only when the run starts\n"
     "  --out <file>                 output path (default output/video-<timestamp>.mp4)\n"
     "  --aspect <W:H>               display aspect (default 16:9; model canvas limits)\n"
     "  --resolution <WxH>           exact canvas instead of an aspect; both axes a\n"
     "                               multiple of 32, ratio 1:4 to 4:1. Not capped to the\n"
     "                               trained 1344x768 area — larger is allowed, warned\n"
     "                               about, and costs attention time quadratically\n"
     "  --frames <n>                 snapped up to 17k+5 (default 124, minimum 6)\n"
     "  --steps <n>                  sigma grid points, n-1 evaluations (inherit recipe; normally 50)\n"
     "  --lora <file>               H3 safetensors adapter (repeatable)\n"
     "  --lora-strength <n>         strength for preceding --lora (default 1)\n"
     "  --schedule <name>           default or taomate-3step\n"
     "  --sampling-settings FILE    JSON sigma shifts and optional fixed base grid\n"
     "  --conditioning-settings FILE  JSON reference and fixed-conditioning policy\n"
     "  --sampler euler|ab2          integrator (default euler)\n"
     "  --seed <n>                   noise seed; negative or absent draws a random one\n"
     "  --count <n>                  generate n videos; explicit seeds increment by one,\n"
     "                               random ones are drawn afresh for each\n"
     "  --raw                        write .y4m + .wav instead of muxing MP4\n"
     "  --inference-backend cuda|vulkan\n"
     "                               neural model backend (default cuda); Vulkan\n"
     "                               supports native text and reference conditioning\n"
     "  --output-accelerator cpu|vulkan\n"
     "                               RGB-to-YUV output conversion only (default cpu);\n"
     "                               model inference remains CUDA\n"
     "  --dry-run                    resolve and print the plan, touch no weights\n"
     "  --synthetic-latents          skip conditioning and denoising and decode seeded\n"
     "                               noise, to exercise the VAEs and the muxer\n"
     "  --attn-band <frames>         frame-banded attention: a video row attends to\n"
     "                               +/- this many latent frames instead of the whole\n"
     "                               sequence. 0 (default) is off, and saves more the\n"
     "                               longer the request.\n"
     "                               Lossy. At +/-9 on the default geometry: ~1.4x\n"
     "                               faster per step; video latents land 1.6-2.0x the\n"
     "                               sampler noise floor away from unbanded, measured\n"
     "                               at 6, 24 and 40 steps with no trend in between.\n"
     "                               The audio cost is NOT characterised -- one seed\n"
     "                               per point leaves its noise floor moving as much\n"
     "                               as the effect. Judge output before relying on it\n"
     "  --attention <backend>        none, flash2, sage2 (default), sol,\n"
     "                               sol-experimental, or exact. Vulkan neural\n"
     "                               inference accepts exact, flash2 and sage2.\n"
     "                               The experimental SM120-only\n"
     "                               path is lossy and fails rather than falling back.\n"
     "  --vulkan-sage-workspace-mib <n>\n"
     "                               extra Sage scratch budget (default 64 MiB);\n"
     "                               0 uses compact preparation; Q/K scratch is extra\n"
     "  --sol-beta <f>               routing threshold multiplier (default 1)\n"
     "  --sol-error-k/v <f>           experimental K-residual/V-dispersion weights\n"
     "  --sol-step-start <n>         first active denoise step (default 10)\n"
     "  --sol-step-end <n>           last active denoise step (default max)\n"
     "  --sol-step-every <n>         activate every nth step in that range\n"
     "  --sol-layer-start <n>        first active main layer (default 2)\n"
     "  --sol-layer-end <n>          last active main layer (default max)\n"
     "  --sol-layer-every <n>        activate every nth layer in that range\n"
     "                               unfused reference implementation. Sage2 is lossy\n"
     "                               INT8/FP8 attention and requires a supported GPU\n"
     "  --dump-latents <f>           the denoiser's own output as fp32 safetensors,\n"
     "                               before either VAE; the diff point for a change\n"
     "                               to the transformer\n"
     "  --init-latents <f>           start the loop from these latents instead of the\n"
     "                               seeded draw; with --synthetic-latents, decode\n"
     "                               them straight to video and audio. Same shape as\n"
     "                               --dump-latents writes. Off by default.\n"
     "  --save-latents <f>           save a versioned AV latent archive for continuation.\n"
     "  --continue-from <f>          extend a saved archive; --frames is NEW frames,\n"
     "                               rounded up to a multiple of 17. Output is joined.\n"
     "  --overlap-frames <n>         hidden context, 17*k+5 frames (default 22).\n"
     "  --animate                   Viggle fixed-conditioning recipe (default 4 steps)\n"
     "  --preserve-driving-audio    pin driving soundtrack as clean target audio\n"
     "  --prompt-embedding <f>       F32 prompt_embedding [L,5120] safetensors;\n"
     "                               reference runs also need text_token_tags [L]\n"
     "\n"
     "step caching (all off by default; each one trades quality for time):\n"
     "  --motion-cache               enable motion-aware video/audio residual reuse\n"
     "  --motion-cache-threshold <x> reuse threshold, 0 disables (default 0.15, 0..1)\n"
     "  --motion-cache-strength <x>  motion weighting (default 1, 0..4)\n"
     "  --motion-cache-warmup <n>    initial computed calls (default 4, 2..20)\n"
     "  --motion-cache-max-skips <n> consecutive reuse limit (default 2, 1..10)\n"
     "  --motion-cache-start <x>     active range start (default 0.15, 0..1)\n"
     "  --motion-cache-end <x>       active range end (default 0.95, 0..1)\n"
     "  --motion-cache-subsample <n> estimator stride (default 8, 1..32)\n"
     "  --motion-cache-verbose       log each compute/reuse decision\n"
     "                               Requires Euler; excludes other caches,\n"
     "                               FastH3 V2, TaoMate and Animate.\n"
     "  --cache-threshold <x>        reuse the previous step's velocity until the\n"
     "                               conditioning has moved by <x>. 0 = off (default).\n"
     "                               Units: accumulated relative L1 of the rank-8 AdaLN\n"
     "                               code c(t) over the pair (video t, audio t) --\n"
     "                               dimensionless, and 1.0 means the conditioning has\n"
     "                               moved, summed over the skipped steps, by as much as\n"
     "                               its own magnitude. Useful values are far below 1.\n"
     "                               LOSSY AND UNCHARACTERISED: no quality comparison\n"
     "                               has been run at any threshold. Skips are strongly\n"
     "                               front-loaded on this checkpoint, which is where\n"
     "                               errors have the most steps left to compound.\n"
     "  --cache-warmup <n>           first n steps always evaluated (default 3, floor 2)\n"
     "  --skip-every <n>             instead of the threshold, evaluate every n-th step.\n"
     "                               0 = off (default). A calibration-free baseline the\n"
     "                               threshold has to beat; the two cannot be combined.\n"
     "\n"
     "block caching (off by default; independent of the step cache above):\n"
     "  --block-cache-span <n>       reuse the combined residual of n consecutive\n"
     "                               transformer blocks instead of evaluating them.\n"
     "                               0 = off (default). Saves (n/50) x (reused steps /\n"
     "                               total steps); measured within 0.1 point of that.\n"
     "                               Costs one buffer the size of the residual stream\n"
     "                               while enabled -- 844 MB at the 248-frame geometry,\n"
     "                               88 MB at 22 frames.\n"
     "                               Cannot be combined with --cache-threshold or\n"
     "                               --skip-every: a step those skip is a step this one\n"
     "                               never sees, so its interval stops counting.\n"
     "                               LOSSY AND UNCHARACTERISED, like the step cache: the\n"
     "                               reused residual is added unscaled to a state that\n"
     "                               has moved since it was captured.\n"
     "  --block-cache-start <n>      first block of the span. Default -1, which centres\n"
     "                               it: the first and last blocks carry the fastest-\n"
     "                               changing residuals and are the worst to reuse.\n"
     "  --block-cache-interval <n>   evaluate the span every n-th step, reuse it on the\n"
     "                               others (default 2 = alternate, minimum 2).\n"
     "  --block-cache-warmup <n>     first n steps always evaluate the span (default 3,\n"
     "                               floor 1)\n"
     "\n"
     "The first steps and the last step are always evaluated whatever these say.\n"
     "The trajectory is most sensitive early, and at the terminal step the sigma\n"
     "ratio is zero, so a reused velocity there lands in the output undamped.\n"
     "Every run that reused anything prints how many of its evaluations it\n"
     "skipped.\n"
     "\n"
     "checkpoints (omitted weights are found under weights/ or downloaded there):\n"
     "  --tokenizer <f>              override the embedded tokenizer.json\n"
     "  --text-encoder <f>           Qwen3-VL conditioner, int8 ConvRot or nvfp4 AWQ\n"
     "  --transformer <f>            H3 omni transformer, fp8, int8 ConvRot, nvfp4 or NF4\n"
     "  --vae <f>                    video VAE decoder\n"
     "  --audio-vae <f>              audio VAE decoder and reference encoder\n"
     "\n"
     "  --bench-load <n>             load --transformer n times and exit, timing each.\n"
     "                               The first pays for reading the file, the rest do\n"
     "                               not; the gap is the cold-start cost.\n"
     "\n"
     "The quantisation of each checkpoint is read out of the file, so there is\n"
     "no flag for it and the two need not match.\n"
     "\n"
     "--sampler ab2 is Adams-Bashforth 2: second order at the same one forward\n"
     "pass per step, so a step costs what it always did and the reason to use it\n"
     "is to lower --steps. Its first step has no velocity history and is Euler.\n"
     "\n"
     "The conditioner and the transformer are loaded and freed in sequence\n"
     "rather than together: at 23.1 GB and 19.3 GB the int8 and fp8 pair cannot\n"
     "co-exist on a 32 GB card. The nvfp4 pair would (13.1 GB and 12.5 GB), but\n"
     "the sequencing costs nothing measurable and is what makes every mixture of\n"
     "the four safe. Expect the first output well after the progress line starts\n"
     "moving.\n"},
    {"inspect", "slopfab inspect <file.safetensors> [options]",
     "summarise a checkpoint's tensors",
     "  --list                       print every tensor, not just a summary\n"
     "  --prefix <str>               only tensors whose name starts with <str>\n"
     "  --limit <n>                  cap listed tensors (default 40, 0 = all)\n"},
    {"compare", "slopfab compare <reference> <actual> [options]",
     "diff two checkpoints tensor by tensor",
     "  --abs-tol <x>                absolute tolerance (default 1e-3)\n"
     "  --rel-tol <x>                relative tolerance (default 1e-2)\n"
     "  --verbose                    report passing tensors too\n"
     "\n"
     "Pass/fail is elementwise: a tensor passes on either tolerance. Beside\n"
     "that, two whole-tensor quality metrics are reported, because \"is any\n"
     "element wrong\" and \"is this the same tensor\" are different questions and\n"
     "the second is the one a quality comparison asks.\n"
     "\n"
     "  rel_L2       ||reference - actual|| / ||reference||\n"
     "               = sqrt(sum (r-a)^2) / sqrt(sum r^2)\n"
     "               Normalised by the REFERENCE -- not by actual, and not by\n"
     "               the mean of the two norms. So it is asymmetric on purpose:\n"
     "               compare(a,b) and compare(b,a) ask different questions.\n"
     "               Zero reference gives 0 if the difference is zero, else inf.\n"
     "\n"
     "  correlation  Pearson over the flattened tensor, MEANS SUBTRACTED:\n"
     "               S_ra / sqrt(S_rr * S_aa), S_xy = sum (x-mean_x)(y-mean_y).\n"
     "               Not cosine similarity. Identical tensors give exactly +1,\n"
     "               a tensor against its own negation exactly -1, and a\n"
     "               constant tensor (zero variance) gives 0, undefined.\n"
     "\n"
     "Both are computed over the flattened tensor across element pairs where\n"
     "both sides are finite. Matching global statistics with falling\n"
     "correlation is this project's signature of a different sample rather\n"
     "than a degraded one -- see the README.\n"},
    {"compare-y4m", "slopfab compare-y4m <expected.y4m> <actual.y4m>",
     "byte-compare deterministic raw video outputs",
     "Reports both headers, sizes, and the first differing byte. The command\n"
     "streams its inputs and returns non-zero for any difference.\n"},
    {"decode", "slopfab decode --vae <f> [--latent <f>] [options]",
     "run the video VAE decoder",
     "  --vae <f>                    video VAE checkpoint\n"
     "  --latent <f>                 latent safetensors; omit for a synthetic one\n"
     "  --shape <T> <H> <W>          synthetic latent shape\n"
     "  --out <f>                    .y4m output\n"
     "  --ppm <f>                    also write frame 0 as a PPM\n"
     "  --dump <f>                   raw fp32 pixels as safetensors\n"},
    {"tokenize", "slopfab tokenize [--tokenizer <f>] <text>",
     "encode text and round-trip it",
     "  --tokenizer <f>              override the embedded tokenizer.json\n"
     "  --pieces                     also print the pre-tokenizer split\n"},
    {"devices", "slopfab devices", "list CUDA inference and Vulkan output devices", ""},
    {"version", "slopfab version", "print the version and exit", ""},
};

const CommandHelp* find_command(std::string_view name) {
  for (const CommandHelp& c : kCommands) {
    if (name == c.name) return &c;
  }
  return nullptr;
}

// True when the argument list asks for help. Checked before any other option,
// so `--help` works even alongside otherwise invalid arguments.
bool wants_help(int argc, char** argv) {
  for (int i = 0; i < argc; ++i) {
    const std::string_view a = argv[i];
    if (a == "--help" || a == "-h" || a == "help") return true;
  }
  return false;
}

int print_command_help(const CommandHelp& c) {
  std::printf("usage: %s\n\n%s\n", c.usage, c.summary);
  if (c.detail[0] != '\0') std::printf("\noptions:\n%s", c.detail);
  return 0;
}

void print_usage() {
  std::printf(
      "slopfab %s - MiniMax H3 video generation\n"
      "\n"
      "usage: slopfab [--cuda-version=auto|13|12] <command> [options]\n"
      "       slopfab <command> --help\n"
      "\n"
      "commands:\n",
      kVersion);
  for (const CommandHelp& c : kCommands) {
    std::printf("  %-9s %s\n", c.name, c.summary);
  }
  std::printf("\nRun `slopfab <command> --help` for that command's options.\n"
              "CUDA defaults to installed version 13, then 12; "
              "SLOPFAB_CUDA_VERSION provides the same override.\n");
}

#if SLOPFAB_WITH_CUDA
void consume_cuda_version_option(int& argc, char** argv) {
  std::string requested;
  int write = 1;
  for (int read = 1; read < argc; ++read) {
    const std::string_view argument = argv[read];
    constexpr std::string_view prefix = "--cuda-version=";
    if (argument.rfind(prefix, 0) == 0) {
      requested = std::string(argument.substr(prefix.size()));
    } else if (argument == "--cuda-version") {
      if (++read >= argc)
        throw std::invalid_argument("--cuda-version requires auto, 13, or 12");
      requested = argv[read];
    } else {
      argv[write++] = argv[read];
    }
  }
  argc = write;
  argv[argc] = nullptr;
  if (!requested.empty()) slopfab::cuda::set_cublas_version_request(requested);
}
#endif


}
