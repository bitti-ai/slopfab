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
#include "media_commands.h"
#include "agent.h"
#include "startup.h"
#include "settings.h"

namespace slopfab::cli {
const CommandHelp kCommands[] = {
    {"jobs", "slopfab jobs COMMAND [options]", "Submit and manage persistent generation jobs.",
     "  submit  queue generation\n  status  inspect a job\n  wait    wait for completion\n  cancel  cancel a job\n  worker  process the queue\n"},
    {"models", "slopfab models COMMAND [options]", "List, inspect, compare and prepare model files.",
     "  list          installed files and aliases\n  inspect       checkpoint tensors\n  compare       compare checkpoints\n  prepare-lora  prepare an adapter\n"},
    {"conditioning", "slopfab conditioning COMMAND [options]", "Prepare text and multimedia conditioning.",
     "  tokenize\n  encode-text\n  encode-refmod\n  bundle-refmods\n"},
    {"system", "slopfab system COMMAND", "Inspect hardware, capabilities and executable information.",
     "  doctor\n  devices\n  capabilities\n  version\n"},
    {"requests", "slopfab requests COMMAND [options]", "Run and validate shared generation requests.",
     "  run       generate from arguments or JSON\n  validate  validate without inference\n  schema    generation JSON schema\n"},
    {"update", "slopfab update [--check]", "Download and install the latest GitHub release.",
     "  --check  report the latest stable version without downloading or installing\n"
     "Requires a pinned-key Ed25519 release signature and matching package SHA256.\n"
     "Replaces executables and bundled download/licenses mods.\n"
     "Preserves weights, projects, settings and other mods. Windows installs after this process exits.\n"
     "Backups and installation status are saved beside the executable.\n"},
    {"video", "slopfab video COMMAND [options]", "Generate, inspect, restore and analyze video.",
     "  generate         generate video with audio\n  frames           extract frames or contact sheets\n"
     "  syncnet          measure lip synchronization with face detection/tracking\n"
     "  upscale          restore Y4M video\n"
     "  upscale-latents  enlarge H3 latent video\n  decode           decode video VAE latents\n"
     "  compare          compare two Y4M videos\nUse slopfab help video COMMAND for options.\n"},
    {"audio", "slopfab audio COMMAND [options]", "Generate music, transcribe audio and separate stems.",
     "  generate    generate songs or instrumental music\n  transcribe  transcribe speech or vocals\n"
     "  separate    extract vocals or four stems\nUse slopfab help audio COMMAND for options.\n"},
    {"image", "slopfab image COMMAND [options]", "Generate, edit and restore still images.",
     "  generate  generate or edit an image\n  upscale   restore an image\n"
     "Use slopfab help image COMMAND for options.\n"},
    {"syncnet", "slopfab video syncnet VIDEO --model syncnet_v2.model --face-model sfd_face.pth [options]",
     "Measure lip synchronization per face track using native SyncNet and S3FD.",
     "  --model FILE          only SHA256 961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442\n"
     "  --face-model FILE     pinned upstream S3FD .pth or standard FP32 S3FD safetensors\n"
     "  --cropped             input is already a single face crop; omit --face-model\n"
     "  --backend NAME        auto (prefer Vulkan), vulkan or cpu\n"
     "  --out FILE            default output/<input-name>-syncnet.json\n"
     "  --start SECONDS       start relative to video stream origin (default 0)\n"
     "  --duration SECONDS    analyze up to 0.24..600 seconds (default 30)\n"
     "  --vshift N            search +/-N video frames at 25 fps (default 15, max 250)\n"
     "  --min-track SECONDS   minimum track length (default 1.5); also needs 2*vshift+6 frames\n"
     "  --detect-every N      detect every 1..10 frames (default 5), interpolate gaps\n"
     "  --detect-size N       detector longest side 128..1280 (default 320)\n"
     "  --face-threshold N    face confidence threshold .5..0.9999 (default .9)\n"
     "  --ffmpeg PATH --ffprobe PATH  external media tools; preserve relative AV timestamps\n"
     "  --overwrite           atomically replace an existing report\n"
     "Negative audio_shift_ms advances audio; positive delays it. LSE-D/LSE-C are not accuracy percentages.\n"
     "Temporary decoded video is bounded to 8 GiB and 1280 pixels per side. See docs/syncnet.md.\n"},
    {"remote", "slopfab remote setup|show|update|remove NAME | remote list | remote fetch RECEIPT",
     "Configure SSH hosts and install/update their executable from the bundled Windows/Linux binaries.",
     "  setup NAME --directory PATH  register a host and provision slopfab in its existing project\n"
     "  --host SSH_HOST              SSH alias or user@hostname (default NAME)\n"
     "  --platform linux|windows     inferred from --directory when omitted; WSL uses linux\n"
     "  --binary-dir PACKAGE_ROOT    local folder containing windows/slopfab.exe and linux/slopfab\n"
     "  --ssh PATH                   local SSH client (default ssh on PATH)\n"
     "  show NAME | list             inspect saved connections\n"
     "  update NAME                 verify/upload the current bundled executable\n"
     "  remove NAME                 forget a connection; retain remote files\n"
     "  fetch RECEIPT               retry downloads from a remote.json receipt\n"
     "Execute regular commands with --host NAME (alias --remote NAME), for example:\n"
     "  slopfab requests run --host gpu image --prompt \"A red apple\"\n"
     "  slopfab system devices --host gpu\n"
     "Each invocation verifies the remote binary checksum and installs an update when needed.\n"
     "Models, sources and inputs stay remote. See docs/remote.md for setup and packaging.\n"},
    {"fetch", "slopfab remote fetch PATH/remote.json", "Retry result downloads without rerunning the remote command.",
     "Uses the saved SSH connection and checksums. Verified local files are skipped.\n"},
    {"separate", "slopfab audio separate FILE --model MODEL.safetensors [options]",
     "Separate vocals or four stems with native HTDemucs from safetensors weights.",
     "  --model FILE                standard HTDemucs state_dict safetensors (or model folder)\n"
     "  --out DIRECTORY             default output/<input-name>-stems\n"
     "  --two-stems NAME            vocals, drums, bass or other; default writes all four\n"
     "                              vocals writes vocals.wav and instrumental.wav\n"
     "  --segment SECONDS           0.05..7.8 (default 7.8); shorter can reduce quality\n"
     "  --overlap FRACTION          0..0.9 (default 0.25), triangular overlap-add\n"
     "  --backend cpu|auto          native FP32 CPU inference; no GPU backend yet\n"
     "  --ffmpeg PATH               decoder/resampler for non-44.1kHz WAV and other media\n"
     "  --overwrite                 replace existing stems\n"
     "Outputs are stereo 44.1kHz float32 WAV, without clipping or per-stem normalization.\n"
     "Two-stem accompaniment is the input minus the selected stem. See docs/separation.md.\n"},
    {"transcribe", "slopfab audio transcribe FILE --model MODEL.safetensors [options]",
     "Transcribe speech or song vocals locally with native Whisper and embedded model assets.",
     "  --model FILE                standard Hugging Face Whisper safetensors (or model folder)\n"
     "  --variant NAME              auto; use large-v1 or large-v2 for their ambiguous shapes\n"
     "  --backend auto|cuda|vulkan|cpu  FP32 inference; CPU is a slow reference path\n"
     "  --language CODE             auto (default), ISO language code or English language name\n"
     "  --prompt TEXT               vocabulary/style context; not an instruction to rewrite\n"
     "  --beam-size N               deterministic search width 1..10 (default 1)\n"
     "  --word-timestamps           estimate word timing using cross-attention alignment\n"
     "  --no-context                do not condition on previous windows\n"
     "  --out FILE --format FORMAT  json, txt, srt or vtt; format defaults to output extension\n"
     "  --overwrite                 replace an existing transcript atomically\n"
     "  --ffmpeg PATH               decoder for non-WAV audio/video; WAV is native\n"
     "Models: tiny/base/small/medium (including .en), large-v1/v2/v3 and large-v3-turbo.\n"
     "Only weights are needed; tokenizer, mel filters and standard profiles are embedded.\n"
     "Songs use the same recognizer; no vocal separation is applied. Word timing is approximate.\n"},
    {"frames", "slopfab video frames VIDEO --out IMAGE.png [options]",
     "Extract a video frame or a downscaled contact sheet; capture generation samples in RAM.",
     "  VIDEO --out IMAGE.png        extract frame zero by default\n"
     "  --frame N | --time SECONDS   zero-based frame index or nearest timestamp\n"
     "  --grid COLSxROWS             evenly sample a grid (at most 256 cells)\n"
     "  --start SECONDS --end SECONDS optional inclusive grid range, relative to first frame\n"
     "  --width N                   maximum frame/cell width; grid default 320; never upscale\n"
     "  --overwrite                 atomically replace an existing PNG\n"
     "  --ffmpeg PATH --ffprobe PATH compressed-video decoder/probe overrides\n"
     "  frames generate --out REVIEW.png [options] -- GENERATION_ARGS\n"
     "                              review generation RGB in RAM, without re-decoding video\n"
     "Native 8-bit 420 Y4M needs no FFmpeg. Other formats need FFmpeg and ffprobe.\n"
     "JavaScript: import {frames} from 'slopfab'; frames.extract({input,output,frame|time});\n"
     "use {columns,rows,width} for grids. frames.fromRGB(bytes,{sourceWidth,sourceHeight,\n"
     "count,fps,output,...}) reads packed RGB24 frames. frames.generate(args,options) runs\n"
     "generation in a child process; capture uses that process's live decoded samples.\n"
     "Results include source frame indices, timestamps and cell positions.\n"},
    {"motions", "slopfab motions render|frame FILE.js --out FILE [options]",
     "Render JavaScript vector Motions with ThorVG; export video or overlay existing video.",
     "  render FILE.js --out VIDEO.mp4|.mov|.webm   render animation; MOV/WebM preserve alpha\n"
     "  frame FILE.js --out IMAGE.png|.svg         render a still; --time SECONDS selects time\n"
     "  --over VIDEO                composite over video; preserve base duration and audio\n"
     "  --audio AUDIO               supply/replace audio (e.g. generated WAV alongside Y4M)\n"
     "  --width N --height N        canvas size (default 1280x720 or script definition)\n"
     "  --fps N --duration SECONDS  timeline (default 30 fps, 5 seconds or script definition)\n"
     "  --x N --y N                 overlay placement (default 0,0)\n"
     "  --overwrite                 replace an existing output atomically\n"
     "  --memory-mb N --timeout-ms N JavaScript limits, as with run\n"
     "  --ffmpeg PATH               external video encoder override\n"
     "Module: export default {width,height,fps,duration,frame:ctx=>SVG_STRING};\n"
     "or export a factory receiving script arguments after --. Import {motions} from 'slopfab'.\n"
     "motions.svg(w,h,body), element(tag,attrs,children), text(text,attrs), font(name,path),\n"
     "and render(options,frame) are available to workflows. ctx: frame,time,progress,width,height,fps,frames.\n"
     "PNG/SVG need no FFmpeg. Video export requires FFmpeg.\n"},
    {"run", "slopfab run FILE.js [-- ARGS...] | run --eval SOURCE",
     "Run an embedded QuickJS workflow with filesystem, HTTP, process and CLI APIs.",
     "  run FILE.js -- ARGS...       run an ES module; optional default export is called with args\n"
     "  run -                       read JavaScript from stdin\n"
     "  run --eval SOURCE           execute inline JavaScript\n"
     "  --memory-mb N               JavaScript heap limit (default 256)\n"
     "  --timeout-ms N              workflow deadline, 0 = unlimited (default)\n"
     "Import APIs from 'slopfab' or use global slopfab.\n"
     "  export default args => VALUE          return JSON-serializable data (async supported)\n"
     "  slopfab.fs.readJson/writeJson         files; readText/writeText/readBytes/writeBytes also available\n"
     "  slopfab.exec([EXE, ...ARGS], OPTIONS)  shell-free process; stdout/stderr/exitCode returned\n"
     "  slopfab.spawn([EXE, ...ARGS])          concurrent child with poll(), wait(), kill()\n"
     "  slopfab.cli(['COMMAND', ...ARGS])      same CLI/config/jobs; parsed events and result\n"
     "  slopfab.http.json(URL)                fetch JSON; download(URL, PATH) streams to disk\n"
     "    download(URL, PATH, {resume:true, connections:4, retries:2}) resumes parallel ranges\n"
     "  slopfab.tool('ffmpeg')                locate external FFmpeg for exec/spawn\n"
     "  slopfab.path.fileUrl(PATH)            absolute percent-encoded file URL\n"
     "  slopfab.frames.extract(OPTIONS)      video frame/grid PNG; see video frames --help\n"
     "Console output goes to stderr. Local ES imports, promises and timers are supported.\n"
     "Scripts and executable mods run with your normal filesystem/process access.\n"},
    {"prompts", "slopfab prompts list|get|select [options]",
     "Retrieve offline model-specific prompting guidance, templates and examples before writing generation prompts.",
     "  prompts list [--model NAME]          index supported models and use cases\n"
     "  prompts get MODEL USE_CASE           load a guide; e.g. h3 fl2va, moss layout, minimax_music3 song\n"
     "  prompts select --model NAME [options] select by intent and return a guide with reference bindings\n"
     "  --task video|image|music             default: model's primary task\n"
     "  --first-frame FILE --last-frame FILE intended endpoints; select I2VA, L2VA or FL2VA\n"
     "  --reference-image FILE               ordinary identity/style/content reference; repeatable\n"
     "  --reference-video FILE --reference-audio FILE   ordinary H3 references; repeatable\n"
     "  --edit-image FILE                    source for local image editing\n"
     "  --layout                            MOSS spatial-layout JSON guide\n"
     "  --instrumental                      music without vocals; otherwise clarify vocal intent\n"
     "  --continuation                      H3 long-shot/latent continuation guide\n"
     "  --character-swap                    H3 character LoRA guide; one source video and identity image\n"
     "  --duration SECONDS --shots N         video template timing; validated sampling window, default 1 shot\n"
     "Model names: h3, krea2, moss, fibo2, yue2, minimax_music3; configured aliases also work.\n"
     "These selectors retrieve context only; they neither read files nor generate media.\n"
     "--first-frame and --last-frame belong to prompts select, not generate. Follow the guide's binding instructions.\n"
     "Ordinary references do not imply first frames. Mixed endpoint/reference roles require explicit Ref2VA mapping.\n"
     "Guides include local capability limits, source links and original examples; fill placeholders for the actual brief.\n"},
    {"projects", "slopfab projects ACTION [NAME] [ITEM_ID] [options]",
     "Organize generation results in an output folder with an offline review page and copyable agent edit prompts.",
     "  projects create NAME [--title TEXT]   create project.json, PROJECT.md and output/NAME.html\n"
     "  projects list                        list projects in the output directory\n"
     "  projects show NAME                   read items, exact filenames, recipes and naming rules\n"
     "  projects open NAME                   open NAME.html in the default browser\n"
     "  projects add NAME --file PATH        import and register media; --id ID sets a stable ID\n"
     "  projects update NAME                build output/NAME.html from the project's project.json\n"
     "  projects update NAME ID [options]    replace media, preview or metadata; retain older files\n"
     "  projects remove NAME ID              remove from the active project; retain files/history\n"
     "  projects restore NAME ID             restore a removed item\n"
     "  projects render NAME                compatibility alias for projects update NAME\n"
     "  --output-dir DIR                     root directory; default: settings.output_directory or ./output\n"
     "  --title TEXT --prompt TEXT           item label and generation prompt (add/update)\n"
     "  --file PATH --preview PATH           source or browser preview; paths relative to working directory\n"
     "  --request FILE                       save a generation JSON request for regeneration (add/update)\n"
     "  --kind image|video|music|clip         default inferred from file; only add sets kind\n"
     "  --parent VIDEO_ID                    required for clips; only add sets parent\n"
     "  --position N --start SECONDS --end SECONDS   clip order and optional source trim (add/update)\n"
     "  --if-revision N                      reject stale add/update/remove/restore operations\n"
     "Use projects for user generation requests. Create a project, then generate with --project NAME.\n"
     "Read PROJECT.md or projects show before writing files: NAME.html uses exact manifest paths, not a folder scan.\n"
     "For page content, edit only project.json; never edit HTML, its layout/template or PROJECT.md.\n"
     "Run projects update NAME after JSON edits or new media/previews. --project generation and registration rebuild automatically.\n"
     "Imports use assets/ID/vVERSION/source.EXT and preview.EXT; registration regenerates the page.\n"
     "Use --project-item ID to regenerate an item under its stable ID. Preserve unrelated items.\n"
     "After clip edits, rebuild the parent video externally and register it with projects update --file.\n"
     "For Y4M/PPM, create a browser preview with external FFmpeg and register --preview.\n"
     "Open NAME.html locally: actions generate prompts to paste into an agent; the page executes no commands.\n"},
    {"upscale-latents", "slopfab video upscale-latents --input FILE --out FILE [options]",
     "upscale an existing H3 latent archive without diffusion or decoding",
     "  --model FILE               H3 3D conv v1 checkpoint (default under weights/)\n"
     "  --scale N                  1..4, default 2; dimensions round to multiples of 32\n"
     "  --inference-backend NAME   cuda or vulkan\n"
     "  --no-chunking              process all latent frames together\n"
     "Output is a new H3 latent archive with resized video and unchanged audio.\n"},
    {"upscale", "slopfab video upscale --input FILE --out FILE --upscale-method NAME [options]",
     "restore native images or Y4M video with Real-ESRGAN or SeedVR2",
     "  --upscale-method NAME       realesrgan (default), seedvr2 (CUDA/Vulkan)\n"
     "  --upscale-model FILE        checkpoint; default path depends on method\n"
     "  --inference-backend NAME    cuda or vulkan (Real-ESRGAN only)\n"
     "  --upscale-tile N            128 input pixels for ESRGAN; 256 VAE output pixels for SeedVR2\n"
     "                             0 disables spatial tiling\n"
     "  --upscale-tile-pad N        Real-ESRGAN tile context, default 10\n"
     "  --upscale-pre-pad N         Real-ESRGAN reflection padding, default 10\n"
     "  --upscale-vae FILE          SeedVR2 VAE safetensors\n"
     "  --upscale-resolution WxH    SeedVR2 output size; default 4x input\n"
     "  --upscale-segment-frames N  SeedVR2 joint frames: 1 or 4n+1, default 5\n"
     "  --upscale-seed N            SeedVR2 seed, default 666\n"
     "  --upscale-device N          SeedVR2 backend device, default 0\n"
     "  --upscale-no-color-match    disable SeedVR2 color correction\n"
     "  --dump FILE                save float pixels for .ppm output\n"
     "  --dry-run                  validate arguments without loading weights\n\n"
     "Output: .png/.ppm for images; .y4m for video. Convert compressed media externally.\n"
     "Video input: BT.709 limited-range progressive Y4M, 8-bit 4:2:0.\n"
     "Decoded input is limited to 2 GiB; use short clips. Audio remains a separate WAV.\n"},
    {"prepare-lora", "slopfab models prepare-lora --adapter FILE --width N",
     "prepare an adapter's AdaLN grid outside inference",
     "  --adapter FILE             adapter to update atomically\n"
     "  --width N                  base transformer hidden width\n"
     "  Requires an embedded grid or a local companion file.\n"},
    {"generate", "slopfab requests run [video|image|music] --prompt TEXT [options]",
     "text to image, video and audio",
     "  --request FILE|-            versioned JSON request; excludes generation flags\n"
     "  --agent                     NDJSON progress and final result; logs on stderr\n"
     "  --duration SECONDS          video/music duration; result reports resolved length\n"
     "  --deadline SECONDS          cooperative generation deadline\n"
     "  --overwrite                 allow replacing output artifacts\n"
     "  --check-ready               with --dry-run, check required model files\n"
     "  Native outputs: .png/.ppm images, .y4m video plus .wav audio, .wav music.\n"
     "  Convert other formats with an external ffmpeg installation.\n"
     "  --clip-projection FILE      H3 ClipProj safetensors with a matching 4B/8B text encoder; text-only\n"
     "  --model NAME                h3, moss, fibo2, krea2, yue2 or minimax_music3\n"
     "  --lyrics TEXT               music lyrics; --prompt supplies style/tags\n"
     "  --music-duration SECONDS    music duration ceiling (default 30, maximum 300)\n"
     "  --music-planning MODE       YuE2 full (default), melody or off\n"
     "  --project NAME              save results in an existing project and refresh its review page\n"
     "  --project-item ID           regenerate an existing item (count 1); preserve previous versions\n"
     "  --output-dir DIR            project root override; otherwise use settings.output_directory or ./output\n"
     "  --music-settings JSON       music sampling settings as inline JSON\n"
     "  YuE2 defaults: CUDA, 32 midpoint steps, stereo 48 kHz .wav output.\n"
     "  MiniMax Music3 defaults: CUDA, 30 Euler steps, stereo 44.1 kHz .wav.\n"
     "  --prompt <text>              the prompt\n"
     "  --negative-prompt <text>     MOSS negative prompt (default empty)\n"
     "  --guidance <number>          CFG scale (MOSS: 3; Krea 2: 0)\n"
     "  --uncond-skip-layers <list>   MOSS negative-branch layers, e.g. 22,24 or none\n"
     "                               (default 22; disabled on the first step)\n"
     "  MOSS defaults: CUDA/flash2, 1024x1024, 50 Euler updates, .ppm output.\n"
     "  --prompt-file <file>         read that same prompt from a UTF-8 text file\n"
     "                               instead; a BOM and surrounding blank space are\n"
     "                               stripped. Cannot be combined with --prompt\n"
     "  --reference-image <file>     ordered Ref2VA image; repeat up to 9 times.\n"
     "  --edit-image <file>          source for still-image inpainting; output is .png or .ppm\n"
     "  --edit-box x,y,w,h           editable rectangle in original-image pixels\n"
     "  --edit-strength <0..1>       fraction of denoising steps, >0 (default 1)\n"
     "  --edit-feather <pixels>      inward edge blending (default 0)\n"
     "  --refmod <file>              standalone or mixed H3 RefMod bundle; repeatable.\n"
     "  --refmod-strength <0..1>     strength of preceding refmod (default 1).\n"
     "  --refmod-copies <1..10>      copies of preceding refmod (default 1).\n"
     "  --reference-video FILE      native Y4M video, 2..15 seconds (up to 3).\n"
     "  --reference-audio FILE      mono/stereo WAV, 2..15 seconds (up to 3).\n"
     "  --driving-audio FILE        attach WAV to exactly one --reference-video, at time zero.\n"
     "                             trim excess audio or pad silence to the video duration.\n"
     "  --reference-image FILE      PNG/JPEG/PPM; Windows also supports platform image codecs.\n"
     "                               References require compatible Ref2VA weights.\n"
     "  --out FILE                  native output (default output/<unique-id>/video.y4m).\n"
     "  --aspect <W:H>               display aspect (default 16:9; model canvas limits)\n"
     "  --resolution <WxH>           exact canvas instead of an aspect; both axes a\n"
     "                               multiple of 32, ratio 1:4 to 4:1. Not capped to the\n"
     "                               trained 1344x768 area — larger is allowed, warned\n"
     "                               about, and costs attention time quadratically\n"
     "  --frames <n>                 snapped up to 17k+5 (default 124, minimum 6)\n"
     "  --steps <n>                  denoising updates (inherit recipe; normally 50)\n"
     "  --audio-steps <n>            independent audio updates (experimental)\n"
     "  --lora <file>               H3 safetensors adapter (repeatable)\n"
     "  --lora-strength <n>         strength for preceding --lora (default 1)\n"
     "  --schedule <name>           default, beta, taomate-3step or dmad-4step\n"
     "  --scheduler <name>          alias for --schedule\n"
     "  --sampling-settings FILE    JSON sampler, sigma shifts and optional fixed base grid\n"
     "  --conditioning-settings FILE|JSON  reference and fixed-conditioning policy\n"
     "  --sampler <name>            euler, ab2, renoise or er_sde (default inherits)\n"
     "  --seed <n>                   noise seed; -1 or absent draws a random one\n"
     "  --count <n>                  generate n videos; explicit seeds increment by one,\n"
     "                               random ones are drawn afresh for each\n"
     "  --raw                        compatibility flag; video always writes Y4M + WAV\n"
     "  --vulkan-arithmetic portable|exact\n"
     "                               portable checks features without CUDA bit parity;\n"
     "                               exact requires a qualified device/driver\n"
     "  --inference-backend cuda|vulkan\n"
     "                               neural backend (CUDA by default when built); Vulkan\n"
     "                               supports native text and reference conditioning\n"
     "  --upscale-model <file>       optional upscaler safetensors checkpoint\n"
     "  --latent-upscale             enable H3 latent upscaling before VAE decode\n"
     "  --latent-upscale-model FILE  enable with a custom 3D conv v1 checkpoint\n"
     "  --latent-upscale-scale N     enable with scale 1..4 (default 2)\n"
     "  --latent-upscale-no-chunking process the full latent clip together\n"
     "  --upscale-method <name>      enable realesrgan or seedvr2 (CUDA/Vulkan)\n"
     "  --upscale-tile <n>           ESRGAN input tile 128 / SeedVR2 VAE tile 256; 0 untiled\n"
     "  --upscale-tile-pad <n>       tile overlap, 10 input pixels\n"
     "  --upscale-pre-pad <n>        right/bottom reflection padding, 10\n"
     "  --upscale-resolution WxH     SeedVR2 target size; default 4x generation size\n"
     "  --upscale-vae FILE           SeedVR2 VAE checkpoint\n"
     "  --upscale-segment-frames N   SeedVR2 joint frames, default 5\n"
     "  --upscale-seed N             SeedVR2 seed, default 666\n"
     "  --upscale-device N           SeedVR2 backend device, default 0\n"
     "  --upscale-no-color-match     disable SeedVR2 color correction\n"
     "  --output-accelerator cpu|vulkan\n"
     "                               RGB-to-YUV output conversion only (default cpu);\n"
     "                               independent of the inference backend\n"
     "  --dry-run                    resolve and print the plan, do not load GPU weights\n"
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
     "                               sol-experimental, exact, or veda. Vulkan neural\n"
     "                               inference accepts exact, flash2, sage2 and veda.\n"
     "                               The experimental SM120-only\n"
     "                               path is lossy and fails rather than falling back.\n"
     "  --veda-predictor <path>      VEDA predictor safetensors (required for veda)\n"
     "  --veda-sparsity <budget>     trained (default), skipped percent (90%), or kept tiles (32)\n"
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
     "  --video-decoder <f>         decoder override (Sol-H3-VAE); --vae encodes references.\n"
     "  --save-latents <f>           save a versioned AV latent archive for continuation.\n"
     "  --lcs <f>                    native H3 LCS calibration archive.\n"
     "  --lcs-settings <json>        LCS controls as inline JSON.\n"
     "  --regenerate-overlap-frames <n> release the last n overlap frames (multiples of 17);\n"
     "                               requires --lock-overlap; join replaces the overlap.\n"
     "  --continue-from <f>          extend a saved archive; --frames is NEW frames,\n"
     "                               rounded up to a multiple of 17. Output is joined.\n"
     "  --overlap-frames <n>         hidden context, 17*k+5 frames (default 22).\n"
     "  --lock-overlap               constrain continuation overlap at every step (experimental).\n"
     "  --continuation-source-noise <s> initialize new video from source; noise in (0,1], experimental.\n"
     "  --keystone-from <archive> persistent original-clip anchors for continuation (experimental).\n"
     "  --keystone-front-only    omit the rear capstone; default anchors both ends.\n"
     "  --capstone-lock-after <n> video updates before appending rear context (default 4; 0 = immediate).\n"
     "  --context-noise <s>      noise held video with a fade to the join; [0,1], default 0 (try 0.8).\n"
     "  --context-noise-fade <f> linear (default) or ease; experimental H3 continuation.\n"
     "  --context-noise-include-first also noise latent 0 unless it is a Keystone.\n"
     "  --end-anchor <image>        guide the ending toward an image (also with continuation).\n"
     "  --end-anchor-latents <f>    retain an ending archive; --frames adds a lead-in/gap.\n"
     "  --end-anchor-context <n>    locked opening context of ending, 17*k+5 (default 22).\n"
     "  --bridge-from <f>            left saved latent clip; use with --bridge-to.\n"
     "  --bridge-to <f>              right clip; --frames is added gap (rounded to 17*k+12).\n"
     "  --bridge-left-margin <n>     replace left tail, multiple of 17 (default 17).\n"
     "  --bridge-right-margin <n>    replace right head, multiple of 17 (default 17).\n"
     "  --bridge-context <n>         locked context per side, 17*k+5 frames (default 22).\n"
     "  --remake swap|edit          Viggle Remake: source video + 1..4 painted frames, 6 steps\n"
     "  --preserve-driving-audio    pin attached driving soundtrack as clean target audio;\n"
     "                             default include_reference_audio=false; explicit conflicts fail\n"
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
     "checkpoints (omitted weights are discovered locally under weights/):\n"
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
    {"inspect", "slopfab models inspect <file.safetensors> [options]", "summarise a checkpoint's tensors",
     "  --list                       print every tensor, not just a summary\n"
     "  --prefix <str>               only tensors whose name starts with <str>\n"
     "  --limit <n>                  cap listed tensors (default 40, 0 = all)\n"},
    {"compare", "slopfab models compare <reference> <actual> [options]",
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
    {"compare-y4m", "slopfab video compare <expected.y4m> <actual.y4m>",
     "byte-compare deterministic raw video outputs",
     "Reports both headers, sizes, and the first differing byte. The command\n"
     "streams its inputs and returns non-zero for any difference.\n"},
    {"decode", "slopfab video decode --vae <f> [--latent <f>] [options]", "run the video VAE decoder",
     "  --inference-backend cuda|vulkan  decoder backend\n"
     "  --vulkan-arithmetic portable|exact  device-feature or qualified arithmetic\n"
     "  --vae <f>                    video VAE checkpoint\n"
     "  --latent <f>                 H3 video archive or [24,T,H,W] latent safetensors\n"
     "                               Omit for synthetic input; archive audio is not decoded.\n"
     "  --shape <T> <H> <W>          synthetic latent shape\n"
     "  --out <f>                    .y4m output\n"
     "  --ppm <f>                    also write frame 0 as a PPM\n"
     "  --dump <f>                   raw fp32 pixels as safetensors\n"},
    {"tokenize", "slopfab conditioning tokenize [--tokenizer <f>] <text>", "encode text and round-trip it",
     "  --tokenizer <f>              override the embedded tokenizer.json\n"
     "  --qwen                      use base Qwen vocabulary without H3 special tokens\n"
     "  --pieces                     also print the pre-tokenizer split\n"},
    {"encode-refmod", "slopfab conditioning encode-refmod --reference-image <file> --output <file> [options]",
     "encode raw images, video and audio into reusable RefMods",
     "  --reference-image FILE     image input; repeatable\n"
     "  --reference-video FILE     video and optional soundtrack; repeatable\n"
     "  --reference-audio FILE     audio input; repeatable\n"
     "  --video-vae FILE           H3 video VAE with encoder weights\n"
     "  --audio-vae FILE           H3 audio VAE with encoder weights\n"
     "  --short-edge N             multiple of 32 in 32..768 (default 768)\n"
     "  --name TEXT                display name (default output stem)\n"
     "  --description TEXT         metadata only\n"
     "  --inference-backend MODE   cuda or vulkan\n"
     "  --vulkan-arithmetic MODE   portable or exact\n"
     "  --output FILE              one member: v4; multiple: mixed v5 bundle\n"
     "  Video/audio input: 2..15 seconds; use native Y4M or WAV.\n"
     "  Loads only VAEs; no text encoder or diffusion transformer.\n"},
    {"encode-text", "slopfab conditioning encode-text --prompt <text> --output <file> [options]",
     "save standalone H3 prompt conditioning without loading a transformer or VAE",
     "  --prompt-file <file>         UTF-8 prompt instead of --prompt\n"
     "  --text-encoder <file>        checkpoint (otherwise discover weights/text_encoder)\n"
     "  --clip-projection FILE      H3 4B/8B projection; student streams layers, projection runs on CPU\n"
     "  --tokenizer <file>           override embedded tokenizer\n"
     "  --inference-backend cuda|vulkan\n"
     "  --residency streaming|resident|auto  CUDA weights; default streaming\n"
     "  --arithmetic shipped|exact   CUDA arithmetic; default shipped\n"
     "  --vulkan-arithmetic portable|exact  device-feature or qualified execution\n"
     "Load the resulting file using generate --prompt-embedding. It replaces the\n"
     "whole prompt conditioning; it is not an upstream text-kind RefMod.\n"},
    {"bundle-refmods",
     "slopfab conditioning bundle-refmods --refmod <file> [--refmod <file> ...] --output <file>",
     "pack existing image/video/audio RefMods into a version-5 mixed bundle",
     "  --name <text>               display name (input bundle name or output stem)\n"
     "  --description <text>        associated text metadata; not added to prompts\n"
     "Input bundles are flattened in order; each member keeps its metadata and dtype.\n"
     "Up to 256 members. Does not encode raw media. No models or GPU are needed.\n"
     "Use generate --refmod to load the bundle; strength/copies apply to all members.\n"},
    {"mods", "slopfab mods list [--all] | load NAME",
     "Find active mods and load their instructions into agent context.",
     "  mods list                 index active mods (alias: mods index)\n"
     "  mods list --all           include disabled mods; report invalid files and conflicts\n"
     "  mods load NAME            return the full prompt and its source directory\n"
     "Create mods/NAME/MOD.md beside the config, beside the executable, or in the working directory.\n"
     "Format: YAML between --- lines, then a Markdown prompt. Required: name (folder name), description.\n"
     "Optional: enabled: true|false (default true), author, version. Names: a-z then a-z, 0-9, - or _.\n"
     "YAML supports scalar values, quotes, comments and |/> multiline text; no nested mappings.\n"
     "Loading returns instructions for the calling agent to follow; it does not execute commands.\n"
     "With --agent, list returns metadata only; load returns metadata, prompt, path and directory.\n"},
    {"settings", "slopfab settings get [KEY] | set KEY VALUE | remove KEY [PATH]",
     "Read or update global settings; changes persist across invocations.",
     "  settings get                         show all settings and the file path\n"
     "  settings get weight_paths            show configured weight paths\n"
     "  settings set output_directory PATH   set the output root used by projects\n"
     "  settings set weight_paths PATH [PATH ...]  replace the list; JSON arrays also work\n"
     "  settings remove weight_paths [PATH ...]   remove selected paths, or clear the list\n"
     "  settings set models.NAME JSON        set a model alias\n"
     "  settings set presets.NAME JSON       set a generation preset\n"
     "  settings remove models.NAME          remove a model alias\n"
     "Writable keys: weight_paths, models, presets; dotted keys access nested fields.\n"
     "Paths may name directories or checkpoint files. Relative paths are saved as absolute paths.\n"
     "Use --config FILE to select another settings file. GPU and model files are not required.\n"},
    {"capabilities", "slopfab system capabilities",
     "Discover tasks, model families, backends and native formats.",
     "No weights or GPU initialization. Next: models list, doctor, schema.\n"},
    {"schema", "slopfab requests schema", "Get the version-1 generation request JSON Schema.",
     "Pass a request with generate --request FILE or --request - for stdin.\n"},
    {"model-list", "slopfab models list", "List installed model files, configured aliases and presets.",
     "Use --config FILE or SLOPFAB_CONFIG to select config. Paths in config are relative to that file.\n"},
    {"doctor", "slopfab system doctor", "Check available GPUs and report configuration paths.",
     "No inference. Next: validate --request FILE --check-ready.\n"},
    {"validate", "slopfab requests validate --request FILE",
     "Resolve and validate a generation without loading GPU weights.",
     "--check-ready also checks required model files. Validation does not guarantee available VRAM.\nSee schema and generate --help for request fields and options.\n"},
    {"submit", "slopfab jobs submit --request FILE --idempotency-key KEY",
     "Queue a persistent generation and start a local worker.",
     "Same requests as generate. Queue capacity: 32. Retrying a key returns its existing job; changed requests fail.\nUse status JOB_ID, wait JOB_ID --timeout 60, or cancel JOB_ID. --jobs-dir selects a store.\n"},
    {"status", "slopfab jobs status JOB_ID", "Read persistent job state, result and event-file path.",
     "States: queued, running, succeeded, failed, cancelled, interrupted.\n"},
    {"wait", "slopfab jobs wait JOB_ID --timeout 60", "Wait for a terminal job state.",
     "Timeout returns exit 8 with timed_out=true. Waiting does not cancel the job.\n"},
    {"cancel", "slopfab jobs cancel JOB_ID",
     "Request cooperative cancellation of a queued or running job.",
     "A running GPU operation may finish before cancellation takes effect. Inspect status afterward.\n"},
    {"worker", "slopfab jobs worker --idle-seconds 60",
     "Process the local queue serially and reuse generation sessions.",
     "Submit starts this automatically. One worker per store. Exits after idle timeout.\nOn restart, abandoned running jobs become interrupted and are never automatically rerun.\n"},
    {"devices", "slopfab system devices", "list CUDA and Vulkan compute devices", ""},
    {"version", "slopfab system version", "print the version and exit", ""},
};

const CommandHelp* find_command(std::string_view name) {
  for (const CommandHelp& c : kCommands) {
    if (name == c.name)
      return &c;
  }
  return media_command_help(name);
}

// True when the argument list asks for help. Checked before any other option,
// so `--help` works even alongside otherwise invalid arguments.
bool option_takes_value(std::string_view a, std::string_view command) {
  if (a == "--host" || a == "--remote" || a == "--download-dir" ||
      a == "--download" || a == "--local-project") return true;
  if (a == "--word-timestamps" || a == "--no-context" || a == "--cropped")
    return false;
  if (command == "update" && a == "--check")
    return false;
  if (command == "prompts" && (a == "--layout" || a == "--instrumental" || a == "--continuation" || a == "--character-swap"))
    return false;
  if (command == "decode" && a == "--bench-load")
    return false;
  if (a.rfind("--", 0) != 0 || a.find('=') != std::string_view::npos)
    return false;
  for (const char* flag : {"--agent",
                           "--help",
                           "--dry-run",
                           "--check-ready",
                           "--overwrite",
                           "--raw",
                           "--motion-cache",
                           "--motion-cache-verbose",
                           "--synthetic-latents",
                           "--keystone-front-only",
                           "--context-noise-include-first",
                           "--lock-overlap",
                           "--preserve-driving-audio",
                           "--latent-upscale",
                           "--latent-upscale-no-chunking",
                           "--upscale-no-color-match",
                           "--no-chunking",
                           "--list",
                           "--pieces",
                           "--qwen",
                           "--verbose",
                           "--no-tiling",
                           "--h3",
                           "--all",
                           "--stop"})
    if (a == flag)
      return false;
  if (a == "--cuda-version" || a == "--config" || a == "--jobs-dir")
    return true;
  for (const auto& c : kCommands) {
    for (const char* text : {c.detail, c.usage}) {
      std::string_view help(text);
      size_t offset = 0;
      while ((offset = help.find(a, offset)) != std::string_view::npos) {
        const size_t end = offset + a.size();
        if (end == help.size() || help[end] == ' ' || help[end] == '\n' || help[end] == '|')
          return true;
        offset = end;
      }
    }
  }
  return false;
}

bool wants_help(int argc, char** argv, std::string_view command) {
  for (int i = 0; i < argc; ++i) {
    const std::string_view a = argv[i];
    if (a == "--") break;
    if (a == "--help" || a == "-h" || (i == 0 && a == "help"))
      return true;
    if (option_takes_value(a, command) && i + 1 < argc)
      ++i;
  }
  return false;
}

int print_command_help(const CommandHelp& c) {
  const std::string next =
      "Remote execution: --host NAME (or --remote NAME) uses a saved SSH host. "
      "--download-dir DIR chooses local results; --download PATH collects an extra remote output. "
      "--local-project NAME imports generation results locally. See slopfab help remote.\n"
      "Add --agent for NDJSON responses and generation progress. Use requests schema for JSON requests; system capabilities for supported tasks; system doctor for setup checks.";
  if (agent().enabled) {
    emit({{"type", str("help")},
          {"command", str(c.name)},
          {"usage", str(c.usage)},
          {"description", str(c.summary)},
          {"options", str(c.detail)},
          {"next", str(next)}},
         true);
  } else {
    std::printf("%s\nusage: %s\n\n%s\n%s\n", c.summary, c.usage, c.detail, next.c_str());
  }
  return 0;
}

void print_usage(bool list_commands, const json::Array& extra_commands) {
  const auto dependencies = list_commands ? StartupDependencies{false, false, true}
                                          : startup_dependencies(agent().executable);
  const bool has_weights = list_commands || !configured_weight_paths().empty();
  const auto text = startup_instructions(dependencies, has_weights);
  const std::string missing = missing_startup_dependency(dependencies, has_weights);
  if (agent().enabled) {
    Object response{{"type", str("help")}, {"instructions", str(text)}};
    if (!missing.empty())
      response["missing_dependency"] = str(missing);
    if (list_commands) {
      Array commands;
      for (const auto& c : kCommands)
        if (!hidden_media_alias(c.name))
        commands.push_back(
            Value(Object{{"command", str(c.name)}, {"description", str(c.summary)}}));
      for (const auto& c : extra_commands) commands.push_back(c);
      response["commands"] = Value(commands);
    }
    emit(std::move(response), true);
  } else {
    std::fputs(text.c_str(), stdout);
    if (list_commands) {
      for (const auto& c : kCommands)
        if (!hidden_media_alias(c.name))
        std::printf("  %-18s %s\n", c.name, c.summary);
      for (const auto& c : extra_commands)
        std::printf("  %-18s %s (mod)\n",c.find("command")->as_string().c_str(),c.find("description")->as_string().c_str());
    }
  }
}

#if SLOPFAB_WITH_CUDA
void consume_cuda_version_option(int& argc, char** argv) {
  std::string requested;
  int write = 1;
  for (int read = 1; read < argc; ++read) {
    const std::string_view argument = argv[read];
    if (argument == "--") {
      for (; read < argc; ++read) argv[write++] = argv[read];
      break;
    }
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
  if (!requested.empty())
    slopfab::cuda::set_cublas_version_request(requested);
}
#endif

}
