// Offline driver for the frame-banding quality probe.
//
// The probe asks whether this model tolerates losing distant temporal
// attention, and answers it *without a kernel*: one request is split into
// several overlapping shorter requests, each is run through the ordinary
// pipeline as its own packed sequence, and the resulting latents are
// cross-faded back together. See `include/vidfab/dit/chunking.h` for why that
// is strictly more damaging than a frame band, which is the point.
//
// **This binary links `vidfab_core` only.** It never initialises CUDA, never
// opens a checkpoint's weights, and can be run while the card is busy with
// something else — the same deliberate choice `vidfab_loadprobe` makes, and for
// the same reason. Everything here is host arithmetic over fp32 safetensors.
//
// Four subcommands:
//
//   plan    resolve and print both geometries and the chunk placement
//   slice   write chunk k's initial latents, sliced from the full noise draw
//   blend   cross-fade the chunks' denoised latents into one full-geometry pair
//   stats   compare two latent dumps: relative L2, correlation, per-channel
//           statistics, and the per-latent-frame profiles that locate a seam

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "vidfab/dit/chunking.h"
#include "vidfab/pipeline.h"
#include "vidfab/safetensors.h"
#include "vidfab/sampler/noise.h"
#include "vidfab/safetensors_write.h"
#include "vidfab/tensor_convert.h"

namespace {

struct Geometry {
  int full_frames = 45;
  int chunk_frames = 15;
  int aspect_w = 1;
  int aspect_h = 1;
  int num_chunks = 3;
  uint64_t seed = 11;
};

vidfab::GenerateRequest request_for(const Geometry& g, int frames) {
  vidfab::GenerateRequest r;
  r.prompt = "chunkprobe";  // only the length-independent geometry is used
  r.aspect_w = g.aspect_w;
  r.aspect_h = g.aspect_h;
  r.num_frames = frames;
  r.seed = g.seed;
  return r;
}

struct Resolved {
  vidfab::GeneratePlan full;
  vidfab::GeneratePlan chunk;
  vidfab::dit::ChunkPlan plan;
};

Resolved resolve(const Geometry& g) {
  Resolved out;
  out.full = vidfab::resolve_plan(request_for(g, g.full_frames));
  out.chunk = vidfab::resolve_plan(request_for(g, g.chunk_frames));
  out.plan = vidfab::dit::resolve_chunk_plan(out.full.layout, out.chunk.layout, g.num_chunks);
  return out;
}

// Parses the geometry flags every subcommand shares. Returns the index of the
// first argument it did not consume, so positional arguments still work.
void parse_geometry(int argc, char** argv, Geometry* g, std::vector<std::string>* positional,
                    std::string* out_path, int* index) {
  for (int i = 0; i < argc; ++i) {
    const std::string_view arg = argv[i];
    auto next = [&](const char* what) -> const char* {
      if (i + 1 >= argc) throw std::runtime_error(std::string(what) + " needs a value");
      return argv[++i];
    };
    if (arg == "--frames") {
      g->full_frames = std::atoi(next("--frames"));
    } else if (arg == "--chunk-frames") {
      g->chunk_frames = std::atoi(next("--chunk-frames"));
    } else if (arg == "--chunks") {
      g->num_chunks = std::atoi(next("--chunks"));
    } else if (arg == "--seed") {
      g->seed = std::strtoull(next("--seed"), nullptr, 10);
    } else if (arg == "--index") {
      if (index == nullptr) throw std::runtime_error("--index is not valid here");
      *index = std::atoi(next("--index"));
    } else if (arg == "--out") {
      if (out_path == nullptr) throw std::runtime_error("--out is not valid here");
      *out_path = next("--out");
    } else if (arg == "--aspect") {
      const std::string v = next("--aspect");
      const size_t colon = v.find(':');
      if (colon == std::string::npos) throw std::runtime_error("--aspect wants W:H, e.g. 1:1");
      g->aspect_w = std::atoi(v.substr(0, colon).c_str());
      g->aspect_h = std::atoi(v.substr(colon + 1).c_str());
    } else if (!arg.empty() && arg.front() == '-') {
      throw std::runtime_error("unrecognised option '" + std::string(arg) + "'");
    } else if (positional != nullptr) {
      positional->emplace_back(arg);
    } else {
      throw std::runtime_error("unexpected argument '" + std::string(arg) + "'");
    }
  }
}

void print_plan(const Geometry& g, const Resolved& r) {
  const vidfab::dit::SequenceLayout& fl = r.full.layout;
  const vidfab::dit::SequenceLayout& cl = r.chunk.layout;
  std::printf("full      --frames %d -> %d aligned, %dx%d px, %d latent frames of %d rows\n",
              g.full_frames, r.full.aligned_frames, r.full.canvas_width, r.full.canvas_height,
              fl.num_latent_frames, fl.rows_per_frame());
  std::printf("          video rows %d, audio latents %d (%d rows), media rows %d\n",
              fl.num_video_rows, fl.num_audio_latents, fl.num_audio_rows, fl.total_rows());
  std::printf("chunk     --frames %d -> %d aligned, %d latent frames, video rows %d, "
              "audio latents %d\n",
              g.chunk_frames, r.chunk.aligned_frames, cl.num_latent_frames, cl.num_video_rows,
              cl.num_audio_latents);
  std::printf("chunks    %d, stride %d latent frames, overlap %d latent frames "
              "(%d audio latents)\n",
              r.plan.num_chunks, r.plan.latent_stride, r.plan.frame_overlap, r.plan.audio_overlap);
  for (int k = 0; k < r.plan.num_chunks; ++k) {
    const int f0 = r.plan.frame_offset[static_cast<size_t>(k)];
    const int a0 = r.plan.audio_offset[static_cast<size_t>(k)];
    std::printf("  chunk %d  latent frames [%2d, %2d)   audio latents [%2d, %2d)\n", k, f0,
                f0 + cl.num_latent_frames, a0, a0 + cl.num_audio_latents);
  }
  std::printf("audio     worst offset drift %.3f latents (%.1f ms at 40 latents/s)\n",
              r.plan.worst_audio_drift_latents, r.plan.worst_audio_drift_latents * 25.0);
  std::printf("seed      %llu, sliced from the full-geometry draw\n",
              static_cast<unsigned long long>(g.seed));
}

// --- statistics --------------------------------------------------------------

struct Moments {
  double mean = 0.0;
  double std_dev = 0.0;
};

Moments moments(const float* v, size_t n, size_t stride = 1) {
  double sum = 0.0;
  size_t count = 0;
  for (size_t i = 0; i < n; i += stride) {
    sum += v[i];
    ++count;
  }
  const double mean = count != 0 ? sum / static_cast<double>(count) : 0.0;
  double var = 0.0;
  for (size_t i = 0; i < n; i += stride) {
    const double d = v[i] - mean;
    var += d * d;
  }
  Moments m;
  m.mean = mean;
  m.std_dev = count != 0 ? std::sqrt(var / static_cast<double>(count)) : 0.0;
  return m;
}

// rel_L2 and correlation now come from `vidfab::compare` rather than from a
// copy here. This tool grew its own pair, as did the AB2 sampler sweep and the
// banding work, which is what put them in the shared header — see
// `include/vidfab/tensor_convert.h` for the exact formulas and the reasons for
// those choices rather than the neighbouring plausible ones.
double relative_l2(const std::vector<float>& a, const std::vector<float>& b) {
  return vidfab::compare(a, b).rel_l2;
}

std::vector<float> read_rows(const vidfab::SafeTensors& st, const char* name, int64_t width) {
  const vidfab::TensorView& v = st.at(name);
  if (v.shape.size() != 2 || v.shape[1] != width) {
    throw std::runtime_error(std::string(name) + " must be [rows, " + std::to_string(width) + "]");
  }
  return vidfab::to_f32(v);
}

void report_pair(const char* label, const std::vector<float>& ref, const std::vector<float>& act) {
  if (ref.size() != act.size()) {
    throw std::runtime_error(std::string(label) + ": the two dumps differ in length");
  }
  const Moments mr = moments(ref.data(), ref.size());
  const Moments ma = moments(act.data(), act.size());
  // One call to the shared implementation for all three of rel_L2, correlation
  // and max|diff|. The last is reported because rel_L2 cannot distinguish
  // "identical" from "identical to four decimals", and one of the runs this
  // tool drives is a byte-for-byte control.
  const vidfab::CompareStats s = vidfab::compare(ref, act);
  double dot = 0.0, ref_sq = 0.0, act_sq = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    if (!std::isfinite(ref[i]) || !std::isfinite(act[i])) continue;
    dot += static_cast<double>(ref[i]) * act[i];
    ref_sq += static_cast<double>(ref[i]) * ref[i];
    act_sq += static_cast<double>(act[i]) * act[i];
  }
  const double cosine = ref_sq > 0.0 && act_sq > 0.0
                            ? dot / std::sqrt(ref_sq * act_sq)
                            : 0.0;
  std::printf("%-6s rel_L2 %.4f   cosine %.4f   correlation %.4f   mean %+.4f vs %+.4f   std %.4f vs %.4f   "
              "max|diff| %.3e\n",
              label, s.rel_l2, cosine, s.correlation, mr.mean, ma.mean, mr.std_dev, ma.std_dev,
              s.max_abs_err);
}

// Norm of the difference between consecutive latent frames, normalised by the
// mean over the sequence. A hard cut in the content shows here as a spike at
// the boundary index; a clean stitch does not. This is the numeric form of
// "look at the seams".
std::vector<double> temporal_step_profile(const std::vector<float>& rows, int frames,
                                          int slice_width) {
  std::vector<double> out;
  for (int f = 0; f + 1 < frames; ++f) {
    const float* a = rows.data() + static_cast<size_t>(f) * slice_width;
    const float* b = rows.data() + static_cast<size_t>(f + 1) * slice_width;
    double acc = 0.0;
    for (int i = 0; i < slice_width; ++i) {
      const double d = static_cast<double>(b[i]) - a[i];
      acc += d * d;
    }
    out.push_back(std::sqrt(acc / slice_width));
  }
  double mean = 0.0;
  for (double v : out) mean += v;
  if (!out.empty()) mean /= static_cast<double>(out.size());
  if (mean > 0.0) {
    for (double& v : out) v /= mean;
  }
  return out;
}

// --- subcommands -------------------------------------------------------------

int cmd_plan(int argc, char** argv) {
  Geometry g;
  parse_geometry(argc, argv, &g, nullptr, nullptr, nullptr);
  print_plan(g, resolve(g));
  return 0;
}

int cmd_slice(int argc, char** argv) {
  Geometry g;
  std::string out_path;
  int index = -1;
  parse_geometry(argc, argv, &g, nullptr, &out_path, &index);
  if (out_path.empty()) throw std::runtime_error("slice needs --out <file.safetensors>");
  if (index < 0) throw std::runtime_error("slice needs --index <k>");

  const Resolved r = resolve(g);
  std::vector<float> video;
  std::vector<float> audio;
  vidfab::dit::slice_chunk_noise(g.seed, r.full.layout, r.chunk.layout, r.plan, index, &video,
                                 &audio);
  vidfab::write_safetensors(
      out_path, {{"video_rows", {r.chunk.layout.num_video_rows, 96}, video},
                 {"audio_rows", {r.chunk.layout.num_audio_rows, 32}, audio}});
  std::printf("chunk %d  latent frames [%d, %d)  audio latents [%d, %d)  -> %s\n", index,
              r.plan.frame_offset[static_cast<size_t>(index)],
              r.plan.frame_offset[static_cast<size_t>(index)] + r.chunk.layout.num_latent_frames,
              r.plan.audio_offset[static_cast<size_t>(index)],
              r.plan.audio_offset[static_cast<size_t>(index)] + r.chunk.layout.num_audio_latents,
              out_path.c_str());
  return 0;
}

// The full request's own seeded draw, in the same dump format. Two uses: it is
// the field the slices are taken from, so blending the three slices back must
// reproduce it exactly — the cheapest end-to-end check that the placement
// arithmetic and the file plumbing agree — and it can be fed to
// `generate --init-latents` to confirm the flag is a no-op on the seed's own
// draw.
int cmd_noise(int argc, char** argv) {
  Geometry g;
  std::string out_path;
  parse_geometry(argc, argv, &g, nullptr, &out_path, nullptr);
  if (out_path.empty()) throw std::runtime_error("noise needs --out <file.safetensors>");

  const Resolved r = resolve(g);
  const vidfab::dit::SequenceLayout& fl = r.full.layout;
  const std::vector<float> field =
      vidfab::sampler::video_noise(g.seed, fl.num_latent_frames, fl.latent_height, fl.latent_width,
                                   24);
  std::vector<float> video(static_cast<size_t>(fl.num_video_rows) * 96);
  vidfab::dit::patchify_video(field.data(), fl, video.data());
  const std::vector<float> audio = vidfab::sampler::audio_noise(g.seed, fl.num_audio_latents, 32);

  vidfab::write_safetensors(out_path, {{"video_rows", {fl.num_video_rows, 96}, video},
                                       {"audio_rows", {fl.num_audio_rows, 32}, audio}});
  std::printf("noise     seed %llu at the full geometry -> %s\n",
              static_cast<unsigned long long>(g.seed), out_path.c_str());
  return 0;
}

int cmd_blend(int argc, char** argv) {
  Geometry g;
  std::string out_path;
  std::vector<std::string> inputs;
  parse_geometry(argc, argv, &g, &inputs, &out_path, nullptr);
  if (out_path.empty()) throw std::runtime_error("blend needs --out <file.safetensors>");

  const Resolved r = resolve(g);
  if (static_cast<int>(inputs.size()) != r.plan.num_chunks) {
    throw std::runtime_error("blend needs " + std::to_string(r.plan.num_chunks) +
                             " latent dumps, got " + std::to_string(inputs.size()));
  }

  std::vector<std::vector<float>> video;
  std::vector<std::vector<float>> audio;
  for (const std::string& path : inputs) {
    vidfab::SafeTensors st;
    st.open(path);
    video.push_back(read_rows(st, "video_rows", 96));
    audio.push_back(read_rows(st, "audio_rows", 32));
  }

  std::vector<float> video_out;
  std::vector<float> audio_out;
  vidfab::dit::blend_chunks(r.full.layout, r.chunk.layout, r.plan, video, audio, &video_out,
                            &audio_out);
  vidfab::write_safetensors(
      out_path, {{"video_rows", {r.full.layout.num_video_rows, 96}, video_out},
                 {"audio_rows", {r.full.layout.num_audio_rows, 32}, audio_out}});
  const Moments mv = moments(video_out.data(), video_out.size());
  const Moments ma = moments(audio_out.data(), audio_out.size());
  std::printf("blended   %d chunks -> %d video rows, %d audio rows -> %s\n", r.plan.num_chunks,
              r.full.layout.num_video_rows, r.full.layout.num_audio_rows, out_path.c_str());
  std::printf("latents   video mean %+.4f std %.4f   audio mean %+.4f std %.4f\n", mv.mean,
              mv.std_dev, ma.mean, ma.std_dev);
  return 0;
}

int cmd_stats(int argc, char** argv) {
  Geometry g;
  std::vector<std::string> inputs;
  parse_geometry(argc, argv, &g, &inputs, nullptr, nullptr);
  if (inputs.size() != 2) throw std::runtime_error("stats needs a reference and an actual dump");

  const Resolved r = resolve(g);
  const vidfab::dit::SequenceLayout& fl = r.full.layout;

  vidfab::SafeTensors ref_file;
  vidfab::SafeTensors act_file;
  ref_file.open(inputs[0]);
  act_file.open(inputs[1]);
  const std::vector<float> ref_v = read_rows(ref_file, "video_rows", 96);
  const std::vector<float> act_v = read_rows(act_file, "video_rows", 96);
  const std::vector<float> ref_a = read_rows(ref_file, "audio_rows", 32);
  const std::vector<float> act_a = read_rows(act_file, "audio_rows", 32);

  std::printf("reference %s\nactual    %s\n", inputs[0].c_str(), inputs[1].c_str());
  report_pair("video", ref_v, act_v);
  report_pair("audio", ref_a, act_a);

  // Per-channel over the 24 latent channels. `feature = c*4 + dh*2 + dw`, so a
  // channel is four strided columns of the row buffer (spec 1.4). The nvfp4
  // section's reading of "a different sample rather than a degraded one" is
  // exactly this table matching while the correlation does not.
  std::printf("\nper video latent channel (24):\n");
  std::printf("  ch    ref mean   act mean    ref std    act std     rel_L2\n");
  for (int c = 0; c < 24; ++c) {
    std::vector<float> rc;
    std::vector<float> ac;
    rc.reserve(ref_v.size() / 24);
    ac.reserve(act_v.size() / 24);
    for (size_t row = 0; row * 96 < ref_v.size(); ++row) {
      for (int k = 0; k < 4; ++k) {
        rc.push_back(ref_v[row * 96 + static_cast<size_t>(c) * 4 + k]);
        ac.push_back(act_v[row * 96 + static_cast<size_t>(c) * 4 + k]);
      }
    }
    const Moments mr = moments(rc.data(), rc.size());
    const Moments ma = moments(ac.data(), ac.size());
    std::printf("  %2d   %+9.4f  %+9.4f  %9.4f  %9.4f  %9.4f\n", c, mr.mean, ma.mean, mr.std_dev,
                ma.std_dev, relative_l2(rc, ac));
  }

  // The seam profile. Video rows are frame-major, so consecutive latent frames
  // are consecutive runs of R*96 floats.
  const int R = fl.rows_per_frame();
  if (static_cast<size_t>(fl.num_latent_frames) * R * 96 == ref_v.size()) {
    const std::vector<double> pr = temporal_step_profile(ref_v, fl.num_latent_frames, R * 96);
    const std::vector<double> pa = temporal_step_profile(act_v, fl.num_latent_frames, R * 96);
    std::printf("\nframe-to-frame step, normalised to each run's own mean:\n");
    std::printf("  f->f+1     reference   actual   seam?\n");
    for (size_t i = 0; i < pr.size(); ++i) {
      bool seam = false;
      for (int k = 1; k < r.plan.num_chunks; ++k) {
        const int start = r.plan.frame_offset[static_cast<size_t>(k)];
        // The overlap the k-th chunk shares with its predecessor spans latent
        // frames [start, start + overlap); the steps that cross it are the
        // ones a hard cut would show up in.
        if (static_cast<int>(i) >= start - 1 && static_cast<int>(i) < start + r.plan.frame_overlap) {
          seam = true;
        }
      }
      std::printf("  %2zu->%2zu     %8.4f  %8.4f   %s\n", i, i + 1, pr[i], pa[i],
                  seam ? "<-- overlap" : "");
    }
  }
  return 0;
}

void print_usage() {
  std::printf(
      "vidfab_chunkprobe - frame-banding quality probe (host only, no CUDA)\n"
      "\n"
      "usage: vidfab_chunkprobe <plan|noise|slice|blend|stats> [options]\n"
      "\n"
      "  plan                            print both geometries and the chunk placement\n"
      "  noise  --out <f>                the full request's own seeded draw, so that\n"
      "                                  blending the slices back can be checked\n"
      "  slice  --index <k> --out <f>    chunk k's initial latents, sliced from the\n"
      "                                  full request's own noise draw\n"
      "  blend  --out <f> <a> <b> <c>    cross-fade the chunks' denoised latents\n"
      "  stats  <reference> <actual>     relative L2, correlation, per-channel\n"
      "                                  statistics and the per-frame seam profile\n"
      "\n"
      "geometry (shared by every subcommand; the defaults are the probe's own):\n"
      "  --frames <n>        the full request, snapped up to 17k+5 (default 45 -> 56)\n"
      "  --chunk-frames <n>  one chunk, snapped the same way   (default 15 -> 22)\n"
      "  --chunks <n>        how many chunks                   (default 3)\n"
      "  --aspect <W:H>      display aspect                    (default 1:1)\n"
      "  --seed <n>          noise seed                        (default 11)\n"
      "\n"
      "This binary links vidfab_core only: it initialises no CUDA context and reads\n"
      "no checkpoint, so it is safe to run while the card is busy.\n");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    print_usage();
    return 2;
  }
  const std::string_view command = argv[1];
  if (command == "--help" || command == "-h" || command == "help") {
    print_usage();
    return 0;
  }
  try {
    if (command == "plan") return cmd_plan(argc - 2, argv + 2);
    if (command == "noise") return cmd_noise(argc - 2, argv + 2);
    if (command == "slice") return cmd_slice(argc - 2, argv + 2);
    if (command == "blend") return cmd_blend(argc - 2, argv + 2);
    if (command == "stats") return cmd_stats(argc - 2, argv + 2);
    std::fprintf(stderr, "vidfab_chunkprobe: unknown command '%s'\n\n", argv[1]);
    print_usage();
    return 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "vidfab_chunkprobe: %s\n", e.what());
    return 1;
  }
}
