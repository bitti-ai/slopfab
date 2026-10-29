#pragma once
#include "slopfab/generate.h"
#include "slopfab/vae/taeh3.h"
#include <cstdio>
#include <algorithm>

namespace slopfab {
// One decoder per denoising phase; released before the full VAE is loaded.
class PreviewDecoder {
  const RunOptions& options_;
  dit::SequenceLayout layout_;
  double duration_;
  bool failed_ = false;
  std::unique_ptr<vae::TaeH3> decoder_;
public:
  PreviewDecoder(const RunOptions& options, const dit::SequenceLayout& layout, double duration)
      : options_(options), layout_(layout), duration_(duration) {}
  void operator()(int step, const std::vector<float>& rows) noexcept {
    try {
      if (failed_ || (options_.preview_cancelled && options_.preview_cancelled())) return;
      const int t = layout_.num_latent_frames, sh = layout_.latent_height, sw = layout_.latent_width;
      if (sh < 1 || sw < 1 || t < 1 || duration_ <= 0) return;
      const float scale = std::min(1.f,128.f/std::max(sh,sw));
      const int h = std::max(1,int(sh*scale)), w = std::max(1,int(sw*scale));
      std::vector<float> latents(rows.size());
      dit::unpatchify_video(rows.data(), layout_, latents.data());
      // Bound preview size for very long generations, retaining the complete span.
      const int frames = std::min(t,127);
      if (frames != t || h != sh || w != sw) {
        std::vector<float> sampled(size_t(24)*frames*h*w);
        for (int c = 0; c < 24; ++c)
          for (int f = 0; f < frames; ++f)
            for (int y = 0; y < h; ++y)
              for (int x = 0; x < w; ++x) {
                const int source_t = frames == 1 ? 0 : f*(t-1)/(frames-1);
                sampled[((size_t(c)*frames+f)*h+y)*w+x] =
                    latents[((size_t(c)*t+source_t)*sh+y*sh/h)*sw+x*sw/w];
              }
        latents = std::move(sampled);
      }
      if (!decoder_) decoder_ = std::make_unique<vae::TaeH3>(options_.preview_vae_path,options_.inference_backend);
      GenerationPreview preview;
      preview.frames = frames; preview.width = w*2; preview.height = h*2;
      preview.fps = frames/duration_; preview.step = step;
      preview.rgb = decoder_->decode(latents,frames,h,w,options_.preview_cancelled);
      if (!preview.rgb.empty()) options_.on_preview(preview);
    } catch (const std::exception& e) {
      std::fprintf(stderr,"TAEH3 preview disabled: %s\n",e.what());
      failed_ = true;
      decoder_.reset();
    } catch (...) { failed_ = true; decoder_.reset(); }
  }
};
}
