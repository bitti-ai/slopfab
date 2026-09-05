#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

#include "slopfab/safetensors.h"
#include "slopfab/text/encoder.h"
#include "slopfab/text/qwen_vision.h"
#include "slopfab/text/tokenizer.h"

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: slopfab_qwenconditionerprobe <qwen.safetensors> <tokenizer.json>\n");
    return 2;
  }
  try {
    slopfab::text::Tokenizer tokenizer; tokenizer.load(argv[2]);
    std::vector<uint8_t> rgb(256 * 256 * 3);
    for (int y=0;y<256;++y) for(int x=0;x<256;++x) {
      size_t i=(static_cast<size_t>(y)*256+x)*3;
      rgb[i]=static_cast<uint8_t>((17*x+3*y)&255);
      rgb[i+1]=static_cast<uint8_t>((5*x+11*y)&255);
      rgb[i+2]=static_cast<uint8_t>((x^y)&255);
    }
    auto pixels=slopfab::text::qwen3vl_patchify_resized_rgb(rgb,256,256);
    auto label=tokenizer.encode("<Picture 1>: ");
    auto ids=slopfab::text::qwen3vl_image_block(label,pixels.grid.merged_token_count());
    auto prompt=tokenizer.encode("A calm camera observes the reference image.");
    ids.insert(ids.end(),prompt.begin(),prompt.end());

    slopfab::SafeTensors checkpoint; checkpoint.open(argv[1]);
    slopfab::text::EncoderConfig cfg; cfg.residency=slopfab::text::Residency::kStreaming;
    slopfab::text::Encoder encoder;
    auto t0=std::chrono::steady_clock::now(); encoder.load(checkpoint,cfg);
    auto t1=std::chrono::steady_clock::now(); auto out=encoder.encode(ids,{pixels});
    auto t2=std::chrono::steady_clock::now();
    if(out.num_tokens!=static_cast<int>(ids.size())||out.hidden_size!=5120||
       out.data.size()!=ids.size()*5120ull||out.modality_tags.size()!=ids.size())
      throw std::runtime_error("unexpected multimodal conditioner shape");
    const size_t vision_start=label.size(), vision_end=vision_start+65;
    for(size_t i=0;i<out.modality_tags.size();++i) {
      const int want=(i>=vision_start&&i<=vision_end)?0:1;
      if(out.modality_tags[i]!=want) throw std::runtime_error("incorrect conditioner modality tag");
    }
    long double sq=0; for(float v:out.data){if(!std::isfinite(v))
      throw std::runtime_error("non-finite conditioner output"); sq+=static_cast<long double>(v)*v;}
    const double value=std::sqrt(static_cast<double>(sq/out.data.size()));
    if(!(value>0)) throw std::runtime_error("degenerate conditioner output");
    std::printf("conditioner [%d,5120] rms %.9g\n",out.num_tokens,value);
    std::printf("load %.3f s  multimodal encode %.3f s  tags text/video verified\n",
      std::chrono::duration<double>(t1-t0).count(),std::chrono::duration<double>(t2-t1).count());
    return 0;
  } catch(const std::exception& e) {
    std::fprintf(stderr,"qwenconditionerprobe: %s\n",e.what()); return 1;
  }
}
