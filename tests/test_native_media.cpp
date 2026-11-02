#include "../src/cli/native_media.h"
#include "slopfab/image.h"
#include "slopfab/audio/wav.h"
#include <cmath>
#include <filesystem>
#include <iostream>
#include <chrono>
#include <stdexcept>

int main() {
  auto root=std::filesystem::temp_directory_path()/ ("slopfab-native-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  std::filesystem::create_directories(root);
  try {
    // Cross-check the standalone PNG writer with the OS/libpng decoder.
    const int w=22000,h=2; // spans multiple DEFLATE blocks per scanline
    std::vector<uint8_t> rgb(size_t(w)*h*3);
    for(size_t i=0;i<rgb.size();++i)rgb[i]=uint8_t(i*31);
    auto png=(root/"image.png").u8string();
    slopfab::cli::write_png(png,rgb.data(),w,h);
    auto image=slopfab::load_platform_image(png);
    if(image.width!=w || image.height!=h || image.pixels!=rgb)throw std::runtime_error("PNG roundtrip mismatch");
    std::vector<float> audio={-1,-.5f,0,.5f,1,0};
    for(auto format:{slopfab::audio::SampleFormat::kPcm16,slopfab::audio::SampleFormat::kFloat32}) {
      auto path=(root/"sound.wav").u8string();
      slopfab::audio::write_wav(path,audio,2,48000,format);
      auto decoded=slopfab::cli::read_wav(path);
      if(decoded.channels!=2 || decoded.sample_rate!=48000 || decoded.interleaved.size()!=audio.size())throw std::runtime_error("WAV metadata mismatch");
      for(size_t i=0;i<audio.size();++i)if(std::abs(decoded.interleaved[i]-audio[i])>0.00004f)throw std::runtime_error("WAV samples mismatch");
    }
    slopfab::PixelBuffer video(3*2*4*4,0.5f);
    auto y4m=(root/"video.y4m").u8string();
    slopfab::video::write_y4m(y4m,video,2,4,4,{24,1});
    auto decoded=slopfab::cli::read_y4m(y4m);
    if(decoded.frames.size()!=2 || decoded.width!=4 || decoded.height!=4)throw std::runtime_error("Y4M geometry mismatch");
    for(auto p:decoded.frames[0])if(std::abs(int(p)-128)>2)throw std::runtime_error("Y4M color mismatch");
    std::filesystem::resize_file(y4m,10);
    bool rejected=false;try{slopfab::cli::read_y4m(y4m);}catch(const std::exception&){rejected=true;}
    if(!rejected)throw std::runtime_error("truncated Y4M accepted");
    std::filesystem::remove_all(root);
    std::cout<<"PASS: native PNG, WAV and Y4M roundtrips and truncation\n";
    return 0;
  }catch(const std::exception& e){std::filesystem::remove_all(root);std::cerr<<e.what()<<'\n';return 1;}
}
