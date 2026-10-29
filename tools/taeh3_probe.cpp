#include "slopfab/vae/taeh3.h"
#include "slopfab/tensor_convert.h"
#include <iostream>
#include <cmath>
#include <chrono>
int main(int argc, char** argv) {
  try {
    if (argc != 4) throw std::invalid_argument("usage: taeh3_probe checkpoint reference.safetensors cuda|vulkan");
    slopfab::SafeTensors reference;
    reference.open(argv[2]);
    const auto& input = reference.at("latents");
    if (input.shape.size() != 4 || input.shape[0] != 24) throw std::invalid_argument("expected CTHW input");
    slopfab::vae::TaeH3 decoder(argv[1],std::string(argv[3]) == "cuda" ? slopfab::DeviceBackend::kCuda : slopfab::DeviceBackend::kVulkan);
    const auto start = std::chrono::steady_clock::now();
    const auto output = decoder.decode(slopfab::to_f32(input),int(input.shape[1]),int(input.shape[2]),int(input.shape[3]));
    const auto expected = slopfab::to_f32(reference.at("rgb"));
    if (expected.size() != output.size()) throw std::runtime_error("output shape mismatch");
    float worst = 0; double mean = 0;
    for (size_t i=0;i<output.size();++i) {
      const float error = std::abs(float(output[i])-expected[i]);
      worst = std::max(worst,error); mean += error;
    }
    std::cout << "frames=" << input.shape[1] << " max_rgb8_error=" << worst << " mean_rgb8_error=" << mean/output.size()
              << " seconds=" << std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count() << '\n';
    if (worst > 1 || mean/output.size() > .05) return 1;
    if (!decoder.decode(slopfab::to_f32(input),int(input.shape[1]),int(input.shape[2]),int(input.shape[3]),[]{return true;}).empty()) return 1;
    // Memory must be reset between steps.
    if (decoder.decode(slopfab::to_f32(input),int(input.shape[1]),int(input.shape[2]),int(input.shape[3])) != output) return 1;
    return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
