#pragma once
#include "slopfab/image.h"
#include <string>
#include <vector>

namespace slopfab::fibo {
int gist_count(int tokens);
std::string edit_prompt(const std::string& prompt, int images,
                        std::vector<std::string>& rendered_text);
RGBImage crop_image(const RGBImage& image, int width, int height);
}
