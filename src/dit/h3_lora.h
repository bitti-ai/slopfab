#pragma once
#include "slopfab/lora.h"
#include "../core/lora_grid.h"
#include <memory>
namespace slopfab::detail::h3 {
std::string projection_name(std::string name);
bool split_qkv(const std::string& name);
bool adaln_target(const std::string& name);
bool supported(const std::string& name);
struct AdaLNBasis {
  int rows = 0, columns = 0, full = 0;
  std::vector<float> grid, table;
  std::vector<double> map;
  std::unique_ptr<detail::LoraGrid> source;
  void load(const SafeTensors& base, const SafeTensors& adapter, int width);
  double project(LoraFactors& factors, std::vector<float>& bias) const;
};
}
