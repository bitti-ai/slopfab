#pragma once
#include "slopfab/lora.h"
#include "../core/lora_grid.h"
#include <memory>
namespace slopfab::detail::h3 {
std::string projection_name(std::string name);
bool split_qkv(const std::string& name);
bool adaln_target(const std::string& name);
bool supported(const std::string& name);
// Returns whether full-width modulation factors need table-basis fitting.
bool validate_target(const SafeTensors& base, const std::string& name, const LoraFactors& factors);
bool needs_output_basis_conversion(const std::string& key);
void convert_output_basis(LoraFactors& factors, bool needed);
struct AdaLNBasis {
  int rows = 0, columns = 0, full = 0;
  std::vector<float> grid, table;
  std::vector<double> map;
  std::unique_ptr<detail::LoraGrid> source;
  void load(const SafeTensors& base, const SafeTensors& adapter, int width);
  double project(LoraFactors& factors, std::vector<float>& bias) const;
};
}
