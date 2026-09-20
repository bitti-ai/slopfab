#pragma once
#include <string>
#include <string_view>
#include <vector>
#include <cstdint>
#include "slopfab/pipeline.h"
namespace slopfab::cli {
inline constexpr const char* kVersion = "0.1.0";
struct CommandHelp {
  const char* name;
  const char* usage;
  const char* summary;
  const char* detail;
};


const CommandHelp* find_command(std::string_view);
bool wants_help(int argc, char** argv);
int print_command_help(const CommandHelp&);
void print_usage();
void consume_cuda_version_option(int&, char**);
std::string format_shape(const std::vector<int64_t>&);
std::string format_bytes(uint64_t);
uint64_t random_seed();
std::string timestamped_output_path();
std::string counted_output_path(const std::string&, int, int);
void discover_generate_checkpoints(GenerateRequest&, const char*);
void ensure_generate_models(GenerateRequest&, const char*, bool);
int cmd_inspect(int, char**);
int cmd_compare(int, char**);
int cmd_compare_y4m(int, char**);
int cmd_decode(int, char**);
int cmd_tokenize(int, char**);
int cmd_generate(int, char**, const char*);
int cmd_devices();
int cmd_prepare_lora(int, char**);
}
