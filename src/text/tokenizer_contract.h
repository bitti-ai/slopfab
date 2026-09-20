#pragma once
#include <string>
#include <string_view>
namespace slopfab::text::detail {
void validate_tokenizer_component(const std::string& name, std::string_view value);
void validate_tokenizer_model_field(const std::string& name, std::string_view value);
}
