#include "slopfab/music.h"
#include "slopfab/text/tokenizer.h"
#include "slopfab/safetensors.h"
#include "slopfab/json.h"
#include <algorithm>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace slopfab::music {
std::vector<int32_t> minimax_prompt(const SafeTensors& checkpoint, const std::string& caption, const std::string& lyrics) {
  const auto& blob = checkpoint.at("tokenizer_json");
  text::Tokenizer tokenizer;
  tokenizer.load_json(std::string_view(static_cast<const char*>(blob.data), blob.nbytes));
  for (const auto& token : std::vector<std::pair<std::string, int>>{
      {"<|im_start|>",151644}, {"<|im_end|>",151645}, {"<|audio_cfg|>",151654},
      {"<|audio_start|>",151669}, {"<|audio_end|>",151670}, {"<|caption_start|>",151671},
      {"<|caption_end|>",151672}, {"<|lyrics_start|>",151673}, {"<|lyrics_end|>",151674}})
    if (tokenizer.token_to_id(token.first) != token.second)
      throw std::invalid_argument("MiniMax Music3 tokenizer mismatch: " + token.first);
  std::string clean;
  const std::regex special(R"(<\|([^|]*)\|>)");
  size_t last=0;
  for(std::sregex_iterator it(caption.begin(),caption.end(),special),end;it!=end;++it) {
    clean+=caption.substr(last,size_t(it->position())-last);
    std::istringstream words((*it)[1].str());std::string first,rest;words>>first;std::getline(words>>std::ws,rest);
    clean+=first+(rest.empty()?"":" is "+rest);last=size_t(it->position()+it->length());
  }
  clean+=caption.substr(last);
  std::istringstream lines(clean);clean.clear();std::string line;
  while(std::getline(lines,line)) {
    line=std::regex_replace(line,std::regex(R"(^\s{0,3}#{1,6}\s+)"),"");
    line=std::regex_replace(line,std::regex(R"(^\s*[*+-]\s+)"),"");
    line=std::regex_replace(line,std::regex(R"(\*\*([^*]+)\*\*)"),"$1");
    line=std::regex_replace(line,std::regex(R"(\*([^*\n]+)\*)"),"$1");
    auto end=line.find_last_not_of(" \t\r");line=end==std::string::npos?"":line.substr(0,end+1);
    if(std::regex_match(line,std::regex(R"(\s*[-*_]{3,}\s*)")))line.clear();
    if(!clean.empty())clean+='\n';clean+=line;
  }
  clean=std::regex_replace(clean,std::regex("\n{2,}"),"\n");
  clean=std::regex_replace(clean,std::regex("    "),"");
  clean=std::regex_replace(clean,std::regex("\xe2\x80\xa2 "),"");
  std::string normalized="[start]\n";
  const std::regex tags(R"(\s*(\[[^\]]+\])\s*)");last=0;
  std::vector<std::string> parts;
  for(std::sregex_iterator it(lyrics.begin(),lyrics.end(),tags),end;it!=end;++it) {
    auto part=lyrics.substr(last,size_t(it->position())-last);if(!part.empty())parts.push_back(part);
    part=(*it)[1].str();for(char& c:part)if(c>='A'&&c<='Z')c+=32;parts.push_back(part);
    last=size_t(it->position()+it->length());
  }
  if(last<lyrics.size())parts.push_back(lyrics.substr(last));
  for(size_t i=0;i<parts.size();++i){if(i)normalized+='\n';normalized+=parts[i];}
  normalized=std::regex_replace(normalized,std::regex(" \\^ "),"\n");
  std::vector<int32_t> ids={151644,151671};
  auto add=[&](const std::string& text){auto more=tokenizer.encode(text,false);ids.insert(ids.end(),more.begin(),more.end());};
  add(clean);ids.push_back(151672);ids.push_back(151673);add(normalized);
  ids.insert(ids.end(),{151674,151645,151669});
  if(ids.size()>5000)throw std::invalid_argument("MiniMax Music3 prompt exceeds 5000 tokens");
  return ids;
}
void load_yue_tokenizer(text::Tokenizer& tokenizer, const SafeTensors& checkpoint) {
  auto tokenizer_blob = checkpoint.at("text_encoders.yue2_tokenizer_json");
  // The bundle exporter double-escaped its Qwen pattern. Restore the upstream
  // pattern before validating the tokenizer; vocab and merges stay untouched.
  std::string source(static_cast<const char*>(tokenizer_blob.data), tokenizer_blob.nbytes);
  auto root = json::parse(source);
  auto object = root.as_object();
  auto pre = object.at("pre_tokenizer").as_object();
  auto seq = pre.at("pretokenizers").as_array();
  auto split = seq[0].as_object();
  auto pat = split.at("pattern").as_object();
  auto regex = pat.at("Regex").as_string();
  for (size_t at = 0; (at = regex.find("\\\\", at)) != std::string::npos; ++at)
    regex.erase(at, 1);
  pat["Regex"] = json::Value(regex);
  split["pattern"] = json::Value(pat);
  seq[0] = json::Value(split);
  pre["pretokenizers"] = json::Value(seq);
  object["pre_tokenizer"] = json::Value(pre);
  tokenizer.load_json(json::stringify(json::Value(object)));
}
}
