#include "internal.h"
#include "slopfab/text/tokenizer.h"
#include <algorithm>
#include <sstream>
#include <set>
#include <stdexcept>

namespace slopfab::whisper {
namespace {
std::string unbase64(const std::string& s) {
  const std::string alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string result;
  unsigned acc=0; int bits=0;
  for(char c:s) {
    if(c=='=') break;
    auto n=alphabet.find(c);
    if(n==std::string::npos) throw std::runtime_error("Whisper: invalid embedded tokenizer");
    acc=(acc<<6)|unsigned(n); bits+=6;
    if(bits>=8) { bits-=8; result.push_back(char((acc>>bits)&255)); }
  }
  return result;
}
// Whisper's GPT-2 regex: case-sensitive contractions, optional ASCII space,
// runs of Unicode letters/numbers/symbols, then whitespace with lookahead.
std::vector<std::string> pieces(const std::string& input) {
  std::vector<std::string> out;
  size_t i=0;
  auto kind=[](uint32_t c) {return text::is_letter(c)?1:text::is_number(c)?2:text::is_space(c)?0:3;};
  while(i<input.size()) {
    bool contraction=false;
    for(const char* c:{"'s","'t","'re","'ve","'m","'ll","'d"}) {
      std::string s(c);
      if(input.compare(i,s.size(),s)==0) {out.push_back(s); i+=s.size(); contraction=true; break;}
    }
    if(contraction) continue;
    size_t first=i, probe=i;
    auto cp=text::utf8_next(input,probe);
    if(cp==' ' && probe<input.size()) {
      size_t next=probe; auto c=text::utf8_next(input,next);
      if(kind(c)!=0) {i=probe; cp=c; probe=next;}
    }
    int category=kind(cp);
    i=probe;
    size_t previous=i;
    while(i<input.size()) {
      size_t next=i; auto c=text::utf8_next(input,next);
      if(kind(c)!=category) break;
      previous=i; i=next;
    }
    if(category==0 && i<input.size() && previous>first) i=previous;
    out.emplace_back(input,first,i-first);
  }
  return out;
}
}
Tokenizer::Tokenizer(const Config& c) {
  std::istringstream input(embedded_asset(c.multilingual?"multilingual.tiktoken":"gpt2.tiktoken"));
  std::string b64; int rank;
  while(input>>b64>>rank) {
    std::string token=unbase64(b64);
    if(rank!=int(vocabulary.size())) throw std::runtime_error("Whisper: tokenizer rank order");
    ranks.emplace(token,rank); vocabulary.push_back(token);
  }
  eot=int(vocabulary.size()); sot=eot+1;
  language_count=c.vocab-51765-int(c.multilingual);
  translate=sot+1+language_count; transcribe=translate+1;
  previous=translate+3; no_speech=translate+4; no_timestamps=translate+5;
  timestamp_begin=translate+6;
  std::set<int> blocked;
  const std::string symbols="\"#()*+/:;<=>@[\\]^_`{|}~";
  std::vector<std::string> strings;
  for(char ch:symbols) strings.push_back(std::string(1,ch));
  for(const char* s:{"<<",">>","<<<",">>>","--","---","-(","-[","('","(\"","((","))","(((",")))","[[","]]","{{","}}","♪♪","♪♪♪"}) strings.emplace_back(s);
  for(const auto& s:strings) for(const std::string& prefix:{std::string(),std::string(" ")}) {
    auto ids=encode(prefix+s);
    if(ids.size()==1) blocked.insert(ids.front());
  }
  for(const auto& s:pieces("♩♪♫♬♭♮♯")) {
    // Musical symbols may share a multi-byte prefix; suppress that prefix too.
    size_t at=0;
    while(at<s.size()) {size_t start=at; text::utf8_next(s,at); auto ids=encode(s.substr(start,at-start)); if(!ids.empty()) blocked.insert(ids[0]);}
  }
  for(const char* s:{" -"," '"}) {auto ids=encode(s); if(!ids.empty()) blocked.insert(ids[0]);}
  for(int id:{sot,translate,transcribe,previous,translate+2,no_speech}) blocked.insert(id);
  suppress.assign(blocked.begin(),blocked.end());
}
std::vector<int> Tokenizer::encode(const std::string& s) const {
  std::vector<int> out;
  for(const auto& piece:pieces(s)) {
    if(auto it=ranks.find(piece);it!=ranks.end()) {out.push_back(it->second);continue;}
    std::vector<std::string> parts;
    for(char c:piece) parts.emplace_back(1,c);
    while(parts.size()>1) {
      int best=INT32_MAX; size_t where=0;
      for(size_t i=0;i+1<parts.size();++i) {
        auto it=ranks.find(parts[i]+parts[i+1]);
        if(it!=ranks.end() && it->second<best) {best=it->second;where=i;}
      }
      if(best==INT32_MAX) break;
      parts[where]+=parts[where+1]; parts.erase(parts.begin()+where+1);
    }
    for(const auto& part:parts) out.push_back(ranks.at(part));
  }
  return out;
}
std::string Tokenizer::decode(const std::vector<int>& ids) const {
  std::string out;
  for(int id:ids) if(id>=0 && id<eot) out+=vocabulary[size_t(id)];
  return out;
}
int Tokenizer::language_id(const std::string& name) const {
  const auto& languages=assets().find("languages")->as_array();
  for(int i=0;i<language_count;++i) {
    const auto& v=languages[size_t(i)].as_array();
    if(name==v[0].as_string() || name==v[1].as_string()) return sot+1+i;
  }
  throw std::invalid_argument("Whisper: unsupported language: "+name);
}
std::string Tokenizer::language_code(int id) const {
  if(id<sot+1 || id>=sot+1+language_count) throw std::invalid_argument("Whisper: language ID");
  return assets().find("languages")->as_array()[size_t(id-sot-1)].as_array()[0].as_string();
}
}
