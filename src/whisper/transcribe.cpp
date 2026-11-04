#include "internal.h"
#include "slopfab/text/tokenizer.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

namespace slopfab::whisper {
namespace {
std::string trim(std::string s) {
  auto begin=s.find_first_not_of(" \t\r\n"),end=s.find_last_not_of(" \t\r\n");
  return begin==std::string::npos?"":s.substr(begin,end-begin+1);
}
double logsum(const std::vector<float>& x,int begin,int end) {
  double mx=-INFINITY;for(int i=begin;i<end;++i)mx=std::max(mx,double(x[i]));
  if(!std::isfinite(mx))return mx;
  double sum=0;for(int i=begin;i<end;++i)sum+=std::exp(x[i]-mx);
  return mx+std::log(sum);
}
void rules(std::vector<float>& logits,const Tokenizer& t,const std::vector<int>& sampled) {
  for(int id:t.suppress)logits[id]=-INFINITY;
  // Language/control IDs are never transcript text.
  for(int id=t.eot+1;id<t.timestamp_begin;++id)logits[id]=-INFINITY;
  if(sampled.empty()) {
    std::fill(logits.begin(),logits.begin()+t.timestamp_begin,-INFINITY);
    std::fill(logits.begin()+t.timestamp_begin+51,logits.end(),-INFINITY);
  } else {
    bool last=sampled.back()>=t.timestamp_begin;
    bool before=sampled.size()<2 || sampled[sampled.size()-2]>=t.timestamp_begin;
    if(last && before)std::fill(logits.begin()+t.timestamp_begin,logits.end(),-INFINITY);
    else if(last)std::fill(logits.begin(),logits.begin()+t.eot,-INFINITY);
    auto found=std::find_if(sampled.rbegin(),sampled.rend(),[&](int v){return v>=t.timestamp_begin;});
    if(found!=sampled.rend()) {
      int lower=*found+int(!(last && !before));
      std::fill(logits.begin()+t.timestamp_begin,logits.begin()+std::min(lower,int(logits.size())),-INFINITY);
    }
  }
  if(logsum(logits,t.timestamp_begin,int(logits.size()))>*std::max_element(logits.begin(),logits.begin()+t.timestamp_begin))
    std::fill(logits.begin(),logits.begin()+t.timestamp_begin,-INFINITY);
}
struct Hypothesis {
  std::vector<int> tokens;
  std::vector<float> logits;
  std::vector<Cache> cache;
  double score=0;
  bool ended=false;
};
struct Decoded {std::vector<int> tokens;double logprob=0;bool complete=false;};
Decoded decode_window(Model& m,const Tokenizer& t,const std::vector<int>& prefix,int beams,float temperature) {
  m.reset();
  Hypothesis first;first.logits=m.decode(prefix);first.cache=m.self_cache;
  std::vector<Hypothesis> active{std::move(first)};
  std::mt19937 random(0);
  const int limit=std::min(224,448-int(prefix.size()));
  for(int step=0;step<limit;++step) {
    std::vector<Hypothesis> next;
    for(auto& h:active) {
      if(h.ended){next.push_back(std::move(h));continue;}
      auto logits=h.logits;rules(logits,t,h.tokens);
      if(temperature>0)for(auto& v:logits)v/=temperature;
      double denominator=logsum(logits,0,int(logits.size()));
      if(!std::isfinite(denominator))throw std::runtime_error("Whisper: decoder has no finite candidate");
      std::vector<int> candidates;
      if(temperature>0) {
        std::vector<double> probs(logits.size());
        for(size_t i=0;i<logits.size();++i)probs[i]=std::exp(logits[i]-denominator);
        std::discrete_distribution<int> dist(probs.begin(),probs.end());candidates.push_back(dist(random));
      } else {
        std::vector<int> indices(logits.size());std::iota(indices.begin(),indices.end(),0);
        std::partial_sort(indices.begin(),indices.begin()+beams,indices.end(),[&](int a,int b){return logits[a]>logits[b];});
        candidates.assign(indices.begin(),indices.begin()+beams);
      }
      for(int id:candidates) if(std::isfinite(logits[id])) {
        Hypothesis v;v.tokens=h.tokens;v.tokens.push_back(id);v.score=h.score+logits[id]-denominator;
        v.ended=id==t.eot;v.cache=h.cache;next.push_back(std::move(v));
      }
    }
    std::sort(next.begin(),next.end(),[](const auto& a,const auto& b){return a.score>b.score;});
    if(next.size()>size_t(beams))next.resize(beams);
    active=std::move(next);
    if(std::all_of(active.begin(),active.end(),[](const auto& h){return h.ended;}))break;
    for(auto& h:active) if(!h.ended) {
      m.self_cache=h.cache;h.logits=m.decode({h.tokens.back()});h.cache=m.self_cache;
    }
  }
  auto best=std::max_element(active.begin(),active.end(),[](const auto& a,const auto& b){return a.score/std::max(size_t(1),a.tokens.size())<b.score/std::max(size_t(1),b.tokens.size());});
  Decoded out{best->tokens,best->score/std::max(size_t(1),best->tokens.size()),best->ended};
  if(!out.tokens.empty() && out.tokens.back()==t.eot)out.tokens.pop_back();
  return out;
}
bool looping(const std::vector<int>& tokens) {
  if(tokens.size()<24)return false;
  for(size_t period=1;period<=8;++period) {
    bool repeated=true;
    for(size_t i=tokens.size()-period*3;i<tokens.size()-period;++i)
      repeated &= tokens[i]==tokens[i+period];
    if(repeated)return true;
  }
  return false;
}
bool complete_utf8(const std::string& s) {
  size_t at=0;
  while(at<s.size()) {
    unsigned c=uint8_t(s[at]);size_t n=c<128?1:(c&0xe0)==0xc0?2:(c&0xf0)==0xe0?3:(c&0xf8)==0xf0?4:0;
    if(!n || at+n>s.size())return false;
    for(size_t j=1;j<n;++j)if((uint8_t(s[at+j])&0xc0)!=0x80)return false;
    at+=n;
  }
  return true;
}
}

std::vector<Word> align_words(Model& m,const Tokenizer& tokenizer,const std::vector<int>& prefix,
                             const std::vector<int>& text,double offset,double duration) {
  if(text.empty())return {};
  std::vector<int> ids=prefix;ids.push_back(tokenizer.no_timestamps);
  size_t text_begin=ids.size();ids.insert(ids.end(),text.begin(),text.end());ids.push_back(tokenizer.eot);
  if(ids.size()>448)throw std::runtime_error("Whisper: alignment context exceeded");
  std::vector<std::vector<float>> heads;m.reset();m.decode(ids,&heads);
  int frames=std::clamp(int(std::ceil(duration*50)),1,1500);
  // Include no-timestamps as the leading alignment row, as in upstream DTW.
  int rows=int(text.size())+1;
  std::vector<double> matrix(size_t(rows)*frames);
  for(auto& h:heads) {
    for(size_t row=0;row<ids.size();++row) {
      double sum=0;for(int f=0;f<frames;++f)sum+=h[row*1500+f];
      for(int f=0;f<frames;++f)h[row*1500+f]=float(h[row*1500+f]/std::max(sum,1e-30));
    }
    for(int f=0;f<frames;++f) {
      double mean=0,var=0;for(size_t r=0;r<ids.size();++r)mean+=h[r*1500+f];mean/=ids.size();
      for(size_t r=0;r<ids.size();++r){double d=h[r*1500+f]-mean;var+=d*d;}
      double inv=1/std::sqrt(std::max(var/ids.size(),1e-20));
      for(size_t r=0;r<ids.size();++r)h[r*1500+f]=float((h[r*1500+f]-mean)*inv);
    }
    for(int r=0;r<rows;++r)for(int f=0;f<frames;++f) {
      float values[7];
      for(int j=-3;j<=3;++j) {
        int at=f+j;
        while(at<0 || at>=frames) {if(at<0)at=-at;if(at>=frames)at=2*(frames-1)-at;if(frames==1){at=0;break;}}
        values[j+3]=h[(text_begin-1+r)*1500+at];
      }
      std::nth_element(values,values+3,values+7);matrix[size_t(r)*frames+f]+=values[3]/double(heads.size());
    }
  }
  std::vector<double> cost(size_t(rows+1)*(frames+1),INFINITY);std::vector<uint8_t> trace(cost.size());cost[0]=0;
  for(int r=1;r<=rows;++r)for(int f=1;f<=frames;++f) {
    size_t at=size_t(r)*(frames+1)+f;
    double diagonal=cost[at-frames-2],up=cost[at-frames-1],left=cost[at-1];
    uint8_t direction=diagonal<up && diagonal<left?0:up<diagonal && up<left?1:2;
    cost[at]=(direction==0?diagonal:direction==1?up:left)-matrix[size_t(r-1)*frames+f-1];trace[at]=direction;
  }
  std::vector<int> starts(rows,frames-1);int r=rows,f=frames;
  while(r>0 && f>0) {
    starts[r-1]=std::min(starts[r-1],f-1);auto direction=trace[size_t(r)*(frames+1)+f];
    if(direction!=2)--r;if(direction!=1)--f;
  }
  std::vector<Word> words;std::string bytes;size_t begin=0;
  bool separate=false;
  for(const char* language:{"zh","ja","th","lo","my","yue"})
    if(prefix.size()>1 && tokenizer.language_code(prefix[1])==language)separate=true;
  for(size_t i=0;i<text.size();++i) {
    bytes+=tokenizer.decode({text[i]});if(!complete_utf8(bytes))continue;
    double start=offset+starts[begin]*0.02,end=offset+starts[i+1]*0.02;
    if(words.empty() || separate || (!bytes.empty() && bytes[0]==' ')) words.push_back({start,end,bytes});
    else {words.back().text+=bytes;words.back().end=end;}
    begin=i+1;bytes.clear();
  }
  return words;
}

Result transcribe(const std::vector<float>& pcm,int rate,int channels,const Options& options) {
  if(options.model.empty())throw std::invalid_argument("Whisper: model weights are required");
  if(options.beam_size<1 || options.beam_size>10)throw std::invalid_argument("Whisper: beam size must be 1..10");
  double progress=0;
  auto check=[&](const std::string& stage){if(options.progress && !options.progress(progress,stage))throw std::runtime_error("Whisper: cancelled");};
  check("load");Model model(options);model.check=check;Tokenizer tokenizer(model.config);
  auto audio=mono_16k(pcm,rate,channels);check("features");auto mel=log_mel(audio,model.config.mels);
  Result result;result.model=model.config.name;result.backend=model.backend_name;result.duration=audio.size()/16000.0;
  int language=tokenizer.language_id(options.language=="auto"?"en":options.language);
  if(!model.config.multilingual && language!=tokenizer.language_id("en"))throw std::invalid_argument("Whisper: English-only checkpoint");
  bool detect=options.language=="auto" && model.config.multilingual;
  std::vector<int> history,initial=tokenizer.encode(options.prompt);
  const size_t content_frames=(audio.size()+159)/160;
  size_t seek=0;
  while(seek<content_frames) {
    progress=double(seek)/content_frames;check("encode");
    size_t count=std::min<size_t>(3000,content_frames-seek);
    std::vector<float> window(3000*model.config.mels);
    std::copy_n(mel.begin()+seek*model.config.mels,window.size(),window.begin());model.encode(window);
    auto start_logits=model.decode({tokenizer.sot});
    double no_speech=std::exp(start_logits[tokenizer.no_speech]-logsum(start_logits,0,int(start_logits.size())));
    if(detect) {
      auto first=start_logits.begin()+tokenizer.sot+1;
      language=tokenizer.sot+1+int(std::max_element(first,first+tokenizer.language_count)-first);detect=false;
    }
    result.language=tokenizer.language_code(language);
    std::vector<int> prefix{tokenizer.sot};
    if(model.config.multilingual){prefix.push_back(language);prefix.push_back(tokenizer.transcribe);}
    std::vector<int> previous=initial;
    if(options.condition_on_previous_text)previous.insert(previous.end(),history.begin(),history.end());
    if(previous.size()>220)previous.erase(previous.begin(),previous.end()-220);
    auto prompted=prefix;
    if(!previous.empty()){prompted={tokenizer.previous};prompted.insert(prompted.end(),previous.begin(),previous.end());prompted.insert(prompted.end(),prefix.begin(),prefix.end());}
    Decoded decoded;
    bool fallback=false;
    for(float temp:{0.f,0.2f,0.4f,0.6f,0.8f,1.f}) {
      decoded=decode_window(model,tokenizer,prompted,temp==0?options.beam_size:1,temp);
      if(no_speech>0.6 && decoded.logprob<-1)break;
      if(decoded.logprob>=-1 && decoded.complete && !looping(decoded.tokens))break;
      fallback=true;
      // A fresh context helps break repetition without deleting real choruses.
      prompted=prefix;
    }
    if(no_speech>0.6 && decoded.logprob<-1){seek+=count;history.clear();continue;}
    const auto& tokens=decoded.tokens;
    std::vector<size_t> boundaries;
    for(size_t i=1;i<tokens.size();++i)if(tokens[i-1]>=tokenizer.timestamp_begin && tokens[i]>=tokenizer.timestamp_begin)boundaries.push_back(i);
    bool single=tokens.size()>=2 && tokens.back()>=tokenizer.timestamp_begin && tokens[tokens.size()-2]<tokenizer.timestamp_begin;
    size_t advance=count,begin=0;
    std::vector<int> accepted;
    auto add_segment=[&](size_t end,double start,double finish) {
      std::vector<int> text;
      for(size_t i=begin;i<end;++i)if(tokens[i]<tokenizer.eot)text.push_back(tokens[i]);
      const double offset=seek/100.0;
      Segment segment;segment.start=std::clamp(offset+start,0.0,result.duration);segment.end=std::clamp(offset+finish,segment.start,result.duration);
      segment.text=trim(tokenizer.decode(text));segment.average_logprob=decoded.logprob;segment.no_speech_probability=no_speech;
      if(!segment.text.empty() && segment.end>segment.start) {
        if(options.word_timestamps) {
          // Align against the complete audio window, then bound to this segment.
          segment.words=align_words(model,tokenizer,prefix,text,offset,count/100.0);
          for(auto& word:segment.words){word.start=std::clamp(word.start,segment.start,segment.end);word.end=std::clamp(word.end,word.start,segment.end);}
        }
        if(!result.text.empty())result.text+=' ';result.text+=segment.text;result.segments.push_back(std::move(segment));
        accepted.insert(accepted.end(),text.begin(),text.end());
      }
      begin=end;
    };
    if(!boundaries.empty()) {
      if(single)boundaries.push_back(tokens.size());
      for(size_t end:boundaries) {
        double start=tokens[begin]>=tokenizer.timestamp_begin?(tokens[begin]-tokenizer.timestamp_begin)*0.02:0;
        double finish=(tokens[end-1]-tokenizer.timestamp_begin)*0.02;
        add_segment(end,start,finish);
      }
      if(!single)advance=std::clamp<size_t>(size_t(std::max(1,tokens[begin-1]-tokenizer.timestamp_begin))*2,1,count);
    } else {
      double end=count/100.0;
      for(auto it=tokens.rbegin();it!=tokens.rend();++it)if(*it>tokenizer.timestamp_begin){end=(*it-tokenizer.timestamp_begin)*0.02;break;}
      add_segment(tokens.size(),0,end);
    }
    if(fallback)history.clear();else history.insert(history.end(),accepted.begin(),accepted.end());
    if(history.size()>220)history.erase(history.begin(),history.end()-220);
    seek+=advance;
  }
  progress=1;check("complete");return result;
}

std::string subtitles(const Result& result,bool webvtt) {
  auto time=[&](double seconds) {
    int64_t ms=int64_t(std::llround(seconds*1000));std::ostringstream s;
    s<<std::setfill('0')<<std::setw(2)<<ms/3600000<<":"<<std::setw(2)<<(ms/60000)%60<<":"<<std::setw(2)<<(ms/1000)%60<<(webvtt?'.':',')<<std::setw(3)<<ms%1000;return s.str();
  };
  std::ostringstream out;if(webvtt)out<<"WEBVTT\n\n";int n=1;
  for(const auto& segment:result.segments) {
    if(!webvtt)out<<n++<<"\n";
    out<<time(segment.start)<<" --> "<<time(segment.end)<<"\n"<<segment.text<<"\n\n";
  }
  return out.str();
}
}
