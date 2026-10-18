#pragma once
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cmath>
namespace eeg {
constexpr size_t Samples=11000, MaxWire=11200;
enum Type:uint8_t {Hello=1,Ready=2,Infer=3,Accepted=4,Result=5,Error=6,Status=7,Log=8};
enum Code:uint32_t {BadRequest=1,BadSession=2,OldRequest=3,Conflict=4,Inference=5,Output=6,RxTimeout=7,UartError=8};
inline uint32_t crc(const uint8_t*p,size_t n){uint32_t c=~0u;while(n--){c^=*p++;for(int i=0;i<8;i++)c=(c>>1)^((c&1)?0xedb88320u:0);}return ~c;}
inline uint32_t get32(const uint8_t*p){return uint32_t(p[0])|uint32_t(p[1])<<8|uint32_t(p[2])<<16|uint32_t(p[3])<<24;}
inline void put32(uint8_t*p,uint32_t n){for(int i=0;i<4;i++)p[i]=uint8_t(n>>(8*i));}
inline uint64_t get64(const uint8_t*p){return uint64_t(get32(p))|uint64_t(get32(p+4))<<32;}
inline void put64(uint8_t*p,uint64_t n){put32(p,uint32_t(n));put32(p+4,uint32_t(n>>32));}
inline size_t encode(const uint8_t*src,size_t n,uint8_t*out,size_t cap){
 if(cap<n+n/254+2)return 0;
 size_t codeAt=0,w=1;uint8_t code=1;
 for(size_t i=0;i<n;i++){if(src[i]==0){out[codeAt]=code;codeAt=w++;code=1;}else{out[w++]=src[i];if(++code==255){out[codeAt]=code;codeAt=w++;code=1;}}}out[codeAt]=code;out[w++]=0;return w;
}
// COBS in-place decoding only writes behind its unread input.
inline size_t decode(uint8_t*buf,size_t n){size_t r=0,w=0;while(r<n){uint8_t c=buf[r++];if(!c||r+size_t(c)-1>n)return 0;for(unsigned j=1;j<c;j++){if(!buf[r])return 0;buf[w++]=buf[r++];}if(c!=255&&r<n)buf[w++]=0;}return w;}
struct Frame{uint8_t type;uint64_t session;uint32_t request,trial;const uint8_t*payload;uint16_t length;};
inline bool parse(uint8_t*b,size_t n,Frame&f){
 if(n<36||std::memcmp(b,"EEG2",4)||b[4]!=1||b[6]||b[7]||get32(b+24)||b[30]||b[31])return false;
 uint16_t len=uint16_t(b[28])|uint16_t(b[29])<<8;
 if(len>Samples||n!=size_t(36)+len||crc(b,n-4)!=get32(b+n-4))return false;
 f={b[5],get64(b+8),get32(b+16),get32(b+20),b+32,len};return true;
}
inline size_t pack(const Frame&f,uint8_t*out,size_t cap){
 // Responses are bounded to256 payload bytes. Full requests are host/test encoded.
 if(f.length>256)return 0;uint8_t raw[292]={};std::memcpy(raw,"EEG2",4);raw[4]=1;raw[5]=f.type;put64(raw+8,f.session);put32(raw+16,f.request);put32(raw+20,f.trial);raw[28]=uint8_t(f.length);raw[29]=uint8_t(f.length>>8);if(f.length)std::memcpy(raw+32,f.payload,f.length);put32(raw+32+f.length,crc(raw,32+f.length));return encode(raw,36+f.length,out,cap);
}
class Receiver {
 uint8_t wire_[MaxWire];size_t used_=0;bool dropping_=false;uint32_t started_=0,last_=0;
 public:
 unsigned rejected=0,timeouts=0;
 void invalidate(){used_=0;dropping_=true;rejected++;}
 void tick(uint32_t now){if(used_&&(uint32_t(now-last_)>=1000||uint32_t(now-started_)>=5000)){invalidate();timeouts++;}}
 template<class Callback>void feed(uint8_t c,uint32_t now,Callback cb){
  tick(now);if(c==0){if(dropping_){dropping_=false;used_=0;return;}if(used_){size_t n=decode(wire_,used_);Frame f{};if(n&&parse(wire_,n,f))cb(f);else rejected++;}used_=0;return;}
  if(dropping_)return;if(!used_)started_=now;last_=now;if(used_==MaxWire){invalidate();return;}wire_[used_++]=c;
 }
};
using Forward=bool(*)(void*,int8_t*,float[2]);
using Send=void(*)(void*,const uint8_t*,size_t);
class Server {
 int8_t* input_;Forward forward_;Send send_;void* context_;uint64_t session_=0;uint32_t last_=0,trial_=0,checksum_=0;bool hasCached_=false;uint8_t cachedType_=Result,cached_[16]={};uint16_t cachedLen_=0;
 void emit(uint8_t type,uint32_t req,uint32_t trial,const uint8_t*p,uint16_t n){uint8_t out[320];size_t size=pack({type,session_,req,trial,p,n},out,sizeof(out));if(size)send_(context_,out,size);}
 void error(const Frame&f,Code c){uint8_t b[4];put32(b,c);emit(Error,f.request,f.trial,b,4);}
 void ready(){uint8_t p[69]={};const char*hex="644bf8865ed515ed41c5418070a0b147008c4326b9da34d3232884459d9b3298";auto nib=[](char c){return c>='a'?c-'a'+10:c-'0';};for(int i=0;i<32;i++)p[i]=uint8_t(nib(hex[2*i])*16+nib(hex[2*i+1]));float scale=0.04422667250037193f;uint32_t bits;std::memcpy(&bits,&scale,4);put32(p+32,bits);put32(p+36,uint32_t(-10));put32(p+40,Samples);const uint16_t dims[4]={1,1,22,500};for(int i=0;i<4;i++){p[44+2*i]=uint8_t(dims[i]);p[45+2*i]=uint8_t(dims[i]>>8);}p[52]=1;std::memcpy(p+53,"eeg-stream-v1.0",15);emit(Ready,0,~0u,p,sizeof(p));}
 public:
 unsigned invocations=0;
 Server(int8_t*input,Forward f,Send s,void*ctx):input_(input),forward_(f),send_(s),context_(ctx){}
 void handle(const Frame&f){
  if(f.type==Hello){if(f.length||f.request||f.trial!=~0u||!f.session)return;if(session_!=f.session){session_=f.session;last_=0;hasCached_=false;}ready();return;}
  if(!session_||f.session!=session_){error(f,BadSession);return;}
  if(f.type==Status){if(!f.length&&hasCached_&&f.request==last_&&f.trial==trial_)emit(cachedType_,last_,trial_,cached_,cachedLen_);else error(f,OldRequest);return;}
  if(f.type!=Infer||f.length!=Samples||f.trial>=144||!f.request){error(f,BadRequest);return;}
  uint32_t sum=crc(f.payload,f.length);
  if(f.request<=last_){if(hasCached_&&f.request==last_&&f.trial==trial_&&sum==checksum_)emit(cachedType_,last_,trial_,cached_,cachedLen_);else error(f,f.request==last_?Conflict:OldRequest);return;}
  last_=f.request;trial_=f.trial;checksum_=sum;hasCached_=true;
  std::memcpy(input_,f.payload,Samples);uint8_t ack[4];put32(ack,sum);emit(Accepted,last_,trial_,ack,4);
  float logits[2];invocations++;
  if(!forward_(context_,input_,logits)){cachedType_=Error;cachedLen_=4;put32(cached_,Inference);}
  else if(!std::isfinite(logits[0])||!std::isfinite(logits[1])){cachedType_=Error;cachedLen_=4;put32(cached_,Output);}
  else{cachedType_=Result;cachedLen_=16;put32(cached_,sum);uint32_t bits;std::memcpy(&bits,logits,4);put32(cached_+4,bits);std::memcpy(&bits,logits+1,4);put32(cached_+8,bits);cached_[12]=logits[1]>logits[0]?1:0;cached_[13]=cached_[14]=cached_[15]=0;}
  emit(cachedType_,last_,trial_,cached_,cachedLen_);
 }
};
}
