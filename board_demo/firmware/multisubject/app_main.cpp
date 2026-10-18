// Copyright 2026 Arm Limited and/or its affiliates.
// SPDX-License-Identifier: Apache-2.0
// Isolated v4 runner: unchanged Moamen PTE, with INT8 wire -> float32 input adapter.
#include <array>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <cstring>
#include <cmath>
#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/platform/runtime.h>
#include "arm_embedded_module.hpp"
#include "model_pte.h"
#include "protocol.hpp"
using namespace executorch::aten;
using executorch::extension::BufferDataLoader;
using executorch::runtime::MemoryAllocator;
#ifndef APP_METHOD_POOL_SIZE
#define APP_METHOD_POOL_SIZE (4*1024*1024)
#endif
#ifndef APP_TEMP_POOL_SIZE
#define APP_TEMP_POOL_SIZE (4*1024*1024)
#endif
#ifdef APP_POOL_SECTION
#define POOL __attribute__((section(APP_POOL_SECTION)))
#else
#define POOL
#endif
namespace {
alignas(16)uint8_t method_pool[APP_METHOD_POOL_SIZE] POOL;
alignas(16)uint8_t temp_pool[APP_TEMP_POOL_SIZE] POOL;
alignas(16)int8_t input_data[eeg::Samples];
alignas(16)float model_input[eeg::Samples];
struct Context{arm::embedded::EmbeddedModule*module;Tensor*input;unsigned results=0,errors=0;};
bool infer(void*ctx,int8_t*wire_input,float logits[2]){
 for(size_t i=0;i<eeg::Samples;i++)model_input[i]=eeg::dequantize_input(wire_input[i]);
 auto*c=(Context*)ctx;auto outputs=c->module->forward(*c->input);
 if(!outputs.ok()||outputs->size()!=1||!outputs->front().isTensor())return false;
 Tensor t=outputs->front().toTensor();
 if(t.scalar_type()!=ScalarType::Float||t.dim()!=2||t.size(0)!=1||t.size(1)!=2)return false;
 const float*p=t.const_data_ptr<float>();if(!std::isfinite(p[0])||!std::isfinite(p[1]))return false;
 logits[0]=p[0];logits[1]=p[1];return true;
}
#ifdef EEG_STREAM_FVP_TEST
#include "test_frames.h"
void transmit(void*ctx,const uint8_t*p,size_t n){
 auto*c=(Context*)ctx;uint8_t decoded[320];
 if(n>0&&n<=sizeof(decoded)){std::memcpy(decoded,p,n-1);size_t size=eeg::decode(decoded,n-1);eeg::Frame f{};if(size&&eeg::parse(decoded,size,f)){if(f.type==eeg::Result){c->results++;
 uint32_t bits0=eeg::get32(f.payload+4),bits1=eeg::get32(f.payload+8);float z0,z1;
 std::memcpy(&z0,&bits0,4);std::memcpy(&z1,&bits1,4);
 printf("STREAM_RESULT request=%lu trial=%lu input_crc=%08lx pred=%u logits=%.9g,%.9g bits=%08lx,%08lx\n",static_cast<unsigned long>(f.request),static_cast<unsigned long>(f.trial),static_cast<unsigned long>(eeg::get32(f.payload)),unsigned(f.payload[12]),double(z0),double(z1),static_cast<unsigned long>(bits0),static_cast<unsigned long>(bits1));
 }if(f.type==eeg::Error)c->errors++;}}
 printf("STREAM_FRAME ");for(size_t i=0;i<n;i++)printf("%02x",p[i]);printf("\n");fflush(stdout);
}
#else
extern "C" uint32_t stream_millis();
extern "C" int stream_read_byte();
extern "C" int stream_uart_fault();
extern "C" bool stream_write(const uint8_t*,size_t);
void transmit(void*,const uint8_t*p,size_t n){(void)stream_write(p,n);}
#endif
}
extern "C" int app_main(){
 executorch::runtime::runtime_init();
 arm::embedded::EmbeddedModule module(model_pte,model_pte_size,std::make_unique<BufferDataLoader>(model_pte,model_pte_size),std::make_unique<MemoryAllocator>(sizeof(method_pool),method_pool),std::make_unique<MemoryAllocator>(sizeof(temp_pool),temp_pool));
 std::array<SizesType,4> sizes{1,1,22,256};std::array<DimOrderType,4> order{0,1,2,3};TensorImpl impl(ScalarType::Float,sizes.size(),sizes.data(),model_input,order.data());Tensor input(&impl);
 Context context{&module,&input};static eeg::Receiver receiver;eeg::Server server(input_data,infer,transmit,&context);
#ifdef EEG_STREAM_FVP_TEST
 for(size_t i=0;i<sizeof(test_frames);i++)receiver.feed(test_frames[i],uint32_t(i/100),[&](const eeg::Frame&f){server.handle(f);});
 const bool ok=context.results==expected_results&&context.errors==0&&server.invocations==expected_results&&receiver.rejected==expected_rejected;
 printf("STREAM_TEST %s results=%u calls=%u rejected=%u\n",ok?"PASS":"FAIL",context.results,server.invocations,receiver.rejected);fflush(stdout);return ok?0:1;
#else
 for(;;){uint32_t now=stream_millis();if(stream_uart_fault())receiver.invalidate();int byte=stream_read_byte();if(byte>=0)receiver.feed(uint8_t(byte),now,[&](const eeg::Frame&f){server.handle(f);});else receiver.tick(now);}
#endif
}
