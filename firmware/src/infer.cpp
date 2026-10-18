// Copyright 2026 Arm Limited and/or its affiliates.
// SPDX-License-Identifier: Apache-2.0
//
// ExecuTorch runner for TinyEEGNet, from Emulator_Deploy's src/app_main.cpp harness: program
// and method loaded once into two bump-allocator pools, one forward() per window, NPU cycles
// from the Ethos-U PMU (driver inference callbacks), CPU cycles from the DWT.

#include "infer.h"

#include <array>
#include <cstdio>
#include <memory>
#include <new>

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/platform/runtime.h>

#include "RTE_Components.h"
#include CMSIS_device_header

#if defined(ARM_MODEL_USE_PMU_COUNTERS)
#include "ethosu_driver.h"
#include "pmu_ethosu.h"
#endif

#include "arm_embedded_module.hpp"
#include "clock.h"
#include "model_pte.h"
#include "params_gen.h"

using arm::embedded::EmbeddedModule;
using executorch::aten::DimOrderType;
using executorch::aten::ScalarType;
using executorch::aten::SizesType;
using executorch::aten::Tensor;
using executorch::aten::TensorImpl;
using executorch::extension::BufferDataLoader;
using executorch::runtime::EValue;
using executorch::runtime::MemoryAllocator;

// Pool sizes and placement are board-overridable (board/DevKit-E8/Board-U85.clayer.yml puts
// them in bulk SRAM via APP_POOL_SECTION). The v2 model needs 28,812 B + 98,688 B
// (Emulator_Deploy results/fvp_v2_metrics.md).
#ifndef APP_METHOD_POOL_SIZE
#define APP_METHOD_POOL_SIZE (256 * 1024)
#endif
#ifndef APP_TEMP_POOL_SIZE
#define APP_TEMP_POOL_SIZE (512 * 1024)  // Ethos-U scratch is drawn from here.
#endif
#ifdef APP_POOL_SECTION
#define APP_POOL_ATTRIBUTES __attribute__((section(APP_POOL_SECTION)))
#else
#define APP_POOL_ATTRIBUTES
#endif

namespace {

constexpr size_t kMethodPoolSize = APP_METHOD_POOL_SIZE;
constexpr size_t kTempPoolSize = APP_TEMP_POOL_SIZE;

alignas(16) uint8_t g_method_pool[kMethodPoolSize] APP_POOL_ATTRIBUTES;
alignas(16) uint8_t g_temp_pool[kTempPoolSize] APP_POOL_ATTRIBUTES;

// Bump allocator that remembers its high-water mark (the runtime resets the temp allocator
// after every delegate call, so used_size() afterwards says nothing).
class PeakAllocator : public MemoryAllocator {
 public:
  PeakAllocator(uint32_t size, uint8_t* base) : MemoryAllocator(size, base) {}

  void* allocate(size_t size, size_t alignment = kDefaultAlignment) override {
    void* p = MemoryAllocator::allocate(size, alignment);
    if (used_size() > peak_) {
      peak_ = used_size();
    }
    return p;
  }

  size_t peak() const {
    return peak_;
  }

 private:
  size_t peak_ = 0;
};

EmbeddedModule* g_module = nullptr;
const PeakAllocator* g_method_alloc = nullptr;
const PeakAllocator* g_temp_alloc = nullptr;

alignas(16) float g_input[N_CH * N_SAMPLES];

uint32_t g_npu_cycles = 0;
uint32_t g_npu_active = 0;

}  // namespace

#if defined(ARM_MODEL_USE_PMU_COUNTERS)
extern "C" void ethosu_inference_begin(struct ethosu_driver* drv, void*) {
  ETHOSU_PMU_Enable(drv);
  ETHOSU_PMU_Set_EVTYPER(drv, 0, ETHOSU_PMU_NPU_ACTIVE);
  ETHOSU_PMU_CYCCNT_Reset(drv);
  ETHOSU_PMU_EVCNTR_ALL_Reset(drv);
  ETHOSU_PMU_CNTR_Enable(drv, ETHOSU_PMU_CCNT_Msk | ETHOSU_PMU_CNT1_Msk);
}

extern "C" void ethosu_inference_end(struct ethosu_driver* drv, void*) {
  ETHOSU_PMU_CNTR_Disable(drv, ETHOSU_PMU_CCNT_Msk | ETHOSU_PMU_CNT1_Msk);
  g_npu_cycles += static_cast<uint32_t>(ETHOSU_PMU_Get_CCNTR(drv));
  g_npu_active += ETHOSU_PMU_Get_EVCNTR(drv, 0);
  ETHOSU_PMU_Disable(drv);
}
#endif

namespace edge {

bool Infer::init() {
  executorch::runtime::runtime_init();

  auto method_allocator = std::make_unique<PeakAllocator>(kMethodPoolSize, g_method_pool);
  auto temp_allocator = std::make_unique<PeakAllocator>(kTempPoolSize, g_temp_pool);
  g_method_alloc = method_allocator.get();
  g_temp_alloc = temp_allocator.get();

  g_module = new (std::nothrow) EmbeddedModule(
      model_pte, model_pte_size,
      std::make_unique<BufferDataLoader>(model_pte, model_pte_size),
      std::move(method_allocator),
      std::move(temp_allocator));
  if (g_module == nullptr) {
    printf("# infer: out of heap for the module\n");
    return false;
  }

  const uint32_t start = cycle_counter();
  if (g_module->load_method("forward") != executorch::runtime::Error::Ok) {
    printf("# infer: load_method(forward) failed\n");
    return false;
  }
  load_cycles_ = cycle_counter() - start;
  ready_ = true;
  return true;
}

bool Infer::run(const int8_t* x, float logits[kClasses], InferStats* stats) {
  if (!ready_) {
    return false;
  }
  // (x - zp) * scale: the float the program's quantize step maps back to x exactly.
  const float scale = static_cast<float>(INPUT_QUANT_SCALE);
  for (size_t i = 0; i < N_CH * N_SAMPLES; ++i) {
    g_input[i] = static_cast<float>(static_cast<int32_t>(x[i]) - INPUT_ZERO_POINT) * scale;
  }
  std::array<SizesType, 4> sizes{1, 1, N_CH, N_SAMPLES};
  std::array<DimOrderType, 4> dim_order{0, 1, 2, 3};
  TensorImpl input_impl(ScalarType::Float, sizes.size(), sizes.data(), g_input, dim_order.data());
  Tensor input(&input_impl);

  g_npu_cycles = 0;
  g_npu_active = 0;
  const uint32_t start = cycle_counter();
  auto outputs = g_module->forward(input);
  const uint32_t cycles = cycle_counter() - start;
  if (!outputs.ok() || outputs->empty()) {
    printf("# infer: forward failed err=%u\n", outputs.ok() ? 0u : static_cast<unsigned>(outputs.error()));
    return false;
  }
  const EValue& out = outputs->front();
  if (!out.isTensor() || out.toTensor().scalar_type() != ScalarType::Float ||
      out.toTensor().numel() != kClasses) {
    printf("# infer: unexpected output\n");
    return false;
  }
  const float* y = out.toTensor().const_data_ptr<float>();
  for (int c = 0; c < kClasses; ++c) {
    logits[c] = y[c];
  }
  if (stats != nullptr) {
    stats->cpu_cycles = cycles;
    stats->npu_cycles = g_npu_cycles;
    stats->npu_active = g_npu_active;
  }
  return true;
}

size_t Infer::method_pool_peak() const {
  return g_method_alloc != nullptr ? g_method_alloc->peak() : 0;
}

size_t Infer::temp_pool_peak() const {
  return g_temp_alloc != nullptr ? g_temp_alloc->peak() : 0;
}

size_t Infer::pte_bytes() const {
  return model_pte_size;
}

}  // namespace edge
