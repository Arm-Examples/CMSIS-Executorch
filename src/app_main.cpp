// Copyright 2026 Arm Limited and/or its affiliates.
// SPDX-License-Identifier: Apache-2.0
//
// Headless ExecuTorch runner for the Ethos-U85 example. Loads the .pte
// embedded by the AI layer (model_pte.h, see create_ai_layer.py), runs one
// inference on the NPU, and prints the output logits. Built entirely from the
// PyTorch::ExecuTorch pack's runtime + operator components; the board layer
// provides main(), stdout and the Ethos-U driver init and then calls app_main.
//
// EmbeddedModule (arm_embedded_module.hpp) manages program loading, method
// memory and execution, like ExecuTorch's Module class does on POSIX hosts.

#include <array>
#include <cstdint>
#include <cstdio>
#include <memory>

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/platform/runtime.h>

#include "arm_embedded_module.hpp"
#include "model_pte.h"

using arm::embedded::EmbeddedModule;
using executorch::aten::DimOrderType;
using executorch::aten::ScalarType;
using executorch::aten::SizesType;
using executorch::aten::Tensor;
using executorch::aten::TensorImpl;
using executorch::extension::BufferDataLoader;
using executorch::runtime::EValue;
using executorch::runtime::MemoryAllocator;

namespace {

// Pool sizes and placement are board-overridable: a board layer that cannot
// fit 8 MB of .bss in its default RAM sets APP_*_POOL_SIZE and, via
// APP_POOL_SECTION, the linker section its linker script routes to a larger
// (NPU-accessible) memory.
#ifndef APP_METHOD_POOL_SIZE
#define APP_METHOD_POOL_SIZE (4 * 1024 * 1024)
#endif
#ifndef APP_TEMP_POOL_SIZE
#define APP_TEMP_POOL_SIZE (4 * 1024 * 1024)  // Ethos-U scratch is drawn from here.
#endif
#ifdef APP_POOL_SECTION
#define APP_POOL_ATTRIBUTES __attribute__((section(APP_POOL_SECTION)))
#else
#define APP_POOL_ATTRIBUTES
#endif

constexpr size_t kMethodPoolSize = APP_METHOD_POOL_SIZE;
constexpr size_t kTempPoolSize = APP_TEMP_POOL_SIZE;

alignas(16) uint8_t g_method_pool[kMethodPoolSize] APP_POOL_ATTRIBUTES;
alignas(16) uint8_t g_temp_pool[kTempPoolSize] APP_POOL_ATTRIBUTES;

}  // namespace

extern "C" int app_main(void) {
  executorch::runtime::runtime_init();

  printf("ExecuTorch Ethos-U85 example: %lu byte model\n", model_pte_size);

  EmbeddedModule module(
      model_pte, model_pte_size,
      std::make_unique<BufferDataLoader>(model_pte, model_pte_size),
      std::make_unique<MemoryAllocator>(kMethodPoolSize, g_method_pool),
      std::make_unique<MemoryAllocator>(kTempPoolSize, g_temp_pool));

  // TinyCNN takes one 16x16 RGB image in NCHW order (see model/model.py),
  // filled here with a deterministic ramp. The tensor only wraps the buffer;
  // no tensor extension (and its std::random_device) is needed for that.
  alignas(16) static float input_data[3 * 16 * 16];
  for (size_t i = 0; i < sizeof(input_data) / sizeof(input_data[0]); ++i) {
    input_data[i] = static_cast<float>(i % 32) / 32.0f - 0.5f;
  }
  std::array<SizesType, 4> sizes{1, 3, 16, 16};
  std::array<DimOrderType, 4> dim_order{0, 1, 2, 3};
  TensorImpl input_impl(ScalarType::Float, sizes.size(), sizes.data(), input_data, dim_order.data());
  Tensor input(&input_impl);

  auto outputs = module.forward(input);
  if (!outputs.ok()) {
    printf("forward failed (err=%u)\n", static_cast<unsigned>(outputs.error()));
    return 1;
  }
  if (outputs->empty()) {
    printf("forward returned no outputs\n");
    return 1;
  }

  const EValue& out = outputs->front();
  if (out.isTensor()) {
    Tensor t = out.toTensor();
    printf("Output: %u element(s):", static_cast<unsigned>(t.numel()));
    if (t.scalar_type() == ScalarType::Float) {
      const float* p = t.const_data_ptr<float>();
      for (size_t i = 0; i < t.numel(); ++i) {
        printf(" %.4f", p[i]);
      }
    }
    printf("\n");
  }

  printf("Test_result: PASS\n");
  printf("\x04");  // EOT stops the FVP simulation
  fflush(stdout);
  return 0;
}
