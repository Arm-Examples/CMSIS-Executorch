// Copyright 2026 Arm Limited and/or its affiliates.
// SPDX-License-Identifier: Apache-2.0
//
// Headless ExecuTorch runner for the Corstone-320 / Ethos-U85 example. Loads
// the .pte embedded by the model-conversion build step (model_pte.h), runs one
// inference on the NPU, and prints the output logits. Built entirely from the
// PyTorch::ExecuTorch pack's runtime + operator components; the board layer
// provides main(), stdout and the Ethos-U driver init and then calls app_main.
//
// EmbeddedModule manages program loading, method memory, and execution.

#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/platform/runtime.h>

#include "arm_embedded_module.hpp"
#include "model_pte.h"

using arm::embedded::EmbeddedModule;
using executorch::aten::Tensor;
using executorch::extension::BufferDataLoader;
using executorch::extension::make_tensor_ptr;
using executorch::runtime::EValue;
using executorch::runtime::MemoryAllocator;

namespace {

constexpr size_t kMethodPoolSize = 4 * 1024 * 1024;
constexpr size_t kTempPoolSize = 4 * 1024 * 1024;  // Ethos-U scratch is drawn from here.

alignas(16) uint8_t g_method_pool[kMethodPoolSize];
alignas(16) uint8_t g_temp_pool[kTempPoolSize];

}  // namespace

extern "C" int app_main(void) {
  executorch::runtime::runtime_init();

  printf("ExecuTorch Ethos-U85 example: %lu byte model\n", model_pte_size);

  EmbeddedModule module(
      model_pte, model_pte_size,
      std::make_unique<BufferDataLoader>(model_pte, model_pte_size),
      std::make_unique<MemoryAllocator>(kMethodPoolSize, g_method_pool),
      std::make_unique<MemoryAllocator>(kTempPoolSize, g_temp_pool));

  // TinyCNN takes one 16x16 RGB image in NCHW order (see model/model.py).
  // Keep the TensorPtr alive until forward() finishes using its data.
  auto input = make_tensor_ptr<float>(
      {1, 3, 16, 16}, std::vector<float>(3 * 16 * 16));
  float* input_data = input->mutable_data_ptr<float>();
  for (size_t i = 0; i < input->numel(); ++i) {
    input_data[i] = static_cast<float>(i % 32) / 32.0f - 0.5f;
  }

  auto outputs = module.forward(*input);
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
    if (t.scalar_type() == executorch::aten::ScalarType::Float) {
      const float* p = t.const_data_ptr<float>();
      for (size_t i = 0; i < t.numel(); ++i) {
        printf(" %.4f", p[i]);
      }
    }
    printf("\n");
  }

  printf("Test_result: PASS\n");
  printf("\x04");   // EOT to stop FVP simulation
  fflush(stdout);
  return 0;
}
