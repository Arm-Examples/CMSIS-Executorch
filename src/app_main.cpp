// Copyright 2026 Arm Limited and/or its affiliates.
// SPDX-License-Identifier: Apache-2.0
//
// Headless ExecuTorch runner for the Corstone-320 / Ethos-U85 example. Loads
// the .pte embedded by the model-conversion build step (model_pte.h), runs one
// inference on the NPU, and prints the output logits. Built entirely from the
// PyTorch::ExecuTorch pack's runtime + operator components; the board layer
// provides main(), stdout and the Ethos-U driver init and then calls app_main.
//
// API usage mirrors backends/arm/.../test/all_ops/main.cpp.

#include <cstdint>
#include <cstdio>

#include <executorch/runtime/core/data_loader.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/hierarchical_allocator.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/core/span.h>
#include <executorch/runtime/executor/memory_manager.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/runtime.h>

#include "model_pte.h"

using executorch::aten::Tensor;
using executorch::runtime::DataLoader;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::HierarchicalAllocator;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::MemoryManager;
using executorch::runtime::Method;
using executorch::runtime::MethodMeta;
using executorch::runtime::Program;
using executorch::runtime::Result;
using executorch::runtime::Span;

namespace {

constexpr size_t kMethodPoolSize = 4 * 1024 * 1024;
constexpr size_t kTempPoolSize = 4 * 1024 * 1024;  // Ethos-U scratch is drawn from here.

alignas(16) uint8_t g_method_pool[kMethodPoolSize];
alignas(16) uint8_t g_temp_pool[kTempPoolSize];

// In-memory DataLoader so we don't need the extension/data-loader tree.
class BufferLoader final : public DataLoader {
 public:
  BufferLoader(const void* data, size_t size) : data_(data), size_(size) {}

  Result<FreeableBuffer> load(
      size_t offset, size_t size, const DataLoader::SegmentInfo&) const override {
    if (offset + size > size_) {
      return Error::InvalidArgument;
    }
    return FreeableBuffer(
        static_cast<const uint8_t*>(data_) + offset, size, nullptr);
  }

  Result<size_t> size() const override { return size_; }

 private:
  const void* data_;
  size_t size_;
};

}  // namespace

extern "C" int app_main(void) {
  executorch::runtime::runtime_init();

  printf("ExecuTorch Ethos-U85 example: %lu byte model\n", model_pte_size);

  BufferLoader loader(model_pte, model_pte_size);
  Result<Program> program = Program::load(&loader);
  if (!program.ok()) {
    printf("Program::load failed (err=%u)\n", static_cast<unsigned>(program.error()));
    return 1;
  }

  const char* method_name = "forward";
  Result<MethodMeta> meta = program->method_meta(method_name);
  if (!meta.ok()) {
    printf("method_meta failed\n");
    return 1;
  }

  MemoryAllocator method_allocator(kMethodPoolSize, g_method_pool);
  MemoryAllocator temp_allocator(kTempPoolSize, g_temp_pool);

  size_t num_planned = meta->num_memory_planned_buffers();
  Span<uint8_t> planned_spans[8];
  for (size_t i = 0; i < num_planned && i < 8; ++i) {
    size_t sz = meta->memory_planned_buffer_size(i).get();
    planned_spans[i] = {static_cast<uint8_t*>(method_allocator.allocate(sz, 16)), sz};
  }
  HierarchicalAllocator planned_memory({planned_spans, num_planned});
  MemoryManager memory_manager(&method_allocator, &planned_memory, &temp_allocator);

  Result<Method> method = program->load_method(method_name, &memory_manager);
  if (!method.ok()) {
    printf("load_method failed (err=%u)\n", static_cast<unsigned>(method.error()));
    return 1;
  }

  // Fill each input tensor with a simple deterministic ramp.
  for (size_t i = 0; i < method->inputs_size(); ++i) {
    EValue in = method->get_input(i);
    if (!in.isTensor()) {
      continue;
    }
    Tensor t = in.toTensor();
    if (t.scalar_type() == executorch::aten::ScalarType::Float) {
      float* p = t.mutable_data_ptr<float>();
      size_t n = t.numel();
      for (size_t j = 0; j < n; ++j) {
        p[j] = static_cast<float>((j % 32)) / 32.0f - 0.5f;
      }
    }
  }

  Error exec_err = method->execute();
  if (exec_err != Error::Ok) {
    printf("execute failed (err=%u)\n", static_cast<unsigned>(exec_err));
    return 1;
  }

  EValue out = method->get_output(0);
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
  return 0;
}
