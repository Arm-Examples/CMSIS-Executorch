/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Customization for embedded systems by Arm.
 */
#include "arm_embedded_module.hpp"

#include <executorch/extension/data_loader/buffer_data_loader.h>

using executorch::extension::BufferDataLoader;

namespace arm
{
  namespace embedded
  {

    EmbeddedModule::EmbeddedModule(const unsigned char *pte_data,
                                   size_t pte_size,
                                   std::unique_ptr<DataLoader> data_loader,
                                   std::unique_ptr<MemoryAllocator> memory_allocator,
                                   std::unique_ptr<MemoryAllocator> temp_allocator)
        : pte_data_(pte_data), pte_size_(pte_size),
          data_loader_(std::move(data_loader)),
          memory_allocator_(std::move(memory_allocator)),
          temp_allocator_(std::move(temp_allocator))
    {
    }

    ET_NODISCARD Error EmbeddedModule::load(
        const Program::Verification verification)
    {
      if (!is_loaded())
      {
        if (!data_loader_)
        {
          return Error::MemoryAllocationFailed;
        }
        Result<Program> res_program = Program::load(data_loader_.get(), verification);
        if (!res_program.ok())
        {
          return res_program.error();
        }
        auto program =
            std::make_unique<std::remove_reference_t<decltype(*res_program)>>(
                std::move(*res_program));
        program_ = std::shared_ptr<Program>(
            program.release(), [](Program *pointer)
            { delete pointer; });
      }
      return Error::Ok;
    }

    Result<size_t> EmbeddedModule::num_methods()
    {
      ET_CHECK_OK_OR_RETURN_ERROR(load());
      return program_->num_methods();
    }

    Result<std::unordered_set<std::string>> EmbeddedModule::method_names()
    {
      ET_CHECK_OK_OR_RETURN_ERROR(load());
      const auto method_count = program_->num_methods();
      std::unordered_set<std::string> result;
      result.reserve(method_count);

      for (auto index = 0; index < method_count; ++index)
      {
        result.emplace(program_->get_method_name(index).get());
      }
      return result;
    }

    Error EmbeddedModule::load_method(
        const std::string &method_name,
        HierarchicalAllocator *planned_memory,
        EventTracer *event_tracer)
    {
      if (!is_method_loaded(method_name))
      {
        ET_CHECK_OK_OR_RETURN_ERROR(load());

        MethodHolder method_holder;

        if (!planned_memory)
        {
          auto method_metadata_result = program_->method_meta(method_name.c_str());
          if (!method_metadata_result.ok())
          {
            return method_metadata_result.error();
          }
          const auto method_metadata = std::move(*method_metadata_result);
          const auto planned_buffers_count =
              method_metadata.num_memory_planned_buffers();
          method_holder.planned_buffers.reserve(planned_buffers_count);
          method_holder.planned_spans.reserve(planned_buffers_count);

          for (auto index = 0; index < planned_buffers_count; ++index)
          {
            const auto buffer_size =
                static_cast<size_t>(method_metadata.memory_planned_buffer_size(index).get());

            uint8_t *buffer = reinterpret_cast<uint8_t *>(
                memory_allocator_->allocate(buffer_size, 16UL));
            if (buffer == nullptr)
            {
              ET_LOG(Error, "Could not allocate memory for memory planned buffer size %d", (int)buffer_size);
              return Error::MemoryAllocationFailed;
            }

            method_holder.planned_buffers.push_back(buffer);
            method_holder.planned_spans.push_back({method_holder.planned_buffers.back(), buffer_size});
          }
          method_holder.planned_memory =
              std::make_unique<HierarchicalAllocator>(Span(
                  method_holder.planned_spans.data(),
                  method_holder.planned_spans.size()));
          planned_memory = method_holder.planned_memory.get();
        }
        method_holder.memory_manager = std::make_unique<MemoryManager>(
            memory_allocator_.get(), planned_memory, temp_allocator_.get());
        auto res_method = program_->load_method(
            method_name.c_str(),
            method_holder.memory_manager.get(),
            event_tracer ? event_tracer : this->event_tracer());
        if (!res_method.ok())
        {
          return res_method.error();
        }
        method_holder.method =
            std::make_unique<std::remove_reference_t<decltype(*res_method)>>(
                std::move(*res_method));
        methods_.emplace(method_name, std::move(method_holder));
      }
      return Error::Ok;
    }

    #if 0
    ET_NODISCARD Result<Method *> EmbeddedModule::method(
        const std::string &method_name)
    {
      ET_CHECK_OK_OR_RETURN_ERROR(load_method(method_name));
      return methods_[method_name].method.get();
    }
    #endif 
    
    Result<MethodMeta> EmbeddedModule::method_meta(const std::string &method_name)
    {
      ET_CHECK_OK_OR_RETURN_ERROR(load());
      return program_->method_meta(method_name.c_str());
    }

    Result<std::vector<EValue>> EmbeddedModule::execute(
        const std::string &method_name,
        const std::vector<EValue> &input_values)
    {
      ET_CHECK_OK_OR_RETURN_ERROR(load_method(method_name));
      auto &method = methods_.at(method_name).method;
      for (auto index = 0; index < input_values.size(); ++index)
      {
        ET_CHECK_OK_OR_RETURN_ERROR(method->set_input(input_values[index], index));
      }
      ET_CHECK_OK_OR_RETURN_ERROR(method->execute());
      const auto outputs_size = method->outputs_size();
      std::vector<EValue> outputs(outputs_size);
      ET_CHECK_OK_OR_RETURN_ERROR(
          method->get_outputs(outputs.data(), outputs_size));

      return outputs;
    }

    Error EmbeddedModule::set_input(
        const std::string &method_name,
        const EValue &input_value,
        size_t input_index)
    {
      ET_CHECK_OK_OR_RETURN_ERROR(load_method(method_name));
      auto &method = methods_.at(method_name).method;
      return method->set_input(input_value, input_index);
    }

    Error EmbeddedModule::set_inputs(
        const std::string &method_name,
        const std::vector<EValue> &input_values)
    {
      ET_CHECK_OK_OR_RETURN_ERROR(load_method(method_name));
      auto &method = methods_.at(method_name).method;
      return method->set_inputs(executorch::aten::ArrayRef<EValue>(
          input_values.data(), input_values.size()));
    }

    Error EmbeddedModule::set_output(
        const std::string &method_name,
        EValue output_value,
        size_t output_index)
    {
      ET_CHECK_OK_OR_RETURN_ERROR(load_method(method_name));
      auto &method = methods_.at(method_name).method;
      ET_CHECK_OR_RETURN_ERROR(
          output_value.isTensor(),
          InvalidArgument,
          "output type: %zu is not tensor",
          (size_t)output_value.tag);
      const auto &output_tensor = output_value.toTensor();
      return method->set_output_data_ptr(
          output_tensor.mutable_data_ptr(), output_tensor.nbytes(), output_index);
    }

    Error EmbeddedModule::set_outputs(
        const std::string &method_name,
        const std::vector<EValue> &output_values)
    {
      ET_CHECK_OK_OR_RETURN_ERROR(load_method(method_name));
      auto &method = methods_.at(method_name).method;
      const auto outputs_size = method->outputs_size();
      ET_CHECK_OR_RETURN_ERROR(
          output_values.size() == outputs_size,
          InvalidArgument,
          "output size: %zu is not equal to method output size: %zu",
          output_values.size(),
          outputs_size);
      for (auto index = 0; index < outputs_size; ++index)
      {
        ET_CHECK_OK_OR_RETURN_ERROR(
            set_output(method_name, output_values[index], index));
      }
      return Error::Ok;
    }

    Result<std::vector<EValue>> EmbeddedModule::get_outputs(
        const std::string &method_name)
    {
      ET_CHECK_OK_OR_RETURN_ERROR(load_method(method_name));
      auto &method = methods_.at(method_name).method;
      const auto outputs_size = method->outputs_size();
      std::vector<EValue> outputs(outputs_size);
      ET_CHECK_OK_OR_RETURN_ERROR(
          method->get_outputs(outputs.data(), outputs_size));
      return outputs;
    }

    Result<EValue> EmbeddedModule::get_output(
        const std::string &method_name,
        size_t output_index)
    {
      ET_CHECK_OK_OR_RETURN_ERROR(load_method(method_name));
      auto &method = methods_.at(method_name).method;
      ET_CHECK_OR_RETURN_ERROR(
          output_index < method->outputs_size(),
          InvalidArgument,
          "output index: %zu is out of range",
          output_index);
      return method->get_output(output_index);
    }

  } // namespace embedded
} // namespace arm
