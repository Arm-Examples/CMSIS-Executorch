/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Customization for embedded systems by Arm.
 */
#pragma once

#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/core/data_loader.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace ::executorch::runtime;

namespace arm
{
  namespace embedded
  {

    /**
     * A facade class for loading programs and executing methods within them.
     */
    class EmbeddedModule
    {
    public:
      /**
       * Constructs an instance with the provided PTE data and memory allocators.
       *
       * @param[in] pte_data Pointer to the ExecuTorch program data to load.
       * @param[in] pte_size Size of the ExecuTorch program data to load.
       * @param[in] memory_allocator A MemoryAllocator used for memory planning.
       * @param[in] temp_allocator A MemoryAllocator to use when allocating
       * temporary data during kernel or delegate execution.
       */
      explicit EmbeddedModule(
          const unsigned char *pte_data,
          size_t pte_size,
          std::unique_ptr<DataLoader> data_loader = nullptr,
          std::unique_ptr<MemoryAllocator> memory_allocator = nullptr,
          std::unique_ptr<MemoryAllocator> temp_allocator = nullptr);

      /**
       * Loads the program if needed.
       *
       * @param[in] verification The type of verification to do before returning
       * success.
       *
       * @returns An Error to indicate success or failure of the loading process.
       */
      ET_NODISCARD Error load(
          const Program::Verification verification =
              Program::Verification::Minimal);

      /**
       * Checks if the program is loaded.
       *
       * @returns true if the program is loaded, false otherwise.
       */
      inline bool is_loaded() const
      {
        return program_ != nullptr;
      }

      /**
       * Get the program. The data loader used by the program is guaranteed to be
       * valid for the lifetime of the program.
       *
       * @returns Shared pointer to the program or nullptr if it's not yet loaded.
       */
      inline std::shared_ptr<Program> program() const
      {
        return program_;
      }

      /**
       * Get the number of methods available in the loaded program.
       *
       * @returns A Result object containing either the number of methods available
       *          or an error to indicate failure.
       */
      Result<size_t> num_methods();

      /**
       * Get a list of method names available in the loaded program.
       * Loads the program and method if needed.
       *
       * @returns A set of strings containing the names of the methods, or an error
       * if the program or method failed to load.
       */
      Result<std::unordered_set<std::string>> method_names();

      /**
       * Load a specific method from the program and set up memory management if
       * needed. The loaded method is cached to reuse the next time it's executed.
       *
       * @param[in] method_name The name of the method to load.
       * @param[in] planned_memory The memory-planned buffers to use for mutable
       * tensor data when executing a method.
       * @param[in] event_tracer Per-method event tracer to profile/trace methods
       * individually. When not given, the event tracer passed to the
       * EmbeddedModule constructor is used. Otherwise, this per-method event tracer
       * takes precedence.
       *
       * @returns An Error to indicate success or failure.
       */
      ET_NODISCARD
      Error load_method(
          const std::string &method_name,
          HierarchicalAllocator *planned_memory = nullptr,
          EventTracer *event_tracer = nullptr);

      #if 0
      ET_DEPRECATED ET_NODISCARD Error inline load_method(
          const std::string &method_name,
          EventTracer *event_tracer)
      {
        return load_method(method_name, nullptr, event_tracer);
      }
      #endif

      /**
       * Unload a specific method from the program.
       *
       * @param[in] method_name The name of the method to unload.
       *
       * @returns True if the method is unloaded, false if no-op.
       */
      inline bool unload_method(const std::string &method_name)
      {
        return methods_.erase(method_name);
      }

      #if 0
      /**
       * DEPRECATED: EmbeddedModule manages each Method exclusively.
       *
       * Get a method by it's name. Not recommended to use this method directly as
       * an end user. It's exposed to allow for composability of module in apis that
       * operate on method.
       *
       * @param[in] method_name The name of the method to get.
       *
       * @returns A Result object containing either a pointer to the requested
       *          method or an error to indicate failure.
       */
      ET_DEPRECATED ET_NODISCARD Result<Method *> method(
          const std::string &method_name);
      #endif

      /**
       * Checks if a specific method is loaded.
       *
       * @param[in] method_name The name of the method to check.
       *
       * @returns true if the method specified by method_name is loaded, false
       * otherwise.
       */
      inline bool is_method_loaded(const std::string &method_name) const
      {
        return methods_.count(method_name);
      }

      /**
       * Get a method metadata struct by method name.
       * Loads the program if needed.
       *
       * @param[in] method_name The name of the method to get the metadata for.
       *
       * @returns A method metadata, or an error if the program or method failed to
       * load.
       */
      Result<MethodMeta> method_meta(const std::string &method_name);

      /**
       * Execute a specific method with the given input values and retrieve the
       * output values. Loads the program and method before executing if needed.
       *
       * @param[in] method_name The name of the method to execute.
       * @param[in] input_values A vector of input values to be passed to the
       * method.
       *
       * @returns A Result object containing either a vector of output values
       *          from the method or an error to indicate failure.
       */
      ET_NODISCARD Result<std::vector<EValue>> execute(
          const std::string &method_name,
          const std::vector<EValue> &input_values);

      /**
       * Execute a specific method with a single input value.
       * Loads the program and method before executing if needed.
       *
       * @param[in] method_name The name of the method to execute.
       * @param[in] input_value A value to be passed to the method.
       *
       * @returns A Result object containing either a vector of output values
       *          from the method or an error to indicate failure.
       */
      ET_NODISCARD inline Result<std::vector<EValue>> execute(
          const std::string &method_name,
          const EValue &input_value)
      {
        return execute(method_name, std::vector<EValue>{input_value});
      }

      /**
       * Execute a specific method without any input values.
       * Loads the program and method before executing if needed.
       *
       * @param[in] method_name The name of the method to execute.
       *
       * @returns A Result object containing either a vector of output values
       *          from the method or an error to indicate failure.
       */
      ET_NODISCARD inline Result<std::vector<EValue>> execute(
          const std::string &method_name)
      {
        return execute(method_name, std::vector<EValue>{});
      }

      /**
       * Retrieve the output value of a specific method with the given input values.
       * Loads the program and method before execution if needed.
       *
       * @param[in] method_name The name of the method to execute.
       * @param[in] input_values A vector of input values to be passed to the
       * method.
       *
       * @returns A Result object containing either the first output value from the
       * method or an error to indicate failure.
       */
      ET_NODISCARD inline Result<EValue> get(
          const std::string &method_name,
          const std::vector<EValue> &input_values)
      {
        auto execute_result = execute(method_name, input_values);
        if (!execute_result.ok())
        {
          return execute_result.error();
        }
        auto result = std::move(*execute_result);
        if (result.empty())
        {
          return Error::InvalidArgument;
        }
        return result[0];
      }

      /**
       * Retrieve the output value of a specific method with a single input value.
       * Loads the program and method before execution if needed.
       *
       * @param[in] method_name The name of the method to execute.
       * @param[in] input_value A value to be passed to the method.
       *
       * @returns A Result object containing either the first output value from the
       * method or an error to indicate failure.
       */
      ET_NODISCARD inline Result<EValue> get(
          const std::string &method_name,
          const EValue &input_value)
      {
        return get(method_name, std::vector<EValue>{input_value});
      }

      /**
       * Retrieve the output value of a specific method without any input values.
       * Loads the program and method before execution if needed.
       *
       * @param[in] method_name The name of the method to execute.
       *
       * @returns A Result object containing either the first output value from the
       * method or an error to indicate failure.
       */
      ET_NODISCARD inline Result<EValue> get(
          const std::string &method_name)
      {
        return get(method_name, std::vector<EValue>{});
      }

      /**
       * Execute the 'forward' method with the given input values and retrieve the
       * output values. Loads the program and method before executing if needed.
       *
       * @param[in] input_values A vector of input values for the 'forward' method.
       *
       * @returns A Result object containing either a vector of output values
       *          from the 'forward' method or an error to indicate failure.
       */
      ET_NODISCARD inline Result<std::vector<EValue>> forward(
          const std::vector<EValue> &input_values)
      {
        return execute("forward", input_values);
      };

      /**
       * Execute the 'forward' method with a single value.
       * Loads the program and method before executing if needed.
       *
       * @param[in] input_value A value for the 'forward' method.
       *
       * @returns A Result object containing either a vector of output values
       *          from the 'forward' method or an error to indicate failure.
       */
      ET_NODISCARD inline Result<std::vector<EValue>> forward(
          const EValue &input_value)
      {
        return forward(std::vector<EValue>{input_value});
      }

      /**
       * Execute the 'forward' method without any input values.
       * Loads the program and method before executing if needed.
       *
       * @returns A Result object containing either a vector of output values
       *          from the method or an error to indicate failure.
       */
      ET_NODISCARD inline Result<std::vector<EValue>> forward()
      {
        return forward(std::vector<EValue>{});
      }

      /**
       * Sets a single input value for a specific method.
       *
       * @param[in] method_name The name of the method.
       * @param[in] input_value The EValue to set as the method input.
       * @param[in] input_index Zero-based index of the input to set.
       *
       * @returns An Error to indicate success or failure.
       */
      ET_NODISCARD
      Error set_input(
          const std::string &method_name,
          const EValue &input_value,
          size_t input_index);

      /**
       * Sets a single input value for the "forward" method.
       *
       * @param[in] input_value The EValue to set as the method input.
       * @param[in] input_index Zero-based index of the input to set.
       *
       * @returns An Error to indicate success or failure.
       */
      ET_NODISCARD
      inline Error set_input(
          const EValue &input_value,
          size_t input_index)
      {
        return set_input("forward", input_value, input_index);
      }

      /**
       * Sets all input values for a specific method.
       *
       * @param[in] method_name The name of the method.
       * @param[in] input_values A vector of EValues to set as the method inputs.
       *
       * @returns An Error to indicate success or failure.
       */
      ET_NODISCARD
      Error set_inputs(
          const std::string &method_name,
          const std::vector<EValue> &input_values);

      /**
       * Sets all input values for the "forward" method.
       *
       * @param[in] input_values A vector of EValues to set as the method inputs.
       *
       * @returns An Error to indicate success or failure.
       */
      ET_NODISCARD
      inline Error set_inputs(
          const std::vector<EValue> &input_values)
      {
        return set_inputs("forward", input_values);
      }

      /**
       * Sets the output tensor for a specific method.
       *
       * @param[in] method_name The name of the method.
       * @param[in] output_value The EValue containing the Tensor to set as the
       * method output.
       * @param[in] output_index Zero-based index of the output to set.
       *
       * @returns An Error to indicate success or failure.
       *
       * @note Only Tensor outputs are currently supported for setting.
       */
      ET_NODISCARD
      Error set_output(
          const std::string &method_name,
          EValue output_value,
          size_t output_index = 0);

      /**
       * Sets the output tensor for the "forward" method.
       *
       * @param[in] output_value The EValue containing the Tensor to set as the
       * method output.
       * @param[in] output_index Zero-based index of the output to set.
       *
       * @returns An Error to indicate success or failure.
       *
       * @note Only Tensor outputs are currently supported for setting.
       */
      ET_NODISCARD
      inline Error set_output(
          EValue output_value,
          size_t output_index = 0)
      {
        return set_output("forward", std::move(output_value), output_index);
      }

      /**
       * Sets all output tensors for a specific method.
       *
       * Loads the program and method if needed, and for each output uses
       * the provided tensor's data buffer as the method's output buffer.
       *
       * @param[in] method_name The name of the method.
       * @param[in] output_values A vector of EValues to set as the method outputs.
       *
       * @returns An Error to indicate success or failure.
       *
       * @note Only Tensor outputs are currently supported for setting.
       * @note Will fail for outputs that are memory-planned or constants.
       */
      ET_NODISCARD
      Error set_outputs(
          const std::string &method_name,
          const std::vector<EValue> &output_values);

      /**
       * Sets all output tensors for the "forward" method.
       *
       * @param[in] output_values A vector of EValues to set as the method outputs.
       *
       * @returns An Error to indicate success or failure.
       *
       * @note Only Tensor outputs are currently supported for setting.
       * @note Will fail for outputs that are memory-planned or constants.
       */
      ET_NODISCARD
      inline Error set_outputs(
          const std::vector<EValue> &output_values)
      {
        return set_outputs("forward", output_values);
      }

      /**
       * Retrieve all current output values of a specific method without executing
       * it. Loads the program and method before retrieval if needed.
       *
       * @param[in] method_name The name of the method.
       *
       * @returns A Result containing the vector of output values, or an error.
       */
      ET_NODISCARD
      Result<std::vector<EValue>> get_outputs(
          const std::string &method_name);

      /**
       * Retrieve all current output values of the "forward" method without
       * executing it. Loads the program and method before retrieval if needed.
       *
       * @returns A Result containing the vector of output values, or an error.
       */
      ET_NODISCARD
      inline Result<std::vector<EValue>> get_outputs()
      {
        return get_outputs("forward");
      }

      /**
       * Retrieve a single current output value of a specific method without
       * executing it. Loads the program and method before retrieval if needed.
       *
       * @param[in] method_name The name of the method.
       * @param[in] output_index Zero-based index of the output to retrieve.
       *
       * @returns A Result containing the requested output value, or an error.
       */
      ET_NODISCARD
      Result<EValue> get_output(
          const std::string &method_name,
          size_t output_index = 0);

      /**
       * Retrieve a single current output value of the "forward" method without
       * executing it. Loads the program and method before retrieval if needed.
       *
       * @param[in] output_index Zero-based index of the output to retrieve.
       *
       * @returns A Result containing the requested output value, or an error.
       */
      ET_NODISCARD
      inline Result<EValue> get_output(size_t output_index = 0)
      {
        return get_output("forward", output_index);
      }

      /**
       * Retrieves the EventTracer instance being used by the EmbeddedModule.
       * EventTracer is used for tracking and logging events during the execution
       * of methods.
       *
       * @returns A pointer to the EventTracer instance. Returns nullptr if no
       * EventTracer is set.
       */
      inline EventTracer *event_tracer() const
      {
        return event_tracer_.get();
      }

    private:
      struct MethodHolder
      {
        std::vector<uint8_t *> planned_buffers;
        std::vector<Span<uint8_t>> planned_spans;
        std::unique_ptr<HierarchicalAllocator> planned_memory;
        std::unique_ptr<MemoryManager> memory_manager;
        std::unique_ptr<Method> method;
      };

      std::shared_ptr<Program> program_;
      std::unique_ptr<DataLoader> data_loader_;
      std::unique_ptr<MemoryAllocator> memory_allocator_;
      std::unique_ptr<MemoryAllocator> temp_allocator_;
      // std::unique_ptr<NamedDataMap> merged_data_map_;

      std::unique_ptr<EventTracer> event_tracer_;

      const unsigned char *pte_data_;
      size_t pte_size_;

    protected:
      std::unordered_map<std::string, MethodHolder> methods_;
    };

  } // namespace embedded
} // namespace arm
