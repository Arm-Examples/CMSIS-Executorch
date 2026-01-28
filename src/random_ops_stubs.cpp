/*
 * Stub implementations for random operations not supported on ARM Clang bare-metal.
 * 
 * ARM Clang V22's libc++ disables std::random_device for bare-metal targets
 * (_LIBCPP_HAS_RANDOM_DEVICE=0). These stubs satisfy the linker for the operator
 * registrations and abort at runtime if called.
 */

#include <executorch/runtime/kernel/kernel_includes.h>
#include <cstdlib>  // for abort()

namespace torch {
namespace executor {
namespace native {

using executorch::aten::IntArrayRef;
using Tensor = executorch::aten::Tensor;

// Stub for rand_out - random operations require std::random_device which is not
// available in ARM Clang bare-metal libc++
Tensor& rand_out(
    KernelRuntimeContext& ctx,
    const IntArrayRef sizes,
    Tensor& out) {
  (void)ctx;
  (void)sizes;
  // Random operations are not supported on bare-metal ARM Clang
  // (no std::random_device available)
  ET_LOG(Error, "rand_out: Random operations not supported on this platform");
  abort();
  return out;
}

// Stub for randn_out - random operations require std::random_device
Tensor& randn_out(
    KernelRuntimeContext& ctx,
    const IntArrayRef sizes,
    Tensor& out) {
  (void)ctx;
  (void)sizes;
  ET_LOG(Error, "randn_out: Random operations not supported on this platform");
  abort();
  return out;
}

// Stub for native_dropout_out - requires std::random_device
std::tuple<Tensor&, Tensor&> native_dropout_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    double p,
    std::optional<bool> train,
    Tensor& out,
    Tensor& mask) {
  (void)ctx;
  (void)input;
  (void)p;
  (void)train;
  ET_LOG(Error, "native_dropout_out: Random operations not supported on this platform");
  abort();
  return std::tie(out, mask);
}

} // namespace native
} // namespace executor
} // namespace torch
