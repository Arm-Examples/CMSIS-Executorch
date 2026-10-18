// TinyEEGNet on the Ethos-U85 through ExecuTorch (the .pte embedded by ai_layer/model_pte.c).
// Float I/O program: the int8 window is fed as (x_int8 - zero_point) * quant_scale, which the
// program's quantize step maps back to exactly x_int8 (Emulator_Deploy harness, 20/20 on FVP).
#ifndef EDGE_EEG_INFER_H_
#define EDGE_EEG_INFER_H_

#include <cstddef>
#include <cstdint>

namespace edge {

struct InferStats {
  uint32_t cpu_cycles;   // DWT cycles for the whole forward() call (CPU + waiting on the NPU)
  uint32_t npu_cycles;   // Ethos-U PMU cycle counter, summed over the command streams
  uint32_t npu_active;   // Ethos-U PMU NPU_ACTIVE events
};

class Infer {
 public:
  static constexpr int kClasses = 2;

  // Loads the program and the forward method once. Prints a '#' line on failure.
  bool init();
  // x: int8 NCHW 1 x 1 x 22 x 256. Writes the two logits.
  bool run(const int8_t* x, float logits[kClasses], InferStats* stats);

  size_t method_pool_peak() const;
  size_t temp_pool_peak() const;
  size_t pte_bytes() const;
  uint32_t load_cycles() const { return load_cycles_; }

 private:
  bool ready_ = false;
  uint32_t load_cycles_ = 0;
};

}  // namespace edge

#endif  // EDGE_EEG_INFER_H_
