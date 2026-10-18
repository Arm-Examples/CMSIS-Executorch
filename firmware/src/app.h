// The stream application: UART bytes -> frames -> per-sample preprocessing -> one inference
// per cue (and every 256 ms in free-run mode) -> motion gate -> LED + JSON line.
// main.cpp feeds it the board's UART bytes; selftest.cpp feeds it an embedded stream through
// the same parser.
#ifndef EDGE_EEG_APP_H_
#define EDGE_EEG_APP_H_

#include <cstdint>

#include "imu.h"
#include "infer.h"
#include "json_out.h"
#include "preprocess.h"
#include "uart_rx.h"

namespace edge {

class App {
 public:
  // Called after every inference with the record (already printed), the int8 window and the
  // NPU/CPU statistics. Used by the self-test.
  using InferenceHook = void (*)(void* ctx, const InferenceRecord& rec, const int8_t* window,
                                 const InferStats& stats);

  static constexpr uint32_t kMaxPendingCues = 8;
  static constexpr uint32_t kFreeRunStride = 32;     // 125 Hz samples = 256 ms
  static constexpr uint32_t kMaxGapFill = 250;       // fill up to 1 s of lost frames

  App(Infer* infer, MotionGate* gate) : infer_(infer), gate_(gate) { reset(); }

  void set_hook(InferenceHook hook, void* ctx) {
    hook_ = hook;
    hook_ctx_ = ctx;
  }
  void set_imu_ok(bool ok) { imu_ok_ = ok; }

  // Bytes from the UART; t_rx_cycles = cycle_counter() when they were taken from the driver.
  void on_bytes(const uint8_t* data, uint32_t n, uint32_t t_rx_cycles);

  uint32_t inferences() const { return inferences_; }
  uint32_t frame_errors() const { return parser_.errors(); }
  uint32_t gap_samples() const { return gap_samples_; }
  const Preprocess& preprocess() const { return pre_; }

 private:
  struct PendingCue {
    bool resolved;
    uint16_t seq;          // seq of the cue's data frame
    uint32_t cue_n250;     // stream index of the cue sample, once resolved
    int32_t trial;
    int32_t label;
  };

  void reset();
  void on_frame(const Frame& f, uint32_t t_rx_cycles);
  void on_control(const Frame& f);
  void on_data(const Frame& f, uint32_t t_rx_cycles);
  void push(const int16_t* sample, uint16_t seq, uint32_t t_rx_cycles);
  void infer_window(uint32_t end_125, int32_t trial, int32_t label, uint16_t seq, uint32_t t_rx_cycles);
  void drop_cue(uint32_t i);

  Infer* infer_;
  MotionGate* gate_;
  InferenceHook hook_ = nullptr;
  void* hook_ctx_ = nullptr;
  bool imu_ok_ = false;

  FrameParser parser_;
  Preprocess pre_;
  int32_t subject_ = 0;
  bool free_run_ = false;
  bool have_seq_ = false;
  uint16_t last_seq_ = 0;
  int16_t last_sample_[N_CH];
  PendingCue cues_[kMaxPendingCues];
  uint32_t n_cues_ = 0;
  int32_t cue_count_ = 0;
  uint32_t inferences_ = 0;
  uint32_t gap_samples_ = 0;
  int8_t window_[Preprocess::kWindowElems];
};

}  // namespace edge

#endif  // EDGE_EEG_APP_H_
