// Per-sample EEG preprocessing of the board pipeline (Reproduce_Baselines/docs/v2_handoff.md).
// Pure C++ (no HAL): built into the firmware and into the host unit test
// (firmware/tests/preprocess_test.cpp).
//
// Per 250 Hz sample (int16 in the 2a native ADC step, channels in params.json channel_order):
//   1. uV = x * UV_PER_LSB, CAR (subtract the mean of the 22 channels)  float64
//   2. 4-section SOS band-pass, DF2-T state per channel, never reset    float64
//   3. keep samples with even stream index (decimation 2, phase 0)
//   4. store the 125 Hz sample in a ring of 2 x 256 samples             float32
// Per window (125 Hz samples [end - 256, end)):
//   5. EA[subject] @ window                                            float64 accumulate -> float32
//   6. / INPUT_SCALE, / INPUT_QUANT_SCALE, round half to even, + zero point, clamp   float32
//
// The precision follows host/pipeline_ref.py (the bit-exact reference): it filters in float64
// (scipy.signal.sosfilt), casts the window to float32, applies EA in float64 and casts back
// to float32, then scales and quantizes in float32. The M55 has a double-precision FPU, so
// the board does the same and the int8 window matches the reference bit for bit
// (tests/preprocess_test.cpp checks it on the generated vectors).
#ifndef EDGE_EEG_PREPROCESS_H_
#define EDGE_EEG_PREPROCESS_H_

#include <cstdint>

#include "params_gen.h"

namespace edge {

// uV per int16 count of the UART data frame: the BCI IV 2a native ADC step, 100/2048 uV
// (value = uV x 20.48). Exact in float32 and float64, so the conversion adds no rounding.
// 2a native step; x10 was lossy: 26 % of int8 elements differed.
constexpr double UV_PER_LSB = 0.048828125;  // = 25/512

class Preprocess {
 public:
  static constexpr uint32_t kChannels = N_CH;
  static constexpr uint32_t kWindow = N_SAMPLES;          // 125 Hz samples per window
  static constexpr uint32_t kRing = 2 * N_SAMPLES;        // late cues still find their window
  static constexpr uint32_t kWindowElems = N_CH * N_SAMPLES;

  Preprocess() { reset(); }

  // Clears filter state, counters and the ring buffer (RESET control frame).
  void reset();

  // One 250 Hz sample. Returns true when it produced a 125 Hz sample (even stream index).
  bool push_sample(const int16_t counts[N_CH]);

  // 250 Hz samples pushed since reset; the next sample pushed has this stream index.
  uint32_t n250() const { return n250_; }
  // 125 Hz samples stored since reset (the newest has index n125() - 1).
  uint32_t n125() const { return n125_; }

  // 125 Hz end index (exclusive) of the window of the trial whose cue is stream sample cue_n250.
  static uint32_t window_end_125(uint32_t cue_n250) {
    return cue_n250 / DECIMATION + WINDOW_START + N_SAMPLES;
  }
  // 250 Hz stream index of the sample that completes that window.
  static uint32_t window_last_n250(uint32_t cue_n250) {
    return (window_end_125(cue_n250) - 1) * DECIMATION;
  }
  bool window_ready(uint32_t cue_n250) const { return n125_ >= window_end_125(cue_n250); }

  // True while the 125 Hz samples [end_125 - 256, end_125) are all in the ring.
  bool window_available(uint32_t end_125) const {
    return end_125 >= kWindow && end_125 <= n125_ && n125_ - (end_125 - kWindow) <= kRing;
  }

  // int8 model input (NCHW 1 x 1 x 22 x 256, channel-major) of the 125 Hz window ending at
  // end_125 (exclusive), aligned with EA[subject - 1]. Returns false if the window is not
  // (or no longer) in the ring or the subject is not 1..9.
  bool make_window_at(int subject, uint32_t end_125, int8_t out[kWindowElems]) const;

  // Window of the trial whose cue is stream sample cue_n250.
  bool make_window(int subject, uint32_t cue_n250, int8_t out[kWindowElems]) const {
    return make_window_at(subject, window_end_125(cue_n250), out);
  }

  // The newest 256 samples (free-run mode).
  bool make_latest_window(int subject, int8_t out[kWindowElems]) const {
    return make_window_at(subject, n125_, out);
  }

 private:
  double z_[N_SOS][N_CH][2];      // DF2-T state per section and channel
  float ring_[kRing][N_CH];       // filtered, decimated samples (time-major)
  uint32_t n250_;
  uint32_t n125_;
};

}  // namespace edge

#endif  // EDGE_EEG_PREPROCESS_H_
