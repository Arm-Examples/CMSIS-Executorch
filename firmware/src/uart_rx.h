// Host -> board UART frames (README "UART contract"), little endian:
//   data    0xAA 0x55, uint8 n_ch (22), uint16 seq, int16[n_ch]   one 250 Hz sample, uV x 20.48
//   control 0xAA 0x56, uint8 kind, uint8 arg, uint16 seq
//     kind 1 SET_SUBJECT arg 1..9 | 2 CUE arg label (0, 1, 255) | 3 RESET | 4 FREE_RUN arg 0/1
// Byte-wise parser, no HAL (host-testable). A bad header byte or an n_ch other than N_CH
// counts one error and the parser resynchronises on the next 0xAA.
// The bytes come from the board layer's interrupt-driven receive ring (stdio_rx_read in
// board/<target>/retarget_stdio.c), polled by the main loop.
#ifndef EDGE_EEG_UART_RX_H_
#define EDGE_EEG_UART_RX_H_

#include <cstdint>

#include "params_gen.h"

namespace edge {

enum class FrameType : uint8_t { kData, kControl };

enum ControlKind : uint8_t {
  kSetSubject = 1,
  kCue = 2,
  kReset = 3,
  kFreeRun = 4,
};

struct Frame {
  FrameType type;
  uint16_t seq;
  uint8_t kind;               // control only
  uint8_t arg;                // control only
  int16_t data[N_CH];         // data only
};

class FrameParser {
 public:
  static constexpr uint8_t kSync0 = 0xAA;
  static constexpr uint8_t kSyncData = 0x55;
  static constexpr uint8_t kSyncControl = 0x56;
  static constexpr uint32_t kDataFrameBytes = 2 + 1 + 2 + 2 * N_CH;   // 49
  static constexpr uint32_t kControlFrameBytes = 2 + 1 + 1 + 2;       // 6

  FrameParser() { reset(); }
  void reset();

  // Feeds one byte; returns true when it completed a frame (copied to *out).
  bool feed(uint8_t byte, Frame* out);

  uint32_t errors() const { return errors_; }
  uint32_t frames() const { return frames_; }

 private:
  enum class State : uint8_t { kSync0, kSync1, kBody };
  State state_;
  FrameType type_;
  uint32_t need_;             // body bytes still expected
  uint32_t pos_;
  uint8_t body_[kDataFrameBytes];
  uint32_t errors_;
  uint32_t frames_;
};

// Appends the encoded frames to buf (used by the self-test to replay a stream through the
// same parser as the UART path). Return the number of bytes written.
uint32_t encode_data_frame(uint8_t* buf, uint16_t seq, const int16_t data[N_CH]);
uint32_t encode_control_frame(uint8_t* buf, uint8_t kind, uint8_t arg, uint16_t seq);

}  // namespace edge

#endif  // EDGE_EEG_UART_RX_H_
