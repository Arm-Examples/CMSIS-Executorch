// UART frame parser; see uart_rx.h for the frame layout.
#include "uart_rx.h"

namespace edge {

void FrameParser::reset() {
  state_ = State::kSync0;
  type_ = FrameType::kData;
  need_ = 0;
  pos_ = 0;
  errors_ = 0;
  frames_ = 0;
}

bool FrameParser::feed(uint8_t byte, Frame* out) {
  switch (state_) {
    case State::kSync0:
      if (byte == kSync0) {
        state_ = State::kSync1;
      } else {
        ++errors_;
      }
      return false;

    case State::kSync1:
      if (byte == kSyncData) {
        type_ = FrameType::kData;
        need_ = kDataFrameBytes - 2;
      } else if (byte == kSyncControl) {
        type_ = FrameType::kControl;
        need_ = kControlFrameBytes - 2;
      } else {
        ++errors_;
        // 0xAA 0xAA ...: the second 0xAA may start the real frame.
        state_ = byte == kSync0 ? State::kSync1 : State::kSync0;
        return false;
      }
      pos_ = 0;
      state_ = State::kBody;
      return false;

    case State::kBody:
      body_[pos_++] = byte;
      if (type_ == FrameType::kData && pos_ == 1 && byte != N_CH) {
        ++errors_;
        state_ = byte == kSync0 ? State::kSync1 : State::kSync0;
        return false;
      }
      if (pos_ < need_) {
        return false;
      }
      state_ = State::kSync0;
      ++frames_;
      out->type = type_;
      if (type_ == FrameType::kData) {
        out->seq = static_cast<uint16_t>(body_[1] | (body_[2] << 8));
        out->kind = 0;
        out->arg = 0;
        for (uint32_t c = 0; c < N_CH; ++c) {
          const uint8_t* p = &body_[3 + 2 * c];
          out->data[c] = static_cast<int16_t>(static_cast<uint16_t>(p[0] | (p[1] << 8)));
        }
      } else {
        out->kind = body_[0];
        out->arg = body_[1];
        out->seq = static_cast<uint16_t>(body_[2] | (body_[3] << 8));
      }
      return true;
  }
  return false;
}

uint32_t encode_data_frame(uint8_t* buf, uint16_t seq, const int16_t data[N_CH]) {
  uint32_t n = 0;
  buf[n++] = FrameParser::kSync0;
  buf[n++] = FrameParser::kSyncData;
  buf[n++] = N_CH;
  buf[n++] = static_cast<uint8_t>(seq & 0xFF);
  buf[n++] = static_cast<uint8_t>(seq >> 8);
  for (uint32_t c = 0; c < N_CH; ++c) {
    const uint16_t v = static_cast<uint16_t>(data[c]);
    buf[n++] = static_cast<uint8_t>(v & 0xFF);
    buf[n++] = static_cast<uint8_t>(v >> 8);
  }
  return n;
}

uint32_t encode_control_frame(uint8_t* buf, uint8_t kind, uint8_t arg, uint16_t seq) {
  uint32_t n = 0;
  buf[n++] = FrameParser::kSync0;
  buf[n++] = FrameParser::kSyncControl;
  buf[n++] = kind;
  buf[n++] = arg;
  buf[n++] = static_cast<uint8_t>(seq & 0xFF);
  buf[n++] = static_cast<uint8_t>(seq >> 8);
  return n;
}

}  // namespace edge
