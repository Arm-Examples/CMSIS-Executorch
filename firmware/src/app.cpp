// Stream application; see app.h and the README "UART contract".
#include "app.h"

#include <cstdio>
#include <cstring>

#include "clock.h"
#include "leds.h"

namespace edge {

void App::reset() {
  pre_.reset();
  have_seq_ = false;
  last_seq_ = 0;
  std::memset(last_sample_, 0, sizeof(last_sample_));
  n_cues_ = 0;
  cue_count_ = 0;
}

void App::on_bytes(const uint8_t* data, uint32_t n, uint32_t t_rx_cycles) {
  Frame f;
  for (uint32_t i = 0; i < n; ++i) {
    if (parser_.feed(data[i], &f)) {
      on_frame(f, t_rx_cycles);
    }
  }
}

void App::on_frame(const Frame& f, uint32_t t_rx_cycles) {
  if (f.type == FrameType::kControl) {
    on_control(f);
  } else {
    on_data(f, t_rx_cycles);
  }
}

void App::on_control(const Frame& f) {
  switch (f.kind) {
    case kSetSubject:
      if (f.arg >= 1 && f.arg <= N_SUBJECTS) {
        subject_ = f.arg;
        printf("# subject %d\n", static_cast<int>(subject_));
      } else {
        printf("# bad subject %u\n", static_cast<unsigned>(f.arg));
      }
      break;
    case kCue:
      if (n_cues_ == kMaxPendingCues) {
        printf("# cue queue full, dropping trial %ld\n", static_cast<long>(cues_[0].trial));
        drop_cue(0);
      }
      cues_[n_cues_++] = PendingCue{false, f.seq, 0, cue_count_++, f.arg == 255 ? -1 : static_cast<int32_t>(f.arg)};
      break;
    case kReset:
      reset();
      printf("# reset\n");
      break;
    case kFreeRun:
      free_run_ = f.arg != 0;
      printf("# free_run %d\n", free_run_ ? 1 : 0);
      break;
    default:
      printf("# unknown control kind %u\n", static_cast<unsigned>(f.kind));
      break;
  }
}

void App::on_data(const Frame& f, uint32_t t_rx_cycles) {
  if (have_seq_) {
    const uint16_t d = static_cast<uint16_t>(f.seq - last_seq_);
    if (d == 0 || d > 0x8000u) {
      printf("# stale frame seq=%u last=%u\n", static_cast<unsigned>(f.seq), static_cast<unsigned>(last_seq_));
      return;
    }
    if (d > 1) {
      // Lost frames: keep the 250 Hz timeline (decimation phase, cue positions) by repeating
      // the last sample; a longer gap is a new stream and is not filled.
      const uint32_t missing = d - 1u;
      if (missing <= kMaxGapFill) {
        for (uint32_t k = 0; k < missing; ++k) {
          push(last_sample_, static_cast<uint16_t>(last_seq_ + 1u + k), t_rx_cycles);
        }
        gap_samples_ += missing;
        printf("# gap %lu frames before seq=%u filled\n", static_cast<unsigned long>(missing),
               static_cast<unsigned>(f.seq));
      } else {
        printf("# seq jump %u -> %u not filled\n", static_cast<unsigned>(last_seq_), static_cast<unsigned>(f.seq));
      }
    }
  }
  have_seq_ = true;
  last_seq_ = f.seq;
  std::memcpy(last_sample_, f.data, sizeof(last_sample_));
  push(f.data, f.seq, t_rx_cycles);
}

void App::push(const int16_t* sample, uint16_t seq, uint32_t t_rx_cycles) {
  // Resolve cues against this frame: the cue sample is the data frame whose seq the CUE named
  // (normally the very next one).
  const uint32_t index = pre_.n250();
  for (uint32_t i = 0; i < n_cues_; ++i) {
    PendingCue& c = cues_[i];
    if (!c.resolved) {
      const int16_t rel = static_cast<int16_t>(static_cast<uint16_t>(c.seq - seq));
      if (rel <= 0 && static_cast<int32_t>(index) + rel >= 0) {
        c.cue_n250 = static_cast<uint32_t>(static_cast<int32_t>(index) + rel);
        c.resolved = true;
      }
    }
  }

  const bool new_125 = pre_.push_sample(sample);
  if (!new_125) {
    return;
  }

  for (uint32_t i = 0; i < n_cues_;) {
    const PendingCue& c = cues_[i];
    if (c.resolved && pre_.window_ready(c.cue_n250)) {
      const uint32_t end = Preprocess::window_end_125(c.cue_n250);
      const PendingCue done = c;
      drop_cue(i);
      if (!pre_.window_available(end)) {
        printf("# trial %ld window no longer buffered\n", static_cast<long>(done.trial));
        continue;
      }
      infer_window(end, done.trial, done.label, seq, t_rx_cycles);
      continue;
    }
    ++i;
  }

  if (free_run_ && pre_.n125() >= Preprocess::kWindow && pre_.n125() % kFreeRunStride == 0) {
    infer_window(pre_.n125(), -1, -1, seq, t_rx_cycles);
  }
}

void App::drop_cue(uint32_t i) {
  for (uint32_t k = i + 1; k < n_cues_; ++k) {
    cues_[k - 1] = cues_[k];
  }
  --n_cues_;
}

void App::infer_window(uint32_t end_125, int32_t trial, int32_t label, uint16_t seq, uint32_t t_rx_cycles) {
  if (subject_ < 1) {
    printf("# no SET_SUBJECT before the window of trial %ld; skipped\n", static_cast<long>(trial));
    return;
  }
  if (!pre_.make_window_at(subject_, end_125, window_)) {
    printf("# trial %ld window unavailable\n", static_cast<long>(trial));
    return;
  }
  InferenceRecord r{};
  InferStats stats{};
  if (!infer_->run(window_, r.logits, &stats)) {
    printf("# trial %ld inference failed\n", static_cast<long>(trial));
    return;
  }
  const uint32_t now_ms = clock_ms();
  r.t_ms = now_ms;
  r.seq = seq;
  r.trial = trial;
  r.subject = subject_;
  r.raw_cls = argmax2(r.logits);
  r.conf = softmax_conf2(r.logits, r.raw_cls);
  r.motion = (imu_ok_ && gate_ != nullptr && gate_->motion(now_ms)) ? 1 : 0;
  r.cls = r.motion ? -1 : r.raw_cls;
  r.label = label;
  r.imu_ok = imu_ok_;
  leds_show(r.motion ? LedColor::kRed : (r.raw_cls == 0 ? LedColor::kGreen : LedColor::kBlue), now_ms);

  char line[256];
  r.lat_us = cycles_to_us(cycle_counter() - t_rx_cycles);
  const size_t len = format_json(line, sizeof(line), r);
  if (len > 0) {
    fputs(line, stdout);
    fflush(stdout);
  }
  ++inferences_;
  if (hook_ != nullptr) {
    hook_(hook_ctx_, r, window_, stats);
  }
}

}  // namespace edge
