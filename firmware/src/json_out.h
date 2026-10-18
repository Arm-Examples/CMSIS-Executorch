// Board -> host result line (README "UART contract"), one per inference, fixed field order:
// {"t":ms,"seq":n,"trial":k,"subject":s,"cls":c,"raw_cls":r,"conf":0.00,"logits":[l0,l1],
//  "lat_us":n,"motion":m,"label":y}   plus ,"imu":0 when the IMU is not running.
// conf has 2 decimals, logits 6. Pure C++ (host-testable).
#ifndef EDGE_EEG_JSON_OUT_H_
#define EDGE_EEG_JSON_OUT_H_

#include <cstddef>
#include <cstdint>

namespace edge {

struct InferenceRecord {
  uint32_t t_ms;       // ms since boot
  uint16_t seq;        // seq of the window's last data frame
  int32_t trial;       // cue index since RESET (0-based), -1 = free run
  int32_t subject;     // 1..9
  int32_t cls;         // raw_cls, or -1 when gated
  int32_t raw_cls;     // argmax(logits)
  float conf;          // softmax probability of raw_cls
  float logits[2];
  uint32_t lat_us;     // RX interrupt that stored the window's last frame -> just before the line
                       // is formatted (excludes snprintf + queueing, a few us); chunk-granular
  int32_t motion;      // 0/1
  int32_t label;       // CUE arg (0, 1) or -1
  bool imu_ok;         // false adds "imu":0
};

// argmax (ties -> class 0, like np.argmax) and the softmax probability of that class.
int32_t argmax2(const float logits[2]);
float softmax_conf2(const float logits[2], int32_t cls);

// Writes the line including the trailing '\n' and a NUL; returns the length without the NUL,
// or 0 if size is too small.
size_t format_json(char* buf, size_t size, const InferenceRecord& r);

}  // namespace edge

#endif  // EDGE_EEG_JSON_OUT_H_
