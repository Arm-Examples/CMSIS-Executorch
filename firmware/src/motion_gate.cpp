// Motion gate over the accelerometer history; pure C++ (see imu.h).
#include <cmath>

#include "imu.h"

namespace edge {

void MotionGate::add(const AccelSample& s) {
  hist_[head_] = s;
  head_ = (head_ + 1) % MOTION_HISTORY;
  if (count_ < MOTION_HISTORY) {
    ++count_;
  }
}

bool MotionGate::motion(uint32_t now_ms) const {
  const AccelSample* prev = nullptr;
  // Oldest to newest.
  for (uint32_t i = 0; i < count_; ++i) {
    const AccelSample& s = hist_[(head_ + MOTION_HISTORY - count_ + i) % MOTION_HISTORY];
    const uint32_t age = now_ms - s.t_ms;  // unsigned: samples from the future look ancient
    if (age > MOTION_WINDOW_MS) {
      prev = nullptr;
      continue;
    }
    const float norm = std::sqrt(s.x * s.x + s.y * s.y + s.z * s.z);
    if (std::fabs(norm - 1.0f) > MOTION_NORM_THRESHOLD_G) {
      return true;
    }
    if (prev != nullptr && s.t_ms - prev->t_ms <= MOTION_MAX_GAP_MS) {
      if (std::fabs(s.x - prev->x) > MOTION_DELTA_THRESHOLD_G ||
          std::fabs(s.y - prev->y) > MOTION_DELTA_THRESHOLD_G ||
          std::fabs(s.z - prev->z) > MOTION_DELTA_THRESHOLD_G) {
        return true;
      }
    }
    prev = &s;
  }
  return false;
}

}  // namespace edge
