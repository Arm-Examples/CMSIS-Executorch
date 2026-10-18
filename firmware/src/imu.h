// On-board IMU (TDK ICM-42670-P on the DevKit-E8's I3C bus) and the motion gate.
//
// MotionGate is pure C++ (host-testable): it keeps the last MOTION_HISTORY accelerometer
// samples and answers "was there motion in the last MOTION_WINDOW_MS".
// imu_init / imu_poll talk to the sensor over the Alif CMSIS I3C driver with bounded waits
// (imu.cpp). On targets without I3C (the Corstone-320 FVP) imu_init returns false and the
// gate never fires.
// Hardware path untested without a board.
#ifndef EDGE_EEG_IMU_H_
#define EDGE_EEG_IMU_H_

#include <cstdint>

#include "motion_config.h"

namespace edge {

struct AccelSample {
  uint32_t t_ms;
  float x, y, z;   // g
};

class MotionGate {
 public:
  MotionGate() { reset(); }
  void reset() {
    count_ = 0;
    head_ = 0;
  }
  void add(const AccelSample& s);
  // Motion within [now_ms - MOTION_WINDOW_MS, now_ms].
  bool motion(uint32_t now_ms) const;
  uint32_t count() const { return count_; }

 private:
  AccelSample hist_[MOTION_HISTORY];
  uint32_t count_;   // valid samples (<= MOTION_HISTORY)
  uint32_t head_;    // next write slot
};

// Board glue. imu_init prints a '#' diagnostics line either way and returns within ~100 ms
// even with no sensor. imu_ok() turns false if reads keep failing.
bool imu_init();
// Reads one accelerometer sample into the gate when IMU_POLL_PERIOD_MS has passed.
void imu_poll(uint32_t now_ms, MotionGate* gate);
bool imu_ok();
uint32_t imu_read_errors();

}  // namespace edge

#endif  // EDGE_EEG_IMU_H_
