// Motion-gate constants (README "Motion gate"). One place for every threshold.
#ifndef EDGE_EEG_MOTION_CONFIG_H_
#define EDGE_EEG_MOTION_CONFIG_H_

// The accelerometer is polled by the main loop every IMU_POLL_PERIOD_MS (100 Hz).
#define IMU_POLL_PERIOD_MS 10u

// An inference is gated (cls = -1, "motion":1) if, within the MOTION_WINDOW_MS before it,
//   max | |a| - 1 g | > MOTION_NORM_THRESHOLD_G, or
//   any axis changed by more than MOTION_DELTA_THRESHOLD_G between consecutive samples.
#define MOTION_WINDOW_MS 500u
#define MOTION_NORM_THRESHOLD_G 0.25f
#define MOTION_DELTA_THRESHOLD_G 0.25f

// Consecutive samples further apart than this (a missed poll) are not compared axis-wise.
#define MOTION_MAX_GAP_MS 50u

// Samples kept: 0.5 s at 100 Hz plus margin.
#define MOTION_HISTORY 64u

#endif  // EDGE_EEG_MOTION_CONFIG_H_
