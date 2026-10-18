#ifndef CAMERA_H_
#define CAMERA_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Starts the MT9M114 (MIPI CSI2) through the CPI driver. Returns 0 on success. */
int camera_init(void);

/* Captures one frame into the camera buffer. 0 = ok, 1 = frame with error events,
 * negative = failed / timed out. Needs the DWT cycle counter running. */
int camera_capture(void);

uint32_t camera_last_events(void);

/* Non-blocking snapshot: start (0 = started, 1 = one already running),
 * poll (1 = a new frame is in camera_frame()), abort. */
int camera_capture_start(void);
int camera_capture_poll(void);
void camera_capture_abort(void);

/* Full power-down and re-init of the CPI and the sensor. 0 on success. */
int camera_restart(void);

/* RGB565 frame, width x height pixels, 2 bytes each */
const uint8_t *camera_frame(void);
int camera_width(void);
int camera_height(void);

#ifdef __cplusplus
}
#endif

#endif /* CAMERA_H_ */
