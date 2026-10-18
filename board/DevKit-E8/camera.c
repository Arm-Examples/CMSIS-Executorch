/*---------------------------------------------------------------------------
 * MT9M114 camera on the DevKit-E8 MIPI-CSI2 connector, through the CPI driver.
 *
 * Follows Alif's camera demo (Boards/Templates/.../demo_camera_mt9m114): the
 * CPI driver brings up MIPI CSI2, the DPHY and the sensor (over I2C1). The
 * sensor is configured in RTE_Device.h for 640x480 RGB565.
 *---------------------------------------------------------------------------*/

#include <stdint.h>
#include <stdio.h>

#include "RTE_Components.h"
#include CMSIS_device_header
#include "RTE_Device.h"

#include "Driver_CPI.h"

#include "camera.h"

#define CAM_W RTE_MT9M114_CAMERA_SENSOR_FRAME_WIDTH
#define CAM_H RTE_MT9M114_CAMERA_SENSOR_FRAME_HEIGHT

/* .bss.ai_pool is routed to the bulk SRAM by the linker scripts of this layer */
static uint8_t cam_buf[CAM_H * CAM_W * 2] __attribute__((section(".bss.ai_pool"), aligned(32)));

extern ARM_DRIVER_CPI Driver_CPI;

static volatile uint32_t cam_events;

static void camera_callback(uint32_t event)
{
    cam_events |= event;
}

int camera_init(void)
{
    int32_t r;

    r = Driver_CPI.Initialize(camera_callback);
    printf("camera: Initialize -> %ld\n", (long)r);
    if (r != ARM_DRIVER_OK) {
        return -1;
    }
    r = Driver_CPI.PowerControl(ARM_POWER_FULL);
    printf("camera: PowerControl -> %ld\n", (long)r);
    if (r != ARM_DRIVER_OK) {
        return -2;
    }
    r = Driver_CPI.Control(CPI_CONFIGURE, 0);
    printf("camera: CPI configure -> %ld\n", (long)r);
    if (r != ARM_DRIVER_OK) {
        return -3;
    }
    r = Driver_CPI.Control(CPI_CAMERA_SENSOR_CONFIGURE, 0);
    printf("camera: sensor configure -> %ld\n", (long)r);
    if (r != ARM_DRIVER_OK) {
        return -4;
    }
    r = Driver_CPI.Control(CPI_EVENTS_CONFIGURE,
                           ARM_CPI_EVENT_CAMERA_CAPTURE_STOPPED |
                               ARM_CPI_EVENT_ERR_CAMERA_INPUT_FIFO_OVERRUN |
                               ARM_CPI_EVENT_ERR_CAMERA_OUTPUT_FIFO_OVERRUN |
                               ARM_CPI_EVENT_ERR_HARDWARE | ARM_CPI_EVENT_MIPI_CSI2_ERROR);
    printf("camera: events configure -> %ld\n", (long)r);
    if (r != ARM_DRIVER_OK) {
        return -5;
    }
    printf("camera: %dx%d RGB565, buffer at 0x%08lx\n", CAM_W, CAM_H, (unsigned long)cam_buf);
    return 0;
}

int camera_capture(void)
{
    cam_events = 0;
    if (Driver_CPI.CaptureFrame(cam_buf) != ARM_DRIVER_OK) {
        return -1;
    }
    /* A frame takes ~33-66 ms; give up after ~1 s */
    uint32_t start = DWT->CYCCNT;
    while (!(cam_events & ARM_CPI_EVENT_CAMERA_CAPTURE_STOPPED)) {
        if ((DWT->CYCCNT - start) > SystemCoreClock) {
            Driver_CPI.Stop();
            return -2;
        }
    }
    SCB_InvalidateDCache_by_Addr((uint32_t *)cam_buf, (int32_t)sizeof(cam_buf));
    return (cam_events & ~ARM_CPI_EVENT_CAMERA_CAPTURE_STOPPED) ? 1 : 0;  /* 1 = frame with errors */
}

/* Non-blocking snapshots: camera_capture_start() kicks off one frame and
 * camera_capture_poll() reports when it has landed, so the caller can do other
 * work meanwhile. (Continuous CaptureVideo raised a bus error and broke the
 * NN's memory, so it is not used.) */
static int cam_busy;
volatile uint32_t g_cam_frames;   /* debugger-readable: snapshots completed */
volatile int32_t g_cam_err;       /* debugger-readable: last CaptureFrame error (0 = none) */
volatile uint32_t g_cam_errs;     /* debugger-readable: CaptureFrame failures */

int camera_capture_start(void)
{
    if (cam_busy) return 1;
    cam_events = 0;
    int32_t r = Driver_CPI.CaptureFrame(cam_buf);
    if (r != ARM_DRIVER_OK) {
        g_cam_err = r;
        ++g_cam_errs;
        return -1;
    }
    cam_busy = 1;
    return 0;
}

int camera_capture_poll(void)
{
    if (!cam_busy || !(cam_events & ARM_CPI_EVENT_CAMERA_CAPTURE_STOPPED)) return 0;
    cam_busy = 0;
    ++g_cam_frames;
    SCB_InvalidateDCache_by_Addr((uint32_t *)cam_buf, (int32_t)sizeof(cam_buf));
    return 1;
}

void camera_capture_abort(void)
{
    if (cam_busy) Driver_CPI.Stop();
    cam_busy = 0;
}

/* Power the CPI + sensor down and bring them up again (the sensor sometimes
 * stops streaming after a warm reset: frames "arrive" but stay all zeros) */
int camera_restart(void)
{
    camera_capture_abort();
    Driver_CPI.PowerControl(ARM_POWER_OFF);
    Driver_CPI.Uninitialize();
    return camera_init();
}

uint32_t camera_last_events(void)
{
    return cam_events;
}

const uint8_t *camera_frame(void)
{
    return cam_buf;
}

int camera_width(void)
{
    return CAM_W;
}

int camera_height(void)
{
    return CAM_H;
}
