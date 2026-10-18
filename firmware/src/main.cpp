// Edge_Deploy firmware entry (app_main, called by the board layer's main() after clocks,
// pins, UART and the Ethos-U driver are up).
//
// Normal build: READY banner, then forever: drain the UART receive ring -> App (frames,
// preprocessing, inference, JSON) -> poll the IMU at 100 Hz -> LED timeout.
// SelfTest build (-DSELFTEST, build-type SelfTest): runs selftest_main() instead, which
// replays the embedded stream of tests/generated/selftest_vectors.h and prints Test_result.

#include <cstdio>

#include "app.h"
#include "clock.h"
#include "imu.h"
#include "infer.h"
#include "leds.h"
#include "main.h"
#include "selftest.h"

#ifndef UART_BAUDRATE
#define UART_BAUDRATE 921600
#endif

namespace {

edge::Infer g_infer;
edge::MotionGate g_gate;

void print_ready() {
  printf("READY n_ch=%d model=tinyeegnet_int8 pte_bytes=%lu fs=%d fs_model=%d win=%d baud=%lu\n", N_CH,
         static_cast<unsigned long>(g_infer.pte_bytes()), FS_HZ, FS_MODEL_HZ, N_SAMPLES,
         static_cast<unsigned long>(UART_BAUDRATE));
  fflush(stdout);
}

}  // namespace

extern "C" int app_main(void) {
  edge::cycle_counter_init();

  printf("# edge_eeg firmware, params %s sha256 %.16s, cpu %lu Hz\n", PARAMS_VERSION, PARAMS_SHA256,
         static_cast<unsigned long>(edge::cpu_hz()));
  if (!g_infer.init()) {
    printf("# FATAL: model did not load\n");
    fflush(stdout);
    return 1;
  }
  printf("# model loaded: pte %lu B, load_cycles %lu\n", static_cast<unsigned long>(g_infer.pte_bytes()),
         static_cast<unsigned long>(g_infer.load_cycles()));

#if defined(SELFTEST)
  return edge::selftest_main(&g_infer);
#else
  static edge::App app(&g_infer, &g_gate);  // ~51 KB: not on the 32 KB stack
  print_ready();
  // After READY: the IMU bring-up is bounded (imu.cpp) and only prints '#' lines; without an
  // IMU the board streams with "imu":0.
  app.set_imu_ok(edge::imu_init());
  (void)edge::leds_init();

  static uint8_t rx[512];
  uint32_t reported_errors = 0;
  uint32_t reported_overruns = 0;
  uint32_t reported_tx_faults = 0;
  uint32_t last_report_ms = 0;
  for (;;) {
    (void)edge::clock_cycles();  // keeps the 64-bit time base across CYCCNT wraps
    // Arrival time of the chunk: the DWT stamp of the last UART RX interrupt that stored
    // bytes (board layer), read before the bytes so it never postdates them.
    const uint32_t t_rx = stdio_rx_cycles();
    const uint32_t n = stdio_rx_read(rx, sizeof(rx));
    if (n > 0) {
      app.on_bytes(rx, n, t_rx);
    }
    const uint32_t now_ms = edge::clock_ms();
    edge::imu_poll(now_ms, &g_gate);
    app.set_imu_ok(edge::imu_ok());
    edge::leds_update(now_ms);

    // Error counters, at most once a second (a resync counts every skipped byte).
    if (now_ms - last_report_ms >= 1000u) {
      const uint32_t overruns = stdio_rx_overruns();
      const uint32_t tx_faults = stdio_tx_faults();
      if (app.frame_errors() != reported_errors || overruns != reported_overruns ||
          tx_faults != reported_tx_faults) {
        reported_errors = app.frame_errors();
        reported_overruns = overruns;
        reported_tx_faults = tx_faults;
        printf("# frame_err %lu rx_overrun %lu tx_fault %lu\n", static_cast<unsigned long>(reported_errors),
               static_cast<unsigned long>(reported_overruns), static_cast<unsigned long>(reported_tx_faults));
      }
      last_report_ms = now_ms;
    }
  }
#endif
}
