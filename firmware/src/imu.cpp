// ICM-42670-P accelerometer on the DevKit-E8's I3C bus, with every wait bounded; stub elsewhere.
//
// Own register-level bring-up over the Alif CMSIS I3C driver (Driver_I3C, interrupt mode:
// RTE_I3C_BLOCKING_MODE_ENABLE 0, so every driver call returns at once and completion comes
// through the event callback). The sequence follows the pack's ICM42670P_IMU.c, but that
// driver spins without a timeout on RSTDAA/SETDASA, so it is not used:
//   RSTDAA (broadcast) -> SETDASA static 0x68 -> normal bus mode -> WHO_AM_I (0x75) == 0x67
//   -> PWR_MGMT0 (0x1F) = 0x03 accel low-noise, gyro off -> ACCEL_CONFIG0 (0x21) = 0x49
//   (+-4 g, 100 Hz ODR; read back). Data: ACCEL_DATA_X1 (0x0B) .. Z0, big endian, 8192 LSB/g.
// Bounded: each transfer/command waits at most kXferTimeoutMs (DWT deadline); init as a whole
// is a fixed number of such steps (< 100 ms worst case). Not bounded by us: the time inside the
// driver calls themselves (non-blocking register setup, a finite DAT slot search). On any
// failure imu_init() prints "# imu init failed" and the gate stays off ("imu":0). At run time,
// kMaxReadErrors consecutive failed reads switch the IMU off the same way.
// Untested without a board.
#include <cstdio>

#include "RTE_Components.h"
#include "clock.h"
#include "imu.h"

#if defined(RTE_Drivers_I3C)
#include "sys_utils.h"   // BIT(), used by the I3C CCC macros
#include "Driver_I3C.h"
extern "C" ARM_DRIVER_I3C Driver_I3C;
#endif

namespace edge {

namespace {
bool g_ok = false;
uint32_t g_last_poll_ms = 0;
uint32_t g_errors = 0;
uint32_t g_consecutive_errors = 0;

#if defined(RTE_Drivers_I3C)
constexpr uint8_t kStaticAddr = 0x68;
constexpr uint8_t kRegAccelData = 0x0B;
constexpr uint8_t kRegPwrMgmt0 = 0x1F;
constexpr uint8_t kRegAccelConfig0 = 0x21;
constexpr uint8_t kRegWhoAmI = 0x75;
constexpr uint8_t kWhoAmI = 0x67;
constexpr uint8_t kPwrAccelLowNoise = 0x03;
constexpr uint8_t kAccel4g100Hz = (0x2u << 5) | 0x9u;  // ACCEL_UI_FS_SEL +-4 g, ACCEL_ODR 100 Hz
constexpr float kLsbPerG = 8192.0f;
constexpr uint32_t kXferTimeoutMs = 5;
constexpr uint32_t kMaxReadErrors = 10;

ARM_DRIVER_I3C* const i3c = &Driver_I3C;
uint8_t g_addr = 0;  // dynamic address
volatile uint32_t g_event = 0;

void i3c_event(uint32_t event) {
  g_event |= event;
}

bool wait_us(uint32_t us) {
  const uint32_t start = cycle_counter();
  const uint32_t cycles = static_cast<uint32_t>((static_cast<uint64_t>(cpu_hz()) * us) / 1000000u);
  while (cycle_counter() - start < cycles) {
  }
  return true;
}

// Waits for TRANSFER_DONE / TRANSFER_ERROR with a DWT deadline.
bool wait_done() {
  const uint32_t start = cycle_counter();
  const uint32_t limit = (cpu_hz() / 1000u) * kXferTimeoutMs;
  for (;;) {
    const uint32_t ev = g_event;
    if (ev & ARM_I3C_EVENT_TRANSFER_ERROR) {
      return false;
    }
    if (ev & ARM_I3C_EVENT_TRANSFER_DONE) {
      return true;
    }
    if (cycle_counter() - start > limit) {
      return false;
    }
  }
}

bool command(ARM_I3C_CMD* cmd, bool assign_da) {
  g_event = 0;
  const int32_t r = assign_da ? i3c->MasterAssignDA(cmd) : i3c->MasterSendCommand(cmd);
  return r == ARM_DRIVER_OK && wait_done();
}

bool write_reg(uint8_t reg, uint8_t value) {
  __attribute__((aligned(4))) uint8_t buf[4] = {reg, value, 0, 0};
  g_event = 0;
  return i3c->MasterTransmit(g_addr, buf, 2) == ARM_DRIVER_OK && wait_done();
}

bool read_regs(uint8_t reg, uint8_t* out, uint16_t n) {
  __attribute__((aligned(4))) uint8_t r[4] = {reg, 0, 0, 0};
  g_event = 0;
  if (i3c->MasterTransmit(g_addr, r, 1) != ARM_DRIVER_OK || !wait_done()) {
    return false;
  }
  g_event = 0;
  return i3c->MasterReceive(g_addr, out, n) == ARM_DRIVER_OK && wait_done();
}

const char* bring_up() {
  if (i3c->Initialize(i3c_event) != ARM_DRIVER_OK) return "i3c Initialize";
  if (i3c->PowerControl(ARM_POWER_FULL) != ARM_DRIVER_OK) return "i3c PowerControl";
  if (i3c->Control(I3C_MASTER_INIT, 0) != ARM_DRIVER_OK) return "i3c master init";
  (void)i3c->Control(I3C_MASTER_SET_BUS_MODE, I3C_BUS_SLOW_MODE);
  (void)i3c->Control(I3C_MASTER_SETUP_HOT_JOIN_ACCEPTANCE, 0);
  (void)i3c->Control(I3C_MASTER_SETUP_MR_ACCEPTANCE, 0);

  ARM_I3C_CMD rst = {};
  rst.rw = 0;
  rst.cmd_id = I3C_CCC_RSTDAA(true);
  rst.len = 0;
  rst.addr = 0;
  if (!command(&rst, false)) return "RSTDAA timeout/error";

  ARM_I3C_CMD dasa = {};
  dasa.rw = 0;
  dasa.cmd_id = I3C_CCC_SETDASA;
  dasa.len = 1;
  dasa.addr = kStaticAddr;
  if (!command(&dasa, true)) return "SETDASA timeout/error (no IMU at 0x68?)";
  if (i3c->GetSlaveDynAddr(kStaticAddr, &g_addr) != ARM_DRIVER_OK) return "no dynamic address";
  (void)i3c->Control(I3C_MASTER_SET_BUS_MODE, I3C_BUS_NORMAL_MODE);

  __attribute__((aligned(4))) uint8_t v[4] = {0, 0, 0, 0};
  if (!read_regs(kRegWhoAmI, v, 1)) return "WHO_AM_I read timeout/error";
  if (v[0] != kWhoAmI) return "WHO_AM_I mismatch";
  if (!write_reg(kRegPwrMgmt0, kPwrAccelLowNoise)) return "PWR_MGMT0 write";
  wait_us(1000);  // >= 200 us after enabling the accelerometer before further writes
  if (!write_reg(kRegAccelConfig0, kAccel4g100Hz)) return "ACCEL_CONFIG0 write";
  if (!read_regs(kRegAccelConfig0, v, 1) || v[0] != kAccel4g100Hz) return "ACCEL_CONFIG0 read-back";
  return nullptr;
}
#endif
}  // namespace

bool imu_init() {
#if defined(RTE_Drivers_I3C)
  const uint32_t start = cycle_counter();
  const char* err = bring_up();
  const uint32_t us = cycles_to_us(cycle_counter() - start);
  if (err != nullptr) {
    printf("# imu init failed: %s (%lu us); motion gate off\n", err, static_cast<unsigned long>(us));
    g_ok = false;
    return false;
  }
  printf("# imu ok who_am_i=0x67 icm42670p dyn_addr=0x%02x range=4g odr=100Hz poll=%uHz init_us=%lu\n",
         static_cast<unsigned>(g_addr), static_cast<unsigned>(1000u / IMU_POLL_PERIOD_MS),
         static_cast<unsigned long>(us));
  g_ok = true;
  return true;
#else
  printf("# imu none on this target; motion gate off\n");
  g_ok = false;
  return false;
#endif
}

void imu_poll(uint32_t now_ms, MotionGate* gate) {
  if (!g_ok || now_ms - g_last_poll_ms < IMU_POLL_PERIOD_MS) {
    return;
  }
  g_last_poll_ms = now_ms;
#if defined(RTE_Drivers_I3C)
  __attribute__((aligned(4))) uint8_t b[8] = {};
  if (!read_regs(kRegAccelData, b, 6)) {
    ++g_errors;
    if (++g_consecutive_errors >= kMaxReadErrors) {
      g_ok = false;
      printf("# imu read failed %lu times in a row; motion gate off\n", static_cast<unsigned long>(kMaxReadErrors));
    }
    return;
  }
  g_consecutive_errors = 0;
  auto axis = [&](int i) {
    return static_cast<float>(static_cast<int16_t>(static_cast<uint16_t>((b[i] << 8) | b[i + 1]))) / kLsbPerG;
  };
  gate->add(AccelSample{now_ms, axis(0), axis(2), axis(4)});
#else
  (void)gate;
#endif
}

bool imu_ok() {
  return g_ok;
}

uint32_t imu_read_errors() {
  return g_errors;
}

}  // namespace edge
