// DWT-based time base; see clock.h.
#include "clock.h"

#include "RTE_Components.h"
#include CMSIS_device_header

namespace edge {

namespace {
uint32_t g_last = 0;
uint64_t g_high = 0;
}  // namespace

void cycle_counter_init() {
  DCB->DEMCR |= DCB_DEMCR_TRCENA_Msk;
  DWT->CYCCNT = 0;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
  g_last = 0;
  g_high = 0;
}

uint32_t cycle_counter() {
  return DWT->CYCCNT;
}

uint64_t clock_cycles() {
  const uint32_t now = DWT->CYCCNT;
  if (now < g_last) {
    g_high += (1ull << 32);
  }
  g_last = now;
  return g_high | now;
}

uint32_t cpu_hz() {
  return SystemCoreClock != 0 ? SystemCoreClock : 1;
}

uint32_t clock_ms() {
  return static_cast<uint32_t>(clock_cycles() / (cpu_hz() / 1000u));
}

uint32_t cycles_to_us(uint32_t cycles) {
  return static_cast<uint32_t>((static_cast<uint64_t>(cycles) * 1000000u) / cpu_hz());
}

}  // namespace edge
