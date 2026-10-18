// Time base from the DWT cycle counter (CPU clock, SystemCoreClock Hz). The 32-bit counter
// wraps after 2^32 cycles (10.7 s at 400 MHz); clock_cycles() extends it to 64 bits and must
// be called at least once per wrap period (the main loop calls it every iteration).
#ifndef EDGE_EEG_CLOCK_H_
#define EDGE_EEG_CLOCK_H_

#include <cstdint>

namespace edge {

void cycle_counter_init();
uint32_t cycle_counter();         // raw 32-bit DWT CYCCNT
uint64_t clock_cycles();          // extended, since cycle_counter_init()
uint32_t clock_ms();              // ms since cycle_counter_init()
uint32_t cycles_to_us(uint32_t cycles);
uint32_t cpu_hz();

}  // namespace edge

#endif  // EDGE_EEG_CLOCK_H_
