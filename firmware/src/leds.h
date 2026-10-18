// Result LED: green = left, blue = right, red = gated, off = idle; each colour held 500 ms.
// DevKit-E8: RGB LED 0 (LEDRGB0_R/G/B from the board's Conductor pin configuration) through
// the Alif GPIO driver (Driver_IO). Other targets: no LED, colour changes are only logged
// by the caller's JSON line. Hardware path untested without a board.
#ifndef EDGE_EEG_LEDS_H_
#define EDGE_EEG_LEDS_H_

#include <cstdint>

namespace edge {

enum class LedColor : uint8_t { kOff, kGreen, kBlue, kRed };

constexpr uint32_t kLedHoldMs = 500;

bool leds_init();                                  // prints a '#' line
void leds_show(LedColor color, uint32_t now_ms);   // light colour, (re)start the hold timer
void leds_update(uint32_t now_ms);                 // turn off after the hold time

}  // namespace edge

#endif  // EDGE_EEG_LEDS_H_
