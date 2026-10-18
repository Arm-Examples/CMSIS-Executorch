// Result LED; see leds.h.
#include "leds.h"

#include <cstdio>

#include "RTE_Components.h"

#if defined(RTE_Drivers_IO) && __has_include("board_config.h")
#define EDGE_HAS_RGB_LED 1
#include "Driver_IO.h"
#include "board_config.h"
#else
#define EDGE_HAS_RGB_LED 0
#endif

// The DevKit-E8 LEDs light with the GPIO driven high (the pack's Blinky example turns them
// on with GPIO_PIN_OUTPUT_STATE_HIGH).
#ifndef LED_ON_STATE
#define LED_ON_STATE GPIO_PIN_OUTPUT_STATE_HIGH
#define LED_OFF_STATE GPIO_PIN_OUTPUT_STATE_LOW
#endif

#if EDGE_HAS_RGB_LED
extern "C" ARM_DRIVER_GPIO ARM_Driver_GPIO_(BOARD_LEDRGB0_R_GPIO_PORT);
extern "C" ARM_DRIVER_GPIO ARM_Driver_GPIO_(BOARD_LEDRGB0_G_GPIO_PORT);
extern "C" ARM_DRIVER_GPIO ARM_Driver_GPIO_(BOARD_LEDRGB0_B_GPIO_PORT);
#endif

namespace edge {

namespace {
bool g_ok = false;
bool g_lit = false;
uint32_t g_on_since_ms = 0;

#if EDGE_HAS_RGB_LED
struct Pin {
  ARM_DRIVER_GPIO* drv;
  uint8_t pin;
};

// Index 0 red, 1 green, 2 blue.
const Pin kPins[3] = {
    {&ARM_Driver_GPIO_(BOARD_LEDRGB0_R_GPIO_PORT), BOARD_LEDRGB0_R_GPIO_PIN},
    {&ARM_Driver_GPIO_(BOARD_LEDRGB0_G_GPIO_PORT), BOARD_LEDRGB0_G_GPIO_PIN},
    {&ARM_Driver_GPIO_(BOARD_LEDRGB0_B_GPIO_PORT), BOARD_LEDRGB0_B_GPIO_PIN},
};

void set_rgb(bool r, bool g, bool b) {
  const bool on[3] = {r, g, b};
  for (int i = 0; i < 3; ++i) {
    (void)kPins[i].drv->SetValue(kPins[i].pin, on[i] ? LED_ON_STATE : LED_OFF_STATE);
  }
}
#endif
}  // namespace

bool leds_init() {
#if EDGE_HAS_RGB_LED
  for (const Pin& p : kPins) {
    if (p.drv->Initialize(p.pin, nullptr) != ARM_DRIVER_OK ||
        p.drv->PowerControl(p.pin, ARM_POWER_FULL) != ARM_DRIVER_OK ||
        p.drv->SetDirection(p.pin, GPIO_PIN_DIRECTION_OUTPUT) != ARM_DRIVER_OK) {
      printf("# led init failed (RGB LED 0); results on UART only\n");
      return false;
    }
  }
  g_ok = true;
  set_rgb(false, false, false);
  printf("# led ok rgb0 green=left blue=right red=gated hold=%lums\n", static_cast<unsigned long>(kLedHoldMs));
  return true;
#else
  printf("# led none on this target; results on UART only\n");
  return false;
#endif
}

void leds_show(LedColor color, uint32_t now_ms) {
  g_on_since_ms = now_ms;
  g_lit = color != LedColor::kOff;
  if (!g_ok) {
    return;
  }
#if EDGE_HAS_RGB_LED
  set_rgb(color == LedColor::kRed, color == LedColor::kGreen, color == LedColor::kBlue);
#endif
}

void leds_update(uint32_t now_ms) {
  if (g_lit && now_ms - g_on_since_ms >= kLedHoldMs) {
    leds_show(LedColor::kOff, now_ms);
  }
}

}  // namespace edge
