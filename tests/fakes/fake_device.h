#ifndef SENTINEL_FAKE_DEVICE_H
#define SENTINEL_FAKE_DEVICE_H
#include <stdint.h>
typedef struct { uint32_t CYCCNT; } fake_dwt_t;
extern fake_dwt_t fake_dwt;
#define DWT (&fake_dwt)
extern uint32_t SystemCoreClock;
#endif
