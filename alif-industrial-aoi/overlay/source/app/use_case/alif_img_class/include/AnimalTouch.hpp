// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
namespace alif::app {
bool AnimalTouchInit();
const char* AnimalTouchStatus();
// -1: no new sample; 0: release; 1: press. Foreground only (shared camera I2C).
int AnimalTouchPoll(int32_t& x, int32_t& y);
}
