// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "lvgl.h"
#include <cstddef>
#include <cstdint>
namespace alif::app {
enum class AnimalMode : uint32_t { STOP, SINGLE, LIVE };
void AnimalUiInit(const void* pixels, size_t bytes, int width, int height, unsigned short zoom);
lv_obj_t* AnimalUiImageObject();
lv_obj_t* AnimalUiLabelObject(int index);
lv_obj_t* AnimalUiHeaderObject();
lv_obj_t* AnimalUiTimeObject();
lv_obj_t* AnimalUiLEDObject();
void AnimalUiService();
uint32_t AnimalUiRequest();
bool AnimalUiClaim(uint32_t request);
bool AnimalUiCurrent(uint32_t request);
void AnimalUiSingle();
void AnimalUiResult(bool animal);
}
