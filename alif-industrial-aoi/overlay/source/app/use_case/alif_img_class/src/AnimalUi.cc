// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0
#include "AnimalUi.hpp"
#include "AnimalTouch.hpp"
#include "AnimalFonts.hpp"
#include "board_utils.h"
#include "lv_port.h"
#include <atomic>
namespace {
std::atomic<uint32_t> s_request{0};
std::atomic<uint32_t> s_pointer{0};
uint32_t s_claimed = 0;
lv_indev_t* s_input;
lv_obj_t* s_labels[5];
lv_obj_t* s_header;
lv_obj_t* s_time;
lv_obj_t* s_led;
lv_obj_t* s_image;
lv_timer_t* s_resultTimer;
// AppKit E8 LEDRGB0: red P12_3, green P7_4; HIGH is ON.
// BOARD_UTILS_Init has already configured these outputs.
// Called with the LVGL lock held, or from its timer handler.
void ClearResultLeds()
{
    BOARD_LED1_Control(BOARD_LED_STATE_LOW);
    BOARD_LED2_Control(BOARD_LED_STATE_LOW);
    if (s_resultTimer) {
        lv_timer_pause(s_resultTimer);
    }
}
void Request(alif::app::AnimalMode mode)
{
    ClearResultLeds();
    uint32_t old = s_request.load();
    s_request.store(((old + 4) & ~3U) | static_cast<uint32_t>(mode));
    if (mode == alif::app::AnimalMode::STOP) {
        for (int i = 0; i < 5; ++i) {
            lv_label_set_text(alif::app::AnimalUiLabelObject(i), "");
        }
        lv_obj_set_style_text_color(alif::app::AnimalUiLabelObject(0), lv_color_white(), 0);
        lv_label_set_text(alif::app::AnimalUiLabelObject(0), "Break time! \xe2\x98\x95");
        lv_obj_set_style_text_font(alif::app::AnimalUiLabelObject(0),
                                  alif::app::AnimalTextFont(), 0);
    }
}
void Click(lv_event_t* event)
{
    Request(static_cast<alif::app::AnimalMode>(reinterpret_cast<uintptr_t>(
        lv_event_get_user_data(event))));
}
void Read(lv_indev_t*, lv_indev_data_t* data)
{
    uint32_t pointer = s_pointer.load();
    data->point.x = pointer & 0x7fff;
    data->point.y = (pointer >> 15) & 0x7fff;
    data->state = pointer >> 31 ? LV_INDEV_STATE_PRESSED : LV_INDEV_STATE_RELEASED;
}
}
namespace alif::app {
void AnimalUiInit(const void* pixels, size_t bytes, int width, int height, unsigned short zoom)
{
    (void)zoom;
    lv_port_disp_init();
    bool touch = AnimalTouchInit();
    uint32_t lock = lv_port_lock();
    lv_obj_t* screen = lv_screen_active();
    lv_obj_set_layout(screen, LV_LAYOUT_NONE);
    lv_obj_set_style_pad_all(screen, 0, 0);
    lv_obj_remove_flag(screen, LV_OBJ_FLAG_SCROLLABLE);
    const int screenWidth = lv_display_get_horizontal_resolution(nullptr);
    const int screenHeight = lv_display_get_vertical_resolution(nullptr);
    lv_obj_t* preview = lv_obj_create(screen);
    lv_obj_remove_style_all(preview);
    lv_obj_set_size(preview, screenWidth, screenHeight);
    lv_obj_set_pos(preview, 0, 0);
    lv_obj_remove_flag(preview, LV_OBJ_FLAG_SCROLLABLE);
    lv_obj_remove_flag(preview, LV_OBJ_FLAG_CLICKABLE);
    s_image = lv_image_create(preview);
    static lv_image_dsc_t image{};
    image.header.magic = LV_IMAGE_HEADER_MAGIC;
    image.header.cf = LV_COLOR_FORMAT_NATIVE;
    image.header.w = width; image.header.h = height;
    image.header.stride = width * LV_COLOR_DEPTH / 8;
    image.data = static_cast<const uint8_t*>(pixels); image.data_size = bytes;
    lv_image_set_src(s_image, &image);
    // Cover the screen with one aspect-preserving transform; the parent clips
    // its centered excess. No 480x800 scaled camera buffer is allocated.
    unsigned scaleX = (screenWidth * 256 + width - 1) / width;
    unsigned scaleY = (screenHeight * 256 + height - 1) / height;
    lv_image_set_scale(s_image, scaleX > scaleY ? scaleX : scaleY);
    lv_image_set_antialias(s_image, false);
    lv_obj_remove_flag(s_image, LV_OBJ_FLAG_CLICKABLE);
    lv_obj_center(s_image);
    s_header = lv_label_create(screen);
    lv_obj_set_style_text_font(s_header, &lv_font_montserrat_28, 0);
    lv_obj_set_style_text_color(s_header, lv_color_white(), 0);
    lv_obj_set_style_bg_color(s_header, lv_color_black(), 0);
    lv_obj_set_style_bg_opa(s_header, LV_OPA_70, 0);
    lv_obj_set_style_pad_all(s_header, 6, 0);
    lv_obj_align(s_header, LV_ALIGN_TOP_MID, 0, 8);
    for (int i = 0; i < 5; ++i) {
        s_labels[i] = lv_label_create(screen);
        lv_obj_set_style_text_align(s_labels[i], LV_TEXT_ALIGN_CENTER, 0);
        lv_obj_set_style_text_color(s_labels[i], lv_color_white(), 0);
        lv_obj_set_style_bg_color(s_labels[i], lv_color_black(), 0);
        lv_obj_set_style_bg_opa(s_labels[i], LV_OPA_70, 0);
        lv_obj_set_style_pad_all(s_labels[i], 8, 0);
        lv_obj_set_width(s_labels[i], LV_PCT(92));
        lv_obj_align(s_labels[i], LV_ALIGN_TOP_MID, 0, 65);
        if (i) { lv_obj_add_flag(s_labels[i], LV_OBJ_FLAG_HIDDEN); }
    }
    s_time = lv_label_create(screen);
    lv_obj_set_style_text_font(s_time, &lv_font_montserrat_16, 0);
    lv_obj_set_style_text_color(s_time, lv_color_white(), 0);
    lv_obj_align(s_time, LV_ALIGN_BOTTOM_MID, 0, -86);
    lv_label_set_text(s_time, "");
    s_led = lv_led_create(screen);
    lv_obj_set_size(s_led, 12, 12);
    lv_obj_align(s_led, LV_ALIGN_TOP_RIGHT, -12, 16);
    lv_led_off(s_led);
    s_resultTimer = lv_timer_create([](lv_timer_t*) { ClearResultLeds(); }, 3000, nullptr);
    lv_timer_pause(s_resultTimer);
    // Created last, above the preview and all noninteractive labels.
    lv_obj_t* controls = lv_obj_create(screen);
    lv_obj_set_size(controls, screenWidth, 78);
    lv_obj_align(controls, LV_ALIGN_BOTTOM_MID, 0, 0);
    lv_obj_set_style_pad_all(controls, 6, 0);
    lv_obj_set_style_bg_opa(controls, LV_OPA_80, 0);
    lv_obj_remove_flag(controls, LV_OBJ_FLAG_SCROLLABLE);
    const char* names[] = {"SINGLE", "LIVE", "STOP"};
    const AnimalMode modes[] = {AnimalMode::SINGLE, AnimalMode::LIVE, AnimalMode::STOP};
    for (int i = 0; i < 3; ++i) {
        lv_obj_t* button = lv_button_create(controls);
        lv_obj_set_size(button, LV_PCT(31), 54);
        lv_obj_align(button, LV_ALIGN_LEFT_MID, i * (screenWidth - 20) / 3, 0);
        lv_obj_add_event_cb(button, Click, LV_EVENT_CLICKED,
                           reinterpret_cast<void*>(static_cast<uintptr_t>(modes[i])));
        lv_obj_t* label = lv_label_create(button);
        lv_obj_set_style_text_font(label, &lv_font_montserrat_24, 0);
        lv_label_set_text(label, names[i]);
        lv_obj_center(label);
    }
    s_input = lv_indev_create();
    lv_indev_set_type(s_input, LV_INDEV_TYPE_POINTER);
    lv_indev_set_read_cb(s_input, Read);
    lv_label_set_text(AnimalUiHeaderObject(), "Animal Detector");
    if (!touch) {
        lv_obj_set_width(s_time, LV_PCT(92));
        lv_obj_set_style_text_align(s_time, LV_TEXT_ALIGN_CENTER, 0);
        lv_label_set_text(s_time, AnimalTouchStatus());
    }
    Request(AnimalMode::STOP);
    lv_port_unlock(lock);
}
lv_obj_t* AnimalUiImageObject() { return s_image; }
lv_obj_t* AnimalUiLabelObject(int index) { return s_labels[index]; }
lv_obj_t* AnimalUiHeaderObject() { return s_header; }
lv_obj_t* AnimalUiTimeObject() { return s_time; }
lv_obj_t* AnimalUiLEDObject() { return s_led; }
void AnimalUiService()
{
    int32_t x = 0, y = 0;
    int state = AnimalTouchPoll(x, y);
    if (state >= 0) {
        uint32_t point = state ? (0x80000000U | (static_cast<uint32_t>(y) << 15) | x)
                               : (s_pointer.load() & 0x7fffffffU);
        s_pointer.store(point);
        uint32_t lock = lv_port_lock();
        lv_indev_read(s_input);
        lv_port_unlock(lock);
    }
}
uint32_t AnimalUiRequest() { return s_request.load(); }
bool AnimalUiClaim(uint32_t request)
{
    auto mode = static_cast<AnimalMode>(request & 3);
    if (mode == AnimalMode::LIVE) {
        ClearResultLeds();
        return true;
    }
    if (mode == AnimalMode::SINGLE && s_claimed != request) {
        ClearResultLeds();
        s_claimed = request; return true;
    }
    return false;
}
bool AnimalUiCurrent(uint32_t request) { return request == s_request.load(); }
void AnimalUiSingle()
{
    uint32_t lock = lv_port_lock(); Request(AnimalMode::SINGLE); lv_port_unlock(lock);
}
void AnimalUiResult(bool animal)
{
    ClearResultLeds();
    if (animal) {
        BOARD_LED2_Control(BOARD_LED_STATE_HIGH);
    } else {
        BOARD_LED1_Control(BOARD_LED_STATE_HIGH);
    }
    lv_timer_reset(s_resultTimer);
    lv_timer_resume(s_resultTimer);
    lv_obj_set_style_text_font(s_labels[0], &lv_font_montserrat_32, 0);
    lv_obj_set_style_text_color(s_labels[0], lv_color_hex(animal ? 0x20e060 : 0xff4040), 0);
    lv_label_set_text(s_labels[0], animal ? "YES" : "NO");
}
}
