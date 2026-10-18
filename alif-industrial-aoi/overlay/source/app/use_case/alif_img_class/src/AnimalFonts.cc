// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0
#include "AnimalFonts.hpp"
#include <array>
namespace {
constexpr std::array<uint8_t, 2240> Glyph(std::array<uint8_t, 7> rows)
{
    std::array<uint8_t, 2240> result{};
    for (int y = 0; y < 56; ++y) {
        for (int x = 0; x < 40; ++x) {
            result[y * 40 + x] = rows[y / 8] & (1 << (4 - x / 8)) ? 255 : 0;
        }
    }
    return result;
}
constexpr std::array<std::array<uint8_t, 2240>, 5> s_letters = {
    Glyph({17,17,10,4,4,4,4}), Glyph({31,16,16,30,16,16,31}),
    Glyph({15,16,16,14,1,1,30}), Glyph({17,25,25,21,19,19,17}),
    Glyph({14,17,17,17,17,17,14})};
constexpr std::array<uint8_t, 256> Coffee()
{
    constexpr uint16_t rows[] = {0x0900,0x1200,0x1200,0x0900,0,0x7fc0,0x7ff0,0x7c18,
                               0x7c18,0x7c18,0x7ff0,0x7fc0,0x3f80,0,0xffe0,0};
    std::array<uint8_t, 256> pixels{};
    for (int y = 0; y < 16; ++y) {
        for (int x = 0; x < 16; ++x) {
            pixels[y * 16 + x] = rows[y] & (1 << (15 - x)) ? 255 : 0;
        }
    }
    return pixels;
}
constexpr auto s_coffee = Coffee();
bool ResultDescriptor(const lv_font_t* font, lv_font_glyph_dsc_t* d, uint32_t code, uint32_t)
{
    const char* letters = "YESNO";
    unsigned index = 0;
    while (index < 5 && static_cast<uint32_t>(letters[index]) != code) { ++index; }
    if (index == 5) { return false; }
    *d = {};
    d->resolved_font = font; d->adv_w = 48; d->box_w = 40; d->box_h = 56;
    d->stride = 40; d->format = LV_FONT_GLYPH_FORMAT_A8; d->gid.index = index;
    return true;
}
const void* ResultBitmap(lv_font_glyph_dsc_t* d, lv_draw_buf_t*)
{
    return s_letters[d->gid.index].data();
}
bool CoffeeDescriptor(const lv_font_t* font, lv_font_glyph_dsc_t* d, uint32_t code, uint32_t)
{
    if (code != 0x2615) { return false; }
    *d = {};
    d->resolved_font = font; d->adv_w = 22; d->box_w = 16; d->box_h = 16;
    d->ofs_y = 4; d->stride = 16; d->format = LV_FONT_GLYPH_FORMAT_A8;
    return true;
}
const void* CoffeeBitmap(lv_font_glyph_dsc_t*, lv_draw_buf_t*) { return s_coffee.data(); }
}
namespace alif::app {
const lv_font_t* AnimalResultFont()
{
    static lv_font_t font = [] {
        lv_font_t f{}; f.get_glyph_dsc = ResultDescriptor; f.get_glyph_bitmap = ResultBitmap;
        f.line_height = 64; f.static_bitmap = 1; return f;
    }();
    return &font;
}
const lv_font_t* AnimalTextFont()
{
    static lv_font_t coffee = [] {
        lv_font_t f{}; f.get_glyph_dsc = CoffeeDescriptor; f.get_glyph_bitmap = CoffeeBitmap;
        f.line_height = 28; f.static_bitmap = 1; return f;
    }();
    static lv_font_t font = lv_font_montserrat_28;
    font.fallback = &coffee;
    return &font;
}
}
