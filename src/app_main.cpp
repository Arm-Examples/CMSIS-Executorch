// Copyright 2026 Arm Limited and/or its affiliates.
// SPDX-License-Identifier: Apache-2.0
//
// pico-faces on Ethos-U85: a face generator built on the ExecuTorch runtime.
// The AI layer (create_ai_layer.py) embeds one program with two methods
// delegated to the NPU:
//
//   dit_step(z, c) -> v      the diffusion transformer's velocity for one step
//   decode(z)      -> img    the latent decoder, 128x128 RGB in [-1, 1]
//
// and model_params.h with the sampling schedule and the conditioning vectors
// c(t_k, class). This file runs the rectified-flow Euler loop (with optional
// classifier-free guidance) around those methods, converts the image to 8-bit
// RGB and prints a CRC and an ASCII preview. With APP_INTERACTIVE (the
// DevKit-E8 board layer) it then serves pico-faces' serial protocol, so that
// pico-faces' viewer (viewer/view_serial.py) can request and display images:
//
//   host -> "G <seed> [k_steps] [class] [w]\n"      "I\n" prints an info line
//   dev  -> "RFI2" u32 seed u16 w u16 h u16 ch u16 class | w*h*ch bytes (HWC)
//           | u32 crc32 | u32 gen_ms
//
// The board layer provides main(), stdout/stdin and the Ethos-U driver init and
// then calls app_main().
//
// EmbeddedModule (arm_embedded_module.hpp) manages program loading, method
// memory and execution, like ExecuTorch's Module class does on POSIX hosts.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <vector>

#include "RTE_Components.h"
#include CMSIS_device_header

#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/platform/platform.h>
#include <executorch/runtime/platform/runtime.h>

#include "arm_embedded_module.hpp"
#include "model_params.h"
#include "model_pte.h"

#if defined(ETHOSU_ARCH)
#include "ethosu_driver.h"
#include "pmu_ethosu.h"
#endif

using arm::embedded::EmbeddedModule;
using executorch::aten::DimOrderType;
using executorch::aten::ScalarType;
using executorch::aten::SizesType;
using executorch::aten::Tensor;
using executorch::aten::TensorImpl;
using executorch::extension::BufferDataLoader;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::MethodMeta;
using executorch::runtime::Result;
using executorch::runtime::Span;
using executorch::runtime::TensorInfo;

extern "C" int stdout_putchar(int ch);  // CMSIS-Compiler retarget hooks
extern "C" int stdin_getchar(void);
#ifdef APP_BUTTONS
// Board layer (board/DevKit-E8/buttons.c): SW2 joystick presses since the last
// call (bit mask), and a console read that does not block (-1 when nothing is
// waiting).
extern "C" unsigned board_buttons(void);
extern "C" int board_console_poll(void);
constexpr unsigned kButtonLeft = 1u << 0;
constexpr unsigned kButtonRight = 1u << 1;
#endif
#ifdef APP_DISPLAY
// Board layer (board/DevKit-E8/display.c): show an 8-bit RGB image on the
// board's display. Returns 0 on success.
extern "C" int board_display_image(const uint8_t* rgb, int width, int height, int channels);
#endif
#ifdef APP_RESULT_DIR
// Board layer (board/Corstone-320/retarget_stdio.c): write a file on the host.
extern "C" int board_save_file(const char* path, const void* data, size_t n);
#endif

namespace {

// Pool sizes and placement are board-overridable: a board layer that cannot
// fit 8 MB of .bss in its default RAM sets APP_*_POOL_SIZE and, via
// APP_POOL_SECTION, the linker section its linker script routes to a larger
// (NPU-accessible) memory.
#ifndef APP_METHOD_POOL_SIZE
#define APP_METHOD_POOL_SIZE (4 * 1024 * 1024)
#endif
#ifndef APP_TEMP_POOL_SIZE
#define APP_TEMP_POOL_SIZE (4 * 1024 * 1024)  // Ethos-U scratch is drawn from here.
#endif
#ifdef APP_POOL_SECTION
#define APP_POOL_ATTRIBUTES __attribute__((section(APP_POOL_SECTION)))
#else
#define APP_POOL_ATTRIBUTES
#endif

// The boot demo: seed 3, 4 steps, class 1 (female, smiling), guidance 4.
#ifndef APP_DEMO_SEED
#define APP_DEMO_SEED 3
#endif
#ifndef APP_DEMO_STEPS
#define APP_DEMO_STEPS 4
#endif
#ifndef APP_DEMO_CLASS
#define APP_DEMO_CLASS 1
#endif
#ifndef APP_DEMO_W
#define APP_DEMO_W 4.0f
#endif

constexpr size_t kMethodPoolSize = APP_METHOD_POOL_SIZE;
constexpr size_t kTempPoolSize = APP_TEMP_POOL_SIZE;
constexpr size_t kLatent = PF_LATENT_CH * PF_LATENT_HW * PF_LATENT_HW;
constexpr size_t kImage = PF_IMG_HW * PF_IMG_HW * PF_IMG_CH;
constexpr const char* kDitStep = "dit_step";
constexpr const char* kDecode = "decode";

alignas(16) uint8_t g_method_pool[kMethodPoolSize] APP_POOL_ATTRIBUTES;
alignas(16) uint8_t g_temp_pool[kTempPoolSize] APP_POOL_ATTRIBUTES;

// Generation state; static so the 32 kB main stack stays small.
float g_z[kLatent];
float g_v[kLatent];
float g_vn[kLatent];
float g_c[PF_COND_DIM];
uint8_t g_img[kImage];

// The method inputs wrap g_z and g_c; EmbeddedModule copies them into the
// method's own input buffers on every call.
std::array<SizesType, 4> g_latent_sizes{1, PF_LATENT_CH, PF_LATENT_HW, PF_LATENT_HW};
std::array<DimOrderType, 4> g_latent_order{0, 1, 2, 3};
std::array<SizesType, 2> g_cond_sizes{1, PF_COND_DIM};
std::array<DimOrderType, 2> g_cond_order{0, 1};
TensorImpl g_z_impl(ScalarType::Float, g_latent_sizes.size(), g_latent_sizes.data(), g_z, g_latent_order.data());
TensorImpl g_c_impl(ScalarType::Float, g_cond_sizes.size(), g_cond_sizes.data(), g_c, g_cond_order.data());

// Per-method NPU counters from the Ethos-U PMU: cycles the NPU was clocked
// (CCNT), cycles it was active, cycles its MAC array was active, and data beats
// read on the two AXI ports (on the Ensemble E8 all traffic, weights from MRAM
// included, arrives on port 0).
struct NpuCounters {
  uint64_t cycles;
  uint64_t active;
  uint64_t mac_active;
  uint64_t axi0_read_beats;
  uint64_t axi1_read_beats;
};

struct Timing {
  uint32_t dit_ms;
  uint32_t dit_calls;
  uint32_t decode_ms;
  uint32_t total_ms;
  NpuCounters dit_npu;
  NpuCounters decode_npu;
};

// ---------------------------------------------------------------------------
// Ethos-U PMU: the driver instance belongs to the board layer; the driver's
// reserve/release API hands it out between inferences.

#if defined(ETHOSU_ARCH)
constexpr uint32_t kPmuCounters = ETHOSU_PMU_CCNT_Msk | ETHOSU_PMU_CNT1_Msk | ETHOSU_PMU_CNT2_Msk |
                                  ETHOSU_PMU_CNT3_Msk | ETHOSU_PMU_CNT4_Msk;

void npu_pmu_start() {
  ethosu_driver* drv = ethosu_reserve_driver();
  ETHOSU_PMU_Enable(drv);
  ETHOSU_PMU_Set_EVTYPER(drv, 0, ETHOSU_PMU_NPU_ACTIVE);
  ETHOSU_PMU_Set_EVTYPER(drv, 1, ETHOSU_PMU_MAC_ACTIVE);
  ETHOSU_PMU_Set_EVTYPER(drv, 2, ETHOSU_PMU_EXT0_RD_DATA_BEAT_RECEIVED);
  ETHOSU_PMU_Set_EVTYPER(drv, 3, ETHOSU_PMU_EXT1_RD_DATA_BEAT_RECEIVED);
  ETHOSU_PMU_CYCCNT_Reset(drv);
  ETHOSU_PMU_EVCNTR_ALL_Reset(drv);
  ETHOSU_PMU_CNTR_Enable(drv, kPmuCounters);
  ethosu_release_driver(drv);
}

void npu_pmu_stop(NpuCounters& c) {
  ethosu_driver* drv = ethosu_reserve_driver();
  ETHOSU_PMU_CNTR_Disable(drv, kPmuCounters);
  c.cycles += ETHOSU_PMU_Get_CCNTR(drv);
  c.active += ETHOSU_PMU_Get_EVCNTR(drv, 0);
  c.mac_active += ETHOSU_PMU_Get_EVCNTR(drv, 1);
  c.axi0_read_beats += ETHOSU_PMU_Get_EVCNTR(drv, 2);
  c.axi1_read_beats += ETHOSU_PMU_Get_EVCNTR(drv, 3);
  ethosu_release_driver(drv);
}
#else
void npu_pmu_start() {}
void npu_pmu_stop(NpuCounters&) {}
#endif

// ---------------------------------------------------------------------------
// Millisecond tick from SysTick, for per-phase timing (the DWT cycle counter
// does not count on every device). Not meaningful on the FVP.

volatile uint32_t g_ms_ticks;

void ticks_init() {
  SysTick_Config(SystemCoreClock / 1000u);
}

inline uint32_t ms_now() { return g_ms_ticks; }

// ---------------------------------------------------------------------------
// PCG32 + CLT-12 gaussian, as pico-faces' engine/src/prng.c. The noise is
// filled in CHW order (upstream fills in token order), so a seed gives other
// faces than on the Pico, but the same as model/verify_export.py on the host.

uint64_t g_pcg_state;

void pcg32_seed(uint64_t seed) {
  const uint64_t mult = 6364136223846793005ULL, inc = 1442695040888963407ULL;
  g_pcg_state = 0;
  g_pcg_state = g_pcg_state * mult + inc;
  g_pcg_state += seed;
  g_pcg_state = g_pcg_state * mult + inc;
}

uint32_t pcg32_next() {
  const uint64_t mult = 6364136223846793005ULL, inc = 1442695040888963407ULL;
  uint64_t x = g_pcg_state;
  unsigned count = static_cast<unsigned>(x >> 59);
  g_pcg_state = x * mult + inc;
  x ^= x >> 18;
  uint32_t out = static_cast<uint32_t>(x >> 27);
  return (out >> count) | (out << ((32 - count) & 31));
}

float gauss() {
  int32_t acc = 0;
  for (int j = 0; j < 12; ++j) {
    acc += static_cast<int32_t>(pcg32_next() >> 20);
  }
  return static_cast<float>(acc - 24576) / 4096.0f;
}

// ---------------------------------------------------------------------------
// CRC-32 (IEEE, reflected), the same as zlib / Python's binascii.crc32.

uint32_t g_crc_table[256];

void crc32_init() {
  for (uint32_t i = 0; i < 256; ++i) {
    uint32_t c = i;
    for (int k = 0; k < 8; ++k) {
      c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
    }
    g_crc_table[i] = c;
  }
}

uint32_t crc32(const uint8_t* data, size_t n) {
  uint32_t c = 0xFFFFFFFFu;
  for (size_t i = 0; i < n; ++i) {
    c = g_crc_table[(c ^ data[i]) & 0xFFu] ^ (c >> 8);
  }
  return c ^ 0xFFFFFFFFu;
}

// ---------------------------------------------------------------------------
// Methods

bool check_shape(const TensorInfo& info, const char* what, std::initializer_list<int32_t> expect) {
  Span<const int32_t> sizes = info.sizes();
  bool ok = sizes.size() == expect.size();
  size_t i = 0;
  for (int32_t d : expect) {
    ok = ok && sizes[i++] == d;
  }
  if (!ok) {
    printf("%s: unexpected shape [", what);
    for (size_t j = 0; j < sizes.size(); ++j) {
      printf("%s%d", j ? "," : "", static_cast<int>(sizes[j]));
    }
    printf("], model_params.h says [");
    i = 0;
    for (int32_t d : expect) {
      printf("%s%d", i++ ? "," : "", static_cast<int>(d));
    }
    printf("]\n");
  }
  return ok;
}

// Loads a method up front (EmbeddedModule would do it on the first call) and
// prints what it needs.
bool load_method(EmbeddedModule& module, const char* name) {
  Result<MethodMeta> meta = module.method_meta(name);
  if (!meta.ok()) {
    printf("method %s not found in the program (err=%u)\n", name, static_cast<unsigned>(meta.error()));
    return false;
  }
  Error err = module.load_method(name);
  if (err != Error::Ok) {
    printf("load_method(%s) failed (err=%u)\n", name, static_cast<unsigned>(err));
    return false;
  }
  size_t planned_bytes = 0;
  for (size_t i = 0; i < meta->num_memory_planned_buffers(); ++i) {
    planned_bytes += static_cast<size_t>(meta->memory_planned_buffer_size(i).get());
  }
  printf("  %-9s %u input(s), %u output(s), %u planned byte(s)\n", name,
         static_cast<unsigned>(meta->num_inputs()), static_cast<unsigned>(meta->num_outputs()),
         static_cast<unsigned>(planned_bytes));
  return true;
}

// Executes a method; returns its first output as floats, or nullptr on failure.
// The data stays valid until the method runs again.
const float* run(EmbeddedModule& module, const char* name, const std::vector<EValue>& inputs) {
  Result<std::vector<EValue>> outputs = module.execute(name, inputs);
  if (!outputs.ok()) {
    printf("%s: execute failed (err=%u)\n", name, static_cast<unsigned>(outputs.error()));
    return nullptr;
  }
  return outputs->at(0).toTensor().const_data_ptr<float>();
}

// v = dit_step(g_z, g_c)
bool dit_step(EmbeddedModule& module, float* v, Timing& tm) {
  uint32_t t0 = ms_now();
  npu_pmu_start();
  const float* out = run(module, kDitStep, {EValue(Tensor(&g_z_impl)), EValue(Tensor(&g_c_impl))});
  if (out == nullptr) {
    return false;
  }
  npu_pmu_stop(tm.dit_npu);
  tm.dit_ms += ms_now() - t0;
  tm.dit_calls++;
  memcpy(v, out, kLatent * sizeof(float));
  return true;
}

// img (HWC uint8) = decode(g_z)
bool decode(EmbeddedModule& module, uint8_t* img, Timing& tm) {
  uint32_t t0 = ms_now();
  npu_pmu_start();
  const float* out = run(module, kDecode, {EValue(Tensor(&g_z_impl))});  // (1, C, H, W) in [-1, 1]
  if (out == nullptr) {
    return false;
  }
  npu_pmu_stop(tm.decode_npu);
  tm.decode_ms += ms_now() - t0;
  const size_t plane = PF_IMG_HW * PF_IMG_HW;
  for (size_t p = 0; p < plane; ++p) {
    for (size_t ch = 0; ch < PF_IMG_CH; ++ch) {
      float x = out[ch * plane + p];
      x = x < -1.0f ? -1.0f : (x > 1.0f ? 1.0f : x);
      img[p * PF_IMG_CH + ch] = static_cast<uint8_t>(lrintf((x + 1.0f) * 127.5f));
    }
  }
  return true;
}

// The rectified-flow Euler sampler: z1 ~ N(0, 1) at t = 1 down to t = 0, then
// decode. k_steps (8, 4, 2 or 1) strides through the PF_K_MAX schedule points.
bool generate(EmbeddedModule& module, uint32_t seed, int k_steps, int cls, float w, uint8_t* img, Timing& tm) {
  memset(&tm, 0, sizeof(tm));
  uint32_t t_start = ms_now();
  if (k_steps != 1 && k_steps != 2 && k_steps != 4 && k_steps != 8) {
    k_steps = 4;
  }
  if (cls < 0 || cls >= PF_N_COND) {
    cls = PF_NULL_CLASS;
  }
  const bool guided = cls != PF_NULL_CLASS && w > 0.0f;
  const int stride = PF_K_MAX / k_steps;

  pcg32_seed(seed);
  for (size_t i = 0; i < kLatent; ++i) {
    g_z[i] = gauss();
  }

  for (int i = 0; i < k_steps; ++i) {
    const int k = i * stride;
    const float t = pf_schedule[k];
    const float t_next = (i + 1 < k_steps) ? pf_schedule[k + stride] : 0.0f;
    const float dt = t - t_next;

    memcpy(g_c, pf_cond[cls][k], sizeof(g_c));
    if (!dit_step(module, g_v, tm)) {
      return false;
    }
    if (guided) {
      memcpy(g_c, pf_cond[PF_NULL_CLASS][k], sizeof(g_c));
      if (!dit_step(module, g_vn, tm)) {
        return false;
      }
      for (size_t j = 0; j < kLatent; ++j) {  // v = v_null + w * (v_cond - v_null)
        g_v[j] = g_vn[j] + w * (g_v[j] - g_vn[j]);
      }
    }
    for (size_t j = 0; j < kLatent; ++j) {
      g_z[j] -= dt * g_v[j];
    }
  }

  if (!decode(module, img, tm)) {
    return false;
  }
  tm.total_ms = ms_now() - t_start;
#ifdef APP_DISPLAY
  board_display_image(img, PF_IMG_HW, PF_IMG_HW, PF_IMG_CH);
#endif
  return true;
}

// 32x16 characters from the luminance of 4x8 pixel blocks.
void ascii_preview(const uint8_t* img) {
  static const char ramp[] = " .:-=+*#%@";
  const int cols = 32, rows = 16;
  const int bw = PF_IMG_HW / cols, bh = PF_IMG_HW / rows;
  for (int r = 0; r < rows; ++r) {
    char line[cols + 1];
    for (int c = 0; c < cols; ++c) {
      uint32_t sum = 0;
      for (int y = r * bh; y < (r + 1) * bh; ++y) {
        for (int x = c * bw; x < (c + 1) * bw; ++x) {
          const uint8_t* px = img + (y * PF_IMG_HW + x) * PF_IMG_CH;
          sum += PF_IMG_CH == 3 ? (299u * px[0] + 587u * px[1] + 114u * px[2]) / 1000u : px[0];
        }
      }
      unsigned lum = sum / (bw * bh);
      line[c] = ramp[(lum * (sizeof(ramp) - 2)) / 255u];
    }
    line[cols] = 0;
    printf("  |%s|\n", line);
  }
}

void print_npu(const char* name, const NpuCounters& c, uint32_t calls, uint64_t macs_per_call) {
  if (c.cycles == 0) {
    return;
  }
  const uint64_t macs = macs_per_call * calls;
  // 16-byte AXI beats on the Ethos-U85
  const unsigned kb0 = static_cast<unsigned>(c.axi0_read_beats * 16 / 1024);
  const unsigned kb1 = static_cast<unsigned>(c.axi1_read_beats * 16 / 1024);
  printf("  %-9s NPU %u kcycles, active %u%%, MAC active %u%%, %u MAC/cycle, "
         "read %u kB on AXI0 + %u kB on AXI1\n",
         name, static_cast<unsigned>(c.cycles / 1000), static_cast<unsigned>(c.active * 100 / c.cycles),
         static_cast<unsigned>(c.mac_active * 100 / c.cycles), static_cast<unsigned>(macs / c.cycles), kb0, kb1);
}

void print_timing(const Timing& tm) {
  printf("  dit_step: %u call(s), %u ms total (%u ms each)\n", static_cast<unsigned>(tm.dit_calls),
         static_cast<unsigned>(tm.dit_ms),
         static_cast<unsigned>(tm.dit_calls ? tm.dit_ms / tm.dit_calls : 0));
  printf("  decode:   %u ms\n", static_cast<unsigned>(tm.decode_ms));
  printf("  total:    %u ms at %u MHz (wall clock; not meaningful on the FVP)\n",
         static_cast<unsigned>(tm.total_ms), static_cast<unsigned>(SystemCoreClock / 1000000u));
  print_npu("dit_step:", tm.dit_npu, tm.dit_calls, PF_MACS_DIT_STEP);
  print_npu("decode:", tm.decode_npu, 1, PF_MACS_DECODE);
}

#if defined(APP_INTERACTIVE) || defined(APP_BUTTONS)
// ---------------------------------------------------------------------------
// After the boot demo (board layers that define APP_INTERACTIVE or APP_BUTTONS)

// Raw output: the image frames bypass stdio's text handling.
void uart_write(const void* data, size_t n) {
  fflush(stdout);
  const uint8_t* p = static_cast<const uint8_t*>(data);
  for (size_t i = 0; i < n; ++i) {
    stdout_putchar(p[i]);
  }
}

void put_u32(uint32_t v) {
  uint8_t b[4] = {static_cast<uint8_t>(v), static_cast<uint8_t>(v >> 8),
                  static_cast<uint8_t>(v >> 16), static_cast<uint8_t>(v >> 24)};
  uart_write(b, 4);
}

void put_u16(uint16_t v) {
  uint8_t b[2] = {static_cast<uint8_t>(v), static_cast<uint8_t>(v >> 8)};
  uart_write(b, 2);
}

void send_frame(uint32_t seed, int cls, const uint8_t* img, uint32_t gen_ms) {
  uart_write("RFI2", 4);
  put_u32(seed);
  put_u16(PF_IMG_HW);
  put_u16(PF_IMG_HW);
  put_u16(PF_IMG_CH);
  put_u16(static_cast<uint16_t>(cls));
  uart_write(img, kImage);
  put_u32(crc32(img, kImage));
  put_u32(gen_ms);
}

// pico-faces' firmware conventions: class defaults to seed % PF_N_COND, and
// without class and w the guidance cycles through {plain, pf_cfg_w...}.
float default_w(uint32_t seed) {
  const int w_idx = static_cast<int>(seed % (PF_N_CFG_W + 1)) - 1;
  return w_idx >= 0 ? pf_cfg_w[w_idx] : 0.0f;
}

#ifdef APP_BUTTONS
// Every joystick press picks the next seed, with class and guidance as for a
// bare "G <seed>".
uint32_t g_next_seed = APP_DEMO_SEED + 1;
uint32_t g_button_images = 0;
bool g_continuous = false;

void generate_next(EmbeddedModule& module) {
  const uint32_t seed = g_next_seed++;
  const int cls = static_cast<int>(seed % PF_N_COND);
  const float w = default_w(seed);
  Timing tm;
  if (!generate(module, seed, APP_DEMO_STEPS, cls, w, g_img, tm)) {
    printf("ERR generate\n");
    return;
  }
  g_button_images++;
  printf("Image %u: seed %u, class %d, w %.1f, %u ms%s\n", static_cast<unsigned>(g_button_images),
         static_cast<unsigned>(seed), cls, static_cast<double>(w), static_cast<unsigned>(tm.total_ms),
         g_continuous ? " (continuous)" : "");
}
#endif

// A console character without blocking when the board layer offers a poll
// (the retarget's stdin_getchar() waits for one, which would stall the joystick).
int console_getchar() {
#ifdef APP_BUTTONS
  return board_console_poll();
#else
  return stdin_getchar();
#endif
}

// "G <seed> [k_steps] [class] [w]": generate and send one frame.
void serve_request(EmbeddedModule& module, const char* line) {
  char *e1, *e2, *e3, *e4;
  uint32_t seed = static_cast<uint32_t>(strtoul(line + 1, &e1, 0));
  int k_steps = static_cast<int>(strtol(e1, &e2, 0));
  if (!k_steps) {
    k_steps = 4;
  }
  long cv = strtol(e2, &e3, 0);
  int cls = (e3 != e2) ? static_cast<int>(cv) : static_cast<int>(seed % PF_N_COND);
  float w = 0.0f;
  long wv = strtol(e3, &e4, 0);
  if (e4 != e3) {
    w = static_cast<float>(wv);
  } else if (e3 == e2) {
    w = default_w(seed);
  }
  Timing tm;
  if (!generate(module, seed, k_steps, cls, w, g_img, tm)) {
    printf("ERR generate\n");
    return;
  }
  if (cls < 0 || cls >= PF_N_COND) {
    cls = PF_NULL_CLASS;
  }
  // Timing as text before the binary frame; the viewer syncs on the magic.
  printf("Generated: seed %u, %u steps, class %u, w %.1f\n", static_cast<unsigned>(seed),
         static_cast<unsigned>(k_steps), static_cast<unsigned>(cls), static_cast<double>(w));
  print_timing(tm);
  send_frame(seed, cls, g_img, tm.total_ms);
}

// Main loop after the boot demo: the serial protocol (APP_INTERACTIVE) and the
// SW2 joystick (APP_BUTTONS: left = one new image, right = start or stop
// continuous generation).
[[noreturn]] void serve(EmbeddedModule& module) {
#ifdef APP_INTERACTIVE
  printf("Interactive: send \"G <seed> [k_steps] [class] [w]\" (viewer/view_serial.py) or \"I\"\n");
#endif
#ifdef APP_BUTTONS
  printf("Joystick: left = one new image, right = start/stop continuous generation\n");
  (void)board_buttons();  // drop presses latched during boot
#endif
  fflush(stdout);
  char line[64];
  int n = 0;
  for (;;) {
#ifdef APP_BUTTONS
    const unsigned pressed = board_buttons();
    if (pressed & kButtonRight) {
      g_continuous = !g_continuous;
      printf("Continuous generation %s\n", g_continuous ? "started" : "stopped");
    }
    if ((pressed & kButtonLeft) || g_continuous) {
      generate_next(module);
    }
#endif
    int ch = console_getchar();
    if (ch < 0) {
      continue;
    }
    if (ch != '\n' && ch != '\r') {
      if (n < static_cast<int>(sizeof line) - 1) {
        line[n++] = static_cast<char>(ch);
      }
      continue;
    }
    line[n] = 0;
    n = 0;
#ifdef APP_INTERACTIVE
    if (line[0] == 'G') {
      serve_request(module, line);
    } else if (line[0] == 'I') {
      printf("pico-faces-executorch %s %s K=%u latent=%ux%ux%u img=%ux%ux%u cond=%u pte=%lu\n",
             PF_VARIANT, PF_QUANT, static_cast<unsigned>(PF_K_MAX), PF_LATENT_CH, PF_LATENT_HW,
             PF_LATENT_HW, PF_IMG_HW, PF_IMG_HW, PF_IMG_CH, static_cast<unsigned>(PF_N_COND),
             model_pte_size);
      fflush(stdout);
    }
#endif
  }
}
#endif

}  // namespace

extern "C" void SysTick_Handler(void) { g_ms_ticks++; }

// ExecuTorch's log sink (the runtime's default prints nothing): with
// ET_LOG_ENABLED the runtime's error messages reach the console, e.g. why
// load_method failed.
extern "C" void et_pal_emit_log_message(et_timestamp_t, et_pal_log_level_t level, const char* filename,
                                        const char*, size_t line, const char* message, size_t) {
  printf("[ET %c] %s:%u %s\n", static_cast<char>(level), filename, static_cast<unsigned>(line), message);
}

extern "C" int app_main(void) {
  executorch::runtime::runtime_init();
  ticks_init();
  crc32_init();

  printf("ExecuTorch pico-faces (%s, %s): %lu byte program\n", PF_VARIANT, PF_QUANT, model_pte_size);

  static EmbeddedModule module(
      model_pte, model_pte_size,
      std::make_unique<BufferDataLoader>(model_pte, model_pte_size),
      std::make_unique<MemoryAllocator>(kMethodPoolSize, g_method_pool),
      std::make_unique<MemoryAllocator>(kTempPoolSize, g_temp_pool));

  printf("Methods:\n");
  if (!load_method(module, kDitStep) || !load_method(module, kDecode)) {
    return 1;
  }
  {
    MethodMeta dit = module.method_meta(kDitStep).get();
    MethodMeta dec = module.method_meta(kDecode).get();
    if (!check_shape(dit.input_tensor_meta(0).get(), "dit_step input z", {1, PF_LATENT_CH, PF_LATENT_HW, PF_LATENT_HW}) ||
        !check_shape(dit.input_tensor_meta(1).get(), "dit_step input c", {1, PF_COND_DIM}) ||
        !check_shape(dit.output_tensor_meta(0).get(), "dit_step output v", {1, PF_LATENT_CH, PF_LATENT_HW, PF_LATENT_HW}) ||
        !check_shape(dec.input_tensor_meta(0).get(), "decode input z", {1, PF_LATENT_CH, PF_LATENT_HW, PF_LATENT_HW}) ||
        !check_shape(dec.output_tensor_meta(0).get(), "decode output img", {1, PF_IMG_CH, PF_IMG_HW, PF_IMG_HW})) {
      return 1;
    }
  }

  // Boot demo: one image with fixed parameters. The CRC identifies the image:
  // the same program gives the same CRC on every run and on the FVP.
  Timing tm;
  printf("Generating: seed %u, %u steps, class %u, w %.1f\n", static_cast<unsigned>(APP_DEMO_SEED),
         static_cast<unsigned>(APP_DEMO_STEPS), static_cast<unsigned>(APP_DEMO_CLASS),
         static_cast<double>(APP_DEMO_W));
  fflush(stdout);
  if (!generate(module, APP_DEMO_SEED, APP_DEMO_STEPS, APP_DEMO_CLASS, APP_DEMO_W, g_img, tm)) {
    return 1;
  }
  print_timing(tm);
  printf("Image: %ux%ux%u, CRC32 %08x\n", PF_IMG_HW, PF_IMG_HW, PF_IMG_CH,
         static_cast<unsigned>(crc32(g_img, kImage)));
  ascii_preview(g_img);
#ifdef APP_RESULT_DIR
  // Simulation: hand the image and the measurements to the host through
  // semihosting (model/verify_export.py --compare reads the image).
  {
    static char text[256];
    int n = snprintf(text, sizeof(text),
                     "seed %u steps %u class %u w %.1f crc32 %08x dit_cycles %u decode_cycles %u\n",
                     static_cast<unsigned>(APP_DEMO_SEED), static_cast<unsigned>(APP_DEMO_STEPS),
                     static_cast<unsigned>(APP_DEMO_CLASS), static_cast<double>(APP_DEMO_W),
                     static_cast<unsigned>(crc32(g_img, kImage)), static_cast<unsigned>(tm.dit_npu.cycles),
                     static_cast<unsigned>(tm.decode_npu.cycles));
    bool ok = board_save_file(APP_RESULT_DIR "/fvp_image.bin", g_img, kImage) == 0 &&
              board_save_file(APP_RESULT_DIR "/fvp_result.txt", text, n > 0 ? static_cast<size_t>(n) : 0) == 0;
    printf("Result files in " APP_RESULT_DIR ": %s\n", ok ? "written" : "FAILED");
  }
#endif
  printf("Test_result: PASS\n");
  printf("\x04");  // EOT: the end marker for a console log
  fflush(stdout);

#if defined(APP_INTERACTIVE) || defined(APP_BUTTONS)
  serve(module);
#endif
  return 0;
}
