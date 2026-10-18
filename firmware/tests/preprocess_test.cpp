// Host unit test of the firmware's pure C++ modules (make -C firmware/tests test).
//
// 1. Preprocess: the A03 int16 stream of tests/generated/preprocess_vectors.h (written by
//    host/make_vectors.py from host/pipeline_ref.py) must give the EXPECTED int8 windows
//    exactly: 0 differing elements (count and max |diff| are printed; the header's
//    PV_TOLERANCE is informational only), and each window must complete on WINDOW_LAST_N250. The header's
//    PV_COUNTS_PER_UV must match the firmware's UV_PER_LSB.
// 2. The same stream, encoded as UART frames (RESET, SET_SUBJECT, CUE before each cue frame)
//    and fed byte by byte through FrameParser, must decode to the same samples and cues; plus
//    resync after garbage.
// 3. MotionGate thresholds; 4. JSON line format.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "imu.h"
#include "json_out.h"
#include "preprocess.h"
#include "uart_rx.h"

#include "preprocess_vectors.h"

namespace {

int g_failures = 0;

#define CHECK(cond)                                                       \
  do {                                                                    \
    if (!(cond)) {                                                        \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);         \
      ++g_failures;                                                       \
    }                                                                     \
  } while (0)

int test_preprocess_windows() {
  static_assert(PV_N_CH == N_CH, "channel count");
  static_assert(PV_WIN == N_SAMPLES, "window length");
  // The stream's int16 scale (from the generated header) must be the firmware's.
  const double scale_err = std::fabs(static_cast<double>(PV_COUNTS_PER_UV) * edge::UV_PER_LSB - 1.0);
  std::printf("scale: header %.4f counts/uV, firmware UV_PER_LSB %.9f uV/count\n",
              static_cast<double>(PV_COUNTS_PER_UV), edge::UV_PER_LSB);
  CHECK(scale_err < 1e-6);
  edge::Preprocess pre;
  static int8_t win[edge::Preprocess::kWindowElems];
  uint32_t trial = 0;
  uint32_t total_mismatch = 0;
  int max_diff = 0;
  for (uint32_t i = 0; i < PV_N; ++i) {
    pre.push_sample(STREAM[i]);
    while (trial < PV_N_TRIALS && pre.window_ready(CUE_N250[trial])) {
      // The window must complete on exactly the documented sample.
      CHECK(i == WINDOW_LAST_N250[trial]);
      CHECK(edge::Preprocess::window_last_n250(CUE_N250[trial]) == WINDOW_LAST_N250[trial]);
      CHECK(pre.make_window(PV_SUBJECT, CUE_N250[trial], win));
      uint32_t mismatch = 0;
      for (uint32_t k = 0; k < edge::Preprocess::kWindowElems; ++k) {
        const int d = std::abs(static_cast<int>(win[k]) - static_cast<int>(EXPECTED[trial][k]));
        if (d != 0) {
          ++mismatch;
          if (d > max_diff) {
            max_diff = d;
          }
        }
      }
      std::printf("trial %u (session-2 index %u) label %u: window mismatches %u / %u\n", trial,
                  static_cast<unsigned>(TRIAL_INDEX[trial]), static_cast<unsigned>(LABEL[trial]), mismatch,
                  static_cast<unsigned>(edge::Preprocess::kWindowElems));
      total_mismatch += mismatch;
      ++trial;
    }
  }
  CHECK(trial == PV_N_TRIALS);
  std::printf("preprocess: %u trials, %u of %u int8 elements differ from host/pipeline_ref.py "
              "(max |diff| %d; required 0, PV_TOLERANCE %d informational)\n",
              trial, total_mismatch, static_cast<unsigned>(PV_N_TRIALS * edge::Preprocess::kWindowElems), max_diff,
              PV_TOLERANCE);
  CHECK(total_mismatch == 0);

  // Late windows: the ring holds 2 x 256 samples, so the last window is still readable
  // 256 samples later, but not after that.
  const uint32_t end = edge::Preprocess::window_end_125(CUE_N250[PV_N_TRIALS - 1]);
  CHECK(pre.window_available(end));
  CHECK(!pre.window_available(end + 1));  // not yet received
  CHECK(!pre.make_window_at(0, end, win));   // subject 0 invalid
  CHECK(!pre.make_window_at(10, end, win));  // subject 10 invalid

  // reset() clears everything: the same stream gives the same first window again.
  pre.reset();
  CHECK(pre.n250() == 0 && pre.n125() == 0);
  for (uint32_t i = 0; i <= WINDOW_LAST_N250[0]; ++i) {
    pre.push_sample(STREAM[i]);
  }
  static int8_t again[edge::Preprocess::kWindowElems];
  CHECK(pre.make_window(PV_SUBJECT, CUE_N250[0], again));
  pre.reset();
  for (uint32_t i = 0; i <= WINDOW_LAST_N250[0]; ++i) {
    pre.push_sample(STREAM[i]);
  }
  CHECK(pre.make_window(PV_SUBJECT, CUE_N250[0], win));
  CHECK(std::memcmp(win, again, sizeof(win)) == 0);
  return 0;
}

int test_uart_frames() {
  edge::FrameParser parser;
  std::vector<uint8_t> bytes;
  uint8_t buf[64];
  auto append = [&](uint32_t n) { bytes.insert(bytes.end(), buf, buf + n); };

  append(edge::encode_control_frame(buf, edge::kReset, 0, 0));
  append(edge::encode_control_frame(buf, edge::kSetSubject, PV_SUBJECT, 0));
  // Garbage between frames: the parser must resync without losing the next frame.
  const uint8_t junk[] = {0x00, 0xAA, 0x13, 0xAA, 0xAA, 0x55, 0x07};  // last: data frame with n_ch 7
  bytes.insert(bytes.end(), junk, junk + sizeof(junk));
  uint32_t trial = 0;
  for (uint32_t i = 0; i < PV_N; ++i) {
    if (trial < PV_N_TRIALS && CUE_N250[trial] == i) {
      append(edge::encode_control_frame(buf, edge::kCue, LABEL[trial], static_cast<uint16_t>(i)));
      ++trial;
    }
    append(edge::encode_data_frame(buf, static_cast<uint16_t>(i), STREAM[i]));
  }
  CHECK(edge::FrameParser::kDataFrameBytes == 49);

  edge::Frame f;
  uint32_t data = 0;
  uint32_t controls = 0;
  uint32_t cues = 0;
  bool ok = true;
  for (uint8_t b : bytes) {
    if (!parser.feed(b, &f)) {
      continue;
    }
    if (f.type == edge::FrameType::kControl) {
      ++controls;
      if (f.kind == edge::kCue) {
        ok = ok && cues < PV_N_TRIALS && f.arg == LABEL[cues] && f.seq == static_cast<uint16_t>(CUE_N250[cues]) &&
             data == CUE_N250[cues];
        ++cues;
      }
    } else {
      ok = ok && data < PV_N && f.seq == static_cast<uint16_t>(data) &&
           std::memcmp(f.data, STREAM[data], sizeof(f.data)) == 0;
      ++data;
    }
  }
  CHECK(ok);
  CHECK(data == PV_N);
  CHECK(controls == 2 + PV_N_TRIALS);
  CHECK(cues == PV_N_TRIALS);
  CHECK(parser.errors() > 0);
  std::printf("uart: %u data + %u control frames from %zu bytes, %u resync errors\n", data, controls,
              bytes.size(), parser.errors());
  return 0;
}

int test_motion_gate() {
  edge::MotionGate g;
  CHECK(!g.motion(1000));  // empty
  for (uint32_t t = 0; t <= 1000; t += 10) {
    g.add(edge::AccelSample{t, 0.02f, -0.01f, 1.0f});  // still, 1 g on z
  }
  CHECK(!g.motion(1000));
  g.add(edge::AccelSample{1010, 0.0f, 0.0f, 1.3f});  // | |a| - 1 | = 0.3 g
  CHECK(g.motion(1010));
  CHECK(g.motion(1500));
  CHECK(!g.motion(1511));  // older than 500 ms
  for (uint32_t t = 1020; t <= 2000; t += 10) {
    g.add(edge::AccelSample{t, 0.0f, 0.0f, 1.0f});
  }
  CHECK(!g.motion(2000));
  // Axis jump with |a| still ~1 g (tilt): x 0 -> 0.3, z 1 -> 0.954.
  g.add(edge::AccelSample{2010, 0.3f, 0.0f, 0.954f});
  CHECK(g.motion(2010));
  // Below both thresholds.
  edge::MotionGate h;
  for (uint32_t t = 0; t <= 600; t += 10) {
    const float x = (t % 20 == 0) ? 0.1f : -0.1f;  // 0.2 g steps
    h.add(edge::AccelSample{t, x, 0.0f, 1.0f});
  }
  CHECK(!h.motion(600));
  std::printf("motion gate: thresholds %.2f g norm, %.2f g step, %u ms window\n",
              static_cast<double>(MOTION_NORM_THRESHOLD_G), static_cast<double>(MOTION_DELTA_THRESHOLD_G),
              MOTION_WINDOW_MS);
  return 0;
}

int test_json() {
  edge::InferenceRecord r{};
  r.t_ms = 123456;
  r.seq = 3134;
  r.trial = 0;
  r.subject = 3;
  r.logits[0] = -0.623177f;
  r.logits[1] = 0.986697f;
  r.raw_cls = edge::argmax2(r.logits);
  r.conf = edge::softmax_conf2(r.logits, r.raw_cls);
  r.motion = 0;
  r.cls = r.raw_cls;
  r.lat_us = 1234;
  r.label = 1;
  r.imu_ok = true;
  char line[256];
  const size_t n = edge::format_json(line, sizeof(line), r);
  const std::string want =
      "{\"t\":123456,\"seq\":3134,\"trial\":0,\"subject\":3,\"cls\":1,\"raw_cls\":1,\"conf\":0.83,"
      "\"logits\":[-0.623177,0.986697],\"lat_us\":1234,\"motion\":0,\"label\":1}\n";
  CHECK(n == want.size());
  CHECK(want == line);
  std::printf("json: %s", line);

  r.motion = 1;
  r.cls = -1;
  r.trial = -1;
  r.label = -1;
  r.imu_ok = false;
  edge::format_json(line, sizeof(line), r);
  CHECK(std::strstr(line, "\"trial\":-1,") != nullptr);
  CHECK(std::strstr(line, "\"cls\":-1,\"raw_cls\":1,") != nullptr);
  CHECK(std::strstr(line, "\"label\":-1,\"imu\":0}\n") != nullptr);
  std::printf("json: %s", line);
  CHECK(edge::format_json(line, 16, r) == 0);  // too small
  return 0;
}

}  // namespace

int main() {
  test_preprocess_windows();
  test_uart_frames();
  test_motion_gate();
  test_json();
  if (g_failures != 0) {
    std::printf("preprocess_test: %d check(s) FAILED\n", g_failures);
    return 1;
  }
  std::printf("preprocess_test: all checks passed\n");
  return 0;
}
