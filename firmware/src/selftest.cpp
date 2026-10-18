// Built-in self-test; see selftest.h.
#include "selftest.h"

#if defined(SELFTEST)

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "app.h"
#include "clock.h"
#include "selftest_vectors.h"

namespace edge {

namespace {

struct TrialResult {
  bool seen;
  uint32_t mismatches;
  int max_diff;
  int32_t argmax;
  float logits[2];
  float logit_err;
  InferStats stats;
  uint32_t lat_us;
  uint16_t seq;
};

struct Ctx {
  TrialResult r[PV_N_TRIALS];
  uint32_t unexpected;
};

void on_inference(void* p, const InferenceRecord& rec, const int8_t* window, const InferStats& stats) {
  Ctx* ctx = static_cast<Ctx*>(p);
  if (rec.trial < 0 || rec.trial >= static_cast<int32_t>(PV_N_TRIALS)) {
    ++ctx->unexpected;
    return;
  }
  TrialResult& t = ctx->r[rec.trial];
  t.seen = true;
  t.mismatches = 0;
  t.max_diff = 0;
  for (uint32_t k = 0; k < PV_N_CH * PV_WIN; ++k) {
    const int d = std::abs(static_cast<int>(window[k]) - static_cast<int>(EXPECTED[rec.trial][k]));
    if (d != 0) {
      ++t.mismatches;
      if (d > t.max_diff) {
        t.max_diff = d;
      }
    }
  }
  t.argmax = rec.raw_cls;
  t.logits[0] = rec.logits[0];
  t.logits[1] = rec.logits[1];
  t.logit_err = std::fmax(std::fabs(rec.logits[0] - BOARD_LOGITS[rec.trial][0]),
                          std::fabs(rec.logits[1] - BOARD_LOGITS[rec.trial][1]));
  t.stats = stats;
  t.lat_us = rec.lat_us;
  t.seq = rec.seq;
}

// Cycles of the CPU-side preprocessing alone: per 250 Hz sample, and per window (EA + quantize).
void benchmark_preprocess() {
  static Preprocess pre;
  static int8_t win[Preprocess::kWindowElems];
  pre.reset();
  const uint32_t n = 2 * Preprocess::kRing;
  uint32_t t0 = cycle_counter();
  for (uint32_t i = 0; i < n; ++i) {
    pre.push_sample(STREAM[i]);
  }
  const uint32_t per_sample = (cycle_counter() - t0) / n;
  t0 = cycle_counter();
  const bool ok = pre.make_latest_window(PV_SUBJECT, win);
  const uint32_t per_window = cycle_counter() - t0;
  printf("bench preprocess_cycles_per_sample=%lu window_cycles=%lu (EA 22x22x256 + quantize)%s\n",
         static_cast<unsigned long>(per_sample), static_cast<unsigned long>(per_window), ok ? "" : " FAILED");
}

}  // namespace

int selftest_main(Infer* infer) {
  static Ctx ctx;
  static App app(infer, nullptr);
  app.set_hook(on_inference, &ctx);

  printf("SELFTEST stream A%02u session 2: %lu samples (%.2f s at 250 Hz), %u trials, window %ux%u, "
         "%.2f counts/uV\n",
         static_cast<unsigned>(PV_SUBJECT), static_cast<unsigned long>(PV_N), static_cast<double>(PV_N) / 250.0,
         static_cast<unsigned>(PV_N_TRIALS), static_cast<unsigned>(PV_N_CH), static_cast<unsigned>(PV_WIN),
         static_cast<double>(PV_COUNTS_PER_UV));

  // RESET, SET_SUBJECT, then the stream with a CUE frame right before each cue sample,
  // through the same byte parser as the UART path.
  uint8_t buf[64];
  uint32_t len = encode_control_frame(buf, kReset, 0, 0);
  app.on_bytes(buf, len, cycle_counter());
  len = encode_control_frame(buf, kSetSubject, static_cast<uint8_t>(PV_SUBJECT), 0);
  app.on_bytes(buf, len, cycle_counter());

  uint32_t next = 0;
  const uint32_t t_start = cycle_counter();
  for (uint32_t i = 0; i < PV_N; ++i) {
    if (next < PV_N_TRIALS && CUE_N250[next] == i) {
      len = encode_control_frame(buf, kCue, LABEL[next], static_cast<uint16_t>(i));
      app.on_bytes(buf, len, cycle_counter());
      ++next;
    }
    len = encode_data_frame(buf, static_cast<uint16_t>(i), STREAM[i]);
    app.on_bytes(buf, len, cycle_counter());
  }
  const uint32_t stream_cycles = cycle_counter() - t_start;

  // The embedded stream's int16 scale must be the firmware's (UV_PER_LSB).
  const bool scale_ok = std::fabs(static_cast<double>(PV_COUNTS_PER_UV) * UV_PER_LSB - 1.0) < 1e-6;
  if (!scale_ok) {
    printf("stream scale %.4f counts/uV does not match UV_PER_LSB %.9f FAIL\n",
           static_cast<double>(PV_COUNTS_PER_UV), UV_PER_LSB);
  }
  bool pass = scale_ok && ctx.unexpected == 0 && app.frame_errors() == 0 && app.gap_samples() == 0;
  uint32_t total_mismatch = 0;
  float max_logit_err = 0.0f;
  for (uint32_t k = 0; k < PV_N_TRIALS; ++k) {
    const TrialResult& t = ctx.r[k];
    if (!t.seen) {
      printf("trial %lu idx %u: NO INFERENCE FAIL\n", static_cast<unsigned long>(k), static_cast<unsigned>(TRIAL_INDEX[k]));
      pass = false;
      continue;
    }
    const bool seq_ok = t.seq == static_cast<uint16_t>(WINDOW_LAST_N250[k]);
    // PASS: the int8 window equals the reference exactly (0 differing elements; PV_TOLERANCE
    // is informational only), and the .pte's argmax equals the host model's (BOARD_ARGMAX).
    const bool ok = t.mismatches == 0 && t.argmax == BOARD_ARGMAX[k] && seq_ok;
    pass = pass && ok;
    total_mismatch += t.mismatches;
    max_logit_err = std::fmax(max_logit_err, t.logit_err);
    printf("trial %lu idx %u label %u: window_mismatch %lu/%u max_diff %d seq %u%s argmax %ld board_exp %d offline %d "
           "logits %.6f %.6f ref %.6f %.6f err %.6f npu_cycles %lu cpu_cycles %lu lat_us %lu %s\n",
           static_cast<unsigned long>(k), static_cast<unsigned>(TRIAL_INDEX[k]), static_cast<unsigned>(LABEL[k]),
           static_cast<unsigned long>(t.mismatches), static_cast<unsigned>(PV_N_CH * PV_WIN), t.max_diff,
           static_cast<unsigned>(t.seq), seq_ok ? "" : " (expected seq differs)", static_cast<long>(t.argmax),
           static_cast<int>(BOARD_ARGMAX[k]), static_cast<int>(EXPECTED_ARGMAX[k]), static_cast<double>(t.logits[0]), static_cast<double>(t.logits[1]),
           static_cast<double>(BOARD_LOGITS[k][0]), static_cast<double>(BOARD_LOGITS[k][1]),
           static_cast<double>(t.logit_err), static_cast<unsigned long>(t.stats.npu_cycles),
           static_cast<unsigned long>(t.stats.cpu_cycles), static_cast<unsigned long>(t.lat_us),
           ok ? "PASS" : "FAIL");
  }
  printf("summary: inferences %lu window_mismatch_total %lu (required 0; PV_TOLERANCE %d informational) max_logit_err %.6f frame_errors %lu gap_samples %lu "
         "stream_cycles %lu method_pool %lu temp_pool %lu pte %lu\n",
         static_cast<unsigned long>(app.inferences()), static_cast<unsigned long>(total_mismatch), PV_TOLERANCE,
         static_cast<double>(max_logit_err), static_cast<unsigned long>(app.frame_errors()),
         static_cast<unsigned long>(app.gap_samples()), static_cast<unsigned long>(stream_cycles),
         static_cast<unsigned long>(infer->method_pool_peak()), static_cast<unsigned long>(infer->temp_pool_peak()),
         static_cast<unsigned long>(infer->pte_bytes()));
  benchmark_preprocess();
  printf("Test_result: %s\n", pass ? "PASS" : "FAIL");
  printf("\x04");  // EOT: ends the FVP run (semihosting exit); a board just sees a 0x04
  fflush(stdout);
  return pass ? 0 : 1;
}

}  // namespace edge

#else  // !SELFTEST

namespace edge {
int selftest_main(Infer*) {
  return 1;
}
}  // namespace edge

#endif
