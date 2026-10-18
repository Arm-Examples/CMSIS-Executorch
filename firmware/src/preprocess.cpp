// Per-sample EEG preprocessing; see preprocess.h for the pipeline and the precision rules.
#include "preprocess.h"

#include <cmath>
#include <cstring>

// No fused multiply-add: keep the operation order of scipy's sosfilt and NumPy, so the
// float64 intermediate values are the reference's to the last bit wherever possible.
#if defined(__clang__)
#pragma clang fp contract(off)
#endif

namespace edge {

void Preprocess::reset() {
  std::memset(z_, 0, sizeof(z_));
  std::memset(ring_, 0, sizeof(ring_));
  n250_ = 0;
  n125_ = 0;
}

bool Preprocess::push_sample(const int16_t counts[N_CH]) {
  // 1. uV (exact: power-of-two step), then common average reference:
  //    mean = (sum over channels in order) / 22.
  double x[N_CH];
  double sum = 0.0;
  for (uint32_t c = 0; c < N_CH; ++c) {
    x[c] = static_cast<double>(counts[c]) * UV_PER_LSB;
    sum += x[c];
  }
  const double mean = sum / static_cast<double>(N_CH);
  for (uint32_t c = 0; c < N_CH; ++c) {
    x[c] = x[c] - mean;
  }

  // 2. Band-pass, DF2-T, the same expression order as scipy.signal.sosfilt:
  //    y = b0 x + z0;  z0 = b1 x - a1 y + z1;  z1 = b2 x - a2 y.
  for (uint32_t s = 0; s < N_SOS; ++s) {
    const double b0 = SOS[s][0];
    const double b1 = SOS[s][1];
    const double b2 = SOS[s][2];
    const double a1 = SOS[s][4];
    const double a2 = SOS[s][5];
    for (uint32_t c = 0; c < N_CH; ++c) {
      double* z = z_[s][c];
      const double xc = x[c];
      const double y = b0 * xc + z[0];
      z[0] = b1 * xc - a1 * y + z[1];
      z[1] = b2 * xc - a2 * y;
      x[c] = y;
    }
  }

  // 3. Decimation by 2, phase 0, counted from the first sample after reset.
  const bool keep = (n250_ % DECIMATION) == DECIMATION_PHASE;
  ++n250_;
  if (!keep) {
    return false;
  }

  // 4. Ring buffer of float32 samples (the reference casts the window to float32).
  float* slot = ring_[n125_ % kRing];
  for (uint32_t c = 0; c < N_CH; ++c) {
    slot[c] = static_cast<float>(x[c]);
  }
  ++n125_;
  return true;
}

bool Preprocess::make_window_at(int subject, uint32_t end_125, int8_t out[kWindowElems]) const {
  if (subject < 1 || subject > N_SUBJECTS || !window_available(end_125)) {
    return false;
  }
  const double(*ea)[N_CH] = EA[subject - 1];
  const float input_scale = static_cast<float>(INPUT_SCALE);
  const float quant_scale = static_cast<float>(INPUT_QUANT_SCALE);
  const float zero_point = static_cast<float>(INPUT_ZERO_POINT);
  const uint32_t start = end_125 - kWindow;

  for (uint32_t t = 0; t < kWindow; ++t) {
    const float* w = ring_[(start + t) % kRing];
    double wd[N_CH];
    for (uint32_t d = 0; d < N_CH; ++d) {
      wd[d] = static_cast<double>(w[d]);
    }
    for (uint32_t c = 0; c < N_CH; ++c) {
      // 5. EA in float64, cast to float32 (np.einsum(...).astype(np.float32)).
      double acc = 0.0;
      for (uint32_t d = 0; d < N_CH; ++d) {
        acc += ea[c][d] * wd[d];
      }
      const float xa = static_cast<float>(acc);
      // 6. float32: / input_scale, / quant_scale, round half to even, + zero point, clamp.
      const float xm = xa / input_scale;
      float q = std::nearbyint(xm / quant_scale) + zero_point;
      if (q < -128.0f) {
        q = -128.0f;
      } else if (q > 127.0f) {
        q = 127.0f;
      }
      out[c * kWindow + t] = static_cast<int8_t>(q);
    }
  }
  return true;
}

}  // namespace edge
