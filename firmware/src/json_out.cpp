// JSON result line; see json_out.h.
#include "json_out.h"

#include <cmath>
#include <cstdio>

namespace edge {

int32_t argmax2(const float logits[2]) {
  return logits[1] > logits[0] ? 1 : 0;
}

float softmax_conf2(const float logits[2], int32_t cls) {
  const float other = logits[1 - cls];
  return 1.0f / (1.0f + std::exp(other - logits[cls]));
}

size_t format_json(char* buf, size_t size, const InferenceRecord& r) {
  const int n = snprintf(
      buf, size,
      "{\"t\":%lu,\"seq\":%u,\"trial\":%ld,\"subject\":%ld,\"cls\":%ld,\"raw_cls\":%ld,"
      "\"conf\":%.2f,\"logits\":[%.6f,%.6f],\"lat_us\":%lu,\"motion\":%ld,\"label\":%ld%s}\n",
      static_cast<unsigned long>(r.t_ms), static_cast<unsigned>(r.seq), static_cast<long>(r.trial),
      static_cast<long>(r.subject), static_cast<long>(r.cls), static_cast<long>(r.raw_cls),
      static_cast<double>(r.conf), static_cast<double>(r.logits[0]), static_cast<double>(r.logits[1]),
      static_cast<unsigned long>(r.lat_us), static_cast<long>(r.motion), static_cast<long>(r.label),
      r.imu_ok ? "" : ",\"imu\":0");
  if (n < 0 || static_cast<size_t>(n) >= size) {
    return 0;
  }
  return static_cast<size_t>(n);
}

}  // namespace edge
