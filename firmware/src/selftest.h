// Built-in self-test (build-type SelfTest, -DSELFTEST): replays the embedded A03 stream of
// tests/generated/selftest_vectors.h as UART frames through the real App (parser,
// preprocessing, Ethos-U inference, JSON). PASS: every trial's int8 window equals the host
// reference exactly (0 differing elements) and its argmax equals BOARD_ARGMAX. Needs no UART input, so it
// runs on the Corstone-320 FVP. Prints Test_result: PASS/FAIL and an EOT (ends the FVP run).
#ifndef EDGE_EEG_SELFTEST_H_
#define EDGE_EEG_SELFTEST_H_

#include "infer.h"

namespace edge {

int selftest_main(Infer* infer);

}  // namespace edge

#endif  // EDGE_EEG_SELFTEST_H_
