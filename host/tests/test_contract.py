"""Host framing constants must equal the firmware's (firmware/src/uart_rx.h, params_gen.h).
The sim board shares the host constants, so only this test catches a host/firmware mismatch."""

import re

import pytest

from common import REPO_DIR
from drivers import replay

UART_H = REPO_DIR / "firmware" / "src" / "uart_rx.h"
PARAMS_H = REPO_DIR / "firmware" / "src" / "params_gen.h"


@pytest.fixture(scope="module")
def uart_h() -> str:
    if not UART_H.exists():
        pytest.skip("firmware/src/uart_rx.h not present")
    return UART_H.read_text()


def _const(text: str, name: str) -> int:
    m = re.search(rf"\b{name}\s*=\s*(0x[0-9A-Fa-f]+|\d+)", text)
    assert m, f"{name} not found"
    return int(m.group(1), 0)


def test_control_kinds_match_firmware(uart_h):
    enum = re.search(r"enum\s+ControlKind[^{]*\{(.*?)\}", uart_h, re.S)
    assert enum, "enum ControlKind not found"
    fw = {k: int(v, 0) for k, v in re.findall(r"(k\w+)\s*=\s*(0x[0-9A-Fa-f]+|\d+)", enum.group(1))}
    host = {"kSetSubject": replay.CTRL_SET_SUBJECT, "kCue": replay.CTRL_CUE,
            "kReset": replay.CTRL_RESET, "kFreeRun": replay.CTRL_FREE_RUN}
    assert fw == host


def test_frame_layout_matches_firmware(uart_h):
    assert _const(uart_h, "kSync0") == replay.SYNC
    assert _const(uart_h, "kSyncData") == replay.DATA
    assert _const(uart_h, "kSyncControl") == replay.CTRL
    m = re.search(r"kDataFrameBytes\s*=\s*([^;]+);", uart_h)
    n_ch = int(re.search(r"#define\s+N_CH\s+(\d+)", PARAMS_H.read_text()).group(1))
    assert n_ch == 22
    assert eval(m.group(1).replace("N_CH", str(n_ch))) == replay.data_frame_len(n_ch) == 49  # noqa: S307
    m = re.search(r"kControlFrameBytes\s*=\s*([^;]+);", uart_h)
    assert eval(m.group(1)) == replay.CTRL_LEN == 6  # noqa: S307
    # and the bytes the host actually packs
    f = replay.pack_control_frame(replay.CTRL_CUE, 1, 0x0102)
    assert f == bytes([_const(uart_h, "kSync0"), _const(uart_h, "kSyncControl"), _const(uart_h, "kCue"), 1, 2, 1])
