import numpy as np

from drivers.replay import (CTRL_CUE, CTRL_RESET, Control, DataBatch, FrameParser, pack_control_frame,
                            pack_data_frame, pack_data_frames)


def test_data_frame_layout():
    f = pack_data_frame(0x1234, np.arange(-11, 11, dtype=np.int16))
    assert len(f) == 49
    assert f[:5] == bytes([0xAA, 0x55, 22, 0x34, 0x12])
    assert np.frombuffer(f[5:], "<i2").tolist() == list(range(-11, 11))


def test_control_frame_layout():
    assert pack_control_frame(CTRL_CUE, 1, 0xBEEF) == bytes([0xAA, 0x56, 2, 1, 0xEF, 0xBE])


def test_batch_pack_equals_single_and_seq_wraps():
    x = np.random.default_rng(0).integers(-32768, 32767, size=(5, 22)).astype(np.int16)
    single = b"".join(pack_data_frame((65534 + i) & 0xFFFF, x[i]) for i in range(5))
    assert pack_data_frames(65534, x) == single


def test_parser_round_trip_with_garbage_and_split_feeds():
    rng = np.random.default_rng(1)
    x = rng.integers(-3000, 3000, size=(7, 22)).astype(np.int16)
    stream = (pack_control_frame(CTRL_RESET) + b"\x00\x13\xAA" + pack_data_frames(10, x[:3])
              + pack_control_frame(CTRL_CUE, 0, 13) + pack_data_frames(13, x[3:]))
    p = FrameParser()
    items = []
    for i in range(0, len(stream), 17):  # arbitrary USB-sized chunks
        items += p.feed(stream[i:i + 17])
    ctrls = [it for it in items if isinstance(it, Control)]
    data = [it for it in items if isinstance(it, DataBatch)]
    assert [(c.kind, c.arg, c.seq) for c in ctrls] == [(CTRL_RESET, 0, 0), (CTRL_CUE, 0, 13)]
    seq = np.concatenate([d.seq for d in data]).tolist()
    assert seq == list(range(10, 17))
    assert np.array_equal(np.concatenate([d.x for d in data]), x)
    assert p.errors == 3
    # the cue control frame sits right before the data frame with its seq
    order = [("c", it.seq) if isinstance(it, Control) else ("d", int(it.seq[0])) for it in items]
    assert order.index(("c", 13)) + 1 == order.index(("d", 13))
