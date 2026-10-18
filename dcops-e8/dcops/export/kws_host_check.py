# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
"""Host-side reference for the alif_kws ExecuTorch firmware: prints what the board should print.

  PY=~/projects/arm-executorch-hack/experiments/blazeface/.venv-et110/bin/python
  PATH=$(dirname $PY):$PATH $PY dcops/export/kws_host_check.py dcops/models/placeholder/kws_random [wav ...]

For a run directory (student.pt, calib.npz, metrics.json, as written by dcops/jobs/kws.py) it rebuilds
the float DSCNN and the int8 model with the SAME PT2E recipe as export_kws.py. The .pte itself cannot
run on the host, because its body is an Ethos-U delegate (Vela command stream), so the int8 PyTorch
model is the closest host stand-in. Vela's int8 kernels round differently from torchao's
fake-quant, so expect the board's logits to be close to the int8 ones, not bit-exact.

Cases (they match KWS_SELFTEST_A / KWS_SELFTEST_B in source/app/use_case/alif_kws/src/UseCaseHandler.cc):
  A  model only: input[i] = ((i*37) % 41) - 30, i in [0, 490)              -> logits
  B  synthetic 1 s int16 tone (440 Hz + 1250 Hz) -> MLEK MFCC (numpy port in dcops/jobs/kws.py)
     -> mfcc frame 0, frame 24, sum of all features, logits
  W  each wav given (16 kHz mono int16; first 1 s, zero-padded) -> MFCC summary + logits + top-1.
     NOTE: on the board, audio_alif auto-gain (up to +80 dB, peak -> full scale) runs before the MFCC.
     Pass --peak-normalise to mimic that roughly (it scales the clip so |peak| = 32767).
"""
import argparse, importlib.util, json, os, wave
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
spec_ = importlib.util.spec_from_file_location("kws", os.path.join(os.path.dirname(HERE), "jobs", "kws.py"))
kws = importlib.util.module_from_spec(spec_); spec_.loader.exec_module(kws)


def mfcc_numpy(audio_i16):
    """The numpy reference from dcops/jobs/kws.py (main(): cross-check), on int16 like the board."""
    x = audio_i16.astype(np.float32) / 32768.0                       # Mfcc.cc normaliser 1/(1<<15)
    win = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(kws.FRAME) / kws.FRAME)).astype(np.float32)
    fr = np.stack([x[i * kws.STRIDE: i * kws.STRIDE + kws.FRAME] for i in range(kws.NFRAMES)]) * win
    mag = np.abs(np.fft.rfft(fr, n=kws.NFFT))[:, : kws.NFFT // 2]
    return (np.log(mag @ kws.mlek_mel_filterbank().T + 1.1754944e-38) @ kws.mlek_dct().T).astype(np.float32)  # [49,10]


def build_models(run):
    from executorch.backends.arm.ethosu import EthosUCompileSpec
    from executorch.backends.arm.quantizer import EthosUQuantizer, get_symmetric_quantization_config
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
    m = json.load(open(f"{run}/metrics.json"))
    model = kws.DSCNN(m["arch"]["ch"]).eval()
    model.load_state_dict(torch.load(f"{run}/student.pt", map_location="cpu"))
    c = np.load(f"{run}/calib.npz"); calib = [torch.from_numpy(np.array(x[None], np.float32)) for x in c["mfcc"]]
    repo = os.path.abspath(os.path.join(HERE, "..", ".."))
    spec = EthosUCompileSpec(target="ethos-u85-256", config_ini=f"{repo}/scripts/vela/ensemble_vela.ini",
                             system_config="Ethos_U85_SRAM_MRAM", memory_mode="Shared_Sram",
                             extra_flags=["--output-format=raw", "--debug-force-regor"])
    q = EthosUQuantizer(spec); q.set_global(get_symmetric_quantization_config(is_per_channel=True))
    p = prepare_pt2e(torch.export.export(model, (calib[0],)).module(), q)
    with torch.no_grad():
        for x in calib: p(x)
    return model, convert_pt2e(p), m


def fmt(v):
    return " ".join(f"{float(a):.4f}" for a in np.asarray(v).ravel())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("wavs", nargs="*")
    ap.add_argument("--peak-normalise", action="store_true")
    a = ap.parse_args()
    fl, q8, meta = build_models(a.run)
    pte = os.path.join(a.run, "kws_ethos-u85-256.pte")
    print(f"run {a.run}  placeholder={meta.get('placeholder', False)}  test_acc={meta.get('test_acc')}"
          f"  pte={'%d B' % os.path.getsize(pte) if os.path.exists(pte) else 'missing'}")

    def run(feat):
        x = torch.from_numpy(np.ascontiguousarray(feat, np.float32).reshape(1, 1, kws.NFRAMES, kws.NMFCC).copy())
        with torch.no_grad():
            return fl(x)[0].numpy(), q8(x)[0].numpy()

    a_in = np.array([((i * 37) % 41) - 30 for i in range(kws.NFRAMES * kws.NMFCC)], np.float32)
    lf, lq = run(a_in)
    print("KWS_SELFTEST_A logits (float):", fmt(lf))
    print("KWS_SELFTEST_A logits (int8): ", fmt(lq))

    n = np.arange(16000) / 16000.0
    tone = np.round(8000 * np.sin(2 * np.pi * 440 * n) + 4000 * np.sin(2 * np.pi * 1250 * n)).astype(np.int16)
    f = mfcc_numpy(tone)
    print("KWS_SELFTEST_B mfcc[0]:", fmt(f[0]))
    print("KWS_SELFTEST_B mfcc[24]:", fmt(f[24]))
    print(f"KWS_SELFTEST_B mfcc sum: {float(f.astype(np.float64).sum()):.4f}")
    lf, lq = run(f)
    print("KWS_SELFTEST_B logits (float):", fmt(lf))
    print("KWS_SELFTEST_B logits (int8): ", fmt(lq))

    for w in a.wavs:
        with wave.open(w) as h:
            assert h.getframerate() == 16000 and h.getnchannels() == 1 and h.getsampwidth() == 2, w
            pcm = np.frombuffer(h.readframes(h.getnframes()), np.int16)
        clip = np.zeros(16000, np.int16); clip[: min(16000, len(pcm))] = pcm[:16000]
        if a.peak_normalise and np.abs(clip).max() > 0:
            clip = np.clip(np.round(clip.astype(np.float32) * 32767 / np.abs(clip.astype(np.float32)).max()), -32768, 32767).astype(np.int16)
        f = mfcc_numpy(clip)
        lf, lq = run(f)
        print(f"WAV {w}")
        print("  mfcc[0]:", fmt(f[0]))
        print(f"  mfcc sum: {float(f.astype(np.float64).sum()):.4f}  c0 range [{f[:, 0].min():.2f}, {f[:, 0].max():.2f}]")
        print("  logits (float):", fmt(lf))
        print("  logits (int8): ", fmt(lq))
        print(f"  top-1 float {kws.LABELS[int(lf.argmax())]}, int8 {kws.LABELS[int(lq.argmax())]}")


if __name__ == "__main__":
    main()
