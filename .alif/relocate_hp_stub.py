"""Relocate the Alif M55_HP MRAM debug stub to the DevKit-E8 HP image base.

The original SETOOLS stub is linked at 0x80200000. Its vector table and a
small number of literal-pool words contain absolute pointers into the stub.
Only those in-stub pointers are adjusted; peripheral and MRAM-end constants
must retain their values.
"""

from hashlib import sha256
from pathlib import Path
from struct import pack_into, unpack_from


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "M55_HP_mram_stub_80200000.bin"
OUTPUT = HERE / "M55_HP_mram_stub.bin"
OLD_BASE = 0x80200000
NEW_BASE = 0x80000000
SOURCE_SHA256 = "f1751eafcdbda93c8763a505bacef7f6fa45faea34bf0c1fc56f176b42ec83c9"


def main() -> None:
    stub = bytearray(SOURCE.read_bytes())
    if sha256(stub).hexdigest() != SOURCE_SHA256 or len(stub) != 3040:
        raise ValueError("Source stub differs from the verified Alif SETOOLS binary")

    relocated = 0
    for offset in range(0, len(stub), 4):
        value = unpack_from("<I", stub, offset)[0]
        if OLD_BASE <= value < OLD_BASE + len(stub):
            pack_into("<I", stub, offset, value - OLD_BASE + NEW_BASE)
            relocated += 1

    if relocated != 434:
        raise ValueError(f"Expected 434 in-stub address words; found {relocated}")
    if unpack_from("<I", stub, 4)[0] != 0x80000989:
        raise ValueError("Reset vector was not relocated")
    if unpack_from("<I", stub, 0xA38)[0] != NEW_BASE:
        raise ValueError("VTOR base literal was not relocated")
    if any(
        OLD_BASE <= unpack_from("<I", stub, offset)[0] < OLD_BASE + len(stub)
        for offset in range(0, len(stub), 4)
    ):
        raise ValueError("An old in-stub address remains")

    OUTPUT.write_bytes(stub)
    print(f"Wrote {OUTPUT.name}: {len(stub)} bytes, {relocated} relocated words")


if __name__ == "__main__":
    main()
