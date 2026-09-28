# pico-faces on Ethos-U85 with ExecuTorch: hackathon guide

The Arm ExecuTorch example, on its `hackathon` branch: a generative model on
a microcontroller. [pico-faces](https://github.com/cpldcpu/pico-faces) by
cpldcpu, a latent diffusion transformer (2.5M parameters) with a small
decoder, generates 128x128 faces; exported through ExecuTorch, every layer of
it runs on the Ethos-U85 NPU while the Cortex-M drives the sampling loop. One
CMSIS solution runs it on the **Alif Ensemble E8 DevKit** (Cortex-M55 HP core;
the faces appear on the board's LCD, a new one per joystick press) and on the
**Corstone-320 FVP**; you switch between them by target-type. The model is
exported from PyTorch in three steps: the CMSIS-Toolbox describes the target,
`create_ai_layer.py` turns that into the AI layer, the toolbox builds the
application. This page takes you from an empty machine to a debug session on
the board. Everything about the example itself is in
[documentation/example.md](documentation/example.md).

## 1. Host tools

1. Install [VS Code](https://code.visualstudio.com/).
2. Install the extensions **Keil Studio Pack** (`Arm.keil-studio-pack`, which
   brings the CMSIS Solution extension 1.70.0 or newer that the project's
   task drop-ins need) and **Python** (`ms-python.python`). Sign in with an
   Arm account when Keil Studio asks; the free Keil MDK Community license is
   enough.
3. Nothing else by hand: when you open the project, the Arm Tools Environment
   Manager offers to install the tools pinned in `vcpkg-configuration.json`
   (CMSIS-Toolbox, Arm Compiler 6, GCC, CMake, Ninja, and on Linux and
   Windows the Corstone-320 FVP; macOS runs it in Docker, see step 6). Accept.
4. Optional: the **CMSIS Developer Assistant** extension lets an AI agent
   (Claude Code or GitHub Copilot Chat) build, flash and debug the board
   through an MCP server. Install it, install one of the agents, run
   **CMSIS Developer Assistant: Configure Agents and Skills** from the command
   palette and pick at least the `cmsis-debug-live` and `cmsis-help` skills.

## 2. Alif and SEGGER tools (board only)

1. **Alif SETOOLS** V1.110.000 or later from the
   [Alif software and tools page](https://alifsemi.com/support/software-tools/ensemble/)
   (login required). Unpack it; on Linux and macOS make the tools executable
   and install the Python packages its README lists. Add the root directory
   (the one with `app-gen-toc` and `app-write-mram`) to your VS Code user
   settings:

   ```json
   "alif.setools.root": "/absolute/path/to/setools"
   ```

2. **SEGGER J-Link Software** V8.42 or later from
   [segger.com](https://www.segger.com/downloads/jlink/). The board has an
   on-board J-Link.

## 3. Board

- Connect a USB-C cable to **PRG USB** (the connector in the corner). It
  powers the board and carries the J-Link and a USB-to-UART bridge. Leave
  **MCU USB** unconnected.
- Jumpers at their defaults: **JP5 on 1-2**, **JP7 on 3-4**. Never move
  jumpers with power applied.
- **SW4** selects what the UART bridge is connected to:

  | SW4 | Connected to | Used for |
  |-----|--------------|----------|
  | `SEUART` (default) | Secure Enclave UART | SETOOLS (step 5) |
  | `UART4` | Application UART4, 115200 8N1 | The example's console (step 6) |

- With the board attached, run these once: in the SETOOLS directory
  `updateSystemPackage -d` (SW4 on `SEUART`; picks the serial port, checks the
  system firmware, offers to make the E8 the default target: answer yes), and
  J-Link Commander (`JLinkExe`, `JLink.exe` on Windows), which updates the
  on-board J-Link firmware and installs its serial-port drivers.

## 4. Project

```bash
git clone https://github.com/Arm-Examples/CMSIS-Executorch.git
cd CMSIS-Executorch
git checkout hackathon
```

Open the folder in VS Code and accept the tool activation and the pack
installation (`PyTorch::ExecuTorch`, `AlifSemiconductor::Ensemble`, CMSIS).
In the CMSIS view open **Manage Solution**, choose the target-type
**DevKit-E8** (or **SSE-320-U85** for the FVP) and click **Apply**.

The model data is generated rather than committed, so a fresh checkout
needs Python once before the first build:

1. **Terminal > Run Task > Setup Python virtual environment** (several GB of
   PyTorch, takes a while; the **(uv)** variant of the task uses uv and can
   download the Python version it asks for). It also downloads the
   pico-faces checkpoints (about 60 MB, checked by SHA-256) into
   `model/pico_faces/`.
2. **Terminal > Run Task > Create AI layer** exports the model for the NPU
   and writes `ai_layer/` (a few minutes). It reads
   `cmsis-executorch.cbuild-mlops.yml`, which the CMSIS Solution extension
   writes when it loads the solution (click **Build** once if the file is
   missing). Run the task again after changing `model/model.py`.

## 5. Prepare the board once

The Secure Enclave boots the M55 cores from a table of contents in MRAM; the
debugger needs that table to point at a debug stub.

1. SW4 to **SEUART**, PRG USB attached.
2. **Terminal > Run Task > Alif: Install M55_HP debug stubs (DevKit-E8, single core configuration)**. Choose COM port
   discovery (`-d`) the first time; SETOOLS remembers the port. The task
   copies the configuration and stub from `.alif/` into the SETOOLS tree and
   runs `app-gen-toc` and `app-write-mram`.
3. SW4 to **UART4**.

Repeat this after another project has reprogrammed the table.

## 6. Build, run, debug

1. With SW4 on **UART4**, open the **Serial Monitor** panel on the PRG USB
   port, 115200 baud.
2. In the CMSIS view click **Build**, then **Debug** (or **Run**). Keil Studio
   starts the J-Link GDB server over SWD, loads the image into MRAM and stops
   at `main`; continue with F5. The console shows the Ethos-U banner, the
   program and its two methods, the timings of the boot demo, the CRC of the
   image, an ASCII preview of the face and the pass marker:

   ```text
   Ethos-U version info:
       Arch:       v2.0.0
       MACs/cc:    256
       Cmd stream: v1
   ExecuTorch pico-faces (m3_long_cfg, a16w8/a8w8): 2796016 byte program
   Methods:
     dit_step  2 input(s), 1 output(s), 16384 planned byte(s)
     decode    1 input(s), 1 output(s), 245760 planned byte(s)
   Generating: seed 3, 4 steps, class 1, w 4.0
   Display: 480x800 RGB888 panel started
     dit_step: 8 call(s), 68 ms total (8 ms each)
     decode:   3 ms
     total:    78 ms at 400 MHz (wall clock; not meaningful on the FVP)
     dit_step: NPU 27132 kcycles, active 95%, MAC active 39%, 32 MAC/cycle, read 21279 kB on AXI0 + 0 kB on AXI1
     decode:   NPU 1303 kcycles, active 50%, MAC active 37%, 83 MAC/cycle, read 434 kB on AXI0 + 0 kB on AXI1
   Image: 128x128x3, CRC32 6b938c66
     |::::::::.......  .    ..........|
     |::::::::...........     .....   |
     ...
   Test_result: PASS
   Interactive: send "G <seed> [k_steps] [class] [w]" (viewer/view_serial.py) or "I"
   Joystick: left = one new image, right = start/stop continuous generation
   ```

   A face with guidance takes 78 ms: eight `dit_step` calls of 8.5 ms on the
   NPU and a 3 ms decode. The same face appears on the LCD, scaled to
   384 x 384 in the centre of the screen; it is bit for bit the image the FVP
   produces (same CRC).
3. Press the **SW2 joystick** to the left for a new face (the next seed; class
   and guidance follow from it), to the right to generate faces back to back
   until you press right again. Each one is reported on the console.
4. Or request faces from the host with pico-faces' viewer (close the Serial
   Monitor first, the viewer needs the port; in a clone of
   [pico-faces](https://github.com/cpldcpu/pico-faces),
   `pip install pyserial pillow`):

   ```bash
   python viewer/view_serial.py --port /dev/tty.usbmodemXXXX --seed 3 --steps 4 --class 1 --cfg 4 --show
   ```

   Classes are 0 (female, neutral), 1 (female, smiling), 2 (male, neutral),
   3 (male, smiling) and 4 (unconditional); `--cfg` is the guidance strength
   (0 = plain), `--steps` 8, 4, 2 or 1.
5. Set a breakpoint after `generate()` in `src/app_main.cpp` and inspect the
   timings in `tm`, or ask the CMSIS Developer Assistant to do it: "Build for
   the DevKit-E8, load it, break after the first generate() and show me the
   NPU cycles per method."

**FVP instead of the board:** choose the **SSE-320-U85** target-type and click
**Run** or **Debug**; the same output appears in the terminal (a run takes a
few minutes), and the image lands in `out/fvp_image.bin`.
`python3 model/verify_export.py --compare out/fvp_image.bin <reference.png>`
compares it with the host's rendering of the same seed (see
[documentation/example.md](documentation/example.md)). On macOS the
FVP runs in Docker (Docker Desktop must be running; the first run builds the
image, about 100 MB). On Windows set `model:` in the csolution's SSE-320-U85
target-set to `FVP_Corstone_SSE-320`.

## 7. If something does not work

- **J-Link connects but never stops at `main`:** either the table of contents
  does not point at the debug stub (repeat step 5), or the image cannot boot.
  Look before reprogramming: in the debugger, read the vector table at
  `0x80200000` and the fault registers (CFSR/HFSR); a PC of `0xEFFFFFFE` is a
  lockup at reset, which an image linked to run from ITCM produces when the
  code that copies it there is itself in ITCM.
- **Debug hangs at "Connecting":** the target-set was switched to
  `protocol: jtag`. Keep SWD: the generated load task blocks on JLinkExe's
  JTAG-chain prompt, and the device stays in SWD mode after any SWD use
  until it is power-cycled.
- **No console output:** SW4 is still on `SEUART`, or the port was opened
  before the switch was moved. Set `UART4` and reopen the port.
- **`app-write-mram` gets no answer:** press reset while it waits, check SW4
  is on `SEUART`, close any terminal holding the port.
- **The build cannot find `ai_layer/model_pte.c`:** the model data is not
  committed; run the tasks **Setup Python virtual environment** and **Create
  AI layer** (step 4).
- **"`.venv` does not exist" or "`model/pico_faces/...` is missing":** run the
  task **Setup Python virtual environment** first; it creates the
  environment and downloads the checkpoints.
- After **Apply**, the extension adds a J-Link entry to `.vscode/launch.json`
  next to the committed FVP entry. That is expected.

## Read on

- [documentation/example.md](documentation/example.md): the example, the
  command-line flow, how the model is generated.
- [board/DevKit-E8/README.md](board/DevKit-E8/README.md): the board layer,
  memory map and RTE configuration.
- [documentation/mlops-flow.md](documentation/mlops-flow.md): the `mlops:`
  node and `*.cbuild-mlops.yml` in detail.
- [documentation/pack-provenance.md](documentation/pack-provenance.md): where
  the `PyTorch::ExecuTorch` pack comes from and how to move to a new version.
