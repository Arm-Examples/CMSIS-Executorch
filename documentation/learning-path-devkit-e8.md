---
title: Run an ExecuTorch model on the Ethos-U85 of the Alif Ensemble E8 DevKit with Keil Studio
minutes_to_complete: 90
who_is_this_for: >
  Embedded and ML developers who want to take a PyTorch model to a real
  Ethos-U85 NPU with the CMSIS-Toolbox MLOps flow, for example at a hackathon,
  and optionally let an AI agent drive the build and the debugger.
learning_objectives:
  - Install Keil Studio for VS Code and the Arm tools it needs
  - Optionally connect the CMSIS Developer Assistant to Claude Code or GitHub Copilot
  - Prepare the Alif Ensemble E8 DevKit once with SETOOLS and J-Link
  - Build the example for the DevKit-E8 target with the three-step MLOps flow
  - Run the TinyCNN model on the Ethos-U85 and read the result on the serial console
  - Explore and modify the model, the MLOps configuration and the board layer
prerequisites:
  - An Alif Ensemble E8 DevKit (Gen 2, revision A1) with a USB-C cable
  - A computer running Windows, macOS or Linux with VS Code and Python 3.10 to 3.14
  - An Arm account for the Keil MDK Community license (free for evaluation)
  - Basic familiarity with VS Code, Git and a terminal
author: Arm
skilllevels: Introductory
subjects: ML
armips:
  - Cortex-M55
  - Ethos-U85
tools_software_languages:
  - Keil Studio
  - CMSIS-Toolbox
  - ExecuTorch
  - PyTorch
  - Python
operatingsystems:
  - Windows
  - macOS
  - Linux
layout: learningpathall
---

## Overview

This example takes a small PyTorch image classifier (TinyCNN), quantizes it to
int8, delegates it to the Ethos-U85 and embeds the resulting ExecuTorch
program into a CMSIS-based firmware image. The firmware runs one inference on
the NPU and prints the output logits on the serial console.

Everything is driven by the CMSIS solution `cmsis-executorch-simple.csolution.yml`:

- The **`mlops:` node** describes the NPU and its Vela settings. `cbuild setup`
  turns it into a `*.cbuild-mlops.yml` file.
- **`create_ai_layer.py`** reads that file, exports the model for the NPU in a
  Python virtual environment and writes the **AI layer**: the runtime and
  operator components the model needs, plus the model as a C array.
- **`cbuild`** compiles the application with a **board layer** for the selected
  target. This branch has two: the Corstone-320 FVP (`SSE-320-U85`) and the
  Alif Ensemble E8 DevKit (`DevKit-E8`).

On the DevKit-E8 the application runs on the Cortex-M55 high-performance core
(M55_HP), talks to the Ethos-U85 through the Ethos-U core driver, and uses
UART4 on the PRG USB connector as its console.

You can complete the whole path from the VS Code user interface. The command
line equivalents are given for every step.

## Section 1: Install Keil Studio for VS Code

1. Install [VS Code](https://code.visualstudio.com/).
2. In the Extensions view, install the
   [Keil Studio Pack](https://marketplace.visualstudio.com/items?itemName=Arm.keil-studio-pack)
   (publisher Arm). The pack brings the CMSIS Solution, Device Manager, Arm
   Tools Environment Manager, Arm Debugger and related extensions.
3. Install the
   [Python extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
   (publisher Microsoft).
4. Sign in with your Arm account when the Keil Studio extensions ask for it.
   Keil Studio manages the tool license; the free Keil MDK Community edition is
   sufficient for this example.

The build tools are not installed by hand. The repository ships a
`vcpkg-configuration.json` that lists the exact versions of the CMSIS-Toolbox,
Arm Compiler 6, GCC, CMake, Ninja and the Corstone-320 FVP. The Arm Tools
Environment Manager extension downloads and activates them when you open the
project (Section 5).

## Section 2 (optional): Install the CMSIS Developer Assistant

The CMSIS Developer Assistant is a VS Code extension that exposes the CMSIS
Solution actions (build, load, run, debug), a live Cortex-M debug session
(breakpoints, stepping, registers, memory, fault decode), the serial ports and
the pack documentation to AI coding agents through an MCP server. With it, you
can ask Claude Code or GitHub Copilot to build the project, flash the board,
set a breakpoint in `app_main` and read the Ethos-U registers, instead of
doing it by hand.

1. Install the **CMSIS Developer Assistant** extension in VS Code, from the
   marketplace or from the VSIX provided for the hackathon.
2. Install at least one agent that the assistant can register with:
   [Claude Code](https://claude.com/claude-code) (CLI or VS Code extension) or
   GitHub Copilot Chat.
3. Open the command palette (Ctrl/Cmd+Shift+P) and run
   **CMSIS Developer Assistant: Configure Agents and Skills**. Pick the agents
   to register the MCP server with (it listens on `http://localhost:3001/mcp`)
   and the AI Skills Pack skills to install. For this example, select at
   least:
   - `cmsis-debug-live` for live debugging on the board,
   - `add-board-layer` and `cmsis-project` if you plan to port the example to
     another board,
   - `cmsis-help` to list everything the assistant offers.
4. Optional settings: `cmsis-developer-assistant.packDocs.enabled` turns on the
   documentation tools (page-cited answers from the Alif reference manual and
   the Arm documents the packs link), and
   `cmsis-developer-assistant.buildInfo.enabled` turns on the build-artefact
   tools (memory usage, symbol lookup, section layout). Both need a window
   reload.

You can skip this section and come back to it after the board runs the demo.
Nothing in the build or the board preparation depends on it.

## Section 3: Install the Alif tools

The Ensemble E8 has a Secure Enclave (SE) that boots the application cores.
Before a debugger can load code into the M55_HP core, the SE needs a valid
application table of contents (ATOC) in MRAM that points at a small debug
stub. Alif's **SETOOLS** program that table over the SE's UART.

1. Download the **Alif Security Toolkit (SETOOLS)**, version V1.110.000 or
   later (the first release that supports the E4/E6/E8 devices), from the
   [Alif software and tools page](https://alifsemi.com/support/software-tools/ensemble/)
   (login required). Unpack it and note its root directory (the one containing
   `app-gen-toc` and `app-write-mram`).
2. On Linux and macOS make the tools executable and install the Python
   packages SETOOLS asks for (see its README).
3. Add the root directory to your VS Code user settings (`settings.json`).
   The project's tasks read it from there:

   ```json
   "alif.setools.root": "/absolute/path/to/setools"
   ```

4. Once the board is connected (Section 4), open a terminal in the SETOOLS
   directory and run `updateSystemPackage -d` once. It picks the DevKit's
   serial port, checks the device's system firmware and offers to make the
   Ensemble E8 the default target of the tools; answer yes.
5. Install the
   [SEGGER J-Link Software and Documentation Pack](https://www.segger.com/downloads/jlink/),
   V8.42 or later. The DevKit-E8 has an on-board J-Link (J-Link OB) in its
   "E1" programming device; Alif tests the E8 DevKit with J-Link GDB server
   V8.88. Start J-Link Commander (`JLinkExe`, or `JLink.exe` on Windows)
   once with the board connected: it detects and updates the J-Link OB
   firmware, which also installs the serial-port drivers the on-board bridge
   needs. The Keil Studio debugger configuration `J-Link Server` then starts
   the J-Link GDB server for you.

These steps follow the Ensemble E8 DevKit User Guide (AUGD0023 v1.2, pages
3 to 7).

## Section 4: Connect and configure the board

One USB-C cable is enough. The connector labelled **PRG USB** (on the bottom
side of the board, in the corner) powers the board and carries both the
on-board J-Link and a USB-to-UART bridge. The switch **SW4** next to it
selects which UART the bridge is connected to:

| SW4 position | Bridge connected to | Use |
|--------------|---------------------|-----|
| **SEUART** (default) | Secure Enclave UART | SETOOLS: device status, programming the ATOC and images |
| **UART2** | Application UART2 | Not used by this example |
| **UART4** | Application UART4 | The console of this example, 115200 baud |

1. Check that the option jumpers are in their default positions: JP5 on 1-2
   (VBATT = main) and JP7 on 3-4 (Click_IO = 3.3 V). Never move jumpers with
   power applied.
2. Connect the PRG USB connector to the host. A green LED lights next to the
   E1 device and SW4, and a serial port appears (`COMx` on Windows,
   `/dev/tty.usbmodem*` on macOS, `/dev/ttyACM*` on Linux).
3. Leave SW4 on **SEUART** for the one-time preparation in Section 7. Only
   one program may use the SE-UART at a time: close any terminal on that port
   before running SETOOLS.

The second USB-C connector, **MCU USB**, is the E8's own USB device port; it
plays no role here.

## Section 5: Get the project

1. Clone the repository and check out the `Meta-hackathon` branch:

   ```bash
   git clone https://github.com/Arm-Examples/CMSIS-Executorch.git
   cd CMSIS-Executorch
   git checkout Meta-hackathon
   ```

2. Open the folder in VS Code. Accept the prompt from the Arm Tools
   Environment Manager to **activate** the tools listed in
   `vcpkg-configuration.json`. The first activation downloads the compilers
   and the toolbox; wait for the status bar to report the environment as
   active.
3. Open the **CMSIS** view in the activity bar. If it reports missing packs,
   accept the offer to install them. The solution needs the packs
   `PyTorch::ExecuTorch@1.4.0`, `AlifSemiconductor::Ensemble`,
   `ARM::ethos-u-core-driver`, `ARM::CMSIS`, `ARM::CMSIS-Compiler` and
   `ARM::SSE_320_BSP`. From a terminal, the same happens with:

   ```bash
   cbuild setup cmsis-executorch-simple.csolution.yml --active DevKit-E8 --packs --update-rte
   ```

4. In the CMSIS view, open **Manage Solution** and select the target-type
   **DevKit-E8** and the build-type **Debug**. Click **Apply**. This writes the
   `.vscode/launch.json` and `tasks.json` for the J-Link debugger and merges
   the project's task drop-ins from `.vscode.d/`.
5. Run **Terminal > Run Task > Setup Python virtual environment** once. It
   creates `.venv/` with the ExecuTorch exporter pinned to the same version as
   the `PyTorch::ExecuTorch` pack. From a terminal:

   ```bash
   ./setup_venv.sh        # Linux, macOS
   .\setup_venv.bat       # Windows
   ```

   The repository ships a generated `ai_layer/`, so you can build without this
   step. You need it as soon as you change the model.

## Section 6: Build the application

The build is the three-step MLOps flow. In Keil Studio the first and last
step are the **Build** button of the CMSIS view; the middle step is a task.

1. **Generate the MLOps information.** Click **Build** once (or run
   `cbuild setup`). This resolves the target and writes
   `cmsis-executorch-simple.cbuild-mlops.yml`. With the shipped AI layer the
   build already succeeds here.
2. **Create the AI layer** (only after changing the model or the `mlops:`
   node). Run **Terminal > Run Task > Create AI layer**. The script exports
   `model/model.py`, compiles it with Vela for the Ethos-U85 and rewrites
   `ai_layer/`.
3. **Build** again.

From a terminal:

```bash
cbuild setup cmsis-executorch-simple.csolution.yml --active DevKit-E8
python3 create_ai_layer.py cmsis-executorch-simple.cbuild-mlops.yml   # optional
cbuild cmsis-executorch-simple.csolution.yml --active DevKit-E8
```

The image is `out/cmsis-executorch-simple/DevKit-E8/Debug/cmsis-executorch-simple.axf`.
A successful build reports a program size close to this:

```text
Program Size: Code=119260 RO-data=19188 RW-data=808 ZI-data=3335484
```

The 3.3 MB of zero-initialized data are the runner's memory pools, placed in
the E8's bulk SRAM0 by the board layer (see Section 9).

## Section 7: Prepare the board once with SETOOLS

Do this once per board, or again after another project has reprogrammed the
ATOC.

1. Set **SW4 to SEUART** and make sure the PRG USB connector is attached.
2. Run **Terminal > Run Task > Alif: Install M55_HP debug stubs (DevKit-E8,
   single core configuration)**. When asked, choose COM port discovery (`-d`)
   the first time; SETOOLS then lists the ports and remembers your choice.
   The task copies the configuration and the stub from `.alif/` into the
   SETOOLS tree, generates the table of contents with `app-gen-toc` and writes
   it with `app-write-mram`.
3. Set **SW4 to UART4**. The PRG USB serial port is now the application
   console.

If `app-write-mram` cannot talk to the board, press the reset button while it
waits, check that SW4 is in position SEUART, and make sure no terminal
program holds the port.

## Section 8: Run and debug

1. Open the serial console: in VS Code, use the **Serial Monitor** panel
   (bundled with Keil Studio) on the PRG USB port at **115200** baud, 8N1.
   Any terminal program works as well.
2. In the CMSIS view, click **Debug** (or **Run**). Keil Studio starts the
   J-Link GDB server (JTAG, 1 MHz, as in Alif's own DevKit-E8 examples), loads
   the image into MRAM and stops at `main`. Continue with F5.
3. The console shows the Ethos-U banner, the model size, the output logits
   and the pass marker:

   ```text
   Ethos-U version info:
       Arch:       v2.0.0
       MACs/cc:    256
       Cmd stream: v1
   ExecuTorch Ethos-U85 example: 8864 byte model
   Output: 10 element(s): 0.0082 0.0489 0.0489 -0.0489 0.0857 ...
   Test_result: PASS
   ```

With the CMSIS Developer Assistant connected (Section 2), the same session
can be driven from the agent. Useful prompts:

- "Build the solution for the DevKit-E8 target and load it on the board."
- "Set a breakpoint after `method->execute()` in `src/app_main.cpp`, run to
  it, and show me the output tensor."
- "Read the Ethos-U85 ID and STATUS registers and tell me whether the NPU
  finished the command stream."
- "Open the serial monitor on the PRG USB port and wait for `Test_result`."

## Section 9: Explore the demo

Now that the model runs, look at how the pieces fit together. Each item names
the file to open.

- **The model**: `model/model.py` defines TinyCNN and its example input. Replace
  the class with your own model, keep `get_model()` and
  `get_example_inputs()`, run **Create AI layer** and build. The script prints
  which operator components the new program needs and warns if the pack has
  none for an operator. Anything the Ethos-U backend cannot delegate stays on
  the Cortex-M55 as a portable kernel.
- **The MLOps information**: the `mlops:` node in
  `cmsis-executorch-simple.csolution.yml` and its resolved form in
  `cmsis-executorch-simple.cbuild-mlops.yml`. The Vela options
  (`--system-config`, `--memory-mode`) are passed straight to ExecuTorch's
  `EthosUCompileSpec`, so the NPU configuration lives in one place. The node
  is solution-wide: both target-types share one exported model, which works
  because both have a 256-MAC Ethos-U85.
- **The AI layer**: `ai_layer/ai_layer.clayer.yml` lists only the runtime,
  the Ethos-U backend and the two quantization operators TinyCNN needs;
  `ai_layer/model_pte.c` is the program as a C array. Compare it with the
  full component list of the `PyTorch::ExecuTorch` pack in the CMSIS view's
  software components.
- **The runner**: `src/app_main.cpp` loads the program from the array, plans
  memory from two static pools, fills the input with a ramp, executes and
  prints the output. It is target-agnostic; the pool sizes and their placement
  are overridable defines that a board layer can set.
- **The board layer**: `board/DevKit-E8/Board-U85.clayer.yml` and its README.
  Points worth reading:
  - `main.c` shows the Alif bring-up order: pins, GPIOs, Secure Enclave
    services, clocks, stdio, NPU.
  - `ethos_setup.c` initializes the Ethos-U85 at `NPU_HG_BASE` with IRQ 366.
  - `linker_ac6_mram.sct.src` places code in the HP application region of
    MRAM, `.data`/`.bss`/stack/heap in DTCM, and the `.bss.ai_pool` section
    (the runner's pools, NPU-accessible) in the 4 MB bulk SRAM0.
  - `ethosu_cb_dcache.c` keeps the Cortex-M55 data cache coherent with the
    NPU for buffers outside the TCMs.

  | Region | Address | Holds |
  |--------|---------|-------|
  | MRAM (HP region) | `0x80200000`, 2 MB | Code, constants, the model |
  | DTCM | `0x20000000`, 1 MB | Globals, 96 kB heap, 32 kB stack |
  | SRAM0 | `0x02000000`, 4 MB | 1 MB method pool, 2 MB temp pool (NPU scratch) |

- **Same code, other target**: select `SSE-320-U85` in Manage Solution and
  press Run. The identical application and AI layer run on the Corstone-320
  FVP with the Corstone board layer. Diff the two layers to see what a board
  port consists of.
- **Measure**: wrap `method->execute()` with the DWT cycle counter (or ask the
  assistant to read it) and compare a model with more channels. Use the PMU
  counters of the Ethos-U driver (`ethosu_pmu.h`) for NPU-side numbers.

Ideas for a hackathon project: swap in a keyword-spotting or a small vision
model and feed real input from the DevKit's microphones or camera through the
full board layer of the Ensemble pack; try a model with an operator the NPU
cannot run and watch it fall back to the CPU; port the example to the M55_HE
core with the pack's `Board_HE-U85` layer.

## Troubleshooting

- **`board 'SSE-320' was not found` from `cbuild setup`**: the `ARM::SSE_320_BSP`
  pack is missing. Run the setup command with `--packs`.
- **`[RETARGET-FAILED] POLLING MODE IS NOT ENABLED`**: the UART4 polling flag
  in `board/DevKit-E8/RTE/Device/AE822FA0E5597LS0_M55_HP/RTE_Device.h` was
  reset to the pack default (for example by an RTE update). Set
  `RTE_UART4_BLOCKING_MODE_ENABLE` back to `1`.
- **`L6636E: Pre-processor step failed` at link time**: the layer's scatter
  file is meant to be preprocessed by the toolbox (`.sct.src` plus the
  `regions:` entry of the layer). Do not rename it back to `.sct`.
- **J-Link cannot connect or the core does not halt at `main`**: the ATOC does
  not point at the debug stub. Repeat Section 7. If JTAG at 1 MHz is
  unreliable on your setup, change the target-set in the csolution to
  `protocol: swd` and `clock: 4000000`, the probe settings the pack's board
  description lists, and click Apply again.
- **No console output**: SW4 is still in position SEUART (or UART2), or the
  terminal was opened before the switch was moved. Set SW4 to UART4 and
  reopen the port.
- **Link fails with `L6815U: Out of memory` on macOS** (larger models only):
  the 32-bit `armlink` runs out of address space. Build with
  `cbuild ... --toolchain CLANG` after adding
  `arm:compilers/arm/llvm-embedded` to `vcpkg-configuration.json`, or use GCC.
  The Alif layer ships linker scripts for both.
- **`create_ai_layer.py` says torch is not installed**: run the venv setup
  task or script first (Section 5, step 5).

## Next steps

- [The MLOps flow in detail](mlops-flow.md), including how the toolbox
  hands over to the exporter.
- [Pack provenance](pack-provenance.md): keeping the ExecuTorch pack and the
  Python exporter at the same version.
- [CMSIS-Toolbox MLOps information](https://open-cmsis-pack.github.io/cmsis-toolbox/build-overview/#mlops-information)
- [ExecuTorch Arm Ethos-U backend](https://docs.pytorch.org/executorch/main/backends-arm-ethos-u.html)
- [Alif Ensemble E8 DevKit](https://alifsemi.com/support/kits/ensemble-e8devkit/)
  and the `AlifSemiconductor::Ensemble` pack overview on
  [keil.arm.com](https://www.keil.arm.com/packs/ensemble-alifsemiconductor)
