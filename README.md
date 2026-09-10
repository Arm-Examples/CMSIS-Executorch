# ExecuTorch on Ethos-U85: hackathon guide

The Arm ExecuTorch example, on its `hackathon` branch with the Alif board
added. One CMSIS solution runs a tiny int8 CNN on the Ethos-U85 of the **Alif Ensemble E8 DevKit** (Cortex-M55
HP core) and on the **Corstone-320 FVP**; you switch between them by
target-type. The model is exported from PyTorch in three steps: the
CMSIS-Toolbox describes the target, `create_ai_layer.py` turns that into the
AI layer, the toolbox builds the application. This page takes you from an
empty machine to a debug session on the board. Everything about the example
itself is in [documentation/example.md](documentation/example.md).

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
   Windows the Corstone-320 FVP; macOS runs it in Docker, see step 8). Accept.
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

The repository ships a generated AI layer, so no Python is needed to build.
To change the model, run **Terminal > Run Task > Setup Python virtual
environment** once (several GB of PyTorch, takes a while; the **(uv)**
variant of the task uses uv and can download the Python version it asks
for), edit `model/model.py`, and run the task **Create AI layer** before
building.

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
   at `main`; continue with F5. The console shows:

   ```text
   Ethos-U version info:
       Arch:       v2.0.0
       MACs/cc:    256
       Cmd stream: v1
   ExecuTorch Ethos-U85 example: 8896 byte model
   Output: 10 element(s): 0.0079 0.0459 0.0475 -0.0475 0.0791 0.0411 -0.0285 -0.0744 -0.2246 -0.0016
   Test_result: PASS
   ```

3. Set a breakpoint after `module.forward(input)` in `src/app_main.cpp` and
   inspect the output tensor, or ask the CMSIS Developer Assistant to do it:
   "Build for the DevKit-E8, load it, break after the inference and show me
   the output logits."

**FVP instead of the board:** choose the **SSE-320-U85** target-type and click
**Run** or **Debug**; the same output appears in the terminal. On macOS the
FVP runs in Docker (Docker Desktop must be running; the first run builds the
image, about 100 MB). On Windows set `model:` in the csolution's SSE-320-U85
target-set to `FVP_Corstone_SSE-320`.

## 7. If something does not work

- **J-Link connects but never stops at `main`:** the table of contents does
  not point at the debug stub. Repeat step 5.
- **Debug hangs at "Connecting":** the target-set was switched to
  `protocol: jtag`. Keep SWD: the generated load task blocks on JLinkExe's
  JTAG-chain prompt, and the device stays in SWD mode after any SWD use
  until it is power-cycled.
- **No console output:** SW4 is still on `SEUART`, or the port was opened
  before the switch was moved. Set `UART4` and reopen the port.
- **`app-write-mram` gets no answer:** press reset while it waits, check SW4
  is on `SEUART`, close any terminal holding the port.
- **"torch is not installed":** run the task **Setup Python virtual
  environment** first; it is only needed to regenerate the AI layer.
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
