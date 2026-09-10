# Board layer: Alif Ensemble E8 DevKit (M55_HP + Ethos-U85)

Device: `Alif Semiconductor::AE822FA0E5597LS0:M55_HP` (Cortex-M55 high-performance
core, 400 MHz) with the Ethos-U85 (256 MACs) of the Ensemble E8.

Derived from the Ensemble pack's `Boards/DevKit-e8/Layers/M55_HP/Board_HP-U85.clayer.yml`
and trimmed to what a headless inference runner needs. Camera, display,
Ethernet, USB, VIO and vStream drivers are left out.

| File | Purpose |
|------|---------|
| `Board-U85.clayer.yml` | Layer: startup, SE services, UART4 stdio, Ethos-U85 driver, memory placement |
| `main.c` | Pin/GPIO config, SE services, clocks, stdio, NPU init, then `app_main()` |
| `retarget_stdio.c` | `stdio_init()` for UART4 through the CMSIS USART driver (115200 8N1) |
| `ethos_setup.c` | Ethos-U85 driver init at `NPU_HG_BASE`, IRQ 366, prints the NPU banner |
| `ethosu_cb_dcache.c` | D-cache clean/invalidate hooks for NPU buffers outside the TCMs |
| `linker_ac6_mram.sct.src`, `linker_gnu_mram.ld.src` | Pack linker scripts plus a 32 kB stack and the `.bss.ai_pool` section in bulk SRAM |
| `RTE/` | Configuration files carried with the layer (see below) |

## Memory layout

| Region | Address | Used for |
|--------|---------|----------|
| MRAM (HP application region) | `0x80200000`, 2 MB | Code, constants, the embedded `.pte` model |
| DTCM (SRAM3) | `0x20000000`, 1 MB | `.data`/`.bss`, 96 kB heap, 32 kB stack |
| SRAM0 (bulk) | `0x02000000`, 4 MB | `.bss.ai_pool`: the runner's 1 MB method pool and 2 MB temp pool (NPU scratch) |

The pool sizes and section come from the `define:` node of the layer
(`APP_METHOD_POOL_SIZE`, `APP_TEMP_POOL_SIZE`, `APP_POOL_SECTION`) and are
consumed by `src/app_main.cpp`.

## RTE configuration

Unlike the Corstone-320 layer, `RTE/` is committed for this layer (see
`.gitignore`): Alif's stdio retarget refuses to build unless UART4 is in
polling mode (`RTE_UART4_BLOCKING_MODE_ENABLE 1` in `RTE_Device.h`), and the
Conductor-generated `pins.h`/`board_defs.h` differ from the pack defaults. The
files are the pack's own DevKit-E8 layer configuration.

## STDIO

UART4 on the PRG USB connector, 115200 baud. Set **SW4 to UART4** for the
console; SW4 in position **SEUART** (the default) routes the same USB serial
port to the Secure Enclave for SETOOLS. The on-board J-Link sits behind the
same connector.

## Before the first debug session

Program the debug stubs into the device's ATOC with Alif SETOOLS once:
**Terminal > Run Task > Alif: Install M55_HP debug stubs**. The task needs
the VS Code setting `alif.setools.root` and SW4 in position SEUART. See
[the hackathon README](../../README.md).
