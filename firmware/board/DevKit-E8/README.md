# Board layer: Alif Ensemble E8 DevKit (M55_HP + Ethos-U85)

Device: `Alif Semiconductor::AE822FA0E5597LS0:M55_HP` (Cortex-M55 high-performance
core, 400 MHz) with the Ethos-U85 (256 MACs) of the Ensemble E8.

Derived from the Ensemble pack's `Boards/DevKit-e8/Layers/M55_HP/Board_HP-U85.clayer.yml`
and trimmed to what a headless inference runner needs. Camera, display,
Ethernet, USB, VIO and vStream drivers are left out. The target-set in the
csolution debugs it through the on-board J-Link (`J-Link Server`, SWD at
4 MHz, `start-pname: M55_HP`).

| File | Purpose |
|------|---------|
| `Board-U85.clayer.yml` | Layer: startup, SE services, UART4 stdio + stream, GPIO (RGB LED), I3C (ICM-42670-P accelerometer, driven by `src/imu.cpp`), Ethos-U85 driver, memory placement |
| `main.c` | Pin/GPIO config, SE services, clocks, stdio, NPU init, then `app_main()` |
| `retarget_stdio.c` | UART4 through the CMSIS USART driver, interrupt driven, 921600 8N1: stdout/stderr line buffers, and `stdio_rx_read()` (double-buffered receive of the EEG stream) |
| `ethos_setup.c` | Ethos-U85 driver init at `NPU_HG_BASE`, IRQ 366, prints the NPU banner |
| `ethosu_cb_dcache.c` | D-cache clean/invalidate hooks for NPU buffers outside the TCMs |
| `linker_ac6_mram.sct.src`, `linker_gnu_mram.ld.src` | Pack linker scripts plus a 32 kB stack and the `.bss.ai_pool` section in bulk SRAM |
| `RTE/` | Configuration files carried with the layer (see below) |

## Memory layout

| Region | Address | Used for |
|--------|---------|----------|
| MRAM (HP application region) | `0x80200000`, 2 MB | Constants, the embedded `.pte` model; with AC6 also the code, which executes in place. With GCC and Clang the startup code executes from MRAM and copies the rest of the code to ITCM (`linker_gnu_mram.ld.src`) |
| DTCM (SRAM3) | `0x20000000` (core alias; `0x50800000` global), 1 MB | `.data`/`.bss`, 96 kB heap, 32 kB stack |
| SRAM0/SRAM1 (bulk) | `0x02000000`, 8 MB combined (`SRAM0_SRAM1_COMBINED` in `app_mem_regions.h`; the GNU script uses SRAM0 alone, 4 MB) | `.bss.ai_pool`: the runner's 1 MB method pool and 2 MB temp pool (NPU scratch) |

The pool sizes and section come from the `define:` node of the layer
(`APP_METHOD_POOL_SIZE`, `APP_TEMP_POOL_SIZE`, `APP_POOL_SECTION`) and are
consumed by `src/app_main.cpp`.

## RTE configuration

`RTE/` is committed for this layer, as for the Corstone-320 one. The files are
the pack's own DevKit-E8 layer configuration (Conductor-generated
`pins.h`/`board_defs.h`, which differ from the pack defaults) with two Edge_Deploy
changes in `RTE_Device.h`: UART4 in interrupt mode
(`RTE_UART4_BLOCKING_MODE_ENABLE 0`; the pack's polled `Services:Retarget IO`
component is dropped, `retarget_stdio.c` replaces it) and an RX FIFO trigger of a
quarter (`RTE_UART4_RX_TRIG_LVL 1`). The ICM-42670-P IMU is read over I3C by
`src/imu.cpp` (polled, every wait bounded); the pack's ICM42670P component is
not used.

## STDIO

UART4 on the PRG USB connector, 921600 baud (the 250 Hz, 49-byte EEG frames
need 122.5 kbit/s). Set **SW4 to UART4** for the
console; SW4 in position **SEUART** (the default) routes the same USB serial
port to the Secure Enclave for SETOOLS. The on-board J-Link sits behind the
same connector.

## Before the first debug session

Program the debug stubs into the device's ATOC with Alif SETOOLS once:
**Terminal > Run Task > Alif: Install M55_HP debug stubs (DevKit-E8, single core configuration)**. The task needs
the VS Code setting `alif.setools.root` and SW4 in position SEUART. See
[the getting-started README](../../README.md).
