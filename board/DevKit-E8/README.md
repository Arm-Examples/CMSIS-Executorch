# Board layer: Alif Ensemble E8 DevKit (M55_HP + Ethos-U85)

Device: `Alif Semiconductor::AE822FA0E5597LS0:M55_HP` (Cortex-M55 high-performance
core, 400 MHz) with the Ethos-U85 (256 MACs) of the Ensemble E8.

Derived from the Ensemble pack's `Boards/DevKit-e8/Layers/M55_HP/Board_HP-U85.clayer.yml`
and trimmed to what the face generator needs: the console, the NPU, the SW2
joystick and the 4.3" MIPI-DSI display (ILI9806E, 480 x 800) on connector
J21, driven by the CDC200 display controller. Camera, Ethernet, USB, VIO and
the vStream drivers are left out. The target-set in the csolution debugs it
through the on-board J-Link (`J-Link Server`, SWD at 4 MHz,
`start-pname: M55_HP`).

| File | Purpose |
|------|---------|
| `Board-U85.clayer.yml` | Layer: startup, SE services, UART4 stdio, Ethos-U85 driver, display, memory placement |
| `main.c` | Cache invalidation, pin/GPIO config, SE services, clocks, MIPI DPHY power-up, stdio, NPU init, then `app_main()` |
| `display.c` | `board_display_image()`: brings up CDC200 + DSI + panel on first use and shows an RGB image scaled and centred in the RGB888 frame buffer |
| `buttons.c` | `board_buttons()`: SW2 joystick presses since the last call; `board_console_poll()`: a console read that does not block |
| `retarget_stdio.c` | `stdio_init()` for UART4 through the CMSIS USART driver (115200 8N1) |
| `ethos_setup.c` | Ethos-U85 driver init at `NPU_HG_BASE`, IRQ 366, prints the NPU banner |
| `ethosu_cb_dcache.c` | D-cache clean/invalidate hooks for NPU buffers outside the TCMs |
| `linker_ac6_mram.sct.src`, `linker_gnu_mram.ld.src` | Pack linker scripts plus a 32 kB stack and the `.bss.ai_pool` section in bulk SRAM, above the A32 boot stub |
| `RTE/` | Configuration files carried with the layer (see below) |

The layer's `define:` node sets what `src/app_main.cpp` does on this board:
the pool sizes and their section (`APP_METHOD_POOL_SIZE`,
`APP_TEMP_POOL_SIZE`, `APP_POOL_SECTION`), `APP_DISPLAY` (show every image
on the LCD, 128x128 scaled 3x to 384x384 and centred on black),
`APP_INTERACTIVE` (serve image requests on the console after the boot demo,
see STDIO) and `APP_BUTTONS` (SW2 left: one new image; right: start or stop
generating back to back).

## Memory layout

| Region | Address | Used for |
|--------|---------|----------|
| MRAM (HP application region) | `0x80200000`, 3.4375 MB | Constants and the embedded 2.8 MB `.pte` model; with AC6 also the code, which executes in place. With GCC and Clang the startup code executes from MRAM and copies the rest of the code to ITCM (`linker_gnu_mram.ld.src`) |
| MRAM (user region) | `0x80570000`, 32 kB | Unused; keeps the pack's MPU setup well-formed |
| DTCM (SRAM3) | `0x20000000` (core alias; `0x50800000` global), 1 MB | `.data`/`.bss` (latents, image buffer), 96 kB heap, 32 kB stack |
| SRAM0 (bulk) | `0x02010000`, 4 MB less 64 kB | `.bss.ai_pool`: the runner's 1 MB method pool and 1.5 MB temp pool (NPU scratch); `.bss.lcd_frame_buf`: the 1.15 MB RGB888 frame buffer |

The numbers are set in `RTE/Device/AE822FA0E5597LS0_M55_HP/app_mem_regions.h`.
The pack's default splits the 5.5 MB of application MRAM into 2 MB for the
HE core, 2 MB for the HP core and a 1.5 MB user region; the model does not
fit 2 MB, and this layer runs only the HP core, so its code region extends
over the user area up to `0x80570000`. The HP vector table stays at
`0x80200000`, where the debug stub's ATOC entry boots from, and the ATOC
package at the top of MRAM is not touched. The A32 boot stub the ATOC loads
to the start of SRAM0 is the reason the pools start 64 kB in
(`APP_SRAM_POOL_OFFSET`).

SRAM1 follows SRAM0 at `0x02400000` (the ATOC device configuration stitches
the two banks together; the device pack's `0x08000000` is not mapped then).
The M55_HP and the debugger can use it, but the Ethos-U85 could not run with
its weights there (`ethosu_invoke` returned -1), so the layer keeps
`SRAM0_SRAM1_COMBINED 0` and places nothing the NPU reads in SRAM1.

> [!Warning]
> Do not change the Secure Enclave run profile from this core. On the
> DevKit-E8, `SERVICES_set_run_cfg` calls that assign `memory_blocks` (the
> pack display demo's `MRAM_MASK | SRAM0_MASK`, for example) returned success
> and then left MRAM or SRAM0 unusable: MRAM reads `0xFF`, writes to
> `0x02xxxxxx` read back zero, and neither a debugger reset nor a power cycle
> recovers it. Recovery is SETOOLS over the SE UART (SW4 = SEUART):
> `maintenance`, Hard Maintenance mode, then the "Alif: Install M55_HP debug
> stubs" task and a new flash.

## RTE configuration

`RTE/` is committed for this layer, as for the Corstone-320 one; here it
matters: Alif's stdio retarget refuses to build unless UART4 is in
polling mode (`RTE_UART4_BLOCKING_MODE_ENABLE 1` in `RTE_Device.h`), and the
Conductor-generated `pins.h`/`board_defs.h` differ from the pack defaults. The
files are the pack's own DevKit-E8 layer configuration, with the memory
regions above changed in `app_mem_regions.h`.

## STDIO

UART4 on the PRG USB connector, 115200 baud. Set **SW4 to UART4** for the
console; SW4 in position **SEUART** (the default) routes the same USB serial
port to the Secure Enclave for SETOOLS. The on-board J-Link sits behind the
same connector.

With `APP_INTERACTIVE` the console carries binary image frames too, in
pico-faces' serial protocol: the host sends `G <seed> [steps] [class] [w]`,
the runner answers with the timing as text and a 49 kB RGB frame, which
takes about 4.3 s at 115200 baud. pico-faces' `viewer/view_serial.py` works
unchanged; close the Serial Monitor first, the viewer needs the port. A
faster console (`UART_BAUDRATE: 921600` in the layer's `define:` node, 0.5 s
per frame) needs the same rate in the viewer's `serial.Serial()` call.

## Before the first debug session

Program the debug stubs into the device's ATOC with Alif SETOOLS once:
**Terminal > Run Task > Alif: Install M55_HP debug stubs (DevKit-E8, single core configuration)**. The task needs
the VS Code setting `alif.setools.root` and SW4 in position SEUART. See
[the getting-started README](../../README.md).
