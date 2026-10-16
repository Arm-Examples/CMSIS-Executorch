# Board layer: Alif Ensemble E8 AI/ML AppKit (M55_HP + Ethos-U85)

Device: `Alif Semiconductor::AE822FA0E5597BS0:M55_HP` (Cortex-M55 high-performance
core, 400 MHz) with the Ethos-U85 (256 MACs) of the Ensemble E8. Board
`Alif Semiconductor::AppKit-E8-AIML` in the Ensemble pack.

The AppKit carries the E8 die of the DevKit in the 216-ball WLCSP package
(the DevKit's `AE822FA0E5597LS0` is the 194-ball FBGA): the same cores, NPU,
5.5 MB MRAM and 9.75 MB SRAM. The pack ships a board entry and the
Conductor configuration for it (`Boards/AppKit-e8/board_defs.h`, `pins.h`,
`gpios.h`, through the component `BSP:AppKit Config&AppKit-E8-AIML`), but no
Board layer and no examples. This layer is therefore the
[DevKit-E8 layer](../DevKit-E8/README.md) with the AppKit's board support:
everything the face generator uses is wired the same on both boards.

| | DevKit-E8 | AppKit-E8 |
|---|---|---|
| Console | UART4 on P12_1/P12_2 to the PRG USB (USB-C) bridge, SW4 = `UART4` | UART4 on the same pins to the PRG USB (Micro-USB) bridge, SW4 = `UART4` (SW4 also has `UART2` for the HE core) |
| Debug | J-Link OB on PRG USB, SWD 4 MHz | same |
| Display | 4.3" ILI9806E 480 x 800 on MIPI-DSI connector J21, CDC200, backlight GPIO6_1 | the same panel soldered on, same controller and pins; capacitive touch (GT911, unused) |
| Joystick | SW2, five-way on LPGPIO pins 0-4 | five-way joystick on the same LPGPIO pins |
| LEDs | 2 RGB | 1 RGB (unused) |
| Not on the AppKit | | Ethernet, mikroBUS, the 2x20 headers, ADC/CMP/DAC, I2C/SPI headers |

| File | Purpose |
|------|---------|
| `Board-U85.clayer.yml` | Layer: startup, SE services, UART4 stdio, Ethos-U85 driver, display, memory placement; `BSP:AppKit Config` instead of `BSP:DevKit Config` |
| `AppKit-E8.h` | CMSIS target header: the USART instance of the console |
| `RTE/` | Configuration files carried with the layer, named after the BS0 part (see below) |
| `../DevKit-E8/*.c`, `*.h`, `linker_*.src` | The board sources and linker scripts, shared with the DevKit-E8 layer |

The `define:` node sets the same pools, sections, `APP_DISPLAY`,
`APP_INTERACTIVE` and `APP_BUTTONS` as the DevKit layer; see
[its README](../DevKit-E8/README.md) for the memory layout, the SRAM1 and
Secure Enclave run-profile notes and the serial protocol. They apply
unchanged.

## RTE configuration

The RTE directories are named after the device, so this layer carries its
own copies under `AE822FA0E5597BS0_M55_HP`. The `Device/` and `Services/`
files are the same pack files for both parts (the pack's condition is the
SoC, `AE822FA0E5597*`), copied from the DevKit layer with its edits:
`RTE_UART4_BLOCKING_MODE_ENABLE 1` in `RTE_Device.h` (Alif's stdio retarget
needs polling mode) and the memory regions in `app_mem_regions.h` (the HP
MRAM region extended to 3.4375 MB for the model, the pools above the A32
boot stub in SRAM0, `SRAM0_SRAM1_COMBINED 0`, 96 kB heap). The `BSP/` files
are the pack's AppKit-e8 Conductor configuration, unchanged.

## STDIO

UART4 on the PRG USB connector (Micro-USB), 115200 baud. Set **SW4 to UART4**
(the bottom position); the top position **SEUART** routes the same USB serial
port to the Secure Enclave for SETOOLS, the middle one to UART2.

## Before the first debug session

Program the debug stubs into the device's ATOC with Alif SETOOLS once:
**Terminal > Run Task > Alif: Install M55_HP debug stubs (AppKit-E8, single core configuration)**.
It is the DevKit task with the AppKit's part, `E8 (AE822FA0E5597BS0) - 5.5
MRAM / 9.75 SRAM`; the ATOC configuration and the HP stub in `.alif/` are the
same. The task needs the VS Code setting `alif.setools.root` and SW4 in
position SEUART. See [the getting-started README](../../README.md).
