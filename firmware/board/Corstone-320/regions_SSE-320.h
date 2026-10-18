// Copyright 2026 Arm Limited and/or its affiliates.
// SPDX-License-Identifier: Apache-2.0
//
// Memory regions of the Corstone-320 FVP for this example. Every region that
// differs from the SSE_320_BSP pack default says so in its comment.
#ifndef REGIONS_SSE_320_H
#define REGIONS_SSE_320_H

#include "sse320_memmap_s.h"
#include "sse320_memmap_ns.h"


//-------- <<< Use Configuration Wizard in Context Menu >>> --------------------
//------ With VS Code: Open Preview for Configuration Wizard -------------------

// <h> ROM Configuration
// =======================
// <h> __ROM0
//   <y> Base address
//   <i> Defines base address of memory region.
//   <i> Contains Startup and Vector Table
//   <i> Pack default: 0x11000000 (boot ROM). Its 128 KB are far too small for
//   <i> the ExecuTorch runtime + kernels, so code and rodata are placed in the
//   <i> 2 MB FPGA SRAM. To boot on the FVP, set INITSVTOR to this base
//   <i> (fvp_config.txt does).
#define __ROM0_BASE   FPGA_SRAM_S_BASE
//   <y> Region size [bytes]
//   <i> Defines size of memory region.
#define __ROM0_SIZE   FPGA_SRAM_S_SIZE
// </h>

// <h> __ROM1
//   <y> Base address
//   <i> Defines base address of memory region.
//   <i> Pack default: 0x01000000 (FPGA SRAM, non-secure alias)
#define __ROM1_BASE   FPGA_SRAM_S_BASE
//   <y> Region size [bytes]
//   <i> Defines size of memory region.
//   <i> Pack default: 0x00200000
#define __ROM1_SIZE   FPGA_SRAM_S_SIZE
// </h>

// <h> __ROM2
//   <y> Base address
//   <i> Defines base address of memory region.
//   <i> Pack default: 0 (region unused)
#define __ROM2_BASE   DDR4_3_S_BASE
//   <y> Region size [bytes]
//   <i> Defines size of memory region.
//   <i> Pack default: 0
#define __ROM2_SIZE   DDR4_3_S_SIZE
// </h>

// <h> __ROM3
//   <y> Base address
//   <i> Defines base address of memory region.
//   <i> Pack default: 0 (region unused)
#define __ROM3_BASE   QSPI_FLASH_S_BASE
//   <y> Region size [bytes]
//   <i> Defines size of memory region.
//   <i> Pack default: 0
#define __ROM3_SIZE   QSPI_FLASH_S_SIZE
// </h>

// </h>

// <h> RAM Configuration
// =======================
// <h> __RAM0
//   <y> Base address
//   <i> Defines base address of memory region.
//   <i> Pack default: 0x10000000 (ITCM). 256 MB DDR4 here holds .data/.bss, the two
//   <i> 4 MB inference pools, the Ethos-U cache buffer, heap and stack
#define __RAM0_BASE   DDR4_1_S_BASE
//   <y> Region size [bytes]
//   <i> Defines size of memory region.
//   <i> Pack default: 0x00008000
#define __RAM0_SIZE   DDR4_1_S_SIZE
// </h>

// <h> __RAM1
//   <y> Base address
//   <i> Defines base address of memory region.
//   <i> Pack default: 0x12000000 (FPGA SRAM)
#define __RAM1_BASE   SRAM_VM0_S_BASE
//   <y> Region size [bytes]
//   <i> Defines size of memory region.
//   <i> Pack default: 0x00200000
#define __RAM1_SIZE   SRAM_VM0_S_SIZE
// </h>

// <h> __RAM2
//   <y> Base address
//   <i> Defines base address of memory region.
//   <i> Pack default: 0x30000000 (DTCM)
#define __RAM2_BASE   SRAM_VM1_S_BASE
//   <y> Region size [bytes]
//   <i> Defines size of memory region.
//   <i> Pack default: 0x00008000
#define __RAM2_SIZE   SRAM_VM1_S_SIZE
// </h>

// <h> __RAM3
//   <y> Base address
//   <i> Defines base address of memory region.
//   <i> Pack default: 0x31000000 (SRAM VM0)
#define __RAM3_BASE   DTCM_S_BASE
//   <y> Region size [bytes]
//   <i> Defines size of memory region.
//   <i> Pack default: 0x00400000
#define __RAM3_SIZE   DTCM_S_SIZE
// </h>

// </h>

// <h> Stack / Heap Configuration
//   <i> Pack defaults: 0x600 stack, 0xC00 heap. The runner's EmbeddedModule
//   <i> keeps its method table and planned buffers on the heap.
//   <o0> Stack Size (in Bytes) <0x0-0xFFFFFFFF:8>
//   <o1> Heap Size (in Bytes) <0x0-0xFFFFFFFF:8>
#define __STACK_SIZE  0x00001000
#define __HEAP_SIZE   0x00018000
// </h>

#endif /* REGIONS_SSE_320_H */
