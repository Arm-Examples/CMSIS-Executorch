# Board: Arm V2M-MPS3-SSE-300-FVP

## Board Layer for Corstone-300 FVP

Device: SSE-300-MPS3

### System Configuration

| System Component        | Setting
|:------------------------|:----------------------------------
| Heap                    | 768 kB (configured in linker file)
| Stack (MSP)             |  32 kB (configured in linker file)

### STDIO mapping

**STDIO** is routed to terminal via **UART0** peripheral

### CMSIS-Driver mapping

| CMSIS-Driver           | Peripheral | Connection
|:-----------------------|:-----------|:----------------------
| Driver_USART0          | UART0      | STDOUT, STDERR
