/*---------------------------------------------------------------------------
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *      Name:    retarget_stdio.c
 *      Purpose: Retarget stdio to Arm semihosting
 *
 * CMSIS-Compiler STDOUT/STDERR/STDIN:Custom backend. The CMSIS-Compiler CORE
 * component provides the toolchain-specific low-level retarget (newlib _write
 * for GCC) and routes every character here. Characters go out over Arm
 * semihosting (BKPT 0xAB), which the FVP serves directly on its stdout --
 * no UART model, base address, or driver involved. Requires
 * mps4_board.subsystem.cpu0.semihosting-enable=1 (see fvp_config.txt).
 * The same channel ends the simulation (stdio_exit) and writes files on the
 * host (board_save_file).
 *---------------------------------------------------------------------------*/

#include <stddef.h>
#include <stdint.h>

/* Semihosting operation numbers (Arm semihosting specification). */
#define SYS_OPEN    0x01
#define SYS_CLOSE   0x02
#define SYS_WRITEC  0x03
#define SYS_WRITE   0x05
#define SYS_READC   0x07
#define SYS_EXIT    0x18

/* SYS_EXIT reason codes */
#define ADP_Stopped_RunTimeErrorUnknown  0x20023
#define ADP_Stopped_ApplicationExit      0x20026

static int semihosting_call (int op, void *param) {
  register int   r0 __asm__("r0") = op;
  register void *r1 __asm__("r1") = param;
  __asm__ volatile ("bkpt 0xAB" : "+r"(r0) : "r"(r1) : "memory");
  return r0;
}

/**
  Initialize stdio

  \return          0 on success, or -1 on error.
*/
int stdio_init (void) {
  /* Semihosting needs no initialization. */
  return 0;
}

/**
  Put a character to the stderr

  \param[in]   ch  Character to output
  \return          The character written, or -1 on write error.
*/
int stderr_putchar (int ch) {
  char c = (char)ch;
  (void)semihosting_call(SYS_WRITEC, &c);
  return ch;
}

/**
  Put a character to the stdout

  \param[in]   ch  Character to output
  \return          The character written, or -1 on write error.
*/
int stdout_putchar (int ch) {
  char c = (char)ch;
  (void)semihosting_call(SYS_WRITEC, &c);
  return ch;
}

/**
  Get a character from the stdio

  \return     The next character from the input, or -1 on read error.
*/
int stdin_getchar (void) {
  return semihosting_call(SYS_READC, 0);
}

/**
  End the simulation through semihosting SYS_EXIT (the FVP stops).

  \param[in]   status  Exit status of the application: 0 reports a normal exit,
                       anything else a run-time error
*/
void stdio_exit (int status) {
  /* On AArch32, R1 holds the reason code itself, not a parameter block. */
  uintptr_t reason = (status == 0) ? ADP_Stopped_ApplicationExit : ADP_Stopped_RunTimeErrorUnknown;
  (void)semihosting_call(SYS_EXIT, (void *)reason);
  for (;;) { __asm__ volatile ("wfi"); }
}

/**
  Save a buffer to a file on the simulation host through semihosting
  (SYS_OPEN "wb", SYS_WRITE, SYS_CLOSE). Relative paths resolve against the
  model's working directory, i.e. the workspace when the CMSIS extension
  starts the FVP. Used by the runner to hand the generated image and the
  measurements to the host for checking (APP_RESULT_DIR).

  \param[in]   path   File to create
  \param[in]   data   Bytes to write
  \param[in]   n      Number of bytes
  \return      0 on success, -1 on failure
*/
int board_save_file (const char *path, const void *data, size_t n) {
  size_t len = 0U;
  while (path[len] != '\0') { len++; }
  uintptr_t open_args[3]  = { (uintptr_t)path, 5U /* "wb" */, len };
  int fd = semihosting_call(SYS_OPEN, open_args);
  if (fd < 0) { return -1; }
  uintptr_t write_args[3] = { (uintptr_t)fd, (uintptr_t)data, n };
  int not_written = semihosting_call(SYS_WRITE, write_args);
  uintptr_t close_args[1] = { (uintptr_t)fd };
  (void)semihosting_call(SYS_CLOSE, close_args);
  return (not_written == 0) ? 0 : -1;
}
