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
 *---------------------------------------------------------------------------*/

/* Semihosting operation numbers (Arm semihosting specification). */
#define SYS_WRITEC  0x03
#define SYS_READC   0x07

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
