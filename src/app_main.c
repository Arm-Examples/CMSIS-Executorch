/*------------------------------------------------------------------------------
 * Executorch Example Application Main Entry Point
 * Copyright (c) 2004-2025 Arm Limited (or its affiliates). All rights reserved.
 *------------------------------------------------------------------------------
 * Name:    app_main.c
 * Purpose: Application main entry point and ExecuTorch runner integration
 *----------------------------------------------------------------------------*/

#include "RTE_Components.h"
#include <stdio.h>

// Forward declaration of ExecuTorch runner main function
extern int executorch_runner_main(int argc, const char* argv[]);


#ifdef RTE_CMSIS_RTOS2 

#include "cmsis_os2.h"

// Custom stack size for ExecuTorch thread (32KB)
#define EXECUTORCH_THREAD_STACK_SIZE (4 * 1024)

// Stack memory for the ExecuTorch thread
static uint64_t executorch_thread_stack[EXECUTORCH_THREAD_STACK_SIZE / sizeof(uint64_t)] __attribute__((aligned(8)));

// Thread attributes with custom stack
static const osThreadAttr_t executorch_thread_attr = {
  .name = "executorch_runner",
  .stack_mem = executorch_thread_stack,
  .stack_size = sizeof(executorch_thread_stack),
  .priority = osPriorityNormal,
};

/*-----------------------------------------------------------------------------
  Application Main Thread 'app_main_thread': Run ExecuTorch inference
 *----------------------------------------------------------------------------*/
__NO_RETURN void app_main_thread (void *argument) {
  (void)argument;

  printf("ExecuTorch Cortex-M Runner\n");

  // Call the ExecuTorch runner with no arguments (model embedded)
  const char* argv[] = {"executorch_runner"};
  int result = executorch_runner_main(1, argv);

  printf("Inference completed with result: %d\n", result);

  // Infinite loop after completion
  while (1) {
    osDelay(1000);
  }
}

#endif

/*-----------------------------------------------------------------------------
 *        Application main function
 *----------------------------------------------------------------------------*/
int app_main (void) {

#ifdef RTE_CMSIS_RTOS2
  osKernelInitialize();
  osThreadNew(app_main_thread, NULL, &executorch_thread_attr);
  osKernelStart();
#else
  executorch_runner_main(0, NULL);
#endif
  return 0;
}
