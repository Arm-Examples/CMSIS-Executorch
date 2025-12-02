/*------------------------------------------------------------------------------
 * MDK Middleware - Component ::Network
 * Copyright (c) 2004-2024 Arm Limited (or its affiliates). All rights reserved.
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
  osThreadNew(app_main_thread, NULL, NULL);
  osKernelStart();
#else
  executorch_runner_main(0, NULL);
#endif
  return 0;
}
