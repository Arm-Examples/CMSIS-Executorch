/* Dedicated streaming UART4 owner; no calls to the vendor stdin/stdout adapters. */
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "RTE_Components.h"
#include CMSIS_device_header
#include "Driver_USART.h"
extern ARM_DRIVER_USART Driver_USART4;
static ARM_DRIVER_USART*const uart=&Driver_USART4;
static volatile uint32_t milliseconds,head,tail,fault;
static uint8_t rx_byte,ring[1024];
static volatile uint32_t rx_idle_events,rx_error_events,rx_ring_overflows,rx_rearm_errors;
static volatile bool transmitting;
void SysTick_Handler(void){milliseconds++;}
uint32_t stream_millis(void){return milliseconds;}
static void event(uint32_t flags){
 /* Alif uses RX_TIMEOUT for FIFO-empty/character gaps while RX remains busy.
    Keep receiving; only the framed receiver decides inter-byte/frame expiry. */
 if(flags&ARM_USART_EVENT_RX_TIMEOUT)rx_idle_events++;
 if(flags&(ARM_USART_EVENT_RX_OVERFLOW|ARM_USART_EVENT_RX_BREAK|ARM_USART_EVENT_RX_FRAMING_ERROR|ARM_USART_EVENT_RX_PARITY_ERROR)){rx_error_events++;fault=1;}
 if(flags&ARM_USART_EVENT_RECEIVE_COMPLETE){uint32_t next=(head+1)%sizeof(ring);if(next==tail){rx_ring_overflows++;fault=1;}else{ring[head]=rx_byte;__DMB();head=next;}if(uart->Receive(&rx_byte,1)!=ARM_DRIVER_OK){rx_rearm_errors++;fault=1;}}
 if(flags&ARM_USART_EVENT_SEND_COMPLETE)transmitting=false;
}
int stdio_init(void){
 SystemCoreClockUpdate();if(SysTick_Config(SystemCoreClock/1000))return -1;
 if(uart->Initialize(event)!=ARM_DRIVER_OK||uart->PowerControl(ARM_POWER_FULL)!=ARM_DRIVER_OK||uart->Control(ARM_USART_MODE_ASYNCHRONOUS|ARM_USART_DATA_BITS_8|ARM_USART_PARITY_NONE|ARM_USART_STOP_BITS_1|ARM_USART_FLOW_CONTROL_NONE,115200)!=ARM_DRIVER_OK||uart->Control(ARM_USART_CONTROL_RX,1)!=ARM_DRIVER_OK||uart->Control(ARM_USART_CONTROL_TX,1)!=ARM_DRIVER_OK)return -1;
 return uart->Receive(&rx_byte,1)==ARM_DRIVER_OK?0:-1;
}
int stream_read_byte(void){if(head==tail)return -1;uint8_t b=ring[tail];__DMB();tail=(tail+1)%sizeof(ring);return b;}
int stream_uart_fault(void){uint32_t state=__get_PRIMASK();__disable_irq();int f=(int)fault;fault=0;if(f){uart->Control(ARM_USART_ABORT_RECEIVE,0);tail=head;if(uart->Receive(&rx_byte,1)!=ARM_DRIVER_OK){rx_rearm_errors++;fault=1;}}__set_PRIMASK(state);return f;}
bool stream_write(const uint8_t*p,size_t n){
 if(transmitting)return false;transmitting=true;
 if(uart->Send(p,(uint32_t)n)!=ARM_DRIVER_OK){transmitting=false;fault=1;return false;}
 uint32_t start=milliseconds;while(transmitting){if((uint32_t)(milliseconds-start)>1000){uart->Control(ARM_USART_ABORT_SEND,0);transmitting=false;fault=1;return false;}}
 return true;
}
/* Runtime diagnostics never interleave raw text with binary frames. A bounded
   diagnostic buffer is inspectable in the debugger; protocol ERROR carries failure. */
static char diagnostics[256];static unsigned diagnostic_position;
int stdout_putchar(int ch){diagnostics[diagnostic_position++%sizeof(diagnostics)]=(char)ch;return ch;}
int stderr_putchar(int ch){return stdout_putchar(ch);}
int stdin_getchar(void){return -1;}
