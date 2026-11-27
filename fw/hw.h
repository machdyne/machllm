/*
 * machllm
 * Copyright (c) 2025 Lone Dynamics Corporation. All rights reserved.
 *
 * The memory map of rtl/mlm_soc.v.
 */
#ifndef HW_H
#define HW_H
#include <stdint.h>
#define REG(a) (*(volatile uint32_t *)(a))

#define MAIN_BASE   0x40000000u
#define MAIN_SIZE   (512u << 20)

#define LED         REG(0x70000000)
#define SYS_ID      REG(0x70000004)
#define SYS_BUILD   REG(0x70000008)
#define SYS_LINE    REG(0x7000000c)  /* bit 0: keep main memory's last read line (rtl/mlm_soc.v) */

#define DDR3_STATUS REG(0x70000700)
#define DDR3_TUNE   REG(0x70000704)
#define DDR3_CTRL   REG(0x70000708)

#define UART_DATA   REG(0x71000000)
#define UART_STATUS REG(0x71000004)
#define UART_DIV    REG(0x71000008)

/* system clock, MHz (the Makefile's SYS_MHZ); the DDR3 clock is twice this */
#ifndef SYS_MHZ
#define SYS_MHZ 48
#endif
/* the memory side's clock (DDR3 / 2; the SD host runs on it): the same
 * as SYS_MHZ unless the SOC has two clock domains */
#ifndef MEM_MHZ
#define MEM_MHZ SYS_MHZ
#endif
/* the SD host runs on the CPU's clock (in every build); its data clock
 * = SYS_MHZ / (2 (div + 1)), at most 25 MHz (default speed) */
#define SD_FAST_DIV ((SYS_MHZ + 49) / 50 - 1)

#define SD_BASE     0x72000000u

#define ENG_BASE    0x80000000u
#define ENG_ADDR    REG(0x80004000)
#define ENG_STRIDE  REG(0x80004004)
#define ENG_ROWS    REG(0x80004008)
#define ENG_GROUPS  REG(0x8000400c)
#define ENG_QBITS   REG(0x80004010)
#define ENG_CTRL    REG(0x80004014)
#define ENG_CYCLES  REG(0x80004018)
#define ENG_WAITS   REG(0x8000401c)
#define ENG_MAXLO   REG(0x80004020)
#define ENG_MAXHI   REG(0x80004024)
#define ENG_SHIFT   REG(0x80004028)
#define ENG_OBANK   REG(0x8000402c)
#define ENG_MAX1LO  REG(0x80004030)
#define ENG_MAX1HI  REG(0x80004034)
#define ENG_GLU_N   REG(0x80004038)
#define ENG_GLU_P   REG(0x8000403c)
#define ENG_GLU_GO  REG(0x80004040)
#define ENG_GLU_RES REG(0x80004044)
#define ENG_CO_GO   REG(0x80004048)   /* copy-out: n rows into the mirror */
#define ENG_CO_ROW  REG(0x8000404c)   /* bank << 12 | first row */
#define ENG_MIRROR  0x74000000u       /* the copy-out's results, read like RAM */
#define ENG_CI_GO   REG(0x80004050)   /* copy-in: nv | nb << 16 from the staging RAM */
#define ENG_XSTAGE  0x75000000u       /* the copy-in's source, written like RAM */

/* vector unit (rtl/mlm_vu.v) */
#define VU_A        REG(0x73000000)
#define VU_B        REG(0x73000004)
#define VU_D        REG(0x73000008)
#define VU_N        REG(0x7300000c)
#define VU_P0       REG(0x73000010)
#define VU_P1       REG(0x73000014)
#define VU_OP       REG(0x73000018)   /* write: run (returns when done) */
#define VU_R0       REG(0x7300001c)
#define VU_SUMLO    REG(0x73000020)
#define VU_SUMHI    REG(0x73000024)
#define VU_ID       REG(0x73000028)   /* "VU01" */
#endif
