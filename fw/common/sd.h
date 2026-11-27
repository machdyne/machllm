/*
 * machllm
 * Copyright (c) 2025 Lone Dynamics Corporation. All rights reserved.
 *
 * SD card (native mode) and FAT16/FAT32, reading into main memory.
 */

#ifndef SD_H
#define SD_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* platform: mlm_sd registers, main memory as the CPU sees it */
uint32_t plat_sd_rd(int reg);
void plat_sd_wr(int reg, uint32_t v);
uint8_t *plat_mem(void);

typedef struct {
	int ok, ccs, wide;
	uint32_t rca;
	uint32_t div;           /* the data clock's divider (sd_init's fast_div) */
} sd_t;

/* fast_div: SCK divider for data (0 = clk/2) */
int sd_init(sd_t *s, uint32_t fast_div);
/* n 512-byte blocks from lba to main memory at mem_addr (16-aligned) */
int sd_read(sd_t *s, uint32_t lba, uint32_t n, uint32_t mem_addr);
extern int sd_last_err;          /* the last failed read's code (-21 cmd timeout, -22 cmd CRC, -30 data timeout, -31 data CRC, -32 stuck) */
extern uint32_t sd_last_status;  /* and the host's STATUS then */
extern uint32_t sd_last_r1;      /* and the card's own status (CMD13; ~0: no answer) */

typedef struct {
	sd_t *sd;
	uint32_t scratch;       /* a 512-byte buffer in main memory */
	uint32_t cached;
	uint32_t spc, fat_lba, data_lba, root_clus, root_lba, root_secs;
	int fat32;
} fat_t;

/* part: which FAT partition of an MBR card (0 = the first); an
 * unpartitioned card has only 0. -2: no such partition. */
int fat_mount(fat_t *f, sd_t *s, uint32_t scratch, int part);
int fat_list(fat_t *f, int (*cb)(void *ctx, const char *name83, uint32_t clus, uint32_t size),
             void *ctx);
int32_t fat_load(fat_t *f, uint32_t clus, uint32_t size, uint32_t mem_addr,
                 void (*prog)(uint32_t done, uint32_t size));
extern uint32_t fat_retries;     /* pieces read again by the last fat_load */
extern uint32_t fat_reinits;     /* and times the card was initialised again */
extern uint32_t fat_fail_lba;    /* where it gave up */

#ifdef __cplusplus
}
#endif

#endif
