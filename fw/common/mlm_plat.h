/*
 * machllm
 * Copyright (c) 2025 Lone Dynamics Corporation. All rights reserved.
 *
 * What the model code needs from where it runs. The firmware (fw/)
 * implements this with the hardware; the PC reference (ref/) with plain
 * C that does exactly what the hardware does. See docs/engine.md.
 */

#ifndef MLM_PLAT_H
#define MLM_PLAT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The engine: x in, one int64 per row out. */
#define ENG_X_MAX    3072   /* int8 entries in the input vector (96 blocks of 32) */
#define ENG_OUT_MAX  4096   /* rows per call */

/* Main memory as the CPU sees it; the model image starts at offset 0. */
uint8_t *plat_mem(void);
uint32_t plat_mem_size(void);

/* The engine's input vector, MX-packed (see eng_mx_pack in eng_spec.h):
 * the int8 values four to a word, lowest first, and one word per block
 * of 32 holding its exponent offset (0..15). */
volatile uint32_t *plat_eng_x(void);
volatile uint32_t *plat_eng_xd(void);
/* The input loaded whole: nw value words, nb offset words (the PC
 * reference and the testbenches; the model code uses plat_eng_xpack). */
void plat_eng_xload(const uint32_t *q, int nw, const uint32_t *d, int nb);
/* v[0..n) MX-packed into the engine's input exactly as eng_mx_pack()
 * (eng_spec.h; zero from n to padn, a multiple of 32); returns its
 * exponent shift (eng_mx_pack's return value). The firmware has the
 * engine's packer read v from the RAM. Not while a run is going. */
int plat_eng_xpack(const int32_t *v, int n, int padn);

/* Main memory and the RAM. plat_mem_rd copies nw words from main memory
 * at off into dst and returns when done (the firmware uses the DMA unit
 * when the engine is not running, else the CPU). The two writes start
 * a copy from src (which must stay unchanged until plat_dma_wait):
 * nblk 16-byte blocks to off (16-byte aligned), or n bytes to off,
 * off + stride, ... . One at a time: each waits for the one before.
 * The engine must not read what they write before plat_dma_wait. */
void plat_mem_rd(void *dst, uint32_t off, uint32_t nw);
void plat_mem_wblk(uint32_t off, const void *src, uint32_t nblk);
void plat_mem_wcol(uint32_t off, const void *src, uint32_t n, uint32_t stride);
void plat_dma_wait(void);

/* rows x DOT(row, x) for rows of a 1-bit matrix at mem offset addr,
 * rows row_stride bytes apart, `groups` groups of 128 (eng_dot_spec in
 * eng_spec.h; qbits is 1). Blocks until done. rows <= ENG_OUT_MAX. */
void plat_eng_dot(uint32_t addr, uint32_t row_stride, uint32_t rows,
                  uint32_t groups, uint32_t qbits);

/* Two output banks: start a run into a bank without waiting, wait for
 * it, and choose the bank the reads below use. plat_eng_dot() is
 * start into bank 0 + wait + read bank 0. Results are the same either
 * way; only the time differs. The engine's input must not be written
 * while a run is going. */
void plat_eng_start(uint32_t addr, uint32_t row_stride, uint32_t rows,
                    uint32_t groups, uint32_t qbits, int bank);
void plat_eng_wait(void);
void plat_eng_rbank(int bank);

/* The GLU unit (see glu_spec() in eng_spec.h): SiLU(gate) x up from
 * banks 0 and 1 into the engine's input, n rows. Blocks; returns the
 * two shifts it chose. */
void plat_eng_glu(uint32_t n, int sg, int su, int sq, int *psh, int *xsh);

/* Result of row i (of the bank being read). */
int64_t plat_eng_out(uint32_t i);

/* The largest |result| of the last plat_eng_dot. */
uint64_t plat_eng_maxabs(void);

/* Row i's result shifted: (acc + 2^(sh-1)) >> sh, low 32 bits (sh 0..63).
 * The engine does this as it is read (OUT32), so a row costs one bus
 * read instead of two. */
int32_t plat_eng_out32(uint32_t i, int sh);

/* Rows i0 .. i0 + n - 1 of the bank being read, as plat_eng_out32 gives
 * them, into dst. The firmware has the engine's copy-out write them into
 * the RAM (dst must be in the RAM). */
void plat_eng_out32_block(int32_t *dst, uint32_t i0, uint32_t n, int sh);

/* Output. */
void plat_putc(int c);

/* Profiling: the model code marks where it is; the platform may count
 * time per section (MLM_PROF_*). Defined empty when not wanted. */
#ifdef MLM_PROFILE
void plat_prof(int section);
int plat_prof_cur(void);
#else
#define plat_prof(s) ((void)0)
#define plat_prof_cur() 0
#endif
/* PACK (inputs into the engine) and READ (results out of it) are counted
 * on their own, wherever they happen */
enum { MLM_PROF_NORM, MLM_PROF_QKV, MLM_PROF_QKROPE, MLM_PROF_KV, MLM_PROF_ATT,
       MLM_PROF_WO, MLM_PROF_FFN1, MLM_PROF_FFN2, MLM_PROF_FFN3, MLM_PROF_HEAD,
       MLM_PROF_SILU, MLM_PROF_PACK, MLM_PROF_READ, MLM_PROF_N };

#ifdef __cplusplus
}
#endif

#endif
