/*
 * machllm
 * Copyright (c) 2025 Lone Dynamics Corporation. All rights reserved.
 *
 * Firmware: DDR3 training, self-tests, model loading from the SD card,
 * and a chat / story prompt on the serial console. See docs/usage.md.
 */

#include <stdint.h>
#include "hw.h"
#include "common/mlm.h"
#include "common/mlm_plat.h"
#include "common/mlm_model.h"
#include "common/sd.h"
#include "common/eng_spec.h"

/* ---------------------------------------------------------------- */
/* console                                                          */

void plat_putc(int c)
{
	if (c == '\n') plat_putc('\r');
	while (!(UART_STATUS & 1)) ;
	UART_DATA = (uint32_t)c;
}
static void puts_(const char *s) { while (*s) plat_putc(*s++); }
static void put_hex(uint32_t v, int n) { while (n--) plat_putc("0123456789abcdef"[(v >> (4 * n)) & 15]); }
static void put_dec(uint32_t v)
{
	char b[12]; int n = 0;
	do { b[n++] = (char)('0' + v % 10); v /= 10; } while (v);
	while (n) plat_putc(b[--n]);
}
/* v / 100 with two decimals */
static void put_dec2(uint32_t v100) { put_dec(v100 / 100); plat_putc('.'); plat_putc('0' + (v100 / 10) % 10); plat_putc('0' + v100 % 10); }
static int getc_(void)
{
	uint32_t d;
	do d = UART_DATA; while (d & 0x100);
	return (int)(d & 0xff);
}
static int readline(char *b, int max)
{
	int n = 0;
	for (;;) {
		int c = getc_();
		if (c == '\r' || c == '\n') { puts_("\n"); b[n] = 0; return n; }
		if ((c == 8 || c == 127) && n) { n--; puts_("\b \b"); continue; }
		if (c >= 32 && n < max - 1) { b[n++] = (char)c; plat_putc(c); }
	}
}
static uint64_t cycles(void)
{
	uint32_t hi, lo, hi2;
	do {
		__asm__ volatile ("rdcycleh %0" : "=r"(hi));
		__asm__ volatile ("rdcycle %0" : "=r"(lo));
		__asm__ volatile ("rdcycleh %0" : "=r"(hi2));
	} while (hi != hi2);
	return ((uint64_t)hi << 32) | lo;
}

/* ---------------------------------------------------------------- */
/* platform: memory, engine, SD                                     */

uint8_t *plat_mem(void) { return (uint8_t *)MAIN_BASE; }
uint32_t plat_mem_size(void) { return MAIN_SIZE; }
volatile uint32_t *plat_eng_x(void) { return (volatile uint32_t *)ENG_BASE; }
volatile uint32_t *plat_eng_xd(void) { return (volatile uint32_t *)(ENG_BASE + 0x1000); }

static uint64_t eng_cycles_total;      /* memory-side cycles */
static uint32_t eng_waits_total;
#define ENG_CPU(c) ((uint64_t)(c) * SYS_MHZ / MEM_MHZ)

static int eng_rb;
void plat_eng_start(uint32_t addr, uint32_t stride, uint32_t rows, uint32_t groups, uint32_t qbits, int bank)
{
	/* each write crosses to the memory clock: the ones that would not
	 * change a register are left out (nothing else writes these) */
	static uint32_t c_stride = ~0u, c_rows = ~0u, c_groups = ~0u, c_qbits = ~0u, c_bank = ~0u;
	ENG_ADDR = addr;           /* engine addresses are offsets in main memory */
	if (stride != c_stride) { ENG_STRIDE = stride; c_stride = stride; }
	if (rows != c_rows) { ENG_ROWS = rows; c_rows = rows; }
	if (groups != c_groups) { ENG_GROUPS = groups; c_groups = groups; }
	if (qbits != c_qbits) { ENG_QBITS = qbits; c_qbits = qbits; }
	if ((uint32_t)bank != c_bank) { ENG_OBANK = (uint32_t)bank; c_bank = (uint32_t)bank; }
	ENG_CTRL = 1;
}

void plat_eng_wait(void)
{
	while (ENG_CTRL & 1) ;
	/* the engine counts its own (memory-side) cycles; ENG_CPU() turns
	 * them into CPU cycles where they are reported (a 64-bit division
	 * here, ~45 times a layer, was a slow library call each) */
	eng_cycles_total += ENG_CYCLES;
	eng_waits_total += ENG_WAITS;
}

void plat_eng_rbank(int bank) { eng_rb = bank; }

/* set-up, tests and the console: optimised for size (RAM is short) */
#define COLD __attribute__((cold))

/* the vector unit: the model's vector loops on the RAM (mlm_vu) */
#ifdef VU_PROF
static uint64_t cycles(void);
static uint32_t vup_c[16], vup_n[16], vup_e[16];
#endif
static uint32_t fw_vu(int op, const void *a, const void *b, void *d, uint32_t n,
                      int32_t p0, int32_t p1, uint64_t *sum)
{
#ifdef VU_PROF
	uint64_t t0 = cycles();
	vup_n[op]++; vup_e[op] += n;
#endif
	VU_A = (uint32_t)a; VU_B = (uint32_t)b; VU_D = (uint32_t)d; VU_N = n;
	VU_P0 = (uint32_t)p0; VU_P1 = (uint32_t)p1;
	VU_OP = (uint32_t)op;                   /* the CPU waits here */
	if (sum) *sum = VU_SUMLO | ((uint64_t)VU_SUMHI << 32);
#ifdef VU_PROF
	uint32_t r = VU_R0;
	vup_c[op] += (uint32_t)(cycles() - t0);
	return r;
#else
	return VU_R0;
#endif
}
static int vu_present(void) { return VU_ID == 0x56553031u; }

/* the unit against the C loops: every operation on random data */
COLD static uint32_t vu_selftest(void)
{
	uint32_t *S = (uint32_t *)mlm_scratch();       /* 4096 words */
	uint32_t *A = S, *B = S + 520, *D = S + 1040;  /* the unit's */
	uint32_t *A2 = S + 1560, *B2 = S + 2080, *D2 = S + 2600;   /* the loops' */
	uint32_t seed = 777, bad = 0;
	for (int t = 0; t < 140; t++) {
		int op = 1 + t % 14, n = 1 + (int)((seed >> 8) % 256);
		int32_t p0 = 0, p1 = 0;
		seed = seed * 1103515245u + 12345u;
		switch (op) {
		case MLM_VU_SQMUL: p0 = (int32_t)(seed >> 27); break;
		case MLM_VU_SCALE15: p0 = (int16_t)(seed >> 8); break;
		case MLM_VU_RSHR: p0 = (int32_t)((seed >> 26) % 40) - 10; break;
		case MLM_VU_RESID: p0 = 1 + (int32_t)(seed >> 28); p1 = 1 + (int32_t)((seed >> 24) & 15); break;
		case MLM_VU_MULSHR30: p0 = (int32_t)(seed >> 2); break;
		case MLM_VU_QPACK: p0 = (int32_t)(seed >> 27); p1 = (n + 9) & ~1; break;
		case MLM_VU_KVQ8: n &= ~3; if (!n) n = 4; p0 = (int32_t)(seed >> 28); p1 = (int32_t)((seed >> 8) % (127u << 16)); break;
		case MLM_VU_ROPE: n = 128; p0 = (int32_t)(seed >> 31); break;
		case MLM_VU_ATTSC: p0 = (int32_t)((seed >> 20) % 61) - 30; break;
		case MLM_VU_QPACKU: p0 = (int32_t)((seed >> 26) % 30) - 10; p1 = (n + 9) & ~1; break;
		case MLM_VU_MXPACK: p1 = (n + 31) & ~31; break;
		default: break;
		}
		for (int i = 0; i < 520; i++) {
			seed = seed * 1103515245u + 12345u;
			uint32_t x = seed ^ (seed << 13), sh = (seed >> 3) % 24;
			/* within each operation's range, as the model uses it (beyond
			 * it, C's signed overflow has no defined result to compare) */
			if (op == MLM_VU_SQMUL && 17 - p0 > (int32_t)sh) sh = (uint32_t)(17 - p0);
			A[i] = A2[i] = (op == MLM_VU_RSHR && p0 < 0) ? (uint32_t)((int32_t)x >> 16) : (uint32_t)((int32_t)x >> sh);
			B[i] = B2[i] = x * 2654435761u;
			D[i] = D2[i] = 0;
			/* attention: scale words (16-bit mantissa, small exponent),
			 * softmax inputs near their maximum, exponentials up to 1.0 */
			if (op == MLM_VU_ATTSC || op == MLM_VU_COEF)
				B[i] = B2[i] = (x & 0xffff) | ((uint32_t)(uint16_t)((int)((x >> 20) % 61) - 40) << 16);
			if (op == MLM_VU_EXPSUM) A[i] = A2[i] = (uint32_t)(-(int32_t)((x >> 8) % (20u << 16)));
			if (op == MLM_VU_COEF) A[i] = A2[i] = (x >> 7) % 65537;
		}
		if (op == MLM_VU_EXPSUM) p0 = 0;           /* every input is <= 0 */
		if (op == MLM_VU_COEF) p0 = 20;            /* every exponent is <= 20 */
		int inpl = (op == MLM_VU_RESID || op == MLM_VU_ROPE || op == MLM_VU_ATTSC ||
		            op == MLM_VU_EXPSUM || op == MLM_VU_COEF);
		uint64_t s1 = 0, s2 = 0;
		uint32_t r1 = fw_vu(op, A, B, inpl ? A : D, (uint32_t)n, p0, p1, &s1);
		uint32_t r2 = mlm_vu_ref(op, A2, B2, inpl ? A2 : D2, (uint32_t)n, p0, p1, &s2);
		int wrong = 0;
		for (int i = 0; i < 520; i++) if (A[i] != A2[i] || D[i] != D2[i]) wrong = 1;
		if (op == MLM_VU_MXPACK) {
			for (int i = 0; i < 520; i++) if (B[i] != B2[i]) wrong = 1;
			if (r1 != r2) wrong = 1;
		}
		if ((op == MLM_VU_MAXABS || op == MLM_VU_RESID || op == MLM_VU_ATTSC || op == MLM_VU_COEF) && r1 != r2) wrong = 1;
		if ((op == MLM_VU_SQMUL || op == MLM_VU_EXPSUM) && s1 != s2) wrong = 1;
		if (wrong) { puts_(" [op "); put_dec((uint32_t)op); puts_(" n "); put_dec((uint32_t)n);
		             puts_(" p0 "); put_dec((uint32_t)p0); puts_("]"); }
		bad += (uint32_t)wrong;
	}
	return bad;
}

void plat_eng_glu(uint32_t n, int sg, int su, int sq, int *psh, int *xsh)
{
	ENG_GLU_N = n;
	ENG_GLU_P = (uint32_t)(sg & 63) | ((uint32_t)(su & 63) << 8) | ((uint32_t)(sq & 255) << 16);
	ENG_GLU_GO = 1;
	while (ENG_CTRL & 1) ;
	uint32_t r = ENG_GLU_RES;
	*psh = (int)(r & 0xff);
	*xsh = (int)(int8_t)(r >> 8);
}

void plat_eng_dot(uint32_t addr, uint32_t stride, uint32_t rows, uint32_t groups, uint32_t qbits)
{
	plat_eng_start(addr, stride, rows, groups, qbits, 0);
	plat_eng_wait();
	eng_rb = 0;
}

int64_t plat_eng_out(uint32_t i)
{
	volatile uint32_t *o = (volatile uint32_t *)(ENG_BASE + (eng_rb ? 0x18000 : 0x8000) + 8 * i);
	uint32_t lo = o[0], hi = o[1];
	return (int64_t)(((uint64_t)hi << 32) | lo);
}

#ifdef MLM_PROFILE
static uint64_t prof_t[MLM_PROF_N], prof_e[MLM_PROF_N], prof_last, prof_elast;
static int prof_cur = -1;
static uint64_t cycles(void);
void plat_prof(int sec)
{
	uint64_t now = cycles();
	if (prof_cur >= 0) {
		prof_t[prof_cur] += now - prof_last;
		prof_e[prof_cur] += eng_cycles_total - prof_elast;
	}
	prof_cur = sec; prof_last = now; prof_elast = eng_cycles_total;
}
int plat_prof_cur(void) { return prof_cur; }
COLD static void prof_report(void)
{
	static const char *nm[MLM_PROF_N] = { "norm", "qkv", "qknorm+rope", "kv store", "attention",
		"wo+res", "ffn gate", "ffn up+mul", "ffn down+res", "head", "silu (w3 running)",
		"(of these: packing inputs)", "(of these: reading results)" };
	plat_prof(MLM_PROF_HEAD);
	prof_cur = -1;
	for (int i = 0; i < MLM_PROF_N; i++) {
		puts_("  "); puts_(nm[i]); puts_(": "); put_dec((uint32_t)(prof_t[i] / 1000));
		puts_(" kcyc, engine "); put_dec((uint32_t)(ENG_CPU(prof_e[i]) / 1000)); puts_("\n");
		prof_t[i] = prof_e[i] = 0;
	}
#ifdef VU_PROF
	for (int i = 1; i < 16; i++) if (vup_n[i]) {
		puts_("  vu op "); put_dec((uint32_t)i); puts_(": "); put_dec(vup_n[i]); puts_(" calls, ");
		put_dec(vup_e[i]); puts_(" elements, "); put_dec(vup_c[i] / 1000); puts_(" kcyc\n");
		vup_c[i] = vup_n[i] = vup_e[i] = 0;
	}
#endif
}
#endif

uint64_t plat_eng_maxabs(void)
{
	return eng_rb ? (((uint64_t)ENG_MAX1HI << 32) | ENG_MAX1LO) : (((uint64_t)ENG_MAXHI << 32) | ENG_MAXLO);
}

static int eng_cur_sh = -1;
int32_t plat_eng_out32(uint32_t i, int sh)
{
	if (sh != eng_cur_sh) { ENG_SHIFT = (uint32_t)sh; eng_cur_sh = sh; }
	return ((volatile int32_t *)(ENG_BASE + (eng_rb ? 0x14000 : 0x10000)))[i];
}

/* the engine's input: into the staging RAM at RAM speed, then the
 * engine copies it (a word a cycle); waits for it, as a run must not
 * start before X is complete */
void plat_eng_xload(const uint32_t *q, int nw, const uint32_t *d, int nb)
{
	volatile uint32_t *st = (volatile uint32_t *)ENG_XSTAGE;
	int j = 0;
	for (; j + 4 <= nw; j += 4) {
		uint32_t a0 = q[j], a1 = q[j + 1], a2 = q[j + 2], a3 = q[j + 3];
		st[j] = a0; st[j + 1] = a1; st[j + 2] = a2; st[j + 3] = a3;
	}
	for (; j < nw; j++) st[j] = q[j];
	for (int b = 0; b < nb; b++) st[768 + b] = d[b];
	ENG_CI_GO = (uint32_t)nw | ((uint32_t)nb << 16);
	while (ENG_CTRL & 4) ;
}

/* n rows from i0 through the engine's copy-out and the mirror, 2048 at a
 * time: one crossing to start, then reads at RAM speed */
void plat_eng_out32_block(int32_t *dst, uint32_t i0, uint32_t n, int sh)
{
	if (sh != eng_cur_sh) { ENG_SHIFT = (uint32_t)sh; eng_cur_sh = sh; }
	const volatile int32_t *mir = (const volatile int32_t *)ENG_MIRROR;
	while (n) {
		uint32_t c = n > 2048 ? 2048 : n;
		ENG_CO_ROW = ((uint32_t)eng_rb << 12) | i0;
		ENG_CO_GO = c;
		while (ENG_CTRL & 2) ;          /* bit 1: the copy-out (a run may go on) */
		/* eight loads, then eight stores: the CPU waits on each load's
		 * answer, not on the loop */
		uint32_t k = 0;
		for (; k + 8 <= c; k += 8) {
			int32_t a0 = mir[k], a1 = mir[k + 1], a2 = mir[k + 2], a3 = mir[k + 3];
			int32_t a4 = mir[k + 4], a5 = mir[k + 5], a6 = mir[k + 6], a7 = mir[k + 7];
			dst[k] = a0; dst[k + 1] = a1; dst[k + 2] = a2; dst[k + 3] = a3;
			dst[k + 4] = a4; dst[k + 5] = a5; dst[k + 6] = a6; dst[k + 7] = a7;
		}
		for (; k < c; k++) dst[k] = mir[k];
		dst += c; i0 += c; n -= c;
	}
}

uint32_t plat_sd_rd(int reg) { return ((volatile uint32_t *)SD_BASE)[reg]; }
void plat_sd_wr(int reg, uint32_t v) { ((volatile uint32_t *)SD_BASE)[reg] = v; }

/* ---------------------------------------------------------------- */
/* DDR3 read-path training: Zeitlos's sw/bios/bios.c (b62a74bf),    */
/* reduced to what the result needs                                 */

#define CTRL_HOLD_REFRESH 1
#define CTRL_TRAINING (4 | 8)

static const uint32_t ddr3_pat[4] = { 0x04030201, 0x08070605, 0x0c0b0a09, 0x100f0e0d };
static uint8_t ddr3_tbl[2][8][8];

static uint32_t tune_of(uint32_t wd, uint32_t rs, uint32_t g, uint32_t o)
{ return wd | (g << 4) | (g << 8) | (rs << 12) | (rs << 16) | (o << 20) | (o << 24); }

static void write_pat(void)
{
	volatile uint32_t *m = (volatile uint32_t *)MAIN_BASE;
	DDR3_CTRL = CTRL_TRAINING;
	for (int i = 0; i < 4; i++) m[i] = ddr3_pat[i];
	DDR3_CTRL = CTRL_TRAINING | CTRL_HOLD_REFRESH;
}

static uint32_t score(void)
{
	volatile uint32_t *m = (volatile uint32_t *)MAIN_BASE;
	uint32_t ok = 3;
	for (int i = 0; i < 4; i++) {
		uint32_t a = m[i], b = m[i];
		if (a != b) return 0;
		if ((a & 0x00ff00ff) != (ddr3_pat[i] & 0x00ff00ff)) ok &= ~1u;
		if ((a & 0xff00ff00) != (ddr3_pat[i] & 0xff00ff00)) ok &= ~2u;
	}
	return ok;
}

static int nb(int lane, int rs, int g, uint8_t v)
{
	if (rs < 0) { rs = 7; g--; }
	if (rs > 7) { rs = 0; g++; }
	if (g < 0 || g > 7) return 0;
	return ddr3_tbl[lane][rs][g] == v;
}

static int pick(int lane, uint32_t *rs_o, uint32_t *g_o, uint32_t *o_o)
{
	int best = -1;
	for (int rs = 0; rs < 8; rs++)
		for (int g = 0; g < 8; g++) {
			uint8_t v = ddr3_tbl[lane][rs][g];
			if (v >= 0xfe) continue;
			int sc = nb(lane, rs - 1, g, v) + nb(lane, rs + 1, g, v);
			if (sc > best) { best = sc; *rs_o = rs; *g_o = g; *o_o = v; }
		}
	return best + 1;
}

COLD static int ddr3_train(void)
{
	volatile uint32_t *m = (volatile uint32_t *)MAIN_BASE;
	uint32_t wd, rs0 = 0, g0 = 0, o0 = 0, rs1 = 0, g1 = 0, o1 = 0;
	int p0 = 0, p1 = 0;
	puts_("ddr3 ");
	for (uint32_t n = 0; n < 2000000 && !(DDR3_STATUS & 1); n++) ;
	put_hex(DDR3_STATUS, 8);
	if (!(DDR3_STATUS & 1)) { puts_(" init timeout\n"); return 0; }
	puts_("\n");
	for (wd = 0; wd < 8; wd++) {
		for (uint32_t rs = 0; rs < 8; rs++) {
			DDR3_TUNE = tune_of(wd, 0, 0, 0);
			write_pat();
			for (uint32_t g = 0; g < 8; g++) {
				ddr3_tbl[0][rs][g] = ddr3_tbl[1][rs][g] = 0xff;
				for (uint32_t o = 0; o < 16; o++) {
					DDR3_TUNE = tune_of(wd, rs, g, o);
					uint32_t sc = score();
					if (sc & 1) ddr3_tbl[0][rs][g] = ddr3_tbl[0][rs][g] == 0xff ? (uint8_t)o : 0xfe;
					if (sc & 2) ddr3_tbl[1][rs][g] = ddr3_tbl[1][rs][g] == 0xff ? (uint8_t)o : 0xfe;
				}
			}
		}
		p0 = pick(0, &rs0, &g0, &o0);
		p1 = pick(1, &rs1, &g1, &o1);
		if (p0 && p1) break;
	}
	DDR3_CTRL = 0;
	if (wd == 8) { puts_("ddr3 training FAILED\n"); return 0; }
	DDR3_TUNE = wd | (g0 << 4) | (g1 << 8) | (rs0 << 12) | (rs1 << 16) | (o0 << 20) | (o1 << 24);
	puts_("ddr3 tune "); put_hex(DDR3_TUNE, 8);
	puts_(p0 == 3 ? " L0:C" : " L0:e");
	puts_(p1 == 3 ? " L1:C\n" : " L1:e\n");
	m[64] = 0xa5c3a5c3;
	m[65] = 0x5a3c5a3c;
	((volatile uint8_t *)MAIN_BASE)[64 * 4 + 1] = 0x77;
	if (m[64] != 0xa5c377c3 || m[65] != 0x5a3c5a3c) { puts_("ddr3 check FAILED\n"); return 0; }
	return (p0 == 3 && p1 == 3) ? 2 : 1;   /* 2: both lanes centred */
}

/* ---------------------------------------------------------------- */
/* self-tests                                                       */

static uint32_t lfsr(uint32_t x) { return (x >> 1) ^ (-(x & 1u) & 0xd0000001u); }

COLD static int memtest(uint32_t off, uint32_t bytes)
{
	volatile uint32_t *m = (volatile uint32_t *)(MAIN_BASE + off);
	uint32_t x = 0x1234567;
	for (uint32_t i = 0; i < bytes / 4; i++) { m[i] = x ^ i; x = lfsr(x); }
	x = 0x1234567;
	for (uint32_t i = 0; i < bytes / 4; i++) {
		if (m[i] != (x ^ i)) return (int)i + 1;
		x = lfsr(x);
	}
	return 0;
}

/* stream bandwidth: the engine reads `mb` MB as Q8 rows of 4096 bytes */
COLD static void bwtest(uint32_t mb)
{
	volatile uint32_t *x = plat_eng_x(), *xd = plat_eng_xd();
	for (int i = 0; i < ENG_X_MAX / 4; i++) x[i] = 0;
	for (int i = 0; i < ENG_X_MAX / 32; i++) xd[i] = 0;
	uint32_t rows = mb * 256, done = 0, cyc = 0, waits = 0;
	if (!mb) rows = 32;
	uint32_t stride = 17 * 8 * 30;   /* 30 chunks of 128 weights at Q8: 4080 bytes */
	while (done < rows) {
		uint32_t n = rows - done > 4096 ? 4096 : rows - done;
		eng_cycles_total = eng_waits_total = 0;
		plat_eng_dot(done * stride, stride, n, 120, 8);
		cyc += (uint32_t)ENG_CPU(eng_cycles_total); waits += (uint32_t)ENG_CPU(eng_waits_total);
		done += n;
	}
	uint64_t bytes = (uint64_t)rows * stride;
	puts_("stream read: "); put_dec((uint32_t)(bytes >> 10)); puts_(" KB in "); put_dec(cyc);
	puts_(" cycles = "); put_dec2((uint32_t)(bytes * (SYS_MHZ * 100ull) / cyc));
	puts_(" MB/s at "); put_dec(SYS_MHZ); puts_(" MHz ("); put_dec((uint32_t)(100ull * (cyc - waits) / cyc)); puts_("% of cycles busy)\n");
	/* the CPU path, for comparison: word loads */
	volatile uint32_t *m = (volatile uint32_t *)MAIN_BASE;
	uint64_t t0 = cycles();
	uint32_t s = 0;
	uint32_t words = mb >= 32 ? 65536 : 1024;
	for (uint32_t i = 0; i < words; i++) s += m[i];
	uint32_t t = (uint32_t)(cycles() - t0);
	puts_("cpu word loads: "); put_dec(words / 256); puts_(" KB in "); put_dec(t); puts_(" cycles = ");
	put_dec2((uint32_t)((uint64_t)words * 4 * (SYS_MHZ * 100ull) / t)); puts_(" MB/s"); if (s == 1) puts_(" ");
	puts_("\n");
}

/* engine vs the specification computed by the CPU (through the CPU's
 * own path to DDR3), on rows of the loaded model's tensors */
static int64_t st_want[64];

static void put_s64(int64_t v)
{
	if (v < 0) { plat_putc('-'); v = -v; }
	put_hex((uint32_t)((uint64_t)v >> 32), 8); plat_putc('_'); put_hex((uint32_t)v, 8);
}

/* :e -- the engine on simple inputs, want (CPU) and got (engine) for
 * the first rows of layer 0's wq: tells a datapath fault from a wrong
 * specification */
COLD static void engdiag(void)
{
	const mlm_header_t *h = (const mlm_header_t *)plat_mem();
	const mlm_layer_t *L = (const mlm_layer_t *)(plat_mem() + h->layers_off);
	const mlm_tensor_t *t = &L[0].wq;
	volatile uint32_t *x = plat_eng_x(), *xd = plat_eng_xd();
	uint32_t *xs = (uint32_t *)mlm_scratch();
	uint8_t *xds = (uint8_t *)(xs + ENG_X_MAX / 4);
	uint32_t gr = t->groups < ENG_X_MAX / 128 ? t->groups : ENG_X_MAX / 128, seed = 777;
	puts_("wq: rows "); put_dec(t->rows); puts_(", groups "); put_dec(t->groups);
	puts_(", qbits "); put_dec(t->qbits); puts_(", stride "); put_dec(t->row_stride); puts_("\n");
	/* X written and read back (the engine idle): its block RAM, both ports */
	{
		static const uint32_t pat[4] = { 0x01020304u, 0x7f80ff00u, 0xa5a55a5au, 0x00000001u };
		for (int i = 0; i < 4; i++) { x[i] = pat[i]; x[700 + i] = ~pat[i]; }
		xd[0] = 5; xd[1] = 15; xd[90] = 7;
		puts_("X read back:");
		for (int i = 0; i < 4; i++) { plat_putc(' '); put_hex(x[i], 8); }
		for (int i = 0; i < 4; i++) { plat_putc(' '); put_hex(x[700 + i], 8); }
		puts_(" | offsets "); put_hex(xd[0], 2); plat_putc(' '); put_hex(xd[1], 2); plat_putc(' '); put_hex(xd[90], 2);
		puts_("  (want 01020304 7f80ff00 a5a55a5a 00000001 fefdfcfb 807f00ff 5a5aa5a5 fffffffe | 05 0f 07)\n");
	}
	static const char *nm[6] = { "q=0", "q=+1", "q=-1", "q=+1 first value only", "random", "random, offsets" };
	for (int k = 0; k < 6; k++) {
		for (uint32_t i = 0; i < gr * 32; i++) {
			uint32_t w;
			if (k == 0) w = 0;
			else if (k == 1) w = 0x01010101u;
			else if (k == 2) w = 0xffffffffu;
			else if (k == 3) w = i == 0 ? 1u : 0u;
			else { seed = seed * 1103515245u + 12345u; w = seed & 0x7f7f7f7fu; if (seed & 0x80000000u) w ^= 0x80808080u; }
			xs[i] = w; x[i] = w;
		}
		for (uint32_t b = 0; b < gr * 4; b++) {
			uint8_t d = 0;
			if (k == 5) { seed = seed * 1103515245u + 12345u; d = (uint8_t)((seed >> 20) & 15); }
			xds[b] = d; xd[b] = d;
		}
		plat_eng_dot(t->off, t->row_stride, 4, gr, t->qbits);
		eng_dot_spec(plat_mem(), (const int8_t *)xs, xds, st_want, t->off, t->row_stride, 4, gr, t->qbits);
		puts_(nm[k]); puts_(":\n");
		for (uint32_t r = 0; r < 4; r++) {
			int64_t g = plat_eng_out(r);
			puts_("  row "); put_dec(r); puts_(" want "); put_s64(st_want[r]);
			puts_(" got "); put_s64(g); puts_(g == st_want[r] ? "\n" : "  DIFF\n");
		}
	}
	/* the last run (random, offsets) again: cycles, the max, and the
	 * results through the mirror (copy-out) instead of the register window */
	plat_eng_dot(t->off, t->row_stride, 4, gr, t->qbits);
	puts_("run: "); put_dec(ENG_CYCLES); puts_(" engine cycles, max "); put_s64((int64_t)plat_eng_maxabs()); puts_("\n");
	{
		int32_t mo[4];
		plat_eng_out32_block(mo, 0, 4, 0);
		puts_("mirror (shift 0):");
		for (int r = 0; r < 4; r++) { plat_putc(' '); put_hex((uint32_t)mo[r], 8); }
		puts_("\n");
	}
	/* 8-bit rows (the KV cache's mode: its own eight multipliers), all
	 * weights +1, at the top of main memory */
	{
		uint32_t base = MAIN_SIZE - 4096;
		volatile uint32_t *w = (volatile uint32_t *)(plat_mem() + base);
		for (uint32_t r = 0; r < 4; r++) {
			w[r * 36] = 0x00010001u; w[r * 36 + 1] = 0x00010001u;
			for (int i = 0; i < 32; i++) w[r * 36 + 2 + i] = r == 1 ? 0xffffffffu : r == 2 ? 0x00000001u : 0x01010101u;
		}
		for (int k = 0; k < 2; k++) {
			for (uint32_t i = 0; i < 32; i++) { uint32_t v = k ? 0x02fe7f81u : 0x01010101u; xs[i] = v; x[i] = v; }
			for (uint32_t b = 0; b < 4; b++) { xds[b] = 0; xd[b] = 0; }
			plat_eng_dot(base, 144, 4, 1, 8);
			eng_dot_spec(plat_mem(), (const int8_t *)xs, xds, st_want, base, 144, 4, 1, 8);
			puts_(k ? "8-bit, mixed x:\n" : "8-bit, x = +1:\n");
			for (uint32_t r = 0; r < 4; r++) {
				int64_t g = plat_eng_out(r);
				puts_("  row "); put_dec(r); puts_(" want "); put_s64(st_want[r]);
				puts_(" got "); put_s64(g); puts_(g == st_want[r] ? "\n" : "  DIFF\n");
			}
		}
	}
	/* the first chunk of row 0 as stored: four multipliers, then bits */
	const volatile uint32_t *c = (const volatile uint32_t *)(plat_mem() + t->off);
	puts_("row 0, first chunk:");
	for (int i = 0; i < 6; i++) { plat_putc(' '); put_hex(c[i], 8); }
	puts_("\n");
}
COLD static int selftest(void)
{
	const mlm_header_t *h = (const mlm_header_t *)plat_mem();
	const mlm_layer_t *L = (const mlm_layer_t *)(plat_mem() + h->layers_off);
	const mlm_tensor_t *ts[4] = { &L[0].wq, &L[0].w2, &L[h->n_layers - 1].w1, &h->head };
	volatile uint32_t *x = plat_eng_x(), *xd = plat_eng_xd();
	uint32_t *xs = (uint32_t *)mlm_scratch();   /* the model's scratch: free between steps */
	uint8_t *xds = (uint8_t *)(xs + ENG_X_MAX / 4);
	uint32_t seed = 12345, bad = 0, rows = 0;
	for (int k = 0; k < 4; k++) {
		const mlm_tensor_t *t = ts[k];
		uint32_t n = t->rows < 64 ? t->rows : 64;
		uint32_t r0 = (t->rows - n) / 2;
		/* at most what X holds: the rows' first groups (Bonsai's down
		 * projection has 48, run by the model in two pieces of 24) */
		uint32_t gr = t->groups < ENG_X_MAX / 128 ? t->groups : ENG_X_MAX / 128;
		/* a random x: int8 values (+-127) and block offsets */
		for (uint32_t i = 0; i < gr * 128 / 4; i++) {
			uint32_t w = 0;
			for (int j = 0; j < 4; j++) {
				seed = seed * 1103515245u + 12345u;
				int v = (int)((seed >> 16) % 255) - 127;
				w |= (uint32_t)(uint8_t)v << (8 * j);
			}
			xs[i] = w;
			x[i] = w;
		}
		for (uint32_t b = 0; b < gr * 4; b++) {
			seed = seed * 1103515245u + 12345u;
			xds[b] = (uint8_t)((seed >> 20) & 15);
			xd[b] = xds[b];
		}
		uint32_t a = t->off + r0 * t->row_stride;
		plat_eng_dot(a, t->row_stride, n, gr, t->qbits);
		eng_dot_spec(plat_mem(), (const int8_t *)xs, xds, st_want, a, t->row_stride, n, gr, t->qbits);
		for (uint32_t r = 0; r < n; r++) {
			rows++;
			if (plat_eng_out(r) != st_want[r]) {
				if (!bad) { puts_("engine self-test: tensor "); put_dec((uint32_t)k); puts_(" row ");
				            put_dec(r0 + r); puts_(" differs\n"); }
				bad++;
			}
		}
	}
	puts_("engine self-test: "); put_dec(rows); puts_(" rows, ");
	put_dec(bad); puts_(bad ? " WRONG\n" : " wrong\n");

	/* the GLU unit: gate and up of the last layer on a random x, then
	 * the unit against the CPU, at three input scales */
	const mlm_tensor_t *w1 = &L[h->n_layers - 1].w1, *w3 = &L[h->n_layers - 1].w3;
	uint32_t n = w1->rows < 1024 ? w1->rows : 1024, gbad = 0;
	for (int k = 0; k < 3; k++) {
		for (uint32_t i = 0; i < w1->groups * 128 / 4; i++) {
			seed = seed * 1103515245u + 12345u;
			x[i] = seed & 0x7f7f7f7fu;
		}
		for (uint32_t b = 0; b < w1->groups * 4; b++) xd[b] = (uint32_t)k;
		plat_eng_start(w1->off, w1->row_stride, n, w1->groups, w1->qbits, 0);
		plat_eng_wait();
		plat_eng_start(w3->off, w3->row_stride, n, w3->groups, w3->qbits, 1);
		plat_eng_wait();
		/* shifts as the forward pass picks them: gate to 30 bits, up
		 * to 15; the sigmoid's input scale around its useful range */
		int sg = 0, su = 0;
		plat_eng_rbank(0);
		for (uint64_t v = plat_eng_maxabs(); v; v >>= 1) sg++;
		plat_eng_rbank(1);
		for (uint64_t v = plat_eng_maxabs(); v; v >>= 1) su++;
		sg = sg > 30 ? sg - 30 : 0;
		su = su > 15 ? su - 15 : 0;
		int sq = (k == 0) ? -20 : (k == 1) ? -14 : -8;
		gbad += mlm_glu_check(n, sg, su, sq);
	}
	puts_("glu self-test: "); put_dec(3 * n); puts_(" elements, ");
	put_dec(gbad); puts_(gbad ? " WRONG (:g uses the CPU instead)\n" : " wrong\n");
	plat_eng_rbank(0);
	uint32_t vbad = 0;
	if (vu_present()) {
		vbad = vu_selftest();
		puts_("vector unit self-test: 140 operations, "); put_dec(vbad);
		puts_(vbad ? " WRONG (:u uses the CPU instead)\n" : " wrong\n");
	}
	return bad == 0 && gbad == 0 && vbad == 0;
}

/* ---------------------------------------------------------------- */
/* models on the card                                               */

#define MAX_MODELS 16
static struct { char name[13]; uint32_t clus, size; } models[MAX_MODELS];
static int n_models;
static sd_t sd;
static fat_t fat;

static int is_mlm(const char *n)
{
	int l = 0; while (n[l]) l++;
	return l > 4 && n[l - 4] == '.' && n[l - 3] == 'M' && n[l - 2] == 'L' && n[l - 1] == 'M';
}
static int list_cb(void *ctx, const char *n, uint32_t c, uint32_t sz)
{
	(void)ctx;
	if (is_mlm(n) && n_models < MAX_MODELS) {
		int i = 0;
		while (n[i] && i < 12) { models[n_models].name[i] = n[i]; i++; }
		models[n_models].name[i] = 0;
		models[n_models].clus = c;
		models[n_models].size = sz;
		n_models++;
	}
	return 0;
}

static mlm_ctx_t M;
static int model_ok;
static uint32_t ctx_len = 512;
static uint32_t sd_div = SD_FAST_DIV;   /* :sdclk N: SCK = SYS_MHZ / (2 (N + 1)) */

/* loading progress, on one line: MB so far, of MB, percent */
static uint32_t prog_last;
static void load_prog(uint32_t done, uint32_t size)
{
	uint32_t pct = (uint32_t)((uint64_t)done * 100 / size);
	if (done < size && pct == prog_last) return;
	prog_last = pct;
	puts_("\rloading "); put_dec(done >> 20); puts_(" / "); put_dec(size >> 20);
	puts_(" MB  "); put_dec(pct); puts_("%   ");
}

COLD static int load_model(int k)
{
	puts_("loading "); puts_(models[k].name); puts_(" ("); put_dec(models[k].size >> 20); puts_(" MB)\n");
	/* a failed load must not leave a half image that looks usable */
	((volatile uint32_t *)plat_mem())[0] = 0;
	model_ok = 0;
	prog_last = ~0u;
	uint64_t t0 = cycles();
	int32_t rc = fat_load(&fat, models[k].clus, models[k].size, 0, load_prog);
	uint32_t t = (uint32_t)(cycles() - t0);
	if (rc < 0) {
		((volatile uint32_t *)plat_mem())[0] = 0;
		puts_("\nload failed: sd error "); put_dec((uint32_t)-rc);
		puts_(" at card sector "); put_dec(fat_fail_lba);
		puts_(" after 6 tries, last at "); put_dec(SYS_MHZ * 1000u / (2u * (sd.div + 1u)) / 1000u);
		puts_(" MHz (host status "); put_hex(sd_last_status, 2);
		puts_(", card status "); put_hex(sd_last_r1, 8); puts_(")\n");
		return 0;
	}
	if (fat_retries) {
		puts_("\n("); put_dec(fat_retries); puts_(" pieces read again");
		if (fat_reinits) {
			puts_(", the card initialised again "); put_dec(fat_reinits); puts_(" times; SD clock now ");
			put_dec(SYS_MHZ * 1000u / (2u * (sd.div + 1u)) / 1000u); puts_(" MHz");
			sd_div = sd.div;
		}
		puts_(")");
	}
	puts_("\ndone in "); put_dec2(t / (SYS_MHZ * 10000u)); puts_(" s, ");
	put_dec2((uint32_t)((uint64_t)models[k].size * (SYS_MHZ * 100ull) / t)); puts_(" MB/s\n");
	int e = mlm_init(&M, ctx_len);
	if (e) { puts_("not a usable model image: "); put_dec((uint32_t)-e); puts_("\n"); return 0; }
	const mlm_header_t *h = M.hdr;
	puts_((const char *)plat_mem() + h->name_off);
	puts_(": dim "); put_dec(h->dim); puts_(", layers "); put_dec(h->n_layers);
	puts_(", heads "); put_dec(h->n_heads); plat_putc('/'); put_dec(h->n_kv_heads);
	puts_(", vocab "); put_dec(h->vocab); puts_(", context "); put_dec(M.ctx); puts_("\n");
	selftest();
	return 1;
}

/* ---------------------------------------------------------------- */

/* generated text: wrapped at word boundaries (many terminals don't
 * wrap themselves), control characters dropped (a terminal may answer
 * some of them, ENQ for one, and the answer would become input) */
static uint32_t wrap_cols = 80;         /* :w N; 0 = no wrapping */
static uint32_t col, wcols, wlen;
static uint8_t wbuf[96];
static void word_flush(void)
{
	if (wrap_cols && col && col + wcols > wrap_cols) { plat_putc('\n'); col = 0; }
	for (uint32_t i = 0; i < wlen; i++) plat_putc(wbuf[i]);
	col += wcols;
	wlen = wcols = 0;
}
static void text_out(uint8_t c)
{
	if (c == '\n') { word_flush(); plat_putc('\n'); col = 0; return; }
	if (c == '\t') c = ' ';
	if (c < 0x20 || c == 0x7f) return;
	if (c == ' ') {
		word_flush();
		if (wrap_cols && col >= wrap_cols) { plat_putc('\n'); col = 0; }
		else { plat_putc(' '); col++; }
		return;
	}
	if (wlen == sizeof wbuf) word_flush();
	wbuf[wlen++] = c;
	if ((c & 0xc0) != 0x80) wcols++;     /* UTF-8: count characters, not bytes */
}
static void text_end(void) { word_flush(); }

static void put_tok(uint32_t prev, uint32_t tok)
{
	uint32_t len;
	const uint8_t *s = mlm_decode(&M, prev, tok, &len);
	for (uint32_t i = 0; s && i < len; i++) text_out(s[i]);
}

static uint32_t toks[600];         /* a 511-character line plus the chat template */
static uint32_t max_new = 256;
static int verbose;

static void generate(const char *line)
{
	col = wcols = wlen = 0;
	const mlm_header_t *h = M.hdr;
	int qwen = (h->flags & MLM_F_CHAT_QWEN) != 0;
	int n = 0;
	if (!qwen) {
		mlm_init(&M, ctx_len);          /* stories: each prompt starts over */
		if (h->flags & MLM_F_BOS) toks[n++] = h->bos_id;
		n += mlm_encode(&M, line, toks + n, 580 - n);
	} else {
		/* non-thinking Qwen3 chat turn; the conversation stays in the cache */
		toks[n++] = h->im_start_id;
		n += mlm_encode(&M, "user\n", toks + n, 580 - n);
		n += mlm_encode(&M, line, toks + n, 580 - n);
		toks[n++] = h->im_end_id;
		n += mlm_encode(&M, "\n", toks + n, 580 - n);
		toks[n++] = h->im_start_id;
		n += mlm_encode(&M, "assistant\n", toks + n, 580 - n);
		if (h->think_id != 0xffffffffu) {
			toks[n++] = h->think_id;
			n += mlm_encode(&M, "\n\n", toks + n, 580 - n);
			toks[n++] = h->think_end_id;
			n += mlm_encode(&M, "\n\n", toks + n, 580 - n);
		}
		if (M.pos + n + 32 > M.ctx) { puts_("[context full: starting over]\n"); mlm_init(&M, ctx_len); }
	}
	if (!n) return;
	eng_cycles_total = eng_waits_total = 0;
	uint64_t t0 = cycles(), t_first = 0;
	uint32_t tok = toks[0], prev = h->bos_id, gen = 0;
	if (!qwen) put_tok(prev, tok);
	/* a chat turn's prompt is not printed: show that it is being read */
	if (qwen) { puts_("(reading "); put_dec((uint32_t)n); puts_(" tokens) "); }
	for (int i = 0; ; i++) {
		uint64_t s0 = cycles();
		uint64_t e0 = eng_cycles_total;
		if (mlm_step_ex(&M, tok, i + 1 >= n)) { puts_("\n[context full]"); break; }
		if (qwen && i < n) plat_putc(i + 1 < n ? '.' : '\n');
		if (verbose) {
			uint32_t dt = (uint32_t)(cycles() - s0), de = (uint32_t)ENG_CPU(eng_cycles_total - e0);
			puts_("{"); put_dec(M.pos - 1); puts_(": "); put_dec(dt / 1000); puts_(" kcyc, engine ");
			put_dec(de / 1000); puts_(" kcyc}");
#ifdef MLM_PROFILE
			puts_("\n"); prof_report();
#endif
		}
		uint32_t next;
		if (i + 1 < n) {
			next = toks[i + 1];
			if (!qwen) put_tok(tok, next);
		} else {
			if (!t_first) t_first = cycles() - t0;
			next = mlm_sample(&M);
			if (next == h->eos_id || next == h->eos2_id || next == h->im_end_id) {
				if (qwen) { mlm_step_ex(&M, next, 0); }   /* keep <|im_end|> in the cache */
				break;
			}
			put_tok(tok, next);
			if (++gen >= max_new) break;
		}
		prev = tok;
		tok = next;
		/* a key stops it; a lone LF (the tail of a CR LF Enter) does not */
		if (UART_STATUS & 2) {
			uint32_t c = UART_DATA & 0xff;
			if (c != '\n') { text_end(); puts_(" [stopped]"); break; }
		}
	}
	text_end();
	(void)prev;
	uint64_t t = cycles() - t0;
	uint64_t tg = t - t_first;
	puts_("\n[");
	put_dec((uint32_t)n); puts_(" prompt tokens in "); put_dec2((uint32_t)(t_first / (SYS_MHZ * 10000u))); puts_(" s, ");
	put_dec(gen); puts_(" generated in "); put_dec2((uint32_t)(tg / (SYS_MHZ * 10000u))); puts_(" s = ");
	put_dec2(gen ? (uint32_t)((uint64_t)gen * (SYS_MHZ * 100000000ull) / tg) : 0);
	puts_(" tok/s; engine busy "); put_dec((uint32_t)(ENG_CPU(eng_cycles_total) / (t / 100 + 1))); puts_("%; position ");
	put_dec(M.pos); puts_("]\n");
}

static uint32_t atou(const char *s) { uint32_t v = 0; while (*s >= '0' && *s <= '9') v = v * 10 + (uint32_t)(*s++ - '0'); return v; }

static void help(void)
{
	puts_(":models  :load N  :ctx N  :temp T (x100, 0 = greedy)  :n N (max new tokens)\n"
	      ":reset (forget the conversation)  :bw (bandwidth)  :gap N (DDR3 fallback)  :v (timing per token)\n:s (engine self-test)  :sd (card again, after re-inserting it)  :rd N (dump card sector N)  :sdclk N (SD clock = 24/(N+1) MHz)  :w N (wrap at N columns, 0 = off)\n:o / :g / :u / :k (engine overlap / GLU unit / vector unit / read line on-off, for debugging)  :help\n");
}

/* sector n of the card: the read's result and its first 64 bytes plus
 * the MBR's partition table area (for :rd and card errors) */
COLD static void dump_sector(uint32_t lba)
{
	int rc = sd_read(&sd, lba, 1, 0x1ff00000);
	puts_("sector "); put_dec(lba); puts_(": ");
	if (rc) { puts_("read error "); put_dec((uint32_t)-rc); puts_(", status "); put_hex(plat_sd_rd(3), 2); puts_("\n"); return; }
	const uint8_t *b = plat_mem() + 0x1ff00000;
	puts_("ok\n");
	for (int o = 0; o < 512; o += 16) {
		if (o == 64) { o = 416; continue; }
		put_hex((uint32_t)o, 3); puts_(":");
		for (int i = 0; i < 16; i++) { puts_(" "); put_hex(b[o + i], 2); }
		puts_("\n");
	}
}


/* the card: initialise, mount, list the models, load the first */
COLD static void card_setup(void)
{
	n_models = 0;
	int rc = sd_init(&sd, sd_div);
	if (rc) {
		puts_("sd card: error "); put_dec((uint32_t)-rc); puts_("\n");
		/* a card that machboot (or anything else) has used in SPI mode
		 * stays in SPI mode until it loses power */
		puts_("  (no card, or the card is still in SPI mode from the boot loader:\n"
		      "   take it out, put it back, and type :sd)\n");
		return;
	}
	puts_("sd card: ok, "); puts_(sd.ccs ? "SDHC" : "SDSC"); puts_(", 4-bit, ");
	put_dec(SYS_MHZ * 1000u / (2u * (sd_div + 1u)) / 1000u); puts_(" MHz\n");
	/* the first FAT partition with models on it (a card can carry
	 * machboot's partition first) */
	int part;
	for (part = 0; part < 4; part++) {
		rc = fat_mount(&fat, &sd, 0x1ff00000, part);
		if (rc) break;
		fat_list(&fat, list_cb, 0);
		if (n_models) break;
	}
	if (!n_models) {
		if (part == 0) {
			puts_("no FAT file system: "); put_dec((uint32_t)-rc);
			if (rc == -1 || rc == -3) {
				puts_(" (sd read error "); put_dec((uint32_t)-sd_last_err);
				puts_(", status "); put_hex(sd_last_status, 2); puts_(")");
			}
			puts_("\n");
			dump_sector(0);
		}
		else { puts_("no .MLM files in the root of any FAT partition ("); put_dec((uint32_t)part); puts_(" looked at)\n"); }
		return;
	}
	puts_("models on FAT partition "); put_dec((uint32_t)part + 1); puts_(":\n");
	for (int i = 0; i < n_models; i++) {
		puts_(" "); put_dec((uint32_t)i); puts_(": "); puts_(models[i].name);
		puts_(" "); put_dec(models[i].size >> 20); puts_(" MB\n");
	}
	if (n_models) model_ok = load_model(0);
}

int main(void)
{
	static char line[512];
	LED = 1;
#ifdef SIM
	UART_DIV = SIM_UART_DIV;        /* the testbench decodes the line at this rate */
	puts_("\nmachllm " MLM_BUILD_STR " (simulation)\n");
	while (!(DDR3_STATUS & 1)) ;
	int bad = memtest(0x10000000, 4096);
	puts_(bad ? "memtest FAILED\n" : "memtest ok\n");
	bwtest(1);
	SYS_LINE = 1;
#else
	puts_("\nmachllm " MLM_BUILD_STR " (zeitlos32ps, Sergei ML2)\n");
	/* train until both lanes are centred and the memory test passes;
	 * a marginal training otherwise corrupts everything after it */
	int bad = 1;
	for (int t = 1; t <= 8; t++) {
		int tr = ddr3_train();
		bad = tr ? memtest(0x10000000, 1 << 20) : 1;
		if (tr) puts_(bad ? "memtest FAILED\n" : "memtest ok\n");
		if (tr == 2 && !bad) break;
		if (t < 8) { puts_("ddr3: training again ("); put_dec((uint32_t)t + 1); puts_(" of 8)\n"); }
	}
	if (bad) { puts_("ddr3: no good training after 8 tries; stopped (please send this log)\n"); for (;;) ; }
	bwtest(32);
	SYS_LINE = 1;                       /* trained: reads may now be answered from the last line */
#endif

	if (vu_present()) { mlm_vu = fw_vu; puts_("vector unit: on\n"); }
	card_setup();
	if (!model_ok && ((const mlm_header_t *)plat_mem())->magic == MLM_MAGIC) {
		/* no card, but an image already in memory (a debugger, or the
		 * simulation's backdoor) */
		puts_("using the model image already in memory\n");
		model_ok = mlm_init(&M, ctx_len) == 0;
		if (model_ok) selftest();
	}
	M.temp_q8 = 128;                    /* Bonsai's recommended sampling: */
	M.topp_q16 = 55706;                 /* temperature 0.5, top-p 0.85, top-k 20 */
#ifdef SIM
	M.temp_q8 = 0;                      /* the testbench compares with greedy mlm_run */
#endif
	M.rng = (uint32_t)cycles() | 1;
	help();
	for (;;) {
		puts_(model_ok ? "> " : "(no model) > ");
		readline(line, sizeof(line));
		if (line[0] == ':') {
			const char *a = line + 1;
			while (*a && *a != ' ') a++;
			while (*a == ' ') a++;
			if (line[1] == 'm') {
				for (int i = 0; i < n_models; i++) { put_dec((uint32_t)i); puts_(": "); puts_(models[i].name); puts_("\n"); }
			} else if (line[1] == 'l') {
				uint32_t k = atou(a);
				if (k < (uint32_t)n_models) model_ok = load_model((int)k);
			} else if (line[1] == 'c') {
				ctx_len = atou(a);
				if (model_ok) { mlm_init(&M, ctx_len); puts_("context "); put_dec(M.ctx); puts_("\n"); }
			} else if (line[1] == 't') {
				M.temp_q8 = atou(a) * 256 / 100;
			} else if (line[1] == 'n') {
				max_new = atou(a);
			} else if (line[1] == 'r' && line[2] != 'd') {
				if (model_ok) mlm_init(&M, ctx_len);
			} else if (line[1] == 'b') {
				bwtest(32);
			} else if (line[1] == 'o') {
				extern int mlm_overlap;
				mlm_overlap = !mlm_overlap;
				puts_(mlm_overlap ? "overlap on\n" : "overlap off\n");
			} else if (line[1] == 'w') {
				const char *a = line + 2;
				while (*a == ' ') a++;
				wrap_cols = atou(a);
				puts_("wrap at "); put_dec(wrap_cols); puts_(wrap_cols ? " columns\n" : " (off)\n");
			} else if (line[1] == 's' && line[2] == 'd' && line[3] == 'c') {
				const char *a = line + 6;          /* :sdclk N */
				while (*a == ' ') a++;
				sd_div = atou(a);
				puts_("sd data clock "); put_dec(SYS_MHZ * 1000u / (2u * (sd_div + 1u)) / 1000u);
				puts_(" MHz (from the next :sd or :load)\n");
				sd.div = sd_div;
				plat_sd_wr(0, sd_div);
			} else if (line[1] == 's' && line[2] == 'd') {
				card_setup();
			} else if (line[1] == 'r' && line[2] == 'd') {
				const char *a = line + 3;
				while (*a == ' ') a++;
				dump_sector(atou(a));
			} else if (line[1] == 'u') {
				mlm_vu = (mlm_vu || !vu_present()) ? 0 : fw_vu;
				puts_(mlm_vu ? "vector loops in the vector unit\n" : "vector loops on the CPU\n");
			} else if (line[1] == 'e' && !line[2]) {
				engdiag();
			} else if (line[1] == 'k') {
				SYS_LINE = !(SYS_LINE & 1);
				puts_((SYS_LINE & 1) ? "read line on\n" : "read line off (or not in this bitstream)\n");
			} else if (line[1] == 'g') {
				extern int mlm_hw_glu;
				mlm_hw_glu = !mlm_hw_glu;
				puts_(mlm_hw_glu ? "SwiGLU in the GLU unit\n" : "SwiGLU on the CPU\n");
			} else if (line[1] == 'x') {
				/* OUT32 reads with the engine idle, then while it streams */
				volatile int32_t *o = (volatile int32_t *)(ENG_BASE + 0x10000);
				int32_t acc = 0;
				uint64_t t0 = cycles();
				for (int i = 0; i < 4096; i++) acc += o[i];
				uint32_t ti = (uint32_t)(cycles() - t0);
				const mlm_layer_t *L0 = (const mlm_layer_t *)(plat_mem() + M.hdr->layers_off);
				plat_eng_start(L0->w3.off, L0->w3.row_stride, L0->w3.rows, L0->w3.groups, L0->w3.qbits, 1);
				t0 = cycles();
				for (int i = 0; i < 4096; i++) acc += o[i];
				uint32_t tb = (uint32_t)(cycles() - t0);
				uint32_t still = ENG_CTRL & 1;
				plat_eng_wait();
				for (int pass = 0; pass < 2; pass++) {
					if (pass) plat_eng_start(L0->w3.off, L0->w3.row_stride, L0->w3.rows, L0->w3.groups, L0->w3.qbits, 1);
					t0 = cycles();
					plat_eng_rbank(0);
					uint64_t mx = plat_eng_maxabs();
					int sh = 0; while (mx >> sh) sh++;
					sh -= 30; if (sh < 0) sh = 0;
					for (int i = 0; i < 3072; i++) acc += plat_eng_out32((uint32_t)i, sh);
					uint32_t tt = (uint32_t)(cycles() - t0);
					uint32_t st = ENG_CTRL & 1;
					plat_eng_wait();
					puts_(pass ? "conversion, engine running " : "conversion, idle "); put_dec(tt);
					puts_(st ? " (still running)\n" : "\n");
				}
				puts_("4096 OUT32 reads: idle "); put_dec(ti); puts_(" cycles, engine running ");
				put_dec(tb); puts_(still ? " (still running after)\n" : " (engine finished first)\n");
				if (acc == 12345) puts_(" ");
			} else if (line[1] == 's') {
				if (model_ok) selftest();
			} else if (line[1] == 'v') {
				verbose = !verbose;
				puts_(verbose ? "timing per token on\n" : "timing per token off\n");
			} else if (line[1] == 'g') {
				/* DDR3 stream gap: 0 back to back, 1-3 idle cycles per burst */
				DDR3_CTRL = (atou(a) & 3) << 4;
				puts_("stream gap "); put_dec(atou(a) & 3); puts_("\n");
			} else help();
			continue;
		}
		if (model_ok && line[0]) generate(line);
	}
}
