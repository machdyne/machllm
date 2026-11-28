/*
 * machllm
 * Copyright (c) 2025 Lone Dynamics Corporation. All rights reserved.
 *
 * tb_mem: the whole model on the RTL engine, reading main memory through
 * the DDR3 controller's stream port and a checking DRAM model
 * (sim/mem/mlm_dram_model.v), whose storage is mem[] below.
 *
 * fw/common/mlm_model.c runs unchanged; plat_eng_dot() drives
 * rtl/mlm_engine.v through its bus, and a memory model answers its
 * stream requests from the same memory image. Every call's results are
 * compared with eng_dot_spec(); the generated token ids are printed and
 * must equal `mlm_run -q` (the Makefile checks).
 *
 *   tb_mem model.mlm "prompt" steps
 */

#ifndef TB_MHZ
#define TB_MHZ 48
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include "Vtb_mem_top.h"
#include "svdpi.h"
#include "verilated.h"

extern "C" {
#include "../../fw/common/mlm.h"
#include "../../fw/common/mlm_plat.h"
#include "../../fw/common/mlm_model.h"
#include "../../fw/common/eng_spec.h"
}

#define MEM_SIZE (512u << 20)
static uint8_t *mem;
static uint32_t xshadow[ENG_X_MAX / 4];       /* int8 x, four to a word */
static uint32_t xdshadow[ENG_X_MAX / 32];     /* exponent offsets */
static int64_t obufs[2][ENG_OUT_MAX];
static int rb;
#define obuf obufs[rb]
static Vtb_mem_top *top;
static uint64_t cycles, busy_cycles, data_blocks, glu_cycles;
static unsigned rnd = 12345;
static unsigned rr() { rnd = rnd * 1103515245u + 12345u; return rnd >> 8; }

static uint64_t rd_blocks;

/* the DRAM model's storage: block index = {row, column block, bank},
 * the same bits as the byte address / 16 */
extern "C" int dram_rd32(int blk, int word)
{
	uint32_t v; memcpy(&v, mem + (uint64_t)(uint32_t)blk * 16 + 4 * word, 4);
	if (word == 0) rd_blocks++;
	return (int)v;
}
extern "C" void dram_wr32(int blk, int word, int val, int mask)
{
	uint8_t *p = mem + (uint64_t)(uint32_t)blk * 16 + 4 * word;
	for (int i = 0; i < 4; i++) if (!(mask & (1 << i))) p[i] = (uint8_t)(val >> (8 * i));
}
static const char *fails[] = { "", "ACT to an active bank", "ACT before precharge done (tRP)",
	"tRC", "tRRD", "tFAW", "READ from an idle bank", "tRCD", "tCCD", "WRITE to an idle bank",
	"REFRESH with a bank active", "tRFC", "refresh interval", "tWTR", "ACT before init", "PRECHARGE" };
extern "C" void dram_fail(int code, long long t)
{
	fprintf(stderr, "DRAM MODEL: %s at %.3f us (cycle %llu)\n", fails[code], t / 1e6,
	        (unsigned long long)cycles);
	exit(3);
}

static void tick()
{
	top->clk = 0; top->eval();
	top->clk = 1; top->eval();
	cycles++;
}

static uint32_t wb(bool we, uint32_t byte_adr, uint32_t wdat, uint32_t sel = 0xf)
{
	top->wb_adr = byte_adr >> 2; top->wb_we = we; top->wb_wdat = wdat; top->wb_sel = sel;
	top->wb_stb = 1;
	int n = 0;
	do { tick(); n++; } while (!top->wb_ack && n < 5000);
	if (!top->wb_ack) { fprintf(stderr, "wishbone timeout at %08x\n", byte_adr); exit(2); }
	uint32_t r = top->wb_rdat;
	top->wb_stb = 0;
	tick();
	return r;
}

static uint32_t bus(bool we, uint32_t adr, uint32_t wdat, uint32_t sel = 0xf)
{
	top->bus_req = 1; top->bus_we = we; top->bus_adr = (adr >> 2) & 0x7fff;
	top->bus_wdat = wdat; top->bus_sel = sel;
	int n = 0;
	do { tick(); n++; } while (!top->bus_ack && n < 100);
	if (!top->bus_ack) { fprintf(stderr, "bus timeout at %04x\n", adr); exit(2); }
	uint32_t r = top->bus_rdat;
	top->bus_req = 0;
	tick();
	return r;
}

extern "C" {
uint8_t *plat_mem(void) { return mem; }
uint32_t plat_mem_size(void) { return MEM_SIZE; }
volatile uint32_t *plat_eng_x(void) { return xshadow; }
volatile uint32_t *plat_eng_xd(void) { return xdshadow; }
int64_t plat_eng_out(uint32_t i) { return obuf[i]; }
static uint32_t rows_b[2];
#define last_rows rows_b[rb]
uint64_t plat_eng_maxabs(void)
{
	uint64_t m = rb ? (((uint64_t)bus(false, 0x4034, 0) << 32) | bus(false, 0x4030, 0))
	                : (((uint64_t)bus(false, 0x4024, 0) << 32) | bus(false, 0x4020, 0)), w = 0;
	for (uint32_t i = 0; i < last_rows; i++) {
		uint64_t a = obuf[i] < 0 ? (uint64_t)(-obuf[i]) : (uint64_t)obuf[i];
		if (a > w) w = a;
	}
	if (m != w) { fprintf(stderr, "MAXABS MISMATCH %llx %llx\n", (unsigned long long)m, (unsigned long long)w); exit(1); }
	return m;
}
int32_t plat_eng_out32(uint32_t i, int sh)
{
	static int cur = -1;
	if (sh != cur) { bus(true, 0x4028, (uint32_t)sh); cur = sh; }
	int32_t got = (int32_t)bus(false, (rb ? 0x14000 : 0x10000) + 4 * i, 0);
	int64_t v = obuf[i];
	if (sh > 0) v = (v + ((int64_t)1 << (sh - 1))) >> sh;
	if (got != (int32_t)v) { fprintf(stderr, "OUT32 MISMATCH row %u sh %d\n", i, sh); exit(1); }
	return got;
}
void plat_putc(int c) { putchar(c); }
void plat_eng_xload(const uint32_t *q, int nw, const uint32_t *d, int nb)
{
	volatile uint32_t *x = plat_eng_x(), *xd = plat_eng_xd();
	for (int j = 0; j < nw; j++) x[j] = q[j];
	for (int b = 0; b < nb; b++) xd[b] = d[b];
}
/* the packer and the DMA unit's calls, in C */
int plat_eng_xpack(const int32_t *v, int n, int padn)
{
	static int8_t q[ENG_X_MAX];
	static uint8_t xd[ENG_X_MAX / 32];
	static uint32_t qw[ENG_X_MAX / 4], dw[ENG_X_MAX / 32];
	int sx = eng_mx_pack(v, n, padn, q, xd);
	memcpy(qw, q, (size_t)padn);
	for (int b = 0; b < padn / 32; b++) dw[b] = xd[b];
	plat_eng_xload(qw, padn / 4, dw, padn / 32);
	return sx;
}
void plat_mem_rd(void *dst, uint32_t off, uint32_t nw) { memcpy(dst, plat_mem() + off, 4 * (size_t)nw); }
void plat_mem_wblk(uint32_t off, const void *src, uint32_t nblk) { memcpy(plat_mem() + off, src, 16 * (size_t)nblk); }
void plat_mem_wcol(uint32_t off, const void *src, uint32_t n, uint32_t stride)
{
	for (uint32_t k = 0; k < n; k++) plat_mem()[off + k * stride] = ((const uint8_t *)src)[k];
}
void plat_dma_wait(void) {}
void plat_poll(void) {}
void plat_eng_out32_block(int32_t *dst, uint32_t i0, uint32_t n, int sh)
{
	for (uint32_t k = 0; k < n; k++) dst[k] = plat_eng_out32(i0 + k, sh);
}

/* runs: start writes the registers; wait polls, then reads every row
 * of the bank back (64-bit) and checks it against the specification */
static struct { uint32_t addr, stride, rows, groups, qbits; int bank; bool on; } run;
static int8_t xrun[ENG_X_MAX];
static uint8_t xdrun[ENG_X_MAX / 32];
void plat_eng_start(uint32_t addr, uint32_t stride, uint32_t rows, uint32_t groups, uint32_t qbits, int bank)
{
	uint32_t n = groups * (qbits == 1 ? 128 : 32);
	for (uint32_t w = 0; w < n / 4; w++) bus(true, 4 * w, xshadow[w]);
	for (uint32_t b = 0; b < n / 32; b++) bus(true, 0x1000 + 4 * b, xdshadow[b] & 15);
	memcpy(xrun, xshadow, sizeof(xrun));
	for (uint32_t b = 0; b < ENG_X_MAX / 32; b++) xdrun[b] = (uint8_t)(xdshadow[b] & 15);
	bus(true, 0x4000, addr);
	bus(true, 0x4004, stride);
	bus(true, 0x4008, rows);
	bus(true, 0x400c, groups);
	bus(true, 0x4010, qbits);
	bus(true, 0x402c, (uint32_t)bank);
	bus(true, 0x4014, 1);
	run.addr = addr; run.stride = stride; run.rows = rows; run.groups = groups;
	run.qbits = qbits; run.bank = bank; run.on = true;
}
void plat_eng_wait(void)
{
	static int64_t want[ENG_OUT_MAX];
	if (!run.on) return;
	uint64_t t0 = cycles;
	while (bus(false, 0x4014, 0) & 1) {
		if (cycles - t0 > 50000000ull) { fprintf(stderr, "engine hang\n"); exit(2); }
	}
	busy_cycles += bus(false, 0x4018, 0);
	uint32_t base = run.bank ? 0x18000 : 0x8000;
	for (uint32_t r = 0; r < run.rows; r++) {
		uint32_t lo = bus(false, base + 8 * r, 0), hi = bus(false, base + 8 * r + 4, 0);
		obufs[run.bank][r] = (int64_t)(((uint64_t)hi << 32) | lo);
	}
	rows_b[run.bank] = run.rows;
	eng_dot_spec(mem, xrun, xdrun, want, run.addr, run.stride, run.rows, run.groups, run.qbits);
	for (uint32_t r = 0; r < run.rows; r++)
		if (want[r] != obufs[run.bank][r]) {
			fprintf(stderr, "MISMATCH addr %08x stride %u rows %u groups %u q%u bank %d: row %u got %lld want %lld\n",
			        run.addr, run.stride, run.rows, run.groups, run.qbits, run.bank, r,
			        (long long)obufs[run.bank][r], (long long)want[r]);
			exit(1);
		}
	run.on = false;
}
void plat_eng_rbank(int bank) { rb = bank; }
#include "../../fw/common/mlm_sigtab.h"
static uint64_t glu_calls;
void plat_eng_glu(uint32_t n, int sg, int su, int sq, int *psh, int *xsh)
{
	static int8_t want[ENG_X_MAX] __attribute__((aligned(4)));
	static uint8_t wantd[ENG_X_MAX / 32];
	int wp, wx;
	glu_spec(obufs[0], obufs[1], want, wantd, n, sg, su, sq, mlm_sigtab, &wp, &wx);
	bus(true, 0x4038, n);
	bus(true, 0x403c, (uint32_t)(sg & 63) | ((uint32_t)(su & 63) << 8) | ((uint32_t)(sq & 255) << 16));
	bus(true, 0x4040, 1);
	uint64_t t0 = cycles;
	while (bus(false, 0x4014, 0) & 1)
		if (cycles - t0 > 5000000ull) { fprintf(stderr, "GLU hang\n"); exit(2); }
	glu_cycles += cycles - t0;
	uint32_t r = bus(false, 0x4044, 0);
	*psh = (int)(r & 0xff);
	*xsh = (int)(int8_t)(r >> 8);
	if (*psh != wp || *xsh != wx) {
		fprintf(stderr, "GLU MISMATCH shifts: got %d %d want %d %d (n %u sg %d su %d sq %d)\n",
		        *psh, *xsh, wp, wx, n, sg, su, sq);
		exit(1);
	}
	const uint32_t *ww = (const uint32_t *)want;
	for (uint32_t w = 0; w < n / 4; w++)
		if (bus(false, 4 * w, 0) != ww[w]) { fprintf(stderr, "GLU MISMATCH x word %u\n", w); exit(1); }
	for (uint32_t b = 0; b < n / 32; b++)
		if (bus(false, 0x1000 + 4 * b, 0) != wantd[b]) { fprintf(stderr, "GLU MISMATCH x block %u\n", b); exit(1); }
	memcpy(xshadow, want, n);
	for (uint32_t b = 0; b < n / 32; b++) xdshadow[b] = wantd[b];
	glu_calls++;
}
void plat_eng_dot(uint32_t addr, uint32_t stride, uint32_t rows, uint32_t groups, uint32_t qbits)
{
	plat_eng_start(addr, stride, rows, groups, qbits, 0);
	plat_eng_wait();
	rb = 0;
}
}

int main(int argc, char **argv)
{
	Verilated::commandArgs(argc, argv);
	if (argc < 4) { fprintf(stderr, "usage: tb_engine model.mlm prompt steps [latency] [gap%%]\n"); return 1; }
	
	mem = (uint8_t *)calloc(1, MEM_SIZE);
	FILE *f = fopen(argv[1], "rb");
	if (!f) { perror(argv[1]); return 1; }
	if (fread(mem, 1, MEM_SIZE, f) == 0) return 1;
	fclose(f);

	top = new Vtb_mem_top;
	svSetScope(svGetScopeFromName("TOP.tb_mem_top.dram"));
	top->rst = 1;
	top->st_gap = getenv("ST_GAP") ? atoi(getenv("ST_GAP")) : 0;
	for (int i = 0; i < 4; i++) tick();
	top->rst = 0;
	tick();
	while (!top->init_done) tick();
	fprintf(stderr, "DDR3 initialised after %llu cycles\n", (unsigned long long)cycles);

	/* the CPU path under the new address map: words, half-words, bytes
	 * through read-modify-write, checked in the backing store and read
	 * back through the controller */
	{
		uint32_t base = 0x1f000000;   /* well above any model */
		uint8_t want[4096];
		for (int i = 0; i < 4096; i++) want[i] = mem[base + i];
		for (int i = 0; i < 600; i++) {
			uint32_t off = (rr() % 4096) & ~3u, v = rr() ^ (rr() << 12);
			uint32_t sel = (i % 3 == 0) ? 0xf : (i % 3 == 1) ? (3u << ((rr() & 1) * 2)) : (1u << (rr() & 3));
			wb(true, base + off, v, sel);
			for (int b = 0; b < 4; b++) if (sel & (1u << b)) want[off + b] = (uint8_t)(v >> (8 * b));
		}
		for (int i = 0; i < 4096; i++)
			if (mem[base + i] != want[i]) { fprintf(stderr, "CPU write mismatch at +%d\n", i); return 1; }
		for (int i = 0; i < 4096; i += 4) {
			uint32_t v = wb(false, base + i, 0), w; memcpy(&w, want + i, 4);
			if (v != w) { fprintf(stderr, "CPU read mismatch at +%d: %08x %08x\n", i, v, w); return 1; }
		}
		fprintf(stderr, "CPU path: 600 stores, 1024 loads OK\n");
	}

	mlm_ctx_t m;
	memset(&m, 0, sizeof(m));
	m.rng = 1;
	m.topp_q16 = 58982;
	if (mlm_init(&m, 0)) { fprintf(stderr, "mlm_init failed\n"); return 1; }
	uint32_t toks[512];
	int nt = 0;
	if (m.hdr->flags & MLM_F_BOS) toks[nt++] = m.hdr->bos_id;
	nt += mlm_encode(&m, argv[2], toks + nt, 512 - nt);
	int steps = atoi(argv[3]);
	uint32_t tok = toks[0];
	for (int i = 0; i < nt - 1 + steps; i++) {
		mlm_step_ex(&m, tok, i + 1 >= nt);
		uint32_t next = (i + 1 < nt) ? toks[i + 1] : mlm_sample(&m);
		printf("%u ", next);
		fflush(stdout);
		tok = next;
	}
	printf("\n");
	fprintf(stderr, "%u engine calls, %llu rows, %llu engine busy cycles, %llu blocks read "
	        "(%.3f words/cycle while busy = %.0f MB/s at %d MHz), %llu total cycles\n",
	        m.n_dot, (unsigned long long)m.dot_rows, (unsigned long long)busy_cycles,
	        (unsigned long long)rd_blocks, 2.0 * rd_blocks / (double)busy_cycles,
	        16.0 * rd_blocks / (double)busy_cycles * TB_MHZ, TB_MHZ, (unsigned long long)cycles);
	delete top;
	return 0;
}
