/*
 * machllm
 * Copyright (c) 2025 Lone Dynamics Corporation. All rights reserved.
 *
 * tb_engine: the whole model on the RTL engine.
 *
 * fw/common/mlm_model.c runs unchanged; plat_eng_dot() drives
 * rtl/mlm_engine.v through its bus, and a memory model answers its
 * stream requests from the same memory image. Every call's results are
 * compared with eng_dot_spec(); the generated token ids are printed and
 * must equal `mlm_run -q` (the Makefile checks).
 *
 *   tb_engine model.mlm "prompt" steps [latency] [gap%]
 *     latency  cycles from request to first block (default 14)
 *     gap%     chance of a missing block slot, to shake the FIFO (default 10)
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include "Vmlm_engine.h"
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
static Vmlm_engine *top;
static uint64_t cycles, busy_cycles, data_blocks, glu_cycles;
static int latency = 14, gap_pct = 10;
static unsigned rnd = 12345;
static unsigned rr() { rnd = rnd * 1103515245u + 12345u; return rnd >> 8; }

struct req { uint32_t blk, len; uint64_t due; };
static std::deque<req> rq;
static uint64_t next_slot;

static void tick()
{
	// memory model, evaluated on the falling edge's inputs
	top->sd_valid = 0;
	if (!rq.empty() && cycles >= rq.front().due && cycles >= next_slot) {
		if ((int)(rr() % 100) >= gap_pct) {
			req &r = rq.front();
			memcpy(&top->sd_data, mem + (uint64_t)r.blk * 16, 16);
			top->sd_valid = 1;
			data_blocks++;
			r.blk++;
			if (--r.len == 0) rq.pop_front();
			next_slot = cycles + 2;           // one block per two cycles
		}
	}
	top->sq_ready = (rr() % 4) != 0;
	top->clk = 0; top->eval();
	bool take = top->sq_valid && top->sq_ready;
	uint32_t blk = top->sq_blk, len = top->sq_len;
	top->clk = 1; top->eval();
	if (take) rq.push_back({blk, len, cycles + (uint64_t)latency});
	cycles++;
}

static uint32_t bus(bool we, uint32_t adr, uint32_t wdat, uint32_t sel = 0xf)
{
	top->bus_req_i = 1; top->bus_we_i = we; top->bus_adr_i = (adr >> 2) & 0x7fff;
	top->bus_wdat_i = wdat; top->bus_sel_i = sel;
	int n = 0;
	do { tick(); n++; } while (!top->bus_ack && n < 100);
	if (!top->bus_ack) { fprintf(stderr, "bus timeout at %04x\n", adr); exit(2); }
	uint32_t r = top->bus_rdat;
	top->bus_req_i = 0;
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
		/* products: the hardware leaves p in bank 0 */
		int nbad = 0;
		for (uint32_t i = 0; i < n && nbad < 5; i++) {
			int32_t g = glu_out32(obufs[0][i], sg);
			uint32_t a2;
			if (sq >= 0) a2 = (sq > 20 || (glu_abs(g) >> (20 - sq))) ? (16u << 16) : glu_abs(g) << sq;
			else a2 = (uint32_t)glu_rshr32((int32_t)glu_abs(g), -sq);
			uint32_t sig;
			if (a2 >= (16u << 16)) sig = 65535;
			else { uint32_t j = a2 >> 10, f = a2 & 1023; sig = mlm_sigtab[j] + (((uint32_t)(mlm_sigtab[j + 1] - mlm_sigtab[j]) * f) >> 10); }
			if (g < 0) sig = 65536 - sig;
			int32_t sv = glu_rshr32(g, 15) * (int32_t)((sig + 1) >> 1);
			int32_t pv = glu_rshr32(sv, wp) * glu_out32(obufs[1][i], su);
			int32_t hw = (int32_t)bus(false, 0x8000 + 8 * i, 0);
			if (hw != pv) { fprintf(stderr, "  row %u: g %d s %d u %d p want %d got %d\n", i, g, sv,
			                        glu_out32(obufs[1][i], su), pv, hw); nbad++; }
		}
		fprintf(stderr, "GLU MISMATCH shifts: got %d %d want %d %d (n %u sg %d su %d sq %d)\n",
		        *psh, *xsh, wp, wx, n, sg, su, sq);
		exit(1);
	}
	const uint32_t *ww = (const uint32_t *)want;
	for (uint32_t w = 0; w < n / 4; w++) {
		uint32_t got = bus(false, 4 * w, 0);
		if (got != ww[w]) {
			fprintf(stderr, "GLU MISMATCH x word %u: got %08x want %08x\n", w, got, ww[w]);
			exit(1);
		}
	}
	for (uint32_t b = 0; b < n / 32; b++) {
		uint32_t got = bus(false, 0x1000 + 4 * b, 0);
		if (got != wantd[b]) {
			fprintf(stderr, "GLU MISMATCH x block %u offset: got %u want %u\n", b, got, wantd[b]);
			exit(1);
		}
	}
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
	if (argc > 4) latency = atoi(argv[4]);
	if (argc > 5) gap_pct = atoi(argv[5]);
	mem = (uint8_t *)calloc(1, MEM_SIZE);
	FILE *f = fopen(argv[1], "rb");
	if (!f) { perror(argv[1]); return 1; }
	if (fread(mem, 1, MEM_SIZE, f) == 0) return 1;
	fclose(f);

	top = new Vmlm_engine;
	top->rst = 1;
	for (int i = 0; i < 4; i++) tick();
	top->rst = 0;
	tick();

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
	fprintf(stderr, "%u engine calls, %llu rows, %llu engine busy cycles, %llu blocks "
	        "(%.2f words/cycle while busy), %llu total cycles\n",
	        m.n_dot, (unsigned long long)m.dot_rows, (unsigned long long)busy_cycles,
	        (unsigned long long)data_blocks, 2.0 * data_blocks / (double)busy_cycles,
	        (unsigned long long)cycles);
	delete top;
	return 0;
}
