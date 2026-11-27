// machllm -- sim/vu: rtl/mlm_vu.v against the C loops (mlm_vu_ref) on
// random data, every operation, many sizes and parameters.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include "Vmlm_vu.h"
#include "verilated.h"
extern "C" {
#include "mlm_model.h"
}

// the model code's platform, unused here
extern "C" {
uint8_t *plat_mem(void) { static uint8_t m[16]; return m; }
uint32_t plat_mem_size(void) { return 16; }
volatile uint32_t *plat_eng_x(void) { static uint32_t x[768]; return x; }
volatile uint32_t *plat_eng_xd(void) { static uint32_t x[96]; return x; }
void plat_eng_dot(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) {}
void plat_eng_start(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, int) {}
void plat_eng_wait(void) {}
void plat_eng_rbank(int) {}
void plat_eng_glu(uint32_t, int, int, int, int *, int *) {}
int64_t plat_eng_out(uint32_t) { return 0; }
uint64_t plat_eng_maxabs(void) { return 0; }
int32_t plat_eng_out32(uint32_t, int) { return 0; }
void plat_eng_out32_block(int32_t *d, uint32_t, uint32_t n, int) { for (uint32_t k = 0; k < n; k++) d[k] = 0; }
void plat_eng_xload(const uint32_t *, int, const uint32_t *, int) {}
void plat_putc(int) {}
}

static Vmlm_vu *top;
static const int WORDS = 24576;
static std::vector<uint32_t> ram(WORDS);
static uint64_t cycles;

static void tick()
{
	top->clk = 0; top->eval();
	uint32_t a = top->ram_a; int we = top->ram_we; uint32_t wd = top->ram_wd; int own = top->ram_own;
	top->clk = 1; top->eval();
	if (own) {
		uint32_t q = ram[a % WORDS];
		if (we) ram[a % WORDS] = wd;
		top->ram_q = q;
	}
	cycles++;
}

static uint32_t bus(int we, int reg, uint32_t v)
{
	top->bus_req = 1; top->bus_we = we; top->bus_adr = reg; top->bus_wdat = v;
	for (int n = 0; n < 1000000; n++) {
		tick();
		if (top->bus_ack) { top->bus_req = 0; uint32_t r = top->bus_rdat; tick(); return r; }
	}
	fprintf(stderr, "bus timeout\n"); exit(2);
}

static uint32_t rnd() { static uint64_t s = 0x9e3779b97f4a7c15ull; s ^= s << 13; s ^= s >> 7; s ^= s << 17; return (uint32_t)s; }
static int32_t rval(int bits) { int32_t v = (int32_t)rnd(); if (bits < 32) v >>= (32 - bits); return v; }

int main(int argc, char **argv)
{
	Verilated::commandArgs(argc, argv);
	top = new Vmlm_vu;
	top->rst = 1; for (int i = 0; i < 4; i++) tick(); top->rst = 0; tick();
	if (bus(0, 0xa, 0) != 0x56553031u) { fprintf(stderr, "no ID\n"); return 1; }
	const uint32_t A = 1000, B = 9000, D = 17000;    // word addresses
	int fails = 0, runs = 0;
	uint64_t elems = 0, cyc = 0;
	for (int t = 0; t < 3000; t++) {
		int op = 1 + t % 14;
		int n = (int)(rnd() % 300);
		if (t % 50 == 0) n = 0;
		if (t % 37 == 0) n = 1;
		int bits = 2 + rnd() % 31;
		int32_t p0 = 0, p1 = 0;
		switch (op) {
		case MLM_VU_SQMUL: p0 = rnd() % 20; bits = 15 + p0 < 32 ? 15 + p0 : 32; break;
		case MLM_VU_SCALE15: p0 = rval(16); bits = 31; break;
		case MLM_VU_RSHR: p0 = (int)(rnd() % 50) - 25; if (p0 < 0) bits = 31 + p0 > 1 ? 31 + p0 : 1; break;
		case MLM_VU_RESID: p0 = 1 + rnd() % 33; p1 = 1 + rnd() % 33; break;
		case MLM_VU_MULSHR30: p0 = rval(31); break;
		case MLM_VU_QPACK: p0 = (int)(rnd() % 34) - 1; p1 = 2 * (int)(rnd() % 200); if (rnd() & 1) p1 = (n + 1) & ~1; break;
		case MLM_VU_KVQ8: n &= ~3; p0 = rnd() % 20; p1 = (int32_t)(rnd() % (127u << 16)); break;
		case MLM_VU_ROPE: n = 2 * (8 + rnd() % 64); p0 = rnd() & 1; bits = 31; break;
		case MLM_VU_ATTSC: p0 = (int)(rnd() % 81) - 40; break;
		case MLM_VU_EXPSUM: bits = 31; break;
		case MLM_VU_COEF: break;
		case MLM_VU_QPACKU: p0 = (int)(rnd() % 36) - 15; p1 = 2 * (int)(rnd() % 200); if (rnd() & 1) p1 = (n + 1) & ~1; break;
		case MLM_VU_MXPACK: p1 = (n + 31) & ~31; if (rnd() & 1) p1 += 32 * (int)(rnd() % 3); break;
		default: break;
		}
		for (int i = 0; i < 300 + 8; i++) {
			ram[A + i] = (uint32_t)rval(bits);
			ram[B + i] = (op == MLM_VU_ROPE || op == MLM_VU_SQMUL) ? rnd() : (uint32_t)rval(bits);
			ram[D + i] = rnd();
		}
		// the attention operations' inputs, in the ranges the model gives them
		if (op == MLM_VU_ATTSC)
			for (int i = 0; i < 308; i++) ram[B + i] = (rnd() & 0xffff) | ((uint32_t)(uint16_t)((int)(rnd() % 71) - 50) << 16);
		if (op == MLM_VU_EXPSUM) {
			int32_t mxv = -0x7fffffff;
			/* scores in Q16 within ~20 of the largest, as a softmax sees them */
			int32_t top = rval(31) >> 2;
			for (int i = 0; i < 308; i++) ram[A + i] = (uint32_t)(top - (int32_t)(rnd() % (20u << 16)));
			for (int i = 0; i < n; i++) if ((int32_t)ram[A + i] > mxv) mxv = (int32_t)ram[A + i];
			p0 = mxv;
		}
		if (op == MLM_VU_COEF) {
			int emax = -32768;
			for (int i = 0; i < 308; i++) {
				ram[A + i] = rnd() % 65537;
				int e = (rnd() & 7) ? (int)(rnd() % 41) - 30 : (int)(rnd() % 141) - 130;
				ram[B + i] = (rnd() & 0xffff) | ((uint32_t)(uint16_t)e << 16);
				if (i < n && e > emax) emax = e;
			}
			p0 = n ? emax : 0;
		}
		std::vector<uint32_t> want = ram;
		uint64_t ws = 0;
		uint32_t wr = mlm_vu_ref(op, &want[A], &want[B],
		                         (op == MLM_VU_RESID || op == MLM_VU_ROPE || op == MLM_VU_ATTSC ||
		                          op == MLM_VU_EXPSUM || op == MLM_VU_COEF) ? &want[A] : &want[D],
		                         (uint32_t)n, p0, p1, &ws);
		bus(1, 0, A * 4); bus(1, 1, B * 4); bus(1, 2, D * 4); bus(1, 3, (uint32_t)n);
		bus(1, 4, (uint32_t)p0); bus(1, 5, (uint32_t)p1);
		uint64_t c0 = cycles;
		bus(1, 6, (uint32_t)op);
		cyc += cycles - c0; elems += (uint32_t)n;
		uint32_t gr = bus(0, 7, 0);
		uint64_t gs = bus(0, 8, 0) | ((uint64_t)bus(0, 9, 0) << 32);
		bool ok = ram == want;
		if ((op == MLM_VU_MAXABS || op == MLM_VU_RESID || op == MLM_VU_ATTSC || op == MLM_VU_COEF ||
		     op == MLM_VU_MXPACK) && gr != wr) ok = false;
		if ((op == MLM_VU_SQMUL || op == MLM_VU_EXPSUM) && gs != ws) ok = false;
		runs++;
		if (!ok) {
			if (fails++ < 10) {
				fprintf(stderr, "op %d n %d p0 %d p1 %d: r %08x/%08x s %llx/%llx\n", op, n, p0, p1, gr, wr,
				        (unsigned long long)gs, (unsigned long long)ws);
				for (int i = 0; i < WORDS; i++) if (ram[i] != want[i]) {
					fprintf(stderr, "  word %d: got %08x want %08x\n", i, ram[i], want[i]);
					break;
				}
			}
			ram = want;
		}
	}
	printf("%s: %d runs, %d wrong; %.1f cycles an element\n", fails ? "FAIL" : "PASS", runs, fails,
	       elems ? (double)cyc / elems : 0.0);
	return fails ? 1 : 0;
}
