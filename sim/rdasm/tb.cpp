// sim/rdasm: rtl/mem/mlm_ddr3_rdasm.v against Zeitlos's ddr3_rdasm, random
// data and capture, offsets changed now and then (and held a few cycles,
// as training does, before the outputs are compared)
#include "Vtb_rdasm.h"
#include <cstdio>
#include <cstdlib>
int main()
{
	Vtb_rdasm *t = new Vtb_rdasm;
	srand(1);
	t->rst = 1; t->clk = 0; t->eval();
	for (int i = 0; i < 4; i++) { t->clk = 1; t->eval(); t->clk = 0; t->eval(); }
	t->rst = 0;
	long bad = 0, n = 0;
	int hold = 0; long lastc = 0;
	for (long c = 0; c < 2000000; c++) {
		if ((rand() % 5000) == 0) { t->off0 = rand() & 15; t->off1 = rand() & 15; hold = 3; lastc = c; }
		t->q0 = rand(); t->q1 = rand(); t->q2 = rand(); t->q3 = rand();
		t->cap = !hold && (rand() & 3) == 0;   // no capture right after an offset change
		t->clk = 1; t->eval(); t->clk = 0; t->eval();
		if (hold) { hold--; continue; }
		n++;
		bool d = t->va != t->vb;
		for (int w = 0; w < 4; w++) d |= t->ra[w] != t->rb[w];
		if (d && bad++ < 8) printf("mismatch at cycle %ld (%ld after an offset change; off %d %d)\n", c, c - lastc, t->off0, t->off1);
	}
	printf("%s: %ld cycles compared, %ld mismatches\n", bad ? "FAIL" : "PASS", n, bad);
	return bad != 0;
}
