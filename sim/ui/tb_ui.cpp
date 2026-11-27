// machllm: the screen, one frame: cells written through the CPU port,
// the TMDS symbols decoded back to pixels (a check of the encoder too),
// out.ppm written, the sync timing counted.
#include "Vmlm_ui.h"
#include "Vmlm_ui___024root.h"
#include "verilated.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
static Vmlm_ui *t;
static void cpu_tick() { t->clk = 1; t->eval(); t->clk = 0; t->eval(); }
static void wr(int a, unsigned v) { t->adr = a; t->wdat = v; t->we = 1; t->stb = 1; cpu_tick(); t->stb = 0; t->we = 0; cpu_tick(); }
static unsigned rd(int a) { t->adr = a; t->we = 0; t->stb = 1; cpu_tick(); t->stb = 0; t->eval(); unsigned v = t->rdat; cpu_tick(); return v; }
static void put(int row, int col, const char *s, int colr) { for (int i = 0; s[i]; i++) wr(4 * (row * 80 + col + i), (colr << 7) | (s[i] & 127)); }
static int dec(unsigned q, int *ctl) {   // TMDS 10 -> 8 bits; -1 for a control symbol
	*ctl = -1;
	switch (q) { case 0x354: *ctl = 0; return -1; case 0x0ab: *ctl = 1; return -1; case 0x154: *ctl = 2; return -1; case 0x2ab: *ctl = 3; return -1; }
	unsigned d = q & 0xff; if (q & 0x200) d = ~d & 0xff;
	unsigned o = d & 1;
	for (int i = 1; i < 8; i++) { unsigned b = ((d >> i) ^ (d >> (i - 1))) & 1; if (!(q & 0x100)) b ^= 1; o |= b << i; }
	return o;
}
int main(int argc, char **argv) {
	Verilated::commandArgs(argc, argv);
	t = new Vmlm_ui;
	t->rst = 1; for (int i = 0; i < 4; i++) cpu_tick(); t->rst = 0;
	for (int i = 0; i < 2000; i++) wr(4 * i, 32);
	put(0, 1, "machllm", 2);
	put(2, 1, "You:", 1); put(3, 1, "How do I bake a pumpkin pie?", 1);
	put(5, 1, "machllm:", 2); put(6, 1, "Here is a simple way to bake a pumpkin pie. 0123456789 !@#$%^&*()", 0);
	put(7, 1, "ABCDEFGHIJKLMNOPQRSTUVWXYZ abcdefghijklmnopqrstuvwxyz {}[]<>~", 0);
	put(23, 1, "> type here", 2); wr(4 * (23 * 80 + 13), 127 | (0 << 7));
	put(24, 1, "Writing...  (Esc stops)", 3);
	put(0, 79, "#", 0); put(24, 79, "#", 0);
	wr(0x2004, (1 << 23) | (1 << 24));
	int bad = 0;
	if (rd(0x2000) != 0x55493031) { printf("ID wrong\n"); bad++; }
	if ((rd(4 * (3 * 80 + 1)) & 0x1ff) != ((1 << 7) | 'H')) { printf("cell read-back wrong\n"); bad++; }
	if (rd(0x2004) != ((1u << 23) | (1u << 24))) { printf("ROWBG read-back wrong\n"); bad++; }
	// two frames of pixels; decode the second
	std::vector<unsigned char> img(640 * 480 * 3, 0);
	int x = 0, y = -1, prev_de = 0, prev_vs = 1, frame = 0, hs_len = 0, max_hs = 0, line_len = 0, lines = 0, de_px = 0;
	for (long cyc = 0; cyc < 2L * 800 * 525 + 2000; cyc++) {
		t->pclk = 1; t->eval(); t->pclk = 0; t->eval();
		unsigned qr = t->rootp->mlm_ui__DOT__t_r, qg = t->rootp->mlm_ui__DOT__t_g, qb = t->rootp->mlm_ui__DOT__t_b;
		int cr, cg, cb;
		int r = dec(qr, &cr), g = dec(qg, &cg), b = dec(qb, &cb);
		int de = cb < 0;
		int vs = cb >= 0 ? (cb >> 1) & 1 : prev_vs;
		int hs = cb >= 0 ? cb & 1 : 1;
		if (prev_vs && !vs) { frame++; y = -1; lines = 0; }
		if (!hs) hs_len++; else { if (hs_len > max_hs) max_hs = hs_len; hs_len = 0; }
		if (de && !prev_de) { y++; x = 0; }
		if (frame == 2 && de && y >= 0 && y < 480 && x < 640) {
			if ((cr < 0) != (cg < 0)) bad++;
			int o = (y * 640 + x) * 3; img[o] = r; img[o + 1] = g; img[o + 2] = b; de_px++;
		}
		if (de) x++;
		prev_de = de; prev_vs = vs;
		if (frame >= 3) break;
	}
	FILE *f = fopen("out.ppm", "wb"); fprintf(f, "P6 640 480 255\n"); fwrite(img.data(), 1, img.size(), f); fclose(f);
	printf("hsync %d pixels, %d visible pixels in frame 2 (want 307200)\n", max_hs, de_px);
	if (max_hs != 96) bad++;
	if (de_px != 307200) bad++;
	printf(bad ? "FAIL (%d)\n" : "PASS\n", bad);
	return bad != 0;
}
