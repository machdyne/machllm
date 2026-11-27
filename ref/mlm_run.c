/*
 * machllm
 * Copyright (c) 2025 Lone Dynamics Corporation. All rights reserved.
 *
 * mlm_run: the PC reference engine. Runs a .mlm model image with
 * exactly the board's arithmetic: fw/common/mlm_model.c is compiled
 * unchanged, and eng_dot() below is the engine's specification.
 *
 *   mlm_run model.mlm [-i prompt] [-n steps] [-t temp] [-p topp]
 *                     [-s seed] [-c ctx] [-x ids] [-d] [-q]
 *
 *   -x "1 2 3"   feed these token ids instead of a prompt
 *   -d           per step: position, chosen token, top 5 with logits
 *   -q           print only the token ids (for comparisons)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../fw/common/mlm.h"
#include "../fw/common/mlm_plat.h"
#include "../fw/common/mlm_model.h"

#define MEM_SIZE (512u << 20)

static uint8_t *mem;
static uint32_t xbuf[ENG_X_MAX / 4];          /* int8 x, four to a word */
static uint32_t xdbuf[ENG_X_MAX / 32];        /* one exponent offset per block */
static uint8_t xd8[ENG_X_MAX / 32];
static int64_t obufs[2][ENG_OUT_MAX];
static uint32_t rows_b[2];
static int rb;
#define obuf obufs[rb]

uint8_t *plat_mem(void) { return mem; }
uint32_t plat_mem_size(void) { return MEM_SIZE; }
volatile uint32_t *plat_eng_x(void) { return xbuf; }
volatile uint32_t *plat_eng_xd(void) { return xdbuf; }
#define last_rows rows_b[rb]
int64_t plat_eng_out(uint32_t i) { return obuf[i]; }
uint64_t plat_eng_maxabs(void)
{
	uint64_t m = 0;
	for (uint32_t i = 0; i < last_rows; i++) {
		uint64_t a = obuf[i] < 0 ? (uint64_t)(-obuf[i]) : (uint64_t)obuf[i];
		if (a > m) m = a;
	}
	return m;
}
int32_t plat_eng_out32(uint32_t i, int sh)
{
	int64_t v = obuf[i];
	if (sh > 0) v = (v + ((int64_t)1 << (sh - 1))) >> sh;
	return (int32_t)v;
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
#ifdef MLM_DUMP
#include <math.h>
void mlm_dump(const char *what, const int32_t *v, int n, int e)
{
	double ss = 0;
	for (int i = 0; i < n; i++) { double x = ldexp((double)v[i], e); ss += x * x; }
	fprintf(stderr, "%s: rms %.5f  ", what, sqrt(ss / n));
	for (int i = 0; i < 6; i++) fprintf(stderr, "%.5f ", ldexp((double)v[i], e));
	fprintf(stderr, "\n");
}
#endif

#include "../fw/common/eng_spec.h"

void plat_eng_start(uint32_t addr, uint32_t stride, uint32_t rows,
                    uint32_t groups, uint32_t qbits, int bank)
{
	for (int b = 0; b < ENG_X_MAX / 32; b++) xd8[b] = (uint8_t)(xdbuf[b] & 15);
	eng_dot_spec(mem, (const int8_t *)xbuf, xd8, obufs[bank], addr, stride, rows, groups, qbits);
	rows_b[bank] = rows;
}
void plat_eng_wait(void) {}
#include "../fw/common/mlm_sigtab.h"
void plat_eng_glu(uint32_t n, int sg, int su, int sq, int *psh, int *xsh)
{
	glu_spec(obufs[0], obufs[1], (int8_t *)xbuf, xd8, n, sg, su, sq, mlm_sigtab, psh, xsh);
	for (uint32_t b = 0; b < n / 32; b++) xdbuf[b] = xd8[b];
}
void plat_eng_rbank(int bank) { rb = bank; }
void plat_eng_dot(uint32_t addr, uint32_t stride, uint32_t rows,
                  uint32_t groups, uint32_t qbits)
{
	plat_eng_start(addr, stride, rows, groups, qbits, 0);
	rb = 0;
}

static void put_tok(mlm_ctx_t *m, uint32_t prev, uint32_t tok)
{
	uint32_t len;
	const uint8_t *s = mlm_decode(m, prev, tok, &len);
	if (s) fwrite(s, 1, len, stdout);
	fflush(stdout);
}

int main(int argc, char **argv)
{
	const char *path = 0, *prompt = "", *ids = 0;
	int steps = 64, ctx = 0, dump = 0, quiet = 0;
	double temp = 0.0, topp = 0.9;
	uint32_t seed = 1;
	for (int i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "-i") && i + 1 < argc) prompt = argv[++i];
		else if (!strcmp(argv[i], "-n") && i + 1 < argc) steps = atoi(argv[++i]);
		else if (!strcmp(argv[i], "-t") && i + 1 < argc) temp = atof(argv[++i]);
		else if (!strcmp(argv[i], "-p") && i + 1 < argc) topp = atof(argv[++i]);
		else if (!strcmp(argv[i], "-s") && i + 1 < argc) seed = (uint32_t)atoi(argv[++i]);
		else if (!strcmp(argv[i], "-c") && i + 1 < argc) ctx = atoi(argv[++i]);
		else if (!strcmp(argv[i], "-x") && i + 1 < argc) ids = argv[++i];
		else if (!strcmp(argv[i], "-d")) dump = 1;
		else if (!strcmp(argv[i], "-q")) quiet = 1;
		else path = argv[i];
	}
	if (!path) { fprintf(stderr, "usage: mlm_run model.mlm [options]\n"); return 1; }

	mem = calloc(1, MEM_SIZE);
	FILE *f = fopen(path, "rb");
	if (!f) { perror(path); return 1; }
	size_t n = fread(mem, 1, MEM_SIZE, f);
	fclose(f);

	mlm_ctx_t m;
	memset(&m, 0, sizeof(m));
	m.temp_q8 = (uint32_t)(temp * 256 + 0.5);
	m.topp_q16 = (uint32_t)(topp * 65536 + 0.5);
	m.rng = seed ? seed : 1;
	int rc = mlm_init(&m, (uint32_t)ctx);
	if (rc) { fprintf(stderr, "mlm_init: %d\n", rc); return 1; }
	const mlm_header_t *h = m.hdr;
	if (!quiet)
		fprintf(stderr, "%s: %u bytes, dim %u, layers %u, heads %u/%u, vocab %u, ctx %u, mem %u MB\n",
		        (const char *)mem + h->name_off, (unsigned)n, h->dim, h->n_layers, h->n_heads,
		        h->n_kv_heads, h->vocab, m.ctx, m.mem_end >> 20);

	uint32_t toks[1024];
	int nt = 0;
	if (h->flags & MLM_F_BOS) toks[nt++] = h->bos_id;
	if (ids) {
		char *s = strdup(ids), *tk = strtok(s, " ,");
		while (tk && nt < 1024) { toks[nt++] = (uint32_t)atoi(tk); tk = strtok(0, " ,"); }
	} else if (prompt[0]) {
		nt += mlm_encode(&m, prompt, toks + nt, 1024 - nt);
	}
	if (!nt) toks[nt++] = h->bos_id;

	clock_t t0 = clock();
	uint32_t prev = h->bos_id, tok = toks[0];
	int gen = 0;
	for (int i = 0; ; i++) {
		if (mlm_step_ex(&m, tok, i + 1 >= nt || dump)) break;
		uint32_t next = (i + 1 < nt) ? toks[i + 1] : mlm_sample(&m);
		if (dump) {
			fprintf(stderr, "pos %u in %u next %u:", m.pos - 1, tok, next);
			for (int j = 0; j < 5 && j < (int)m.top_n; j++) {
				double lg = (double)m.top_acc[j];
				int e = m.logit_exp;
				while (e > 0) { lg *= 2; e--; }
				while (e < 0) { lg /= 2; e++; }
				fprintf(stderr, " %u:%.4f", m.top_id[j], lg);
			}
			fprintf(stderr, "\n");
		}
		if (quiet) printf("%u ", next);
		else if (!dump) {
			if (i == 0) put_tok(&m, h->bos_id, tok);
			if (i + 1 < nt) put_tok(&m, tok, next);
		}
		if (i + 1 >= nt) {
			if (next == h->eos_id || next == h->eos2_id) break;
			if (!quiet && !dump) put_tok(&m, tok, next);
			if (++gen >= steps) break;
		}
		prev = tok;
		tok = next;
	}
	double dt = (double)(clock() - t0) / CLOCKS_PER_SEC;
	if (quiet) printf("\n");
	else printf("\n");
	if (!quiet)
		fprintf(stderr, "%d tokens in %.2f s; %u engine calls, %llu rows\n",
		        m.pos, dt, m.n_dot, (unsigned long long)m.dot_rows);
	return 0;
}
