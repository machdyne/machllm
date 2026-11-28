/*
 * machllm
 * Copyright (c) 2025 Lone Dynamics Corporation. All rights reserved.
 *
 * Forward pass, sampling and tokenizer, in integer arithmetic only.
 *
 * Every vector is int32 with one exponent for the whole vector (block
 * floating point): real = v * 2^e. Matrix-vector products are done by
 * the engine (plat_eng_dot), which takes int16 inputs and returns exact
 * int64 sums. Everything else is here, and is the same code in the
 * firmware and in the PC reference, so the two cannot disagree.
 */

#include "mlm_model.h"
#include "mlm_plat.h"
#include "eng_spec.h"
#include "mlm_exp2tab.h"
#include "mlm_sigtab.h"

/* ---------------------------------------------------------------- */
/* helpers                                                          */

static int bitlen64(uint64_t v)
{
	int n = 0;
	if (v >> 32) { n = 32; v >>= 32; }
	while (v) { n++; v >>= 1; }
	return n;
}

static uint64_t uabs64(int64_t v) { return v < 0 ? (uint64_t)(-v) : (uint64_t)v; }

/* v * 2^-s, rounded half up; s <= 0 shifts left */
static int64_t rshr64(int64_t v, int s)
{
	if (s <= 0) {
		if (s < -62) s = -62;
		return (int64_t)((uint64_t)v << (-s));
	}
	if (s > 62) return v < 0 ? -1 : 0;
	return (v + ((int64_t)1 << (s - 1))) >> s;
}

static int32_t sat32(int64_t v)
{
	if (v > 0x7fffffffLL) return 0x7fffffff;
	if (v < -0x7fffffffLL) return -0x7fffffff;
	return (int32_t)v;
}

static uint32_t rd32u(const uint8_t *p)
{
	return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
	       ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint8_t *MEM(uint32_t off) { return plat_mem() + off; }

static void wr32(uint32_t off, uint32_t v) { *(volatile uint32_t *)MEM(off) = v; }

/* sqrt(v) to 16 bits or more: v is brought under 2^32 by k pairs of
 * bits and its 32-bit square root shifted back by k (rmsnorm keeps 15
 * bits of the reciprocal; the 64-bit loop cost ~600 CPU cycles a call) */
uint32_t mlm_isqrt64(uint64_t v)
{
	int k = 0;
	while (v >> 32) { v >>= 2; k++; }
	uint32_t x = (uint32_t)v, r = 0, b = 1u << 30;
	while (b > x) b >>= 2;
	while (b) {
		if (x >= r + b) { x -= r + b; r = (r >> 1) + b; }
		else r >>= 1;
		b >>= 2;
	}
	return r << k;
}

/* exp(-x), x >= 0 in Q16, result Q16 */
uint32_t mlm_exp_neg_q16(uint32_t x)
{
	uint64_t y = ((uint64_t)x * 94548u) >> 16;      /* x * log2(e) */
	uint32_t k = (uint32_t)(y >> 16);
	if (k >= 17) return 0;
	uint32_t f = (uint32_t)y & 0xffff;
	uint32_t i = f >> 8, r = f & 0xff;
	uint32_t a = mlm_exp2tab[i], b = mlm_exp2tab[i + 1];
	uint32_t v = a - (((a - b) * r + 128) >> 8);
	if (k) v = (v + (1u << (k - 1))) >> k;
	return v;
}

/* ---------------------------------------------------------------- */
/* buffers                                                          */

static int32_t H[MLM_MAX_DIM];          /* residual stream */
/* max |H[i]|, as the last residual() into H left it (h_max_ok): the
 * norm that follows needs no pass of its own to find it */
static uint32_t h_max;
static int h_max_ok;
static int32_t S[MLM_MAX_SCRATCH];      /* q|k|v, norm out, ffn */
static int32_t ATT[MLM_MAX_CTX];        /* scores, then exp, then coef */
/* this KV head's K and V scales, copied from main memory once for the
 * query heads that share it (GQA) */
static uint32_t KS[MLM_MAX_CTX], VS[MLM_MAX_CTX];
static uint32_t ks_row = ~0u, ks_n;
static int vs_emax;
/* per cache row (layer x KV head), the largest V-scale exponent of the
 * positions stored so far (kept as they are stored; attend() needs it
 * and would otherwise scan them all) */
#define VEM_ROWS 256
static int16_t VEM[VEM_ROWS];

static int eqh[MLM_MAX_HEADS], ekh[MLM_MAX_HEADS], eoh[MLM_MAX_HEADS];
static int32_t qf[MLM_MAX_HEADS], kf[MLM_MAX_HEADS];   /* deferred QK-norm factors (0: none) */
static int16_t NW[MLM_MAX_DIM] __attribute__((aligned(4)));         /* the norm weights in use */
static uint32_t nw_off = ~0u;
static int16_t RT[256] __attribute__((aligned(4)));                 /* this position's RoPE row */
int mlm_overlap = 1;                    /* engine runs while the CPU works (0: every run waits) */
int mlm_hw_glu = 1;                     /* SwiGLU in the engine's GLU unit (0: on the CPU) */

/* v * 2^-s for int32, rounded as rshr64; s <= 0 shifts left (the
 * caller knows it fits) */
static int32_t rshr32(int32_t x, int s)
{
	if (s <= 0) return (int32_t)((uint32_t)x << (-s));
	if (s > 31) return 0;
	return ((x >> (s - 1)) + 1) >> 1;
}

static uint32_t uabs32(int32_t v) { return v < 0 ? (uint32_t)(-(int64_t)v) : (uint32_t)v; }

/* ---------------------------------------------------------------- */
/* vector operations                                                */
/*                                                                  */
/* These loops are the specification of the vector unit              */
/* (rtl/mlm_vu.v, docs/vu.md): with mlm_vu set, the platform runs     */
/* them in hardware on the same arrays and the result must be the    */
/* same bit for bit. Arrays are word aligned.                        */

uint32_t (*mlm_vu)(int op, const void *a, const void *b, void *d, uint32_t n,
                   int32_t p0, int32_t p1, uint64_t *sum) = 0;

/* max |a[i]| (0 for n = 0) */
static uint32_t v_maxabs(const int32_t *a, int n)
{
	if (mlm_vu) return mlm_vu(MLM_VU_MAXABS, a, 0, 0, (uint32_t)n, 0, 0, 0);
	uint32_t mx = 0;
	for (int i = 0; i < n; i++) { uint32_t x = uabs32(a[i]); if (x > mx) mx = x; }
	return mx;
}

/* u = a[i] >> s (rounded), d[i] = u g[i]; returns the sum of u^2 */
static uint64_t v_sqmul(int32_t *d, const int32_t *a, const int16_t *g, int n, int s)
{
	if (mlm_vu) { uint64_t ss; mlm_vu(MLM_VU_SQMUL, a, g, d, (uint32_t)n, s, 0, &ss); return ss; }
	uint64_t ss = 0;
	for (int i = 0; i < n; i++) {
		int32_t u = rshr32(a[i], s);
		ss += (uint32_t)(u * u);
		d[i] = u * g[i];
	}
	return ss;
}

/* d[i] = (a[i] >> 15) k */
static void v_scale15(int32_t *d, const int32_t *a, int n, int32_t k)
{
	if (mlm_vu) { mlm_vu(MLM_VU_SCALE15, a, 0, d, (uint32_t)n, k, 0, 0); return; }
	for (int i = 0; i < n; i++) d[i] = rshr32(a[i], 15) * k;
}

/* d[i] = a[i] >> s, rounded (s < 0: shifted left) */
static void v_rshr(int32_t *d, const int32_t *a, int n, int s)
{
	if (mlm_vu) { mlm_vu(MLM_VU_RSHR, a, 0, d, (uint32_t)n, s, 0, 0); return; }
	for (int i = 0; i < n; i++) d[i] = rshr32(a[i], s);
}

/* h[i] = (h[i] >> sh) + (v[i] >> sv); returns max |h[i]| */
static uint32_t v_resid(int32_t *h, const int32_t *v, int n, int sh, int sv)
{
	if (mlm_vu) return mlm_vu(MLM_VU_RESID, h, v, h, (uint32_t)n, sh, sv, 0);
	uint32_t mx = 0;
	for (int i = 0; i < n; i++) {
		h[i] = rshr32(h[i], sh) + rshr32(v[i], sv);
		uint32_t x = uabs32(h[i]);
		if (x > mx) mx = x;
	}
	return mx;
}

/* d[i] = (a[i] k) >> 30 (rounded, 64-bit product) */
static void v_mulshr30(int32_t *d, const int32_t *a, int n, int32_t k)
{
	if (mlm_vu) { mlm_vu(MLM_VU_MULSHR30, a, 0, d, (uint32_t)n, k, 0, 0); return; }
	for (int i = 0; i < n; i++) d[i] = (int32_t)rshr64((int64_t)a[i] * k, 30);
}

/* int8 for the KV cache: ((a[i] >> s) r) >> 16, clamped to +-127, four
 * to a word, lowest byte first (n a multiple of 4) */
static void v_kvq8(uint32_t *d, const int32_t *a, int n, int s, int32_t r)
{
	if (mlm_vu) { mlm_vu(MLM_VU_KVQ8, a, 0, d, (uint32_t)n, s, r, 0); return; }
	for (int i = 0; i < n; i += 4) {
		uint32_t w = 0;
		for (int j = 0; j < 4; j++) {
			int64_t t = rshr64((int64_t)rshr32(a[i + j], s) * r, 16);
			if (t > 127) t = 127;
			if (t < -127) t = -127;
			w |= ((uint32_t)t & 0xff) << (8 * j);
		}
		d[i >> 2] = w;
	}
}

/* RoPE on one head of hd values: pairs (2i, 2i+1), or (i, i + hd/2)
 * with split; rt word i = cos (low half), sin (high half), Q15 */
static void v_rope(int32_t *v, const int16_t *rt, int hd, int split)
{
	if (mlm_vu) { mlm_vu(MLM_VU_ROPE, v, rt, v, (uint32_t)hd, split, 0, 0); return; }
	int half = hd / 2;
	for (int i = 0; i < half; i++) {
		int ia = split ? i : 2 * i, ib = split ? i + half : 2 * i + 1;
		int32_t c = rt[2 * i], s = rt[2 * i + 1];
		int32_t a = rshr32(v[ia], 15), b = rshr32(v[ib], 15);
		v[ia] = a * c - b * s;
		v[ib] = a * s + b * c;
	}
}

/* the norm weights, copied from main memory once per use (the DMA unit
 * when the engine is idle; fetching them while it streams was tried and
 * slows the engine by as much as it saves) */
static const int16_t *norm_w(const mlm_norm_t *nw)
{
	if (nw->off != nw_off) {
		plat_mem_rd(NW, nw->off, (nw->n + 1) / 2);
		nw_off = nw->off;
	}
	return NW;
}


/* attention scores: a[t] (the engine's raw q.k) times the position's K
 * scale (word b[t]: mantissa low half, exponent high half), shifted by
 * k = p0 + that exponent, saturated to +-2^30; returns the largest */
static int32_t v_attsc(int32_t *a, const uint32_t *b, int n, int k0)
{
	if (mlm_vu) return (int32_t)mlm_vu(MLM_VU_ATTSC, a, b, a, (uint32_t)n, k0, 0, 0);
	int32_t smax = -0x7fffffff;
	for (int t = 0; t < n; t++) {
		uint32_t w = b[t];
		int k = k0 + (int16_t)(w >> 16);
		int64_t v = (int64_t)a[t] * (int32_t)(w & 0xffff);
		int64_t sc;
		if (k >= 0) sc = (k > 30 || uabs64(v) >> (62 - k)) ? (v < 0 ? -(1LL << 40) : (1LL << 40)) : v << k;
		else sc = rshr64(v, -k);
		if (sc > (1 << 30)) sc = 1 << 30;
		if (sc < -(1 << 30)) sc = -(1 << 30);
		a[t] = (int32_t)sc;
		if (a[t] > smax) smax = a[t];
	}
	return smax;
}

/* softmax numerators with the V scales folded in, one pass: a[t] =
 * exp(a[t] - smax) (Q16), *sum their sum, then a[t] times the mantissa
 * of b[t] shifted right by vemax - its exponent; returns the largest */
static uint32_t v_expcoef(int32_t *a, const uint32_t *b, int n, int32_t smax, int vemax, uint64_t *sum)
{
	if (mlm_vu) return mlm_vu(MLM_VU_EXPCOEF, a, b, a, (uint32_t)n, smax, vemax, sum);
	uint64_t s = 0;
	uint32_t cmax = 0;
	for (int t = 0; t < n; t++) {
		uint32_t e = mlm_exp_neg_q16((uint32_t)(smax - a[t]));
		s += e;
		uint32_t w = b[t];
		int d = vemax - (int16_t)(w >> 16);
		uint64_t c = (uint64_t)e * (w & 0xffff);
		c = (d > 63) ? 0 : (c >> d);
		a[t] = (int32_t)(uint32_t)c;
		if ((uint32_t)c > cmax) cmax = (uint32_t)c;
	}
	*sum = s;
	return cmax;
}

/* the C loops themselves, whatever mlm_vu is: for tests of the unit */
uint32_t mlm_vu_ref(int op, const void *a, const void *b, void *d, uint32_t n,
                    int32_t p0, int32_t p1, uint64_t *sum)
{
	uint32_t (*save)(int, const void *, const void *, void *, uint32_t, int32_t, int32_t, uint64_t *) = mlm_vu;
	uint32_t r = 0;
	mlm_vu = 0;
	switch (op) {
	case MLM_VU_MAXABS: r = v_maxabs((const int32_t *)a, (int)n); break;
	case MLM_VU_SQMUL: *sum = v_sqmul((int32_t *)d, (const int32_t *)a, (const int16_t *)b, (int)n, p0); break;
	case MLM_VU_SCALE15: v_scale15((int32_t *)d, (const int32_t *)a, (int)n, p0); break;
	case MLM_VU_RSHR: v_rshr((int32_t *)d, (const int32_t *)a, (int)n, p0); break;
	case MLM_VU_RESID: r = v_resid((int32_t *)d, (const int32_t *)b, (int)n, p0, p1); break;
	case MLM_VU_MULSHR30: v_mulshr30((int32_t *)d, (const int32_t *)a, (int)n, p0); break;
	case MLM_VU_KVQ8: v_kvq8((uint32_t *)d, (const int32_t *)a, (int)n, p0, p1); break;
	case MLM_VU_ROPE: v_rope((int32_t *)d, (const int16_t *)b, (int)n, p0); break;
	case MLM_VU_ATTSC: r = (uint32_t)v_attsc((int32_t *)d, (const uint32_t *)b, (int)n, p0); break;
	case MLM_VU_EXPCOEF: r = v_expcoef((int32_t *)d, (const uint32_t *)b, (int)n, p0, p1, sum); break;
	default: break;
	}
	mlm_vu = save;
	return r;
}

/* ---------------------------------------------------------------- */
/* engine glue                                                      */

static void dot(mlm_ctx_t *m, const mlm_tensor_t *t, uint32_t r0, uint32_t n)
{
	plat_eng_dot(t->off + r0 * t->row_stride, t->row_stride, n,
	             t->groups, t->qbits);
	m->n_dot++;
	m->dot_rows += n;
}

/* start a run into a bank without waiting */
static void dot_start(mlm_ctx_t *m, const mlm_tensor_t *t, uint32_t n, int bank)
{
	plat_eng_start(t->off, t->row_stride, n, t->groups, t->qbits, bank);
	if (!mlm_overlap) plat_eng_wait();
	m->n_dot++;
	m->dot_rows += n;
}

/* engine results -> int32 vector; returns its exponent */
static int out_to_i32(int32_t *dst, int n, int eacc)
{
	int ps = plat_prof_cur();
	plat_prof(MLM_PROF_READ);
	int sh = bitlen64(plat_eng_maxabs()) - 30;
	if (sh < 0) sh = 0;
	plat_eng_out32_block(dst, 0, (uint32_t)n, sh);
	plat_prof(ps);
	return eacc + sh;
}

/* int32 vector -> the engine's input, MX-packed (eng_mx_pack; the
 * firmware: the engine's packer, which reads v from the RAM); zero pads
 * to padn (a multiple of 32); returns the input's exponent */
#define PAD128(n) (((n) + 127) & ~127)   /* a 1-bit tensor's input: whole groups */
static int to_x(const int32_t *v, int n, int e, int padn)
{
	int ps = plat_prof_cur();
	plat_prof(MLM_PROF_PACK);
	int sx = plat_eng_xpack(v, n, padn);
	plat_prof(ps);
	return e + sx;
}

/* ---------------------------------------------------------------- */
/* building blocks                                                  */

/* 2^(14+b) / v for v of b bits, in 15 bits: v brought to 16 bits and
 * one 32-bit division (a 64-bit one is a slow library call on RV32) */
static int32_t recip15(uint64_t v, int b)
{
	uint32_t v16 = b > 16 ? (uint32_t)(((v >> (b - 17)) + 1) >> 1) : (uint32_t)v << (16 - b);
	return (int32_t)((0x40000000u + v16 / 2) / v16);
}

/* dst = rmsnorm(src) * g; dst may be src. Returns dst's exponent.
 *   u = src >> s (15 bits), w = u g (30 bits), mean of u^2 in Q16,
 *   inv = 2^(29+b) / rms (30 bits), dst = w inv >> sh (30 bits)
 * With defer set the last step is left to the caller: dst = w, and
 * *defer = inv15, the value being dst inv15 2^(exponent - 15) (the
 * QK norms: q's factor goes into the 1/sqrt(d) multiply, k's into
 * its cache step; a vector unit pass less a head) */
static int rmsnorm(mlm_ctx_t *m, int32_t *dst, const int32_t *src, int n,
                   int e_in, const mlm_norm_t *nw, int32_t *defer)
{
	const mlm_header_t *hd = m->hdr;
	const int16_t *g = norm_w(nw);
	int s = bitlen64((src == H && h_max_ok) ? h_max : v_maxabs(src, n)) - 15;
	if (s < 0) s = 0;
	uint64_t ss = v_sqmul(dst, src, g, n, s);
	uint64_t ms;
	if (!(n & (n - 1))) {                 /* a power of two: no 64-bit division */
		int l = 0;
		while ((1 << l) < n) l++;
		ms = (ss << 16) >> l;
	} else ms = (ss << 16) / (uint32_t)n;
	/* eps in the same units: eps_m * 2^(eps_e + 16 - 2(e_in + s)) */
	int es = hd->eps_e + 16 - 2 * (e_in + s);
	uint64_t eps;
	if (es >= 0) eps = (es > 40) ? ((uint64_t)1 << 60) : ((uint64_t)hd->eps_m << es);
	else eps = (-es > 62) ? 0 : ((uint64_t)hd->eps_m >> (-es));
	uint64_t tot = ms + eps;
	if (tot < ms) tot = ~(uint64_t)0 >> 2;
	uint32_t rms = mlm_isqrt64(tot);
	if (!rms) rms = 1;
	int b = bitlen64(rms);
	/* 2^(29+b) / rms to 15 bits: both factors of the scaling are 15
	 * bits, one 32-bit multiply, result <= 2^30 */
	int32_t inv15 = recip15(rms, b);
	if (defer) *defer = inv15;
	else v_scale15(dst, dst, n, inv15);
	return nw->exp + 8 - 29 - b + 30;
}

/* SwiGLU on the CPU, exactly glu_spec() (fw/common/eng_spec.h) and the
 * GLU unit: gate in bank 0, up in bank 1; leaves the products in S
 * (exponent: gate's + psh + up's) and returns the shifts. */
static void glu_cpu(uint32_t n, int sg, int su, int sq, int *psh_o, int *xsh_o)
{
	uint32_t smx = 0, pmx = 0;
	plat_eng_rbank(0);
	for (uint32_t i = 0; i < n; i++) {
		int32_t x = plat_eng_out32(i, sg);
		uint32_t a;
		if (sq >= 0) a = (sq > 20 || (uabs32(x) >> (20 - sq))) ? (16u << 16) : uabs32(x) << sq;
		else a = (uint32_t)rshr32((int32_t)uabs32(x), -sq);
		uint32_t g;
		if (a >= (16u << 16)) g = 65535;
		else {
			uint32_t j = a >> 10, f = a & 1023;
			g = mlm_sigtab[j] + (((uint32_t)(mlm_sigtab[j + 1] - mlm_sigtab[j]) * f) >> 10);
		}
		if (x < 0) g = 65536 - g;
		S[i] = rshr32(x, 15) * (int32_t)((g + 1) >> 1);
		if (uabs32(S[i]) > smx) smx = uabs32(S[i]);
	}
	int psh = bitlen64(smx) - 15;
	if (psh < 0) psh = 0;
	plat_eng_rbank(1);
	for (uint32_t i = 0; i < n; i++) {
		S[i] = rshr32(S[i], psh) * plat_eng_out32(i, su);
		if (uabs32(S[i]) > pmx) pmx = uabs32(S[i]);
	}
	plat_eng_rbank(0);
	*psh_o = psh;
	*xsh_o = bitlen64(pmx) - 15;
}

/* the GLU unit against glu_cpu on whatever banks 0 and 1 hold now
 * (bank 0 is overwritten); returns the number of differences */
void *mlm_scratch(void) { return S; }

uint32_t mlm_glu_check(uint32_t n, int sg, int su, int sq)
{
	int cp, cx, hp, hx;
	glu_cpu(n, sg, su, sq, &cp, &cx);
	plat_eng_glu(n, sg, su, sq, &hp, &hx);
	uint32_t bad = (cp != hp);
	(void)cx;
	const volatile uint32_t *xw = plat_eng_x(), *xdw = plat_eng_xd();
	/* the packing's scratch: S beyond the n products (n <= 3072) */
	uint32_t *xq = (uint32_t *)(S + ENG_X_MAX);
	uint8_t *xd = (uint8_t *)(xq + ENG_X_MAX / 4);
	if (n > ENG_X_MAX) return 1;
	int sx = eng_mx_pack(S, (int)n, (int)n, (int8_t *)xq, xd);
	bad += (sx != hx);
	for (uint32_t i = 0; i < n / 4; i++) bad += (xw[i] != xq[i]);
	for (uint32_t b = 0; b < n / 32; b++) bad += ((xdw[b] & 15) != xd[b]);
	return bad;
}

/* rotates v in place; the result's exponent is the input's + 1 */
static void rope(mlm_ctx_t *m, int32_t *v)
{
	v_rope(v, RT, m->hdr->head_dim, (m->hdr->flags & MLM_F_ROPE_HALF) != 0);
}

/* quantise one head's k or v: q[i] = round(v[i] * 127 / amax), on the
 * values brought to 16 bits; returns the packed step (real step =
 * sm * 2^se) */
static uint32_t kv_quant(const int32_t *v, int n, int e, int8_t *q, int32_t mul)
{
	uint32_t amax = v_maxabs(v, n);
	if (amax < 1) amax = 1;
	int s0 = bitlen64(amax) - 16;
	if (s0 < 0) s0 = 0;
	int32_t am = rshr32((int32_t)amax, s0);
	if (am < 1) am = 1;
	int32_t r = (int32_t)((127u << 16) / (uint32_t)am);
	v_kvq8((uint32_t *)q, v, n, s0, r);
	uint64_t sm = ((uint64_t)amax * 33818641u) >> 16;     /* amax 2^16 / 127 (2^32 / 127, rounded up) */
	int se = e - 16;
	if (mul) { sm *= (uint32_t)mul; se -= 15; }          /* a deferred norm factor (rmsnorm) */
	int b = bitlen64(sm);
	if (b > 16) { sm = (uint64_t)rshr64((int64_t)sm, b - 16); se += b - 16; }
	if (sm >> 16) { sm >>= 1; se++; }
	if (!sm) sm = 1;
	return (uint32_t)sm | ((uint32_t)(se & 0xffff) << 16);
}

/* this position's K and V for every KV head of layer l: per head the
 * K row (headers, values, padding: k_stride bytes) and the V column (a
 * byte per dimension, v_stride apart) are built in the RAM and written
 * by the DMA unit, while the next head is quantised into the other
 * buffer; the scales after the last. */
static uint32_t KB[2][(16 + 256 + 32) / 4], VB[2][256 / 4];
static void kv_store_all(mlm_ctx_t *m, int l, const int32_t *k, const int *ek, const int32_t *kmul,
                         const int32_t *v, int ev)
{
	const mlm_header_t *hd = m->hdr;
	int n = hd->head_dim, hp = m->hd_pad, nkv = hd->n_kv_heads;
	/* in ATT, free until attention (not on the stack: the firmware's
	 * stack is short, and a step runs under the line editor) */
	uint32_t *ks = (uint32_t *)ATT, *vs = ks + MLM_MAX_HEADS;
	uint32_t *qw = vs + MLM_MAX_HEADS;  /* 64 words, aligned for v_kvq8 */
	int8_t *q = (int8_t *)qw;
	uint32_t col = (m->pos / 128) * 136 + 8 + (m->pos % 128);
	for (int i = 0; i < nkv; i++) {
		uint32_t row = (uint32_t)(l * nkv + i);
		uint32_t *kb = KB[i & 1];
		/* K: one row per position, Q8 chunks with multipliers of 1 */
		ks[i] = kv_quant(k + i * n, n, ek[i], q, kmul[i]);
		for (int j = n; j < hp; j++) q[j] = 0;
		uint32_t w = 0;
		for (int g = 0; g < hp / 32; g++) {
			if ((g & 3) == 0) { kb[w++] = 0x00010001u; kb[w++] = 0x00010001u; }
			for (int j = 0; j < 8; j++) kb[w++] = qw[g * 8 + j];
		}
		while (w < m->k_stride / 4) kb[w++] = 0;
		/* V: transposed, one row per dimension, one byte per position */
		vs[i] = kv_quant(v + i * n, n, ev, (int8_t *)VB[i & 1], 0);
		plat_mem_wblk(m->kcache + (row * m->ctx + m->pos) * m->k_stride, kb, m->k_stride / 16);
		plat_mem_wcol(m->vcache + row * n * m->v_stride + col, VB[i & 1], (uint32_t)n, m->v_stride);
	}
	plat_dma_wait();
	for (int i = 0; i < nkv; i++) {
		uint32_t row = (uint32_t)(l * nkv + i);
		wr32(m->kscale + (row * m->ctx + m->pos) * 4, ks[i]);
		wr32(m->vscale + (row * m->ctx + m->pos) * 4, vs[i]);
		int16_t ve = (int16_t)(vs[i] >> 16);
		if (row < VEM_ROWS && (m->pos == 0 || ve > VEM[row])) VEM[row] = ve;
	}
	ks_row = ~0u;                       /* the copies in attend() are stale now */
}

/* attention for every q head of layer l (q in S, head h at q + h hd,
 * exponents eqh) -> o into the same place, exponents eoh. Per head: the
 * scores (K . q, an 8-bit run), their softmax with the V scales folded
 * in (the vector unit), then V^T . coef (an 8-bit run) and the 1/sum.
 * The engine and the CPU overlap across heads: K of head h + 1 runs
 * during head h's softmax, V of head h during head h + 1's scores. Head
 * h's two runs go into bank h & 1, so each bank is read before the
 * next run into it starts. The results are the same as head by head. */
static void att_load(mlm_ctx_t *m, uint32_t row, uint32_t T)
{
	if (row == ks_row && T == ks_n) return;   /* the GQA group's other head */
	int ps_ = plat_prof_cur(); plat_prof(MLM_PROF_READ);
	/* the engine is idle here: the DMA unit copies them */
	int vemax = -32768;
	plat_mem_rd(KS, m->kscale + row * m->ctx * 4, T);
	plat_mem_rd(VS, m->vscale + row * m->ctx * 4, T);
	if (row < VEM_ROWS) vemax = VEM[row];
	else for (uint32_t t = 0; t < T; t++) {
		int ve = (int16_t)(VS[t] >> 16);
		if (ve > vemax) vemax = ve;
	}
	ks_row = row; ks_n = T; vs_emax = vemax;
	plat_prof(ps_);
}
static void attend_all(mlm_ctx_t *m, int l, int32_t *q)
{
	const mlm_header_t *hd = m->hdr;
	int n = hd->head_dim, hp = m->hd_pad, nh = hd->n_heads, g = nh / hd->n_kv_heads;
	uint32_t T = m->pos + 1, Tp = (T + 31) & ~31u;
	int exq[MLM_MAX_HEADS];
	uint64_t sum = 0;
	int ec = 0;
	#define ROW(h) ((uint32_t)(l * hd->n_kv_heads + (h) / g))
	/* q scaled by 1/sqrt(head_dim) (and its deferred norm factor) */
	#define PREP(h) do { int32_t as_ = (int32_t)hd->attn_scale; \
		if (qf[h]) as_ = (int32_t)(((int64_t)as_ * qf[h] + (1 << 14)) >> 15); \
		v_mulshr30(q + (h) * n, q + (h) * n, n, as_); } while (0)
	/* K . q into the head's bank (the engine must be idle: X) */
	#define START_K(h) do { exq[h] = to_x(q + (h) * n, n, eqh[h], hp); \
		plat_eng_start(m->kcache + ROW(h) * m->ctx * m->k_stride, m->k_stride, T, (uint32_t)hp / 32, 8, (h) & 1); \
		if (!mlm_overlap) plat_eng_wait(); \
		m->n_dot++; m->dot_rows += T; } while (0)
	/* the scores out of the head's bank into ATT */
	#define SCORES(h) do { plat_eng_rbank((h) & 1); \
		int s1_ = bitlen64(plat_eng_maxabs()) - 31; if (s1_ < 0) s1_ = 0; \
		plat_eng_out32_block(ATT, 0, T, s1_); exq[h] += s1_; } while (0)

	PREP(0);
	START_K(0);
	plat_eng_wait();
	att_load(m, ROW(0), T);
	SCORES(0);
	if (nh > 1) PREP(1);
	for (int h = 0; h < nh; h++) {
		if (h + 1 < nh) START_K(h + 1);
		/* head h's softmax weights, the V scales folded in (one pass) */
		int32_t smax = v_attsc(ATT, KS, (int)T, exq[h] + 16);
		uint32_t cmax = v_expcoef(ATT, VS, (int)T, smax, vs_emax, &sum);
		/* the coefficients are unsigned (up to 2^32): halved first if
		 * they would not fit the packer's int32 */
		int ce = vs_emax;
		if (cmax >> 31) {
			for (uint32_t t = 0; t < T; t++) ATT[t] = (int32_t)((uint32_t)ATT[t] >> 1);
			ce++;
		}
		if (h + 1 < nh) {
			plat_eng_wait();
			/* head h + 1's scales (after head h's softmax, before its scores) */
			if (ROW(h + 1) != ROW(h)) att_load(m, ROW(h + 1), T);
		}
		/* o = (V^T . coef) / sum, into the head's bank */
		ec = to_x(ATT, (int)T, ce, (int)Tp);
		plat_eng_start(m->vcache + ROW(h) * n * m->v_stride, m->v_stride, (uint32_t)n, Tp / 32, 8, h & 1);
		if (!mlm_overlap) plat_eng_wait();
		m->n_dot++; m->dot_rows += (uint32_t)n;
		int rs = bitlen64(sum);
		int32_t r15 = recip15(sum, rs);      /* 2^(29+rs) / sum, to 15 bits */
		if (h + 1 < nh) {
			SCORES(h + 1);
			if (h + 2 < nh) PREP(h + 2);
		}
		plat_eng_wait();
		/* read to 30 bits, then (o >> 15) r15 in the vector unit */
		plat_eng_rbank(h & 1);
		int s2 = bitlen64(plat_eng_maxabs()) - 30;
		if (s2 < 0) s2 = 0;
		plat_eng_out32_block(q + h * n, 0, (uint32_t)n, s2);
		v_scale15(q + h * n, q + h * n, n, r15);
		eoh[h] = ec - 29 - rs + s2 + 30;
	}
	plat_eng_rbank(0);
	#undef ROW
	#undef PREP
	#undef START_K
	#undef SCORES
}

/* h (eh) += v (ev); returns the new exponent. The sum is renormalised
 * to 30 bits again: without that each add cost h a bit, and 56 adds
 * (Qwen3-0.6B) left it nothing */
static int residual(int32_t *h, int eh, const int32_t *v, int ev, int n)
{
	int e = (eh > ev ? eh : ev) + 1;
	uint32_t mx = v_resid(h, v, n, e - eh, e - ev);
	int sh = 30 - bitlen64(mx);
	if (mx && sh > 0) {
		v_rshr(h, h, n, -sh);       /* a left shift: exact, so the max shifts with it */
		e -= sh;
		mx <<= sh;
	}
	if (h == H) { h_max = mx; h_max_ok = 1; }
	return e;
}

/* the token's row of the (1-bit) embedding: +-m per weight */
static int embed(mlm_ctx_t *m, uint32_t tok, int32_t *dst)
{
	const mlm_tensor_t *t = &m->hdr->emb;
	const uint8_t *row = MEM(t->off + tok * t->row_stride);
	for (uint32_t g = 0; g < t->groups; g++) {
		const uint8_t *ch = row + (g / 4) * ENG_CHUNK;
		int32_t mg = (int32_t)((uint32_t)ch[2 * (g % 4)] | ((uint32_t)ch[2 * (g % 4) + 1] << 8));
		const uint8_t *d = ch + 8 + (g % 4) * 16;
		for (int k = 0; k < 128; k++)
			dst[g * 128 + k] = ((d[k >> 3] >> (k & 7)) & 1) ? mg : -mg;
	}
	return t->exp;
}

/* ---------------------------------------------------------------- */

void mlm_rewind(mlm_ctx_t *m, uint32_t pos)
{
	const mlm_header_t *hd = m->hdr;
	if (pos >= m->pos) return;
	m->pos = pos;
	ks_row = ~0u;
	h_max_ok = 0;
	/* the cache rows beyond pos are stale but never read (and written
	 * again as positions are stored); each row's largest V exponent is
	 * over the positions that remain */
	uint32_t rows = hd->n_layers * hd->n_kv_heads;
	for (uint32_t row = 0; pos && row < rows && row < VEM_ROWS; row++) {
		plat_mem_rd(VS, m->vscale + row * m->ctx * 4, pos);
		int vemax = -32768;
		for (uint32_t t = 0; t < pos; t++) {
			int ve = (int16_t)(VS[t] >> 16);
			if (ve > vemax) vemax = ve;
		}
		VEM[row] = (int16_t)vemax;
	}
}

int mlm_init(mlm_ctx_t *m, uint32_t ctx)
{
	/* the header in block RAM: it is read all through each step, and
	 * from main memory every field costs a slow access */
	static mlm_header_t HDR;
	HDR = *(const mlm_header_t *)plat_mem();
	const mlm_header_t *hd = &HDR;
	m->hdr = hd;
	if (hd->magic != MLM_MAGIC || hd->version != MLM_VERSION) return -1;
	if (hd->arch != MLM_ARCH_LLAMA) return -2;
	if (!ctx || ctx > hd->ctx_max) ctx = hd->ctx_max;
	if (ctx > MLM_MAX_CTX) ctx = MLM_MAX_CTX;
	if (ctx > ENG_OUT_MAX || ctx > ENG_X_MAX) return -3;
	if (hd->dim > MLM_MAX_DIM || hd->n_heads > MLM_MAX_HEADS || hd->head_dim > 256) return -4;
	nw_off = ~0u;                       /* a new model: no weights cached */
	if (hd->n_heads * hd->head_dim + 2 * hd->n_kv_heads * hd->head_dim > MLM_MAX_SCRATCH ||
	    2 * hd->dim > MLM_MAX_SCRATCH) return -5;
	{
		/* the FFN in one or two pieces (see mlm_step_ex) */
		uint32_t np = (hd->hidden_dim + ENG_X_MAX - 1) / ENG_X_MAX;
		if (np > 2 || hd->hidden_dim % np || (hd->hidden_dim / np) % 128 ||
		    (np > 1 && (hd->hidden_dim / np) % 512) ||
		    PAD128(hd->dim) > ENG_X_MAX || PAD128(hd->n_heads * hd->head_dim) > ENG_X_MAX) return -5;
	}
	if (hd->head_dim > 256) return -6;
	m->ctx = ctx;
	m->hd_pad = (hd->head_dim + 31) & ~31u;
	m->k_stride = (mlm_row_words(m->hd_pad / 32, 8) * 8 + 15) & ~15u;
	m->v_stride = (mlm_row_words((ctx + 31) / 32, 8) * 8 + 15) & ~15u;
	uint32_t rows = hd->n_layers * hd->n_kv_heads;
	m->kv_base = (hd->file_size + 0xffff) & ~0xffffu;
	m->kcache = m->kv_base;
	m->vcache = m->kcache + rows * ctx * m->k_stride;
	m->kscale = m->vcache + rows * hd->head_dim * m->v_stride;
	m->vscale = m->kscale + rows * ctx * 4;
	m->mem_end = m->vscale + rows * ctx * 4;
	if (m->mem_end > plat_mem_size()) return -7;
	/* V^T chunk headers: multipliers of 1 */
	uint32_t nch = (ctx + 127) / 128;
	for (uint32_t r = 0; r < rows * hd->head_dim; r++)
		for (uint32_t c = 0; c < nch; c++) {
			uint32_t a = m->vcache + r * m->v_stride + c * 136;
			wr32(a, 0x00010001u); wr32(a + 4, 0x00010001u);
		}
	m->pos = 0;
	m->n_dot = 0;
	m->dot_rows = 0;
	if (!m->rng) m->rng = 0x2545f491u;
	return 0;
}

int mlm_step(mlm_ctx_t *m, uint32_t token) { return mlm_step_ex(m, token, 1); }

#ifdef MLM_DUMP
void mlm_dump(const char *what, const int32_t *v, int n, int e);
#endif
int mlm_step_ex(mlm_ctx_t *m, uint32_t token, int logits)
{
	const mlm_header_t *hd = m->hdr;
	int dim = hd->dim, hdim = hd->head_dim, nh = hd->n_heads, nkv = hd->n_kv_heads;
	int qd = nh * hdim, kvd = nkv * hdim, hid = hd->hidden_dim;
	if (m->pos >= m->ctx) return -1;

	int eh = embed(m, token, H);
	h_max_ok = 0;
#ifdef MLM_DUMP
	mlm_dump("emb", H, dim, eh);
#endif
	plat_mem_rd(RT, hd->rope_off + m->pos * (hdim / 2) * 4, (uint32_t)hdim / 2);
	int32_t *q = S, *k = S + qd, *v = S + qd + kvd;

	for (uint32_t l = 0; l < hd->n_layers; l++) {
		plat_poll();
		/* the descriptor in block RAM: main memory is the engine's
		 * while it streams */
		mlm_layer_t Lc;
		plat_mem_rd(&Lc, hd->layers_off + l * (uint32_t)sizeof(mlm_layer_t), (uint32_t)(sizeof(mlm_layer_t) / 4));
		const mlm_layer_t *L = &Lc;
		plat_prof(MLM_PROF_NORM);
		int en = rmsnorm(m, S, H, dim, eh, &L->attn_norm, 0);
#ifdef MLM_DUMP
		if (l == 0) mlm_dump("norm", S, dim, en);
#endif
		int ex = to_x(S, dim, en, PAD128(dim));
		plat_prof(MLM_PROF_QKV);
		/* q, k, v: each converted while the engine computes the next */
		dot_start(m, &L->wq, qd, 0);
		plat_eng_wait();
		dot_start(m, &L->wk, kvd, 1);
		plat_eng_rbank(0);
		int eq = out_to_i32(q, qd, L->wq.exp + ex);
#ifdef MLM_DUMP
		if (l == 0) mlm_dump("q", q, qd, eq);
#endif
		/* q's QK-norm and RoPE while the engine computes k */
		plat_prof(MLM_PROF_QKROPE);
		for (int i = 0; i < nh; i++) {
			eqh[i] = eq; qf[i] = 0;
			if (hd->flags & MLM_F_QK_NORM)
				eqh[i] = rmsnorm(m, q + i * hdim, q + i * hdim, hdim, eq, &L->q_norm, &qf[i]);
			rope(m, q + i * hdim);
			eqh[i]++;
		}
		plat_prof(MLM_PROF_QKV);
		plat_eng_wait();
		dot_start(m, &L->wv, kvd, 0);
		plat_eng_rbank(1);
		int ek = out_to_i32(k, kvd, L->wk.exp + ex);
		/* and k's while it computes v */
		plat_prof(MLM_PROF_QKROPE);
		for (int i = 0; i < nkv; i++) {
			ekh[i] = ek; kf[i] = 0;
			if (hd->flags & MLM_F_QK_NORM)
				ekh[i] = rmsnorm(m, k + i * hdim, k + i * hdim, hdim, ek, &L->k_norm, &kf[i]);
			rope(m, k + i * hdim);
			ekh[i]++;
		}
		plat_prof(MLM_PROF_QKV);
		plat_eng_wait();
		plat_eng_rbank(0);
		int ev = out_to_i32(v, kvd, L->wv.exp + ex);
#ifdef MLM_DUMP
		if (l == 0) mlm_dump("v", v, kvd, ev);
#endif
		plat_prof(MLM_PROF_KV);
		kv_store_all(m, (int)l, k, ekh, kf, v, ev);
		plat_prof(MLM_PROF_ATT);
		attend_all(m, (int)l, q);
		plat_prof(MLM_PROF_WO);
		int eo = -10000;
		for (int i = 0; i < nh; i++) if (eoh[i] > eo) eo = eoh[i];
		for (int i = 0; i < nh; i++)
			if (eo != eoh[i]) v_rshr(q + i * hdim, q + i * hdim, hdim, eo - eoh[i]);
#ifdef MLM_DUMP
		if (l == 0) mlm_dump("o", q, qd, eo);
#endif
		ex = to_x(q, qd, eo, PAD128(qd));
		dot(m, &L->wo, 0, dim);
		int ea = out_to_i32(S, dim, L->wo.exp + ex);
		eh = residual(H, eh, S, ea, dim);
#ifdef MLM_DUMP
		if (l < 2) mlm_dump("attn", H, dim, eh);
#endif

		/* feed-forward, in pieces of at most ENG_X_MAX hidden units
		 * (Bonsai: two of 3072): per piece, gate into bank 0 and up
		 * into bank 1, the GLU unit makes SiLU(gate) x up the engine's
		 * next input, and down runs over that piece's columns. The
		 * normalised input stays in S[0..dim) for the next piece; the
		 * down outputs add up in S[dim..2 dim). */
		plat_prof(MLM_PROF_NORM);
		en = rmsnorm(m, S, H, dim, eh, &L->ffn_norm, 0);
		int np = (hid + ENG_X_MAX - 1) / ENG_X_MAX, ph = hid / np;
		int32_t *P = S + dim;
		int ep = 0;
		int exd0 = 0;
		for (int pc = 0; pc < np; pc++) {
			plat_prof(MLM_PROF_NORM);
			if (pc == 0) ex = to_x(S, dim, en, PAD128(dim));
			else {
				/* the same input again (the GLU unit wrote X): S[0..dim)
				 * still holds it, and packs to the same values */
				int ps = plat_prof_cur();
				plat_prof(MLM_PROF_PACK);
				plat_eng_xpack(S, dim, PAD128(dim));
				plat_prof(ps);
			}
			plat_prof(MLM_PROF_FFN1);
			uint32_t r0 = (uint32_t)(pc * ph);
			plat_eng_start(L->w1.off + r0 * L->w1.row_stride, L->w1.row_stride, (uint32_t)ph,
			               L->w1.groups, L->w1.qbits, 0);
			m->n_dot++; m->dot_rows += (uint32_t)ph;
			if (!mlm_overlap) plat_eng_wait();
			if (pc > 0) {
				/* the first piece's down outputs (bank 1) while this
				 * gate streams into bank 0 */
				plat_eng_rbank(1);
				ep = out_to_i32(P, dim, L->w2.exp + exd0);
				plat_eng_rbank(0);
			}
			plat_eng_wait();
			plat_eng_start(L->w3.off + r0 * L->w3.row_stride, L->w3.row_stride, (uint32_t)ph,
			               L->w3.groups, L->w3.qbits, 1);
			m->n_dot++; m->dot_rows += (uint32_t)ph;
			plat_prof(MLM_PROF_SILU);
			plat_eng_wait();
			plat_prof(MLM_PROF_FFN2);
			plat_eng_rbank(0);
			int sg = bitlen64(plat_eng_maxabs()) - 30;
			if (sg < 0) sg = 0;
			plat_eng_rbank(1);
			int su = bitlen64(plat_eng_maxabs()) - 15;
			if (su < 0) su = 0;
			plat_eng_rbank(0);
			int eg = L->w1.exp + ex + sg;
			int sq = eg + 16;
			if (sq > 21) sq = 21;
			if (sq < -40) sq = -40;
			int psh, xsh, exd;
			if (mlm_hw_glu || np > 1) {
				plat_eng_glu((uint32_t)ph, sg, su, sq, &psh, &xsh);
				exd = eg + psh + (L->w3.exp + ex + su) + xsh;
			} else {
				glu_cpu((uint32_t)ph, sg, su, sq, &psh, &xsh);
				exd = to_x(S, ph, eg + psh + (L->w3.exp + ex + su), ph);
			}
			plat_prof(MLM_PROF_FFN3);
			/* down over this piece's columns: ph / 128 groups from
			 * column r0 (ph a multiple of 512: whole chunks) */
			uint32_t g0 = r0 / ENG_GROUP, ng = (uint32_t)ph / ENG_GROUP;
			if (pc + 1 < np) {
				/* into bank 1, read under the next piece's gate */
				plat_eng_start(L->w2.off + (g0 / 4) * ENG_CHUNK, L->w2.row_stride, (uint32_t)dim,
				               ng, L->w2.qbits, 1);
				plat_eng_wait();
				exd0 = exd;
				m->n_dot++; m->dot_rows += (uint32_t)dim;
				continue;
			}
			plat_eng_dot(L->w2.off + (g0 / 4) * ENG_CHUNK, L->w2.row_stride, (uint32_t)dim,
			             ng, L->w2.qbits);
			m->n_dot++; m->dot_rows += (uint32_t)dim;
			if (pc == 0) {
				ep = out_to_i32(P, dim, L->w2.exp + exd);
			} else {
				/* the last piece (at most two): its input is in X
				 * already, so S[0..dim) is free */
				int ea = out_to_i32(S, dim, L->w2.exp + exd);
				ep = residual(P, ep, S, ea, dim);
			}
		}
		eh = residual(H, eh, P, ep, dim);
#ifdef MLM_DUMP
		if (l < 2 || l == (uint32_t)hd->n_layers - 1) mlm_dump("ffn", H, dim, eh);
#endif
	}

	/* output head, top-k */
	m->top_n = 0;
	plat_prof(MLM_PROF_HEAD);
	if (!logits) { m->pos++; return 0; }
	int en = rmsnorm(m, S, H, dim, eh, &hd->final_norm, 0);
	int ex = to_x(S, dim, en, PAD128(dim));
	m->top_n = 0;
	/* chunks of ENG_OUT_MAX rows, alternating banks: the CPU scans one
	 * chunk while the engine computes the next */
	const mlm_tensor_t *ht = &hd->head;
	uint32_t nch = (hd->vocab + ENG_OUT_MAX - 1) / ENG_OUT_MAX;
	for (uint32_t c = 0; c < nch; c++) {
		uint32_t r0 = c * ENG_OUT_MAX, n = hd->vocab - r0;
		if (n > ENG_OUT_MAX) n = ENG_OUT_MAX;
		if (c == 0) {
			plat_eng_start(ht->off, ht->row_stride, n, ht->groups, ht->qbits, 0);
			m->n_dot++; m->dot_rows += n;
		}
		plat_eng_wait();
		if (c + 1 < nch) {
			uint32_t r1 = r0 + ENG_OUT_MAX, n1 = hd->vocab - r1;
			if (n1 > ENG_OUT_MAX) n1 = ENG_OUT_MAX;
			plat_eng_start(ht->off + r1 * ht->row_stride, ht->row_stride, n1, ht->groups,
			               ht->qbits, (int)((c + 1) & 1));
			m->n_dot++; m->dot_rows += n1;
		}
		plat_eng_rbank((int)(c & 1));
		/* one 32-bit read per row: the chunk's results shifted to 30
		 * bits, compared back at full scale */
		int hsh = bitlen64(plat_eng_maxabs()) - 30;
		if (hsh < 0) hsh = 0;
		/* the chunk's results into S (free here: x is in the engine) */
		plat_eng_out32_block(S, 0, n, hsh);
		/* the k-th best at this chunk's scale: S[i] << hsh <= top
		 * exactly when S[i] <= floor(top / 2^hsh), so most rows cost
		 * one 32-bit compare (the 64-bit shift and compare were ~60
		 * cycles a row, 151,669 rows a token) */
		int32_t thr = INT32_MIN;
		if (m->top_n == MLM_TOPK) {
			int64_t t = m->top_acc[MLM_TOPK - 1] >> hsh;
			thr = t > INT32_MAX ? INT32_MAX : t < INT32_MIN ? INT32_MIN : (int32_t)t;
		}
		for (uint32_t i = 0; i < n; i++) {
			if (S[i] <= thr && m->top_n == MLM_TOPK) continue;
			int64_t a = (int64_t)((uint64_t)(int64_t)S[i] << hsh);
			if (m->top_n == MLM_TOPK && a <= m->top_acc[MLM_TOPK - 1]) continue;
			int j = (m->top_n < MLM_TOPK) ? (int)m->top_n++ : MLM_TOPK - 1;
			while (j > 0 && m->top_acc[j - 1] < a) {
				m->top_acc[j] = m->top_acc[j - 1];
				m->top_id[j] = m->top_id[j - 1];
				j--;
			}
			m->top_acc[j] = a;
			m->top_id[j] = r0 + i;
			if (m->top_n == MLM_TOPK) {
				int64_t t = m->top_acc[MLM_TOPK - 1] >> hsh;
				thr = t > INT32_MAX ? INT32_MAX : t < INT32_MIN ? INT32_MIN : (int32_t)t;
			}
		}
	}
	plat_eng_rbank(0);
	m->logit_exp = hd->head.exp + ex;
	m->pos++;
	return 0;
}

static uint32_t xorshift(uint32_t *s)
{
	uint32_t x = *s;
	x ^= x << 13; x ^= x >> 17; x ^= x << 5;
	return *s = x;
}

uint32_t mlm_sample(mlm_ctx_t *m)
{
	if (!m->temp_q8 || m->top_n < 2) return m->top_id[0];
	uint32_t w[MLM_TOPK];
	uint64_t tot = 0;
	int s = m->logit_exp + 16 + 8;
	for (uint32_t i = 0; i < m->top_n; i++) {
		uint64_t d = (uint64_t)(m->top_acc[0] - m->top_acc[i]);
		if (s >= 0) d = (s > 30 || (d >> (62 - s))) ? ((uint64_t)1 << 62) : d << s;
		else d = (uint64_t)rshr64((int64_t)d, -s);
		d /= m->temp_q8;
		w[i] = mlm_exp_neg_q16(d > 0xffffffffu ? 0xffffffffu : (uint32_t)d);
		tot += w[i];
	}
	uint64_t lim = (tot * m->topp_q16) >> 16, cum = 0;
	uint32_t n = 0;
	while (n < m->top_n && (cum < lim || n == 0)) cum += w[n++];
	uint64_t r = ((uint64_t)xorshift(&m->rng) * cum) >> 32;
	for (uint32_t i = 0; i < n; i++) {
		if (r < w[i]) return m->top_id[i];
		r -= w[i];
	}
	return m->top_id[n - 1];
}

/* ---------------------------------------------------------------- */
/* tokenizer                                                        */

typedef struct { const uint8_t *base; uint32_t n, maxlen; const uint8_t *offs, *sorted; } tok_t;

static void tok_open(mlm_ctx_t *m, tok_t *t)
{
	t->base = MEM(m->hdr->tok_off);
	t->n = rd32u(t->base);
	t->maxlen = rd32u(t->base + 4);
	t->offs = t->base + rd32u(t->base + 12);
	t->sorted = t->base + rd32u(t->base + 16);
}

static const uint8_t *tok_entry(const tok_t *t, uint32_t id, uint32_t *len, int32_t *score)
{
	const uint8_t *e = t->base + rd32u(t->offs + 4 * id);
	if (score) *score = (int32_t)rd32u(e);
	*len = rd32u(e + 4);
	return e + 8;
}

static int tok_cmp(const uint8_t *a, uint32_t al, const uint8_t *b, uint32_t bl)
{
	uint32_t n = al < bl ? al : bl;
	for (uint32_t i = 0; i < n; i++)
		if (a[i] != b[i]) return (int)a[i] - (int)b[i];
	return (int)al - (int)bl;
}

static int tok_lookup(const tok_t *t, const uint8_t *s, uint32_t len)
{
	int lo = 0, hi = (int)t->n - 1;
	while (lo <= hi) {
		int mid = (lo + hi) >> 1;
		uint32_t id = rd32u(t->sorted + 4 * mid), el;
		const uint8_t *e = tok_entry(t, id, &el, 0);
		int c = tok_cmp(e, el, s, len);
		if (!c) return (int)id;
		if (c < 0) lo = mid + 1; else hi = mid - 1;
	}
	return -1;
}

int mlm_encode(mlm_ctx_t *m, const char *text, uint32_t *out, int max)
{
	tok_t t;
	tok_open(m, &t);
	int bpe = (m->hdr->flags & MLM_F_TOK_BPE) != 0;
	int n = 0;
	static uint8_t buf[256];             /* two tokens joined (none is longer) */
	if (!bpe && text[0] && n < max) {
		int id = tok_lookup(&t, (const uint8_t *)" ", 1);
		if (id >= 0) out[n++] = (uint32_t)id;
	}
	const uint8_t *p = (const uint8_t *)text;
	while (*p && n < max) {
		uint32_t cl = 1;
		if (!bpe) {
			if ((*p & 0xe0) == 0xc0) cl = 2;
			else if ((*p & 0xf0) == 0xe0) cl = 3;
			else if ((*p & 0xf8) == 0xf0) cl = 4;
			for (uint32_t i = 1; i < cl; i++) if (!p[i]) { cl = i; break; }
		}
		int id = tok_lookup(&t, p, cl);
		if (id >= 0) out[n++] = (uint32_t)id;
		else for (uint32_t i = 0; i < cl && n < max; i++) {
			uint8_t b = p[i];
			id = tok_lookup(&t, &b, 1);
			out[n++] = (id >= 0) ? (uint32_t)id : (uint32_t)b + 3;
		}
		p += cl;
	}
	for (;;) {
		int32_t best = -0x7fffffff - 1;
		int bi = -1, bid = -1;
		for (int i = 0; i + 1 < n; i++) {
			uint32_t la, lb;
			const uint8_t *a = tok_entry(&t, out[i], &la, 0);
			const uint8_t *b = tok_entry(&t, out[i + 1], &lb, 0);
			if (la + lb > sizeof(buf)) continue;
			for (uint32_t j = 0; j < la; j++) buf[j] = a[j];
			for (uint32_t j = 0; j < lb; j++) buf[la + j] = b[j];
			int id = tok_lookup(&t, buf, la + lb);
			if (id < 0) continue;
			int32_t sc;
			uint32_t l2;
			tok_entry(&t, (uint32_t)id, &l2, &sc);
			if (sc > best) { best = sc; bi = i; bid = id; }
		}
		if (bi < 0) break;
		out[bi] = (uint32_t)bid;
		for (int i = bi + 1; i + 1 < n; i++) out[i] = out[i + 1];
		n--;
	}
	return n;
}

const uint8_t *mlm_decode(mlm_ctx_t *m, uint32_t prev, uint32_t tok, uint32_t *len)
{
	static uint8_t byte;
	tok_t t;
	tok_open(m, &t);
	if (tok >= t.n) { *len = 0; return 0; }
	const uint8_t *s = tok_entry(&t, tok, len, 0);
	if (m->hdr->flags & MLM_F_TOK_BPE) return s;
	if (prev == m->hdr->bos_id && *len && s[0] == ' ') { s++; (*len)--; }
	if (*len == 6 && s[0] == '<' && s[1] == '0' && s[2] == 'x' && s[5] == '>') {
		int v = 0;
		for (int i = 3; i < 5; i++) {
			int c = s[i];
			v = v * 16 + (c <= '9' ? c - '0' : (c | 32) - 'a' + 10);
		}
		byte = (uint8_t)v;
		*len = 1;
		return &byte;
	}
	return s;
}
