/*
 * machllm
 * Copyright (c) 2025 Lone Dynamics Corporation. All rights reserved.
 *
 * The engine's specification: what rtl/mlm_engine.v computes for one
 * call. Used by the PC reference (as its engine) and by the testbenches
 * (to check the RTL).
 */

#ifndef ENG_SPEC_H
#define ENG_SPEC_H

#include <stdint.h>

/* Bonsai-style weights: 1 bit each (1 = +m, 0 = -m), groups of 128
 * along a row with one 16-bit multiplier m per group. A row is chunks of
 * up to four groups: a u64 of four u16 multipliers (group 0 in bits
 * 15:0), then the groups' bits, 16 bytes each, least significant bit
 * first (the GGUF Q1_0 order). A chunk is nine 64-bit words.
 *
 * The input x is MX-style: int8 values in blocks of 32, each block with
 * a 4-bit exponent offset d (0..15): x[i] = q[i] 2^(-d) relative to the
 * vector's scale. For each block b of a group, S_b = sum (bit ? q : -q),
 * and
 *
 *   acc += (S_b * m_g) << (15 - d_b)
 *
 * exactly (|S_b| <= 32 * 127, 16-bit m: 30 bits before the shift).
 *
 * qbits 8 (the KV cache, for attention): int8 weights in groups of 32,
 * one x block per group; chunks of four groups are the u64 of
 * multipliers (ignored: the KV cache writes 1) then 4 x 32 bytes (17
 * words):
 *
 *   acc += (sum w q) << (15 - d_g)     (|sum| <= 32 * 127 * 127) */
#define ENG_GROUP   128
#define ENG_CHUNK   72      /* bytes in a chunk of four groups */
#define ENG_CHUNK8  136     /* the same at 8 bits (groups of 32) */

static inline uint32_t eng_row_bytes(uint32_t groups)
{
	uint32_t full = groups / 4, rem = groups % 4;
	return full * ENG_CHUNK + (rem ? 8 + rem * 16 : 0);
}

static inline void eng_dot_spec(const uint8_t *m, const int8_t *xq, const uint8_t *xd,
                  int64_t *out, uint32_t addr, uint32_t stride, uint32_t rows,
                  uint32_t groups, uint32_t qbits)
{
	if (qbits == 8) {
		/* the KV cache: int8 weights in groups of 32 (one block of x
		 * each); chunks of four: the multipliers, then 4 x 32 bytes */
		for (uint32_t r = 0; r < rows; r++) {
			const uint8_t *p = m + addr + r * stride;
			int64_t acc = 0;
			for (uint32_t g = 0; g < groups; g++) {
				const uint8_t *hdr = p + (g / 4) * ENG_CHUNK8;
				const int8_t *d = (const int8_t *)(hdr + 8 + (g & 3) * 32);
				int64_t sm = 0;
				for (int k = 0; k < 32; k++) sm += (int32_t)d[k] * xq[g * 32 + k];
				acc += (int64_t)((uint64_t)sm << (15 - (xd[g] & 15)));
			}
			out[r] = acc;
		}
		return;
	}
	for (uint32_t r = 0; r < rows; r++) {
		const uint8_t *p = m + addr + r * stride;
		int64_t acc = 0;
		for (uint32_t g = 0; g < groups; g++) {
			const uint8_t *hdr = p + (g / 4) * ENG_CHUNK;
			int64_t mg = (int64_t)((uint32_t)hdr[2 * (g & 3)] | ((uint32_t)hdr[2 * (g & 3) + 1] << 8));
			const uint8_t *d = hdr + 8 + (g & 3) * 16;
			for (int b = 0; b < 4; b++) {
				int32_t s = 0;
				for (int k = 0; k < 32; k++) {
					int j = 32 * b + k;
					int q = xq[g * 128 + j];
					s += ((d[j >> 3] >> (j & 7)) & 1) ? q : -q;
				}
				acc += (int64_t)((uint64_t)(s * mg) << (15 - (xd[g * 4 + b] & 15)));
			}
		}
		out[r] = acc;
	}
}

/* The input's packing (the CPU's to_x and the GLU unit's last pass):
 * int32 v[0..n) (zero from n to padn, padn a multiple of 32) into int8
 * blocks of 32 with exponent offsets. Per block, s_b = max(0, bitlen(max
 * |v|) - 7); s = max s_b; d_b = min(15, s - s_b); q = clamp(rshr(v, s -
 * d_b), +-127). Returns s - 15: x's exponent relative to v's (the
 * engine's (S m) << (15 - d) carries the 2^15). */
static inline int eng_mx_bitlen(uint32_t v) { int n = 0; while (v) { n++; v >>= 1; } return n; }
static inline int32_t eng_mx_rshr(int32_t x, int s)
{
	if (s <= 0) return x;
	if (s > 31) return 0;
	return (int32_t)((((int64_t)x >> (s - 1)) + 1) >> 1);
}
static inline int eng_mx_pack(const int32_t *v, int n, int padn, int8_t *q, uint8_t *xd)
{
	int nb = padn / 32, s = 0;
	for (int b = 0; b < nb; b++) {
		uint32_t mx = 0;
		for (int k = 0; k < 32; k++) {
			int i = 32 * b + k;
			if (i < n) {
				uint32_t a = v[i] < 0 ? (uint32_t)(-(int64_t)v[i]) : (uint32_t)v[i];
				if (a > mx) mx = a;
			}
		}
		int sb = eng_mx_bitlen(mx) - 7;
		if (sb < 0) sb = 0;
		xd[b] = (uint8_t)sb;          /* s_b for now */
		if (sb > s) s = sb;
	}
	for (int b = 0; b < nb; b++) {
		int d = s - xd[b];
		if (d > 15) d = 15;
		xd[b] = (uint8_t)d;
		for (int k = 0; k < 32; k++) {
			int i = 32 * b + k;
			int32_t x = (i < n) ? eng_mx_rshr(v[i], s - d) : 0;
			if (x > 127) x = 127;
			if (x < -127) x = -127;
			q[i] = (int8_t)x;
		}
	}
	return s - 15;
}

/* The GLU unit: SwiGLU's elementwise part, on the two output banks.
 *
 *   bank 0 holds the gate projection, bank 1 the up projection (n rows,
 *   exact sums). With sg and su the shifts the CPU chose from MAXABS0/1
 *   and sq the gate's exponent + 16 (clamped to -40..21):
 *
 *   g    = (b0 + 2^(sg-1)) >> sg                    as OUT32
 *   a    = |g| in Q16 (shifted by sq), 16.0 at most
 *   sig  = table[a>>10] + ((table[a>>10 + 1] - table[a>>10]) (a & 1023)) >> 10
 *          mirrored (65536 - sig) for negative g
 *   s    = rshr32(g, 15) x ((sig + 1) >> 1)          SiLU, <= 2^30
 *   psh  = max(0, bitlen(max |s|) - 15)
 *   p    = rshr32(s, psh) x ((b1 + 2^(su-1)) >> su)  the product, <= 2^30
 *   x    = eng_mx_pack(p): the engine's next input; xsh = its exponent
 *
 * Exactly the firmware's former SiLU / product / to_x on the CPU. */
static inline int32_t glu_rshr32(int32_t x, int s)
{
	if (s <= 0) return (int32_t)((uint32_t)x << (-s));
	if (s > 31) return 0;
	return ((x >> (s - 1)) + 1) >> 1;
}
static inline int glu_bitlen(uint32_t v) { int n = 0; while (v) { n++; v >>= 1; } return n; }
static inline uint32_t glu_abs(int32_t v) { return v < 0 ? (uint32_t)(-(int64_t)v) : (uint32_t)v; }
static inline int32_t glu_out32(int64_t acc, int sh)
{
	if (sh > 0) acc = (acc + ((int64_t)1 << (sh - 1))) >> sh;
	return (int32_t)acc;
}

static inline void glu_spec(const int64_t *b0, const int64_t *b1, int8_t *xq, uint8_t *xd, uint32_t n,
                            int sg, int su, int sq, const uint16_t *tab, int *psh_o, int *xsh_o)
{
	static int32_t s[4096];
	uint32_t smx = 0, pmx = 0;
	for (uint32_t i = 0; i < n; i++) {
		int32_t g = glu_out32(b0[i], sg);
		uint32_t a;
		if (sq >= 0) a = (sq > 20 || (glu_abs(g) >> (20 - sq))) ? (16u << 16) : glu_abs(g) << sq;
		else a = (uint32_t)glu_rshr32((int32_t)glu_abs(g), -sq);
		uint32_t sig;
		if (a >= (16u << 16)) sig = 65535;
		else {
			uint32_t j = a >> 10, f = a & 1023;
			sig = tab[j] + (((uint32_t)(tab[j + 1] - tab[j]) * f) >> 10);
		}
		if (g < 0) sig = 65536 - sig;
		s[i] = glu_rshr32(g, 15) * (int32_t)((sig + 1) >> 1);
		if (glu_abs(s[i]) > smx) smx = glu_abs(s[i]);
	}
	int psh = glu_bitlen(smx) - 15;
	if (psh < 0) psh = 0;
	for (uint32_t i = 0; i < n; i++) {
		s[i] = glu_rshr32(s[i], psh) * glu_out32(b1[i], su);
		if (glu_abs(s[i]) > pmx) pmx = glu_abs(s[i]);
	}
	(void)pmx;
	/* the engine's next input, MX-packed (n a multiple of 32) */
	*psh_o = psh;
	*xsh_o = eng_mx_pack(s, (int)n, (int)n, xq, xd);
}

#endif
