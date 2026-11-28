/*
 * machllm
 * Copyright (c) 2025 Lone Dynamics Corporation. All rights reserved.
 *
 * The forward pass and sampling, in integer arithmetic. This file and
 * mlm_model.c ARE the arithmetic specification: the firmware and the PC
 * reference compile the same source. See docs/arithmetic.md.
 */

#ifndef MLM_MODEL_H
#define MLM_MODEL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
#include "mlm.h"

/* Compile-time ceilings for the static buffers. */
#ifndef MLM_MAX_DIM
#define MLM_MAX_DIM     2048
#endif
#ifndef MLM_MAX_SCRATCH          /* int32s: max(q+k+v dims, hidden) */
#define MLM_MAX_SCRATCH 4096
#endif
#ifndef MLM_MAX_HEADS
#define MLM_MAX_HEADS   32
#endif
#ifndef MLM_MAX_CTX
#define MLM_MAX_CTX     512
#endif
#define MLM_TOPK        20

typedef struct {
	const mlm_header_t *hdr;
	uint32_t ctx;           /* runtime context length <= hdr->ctx_max */
	uint32_t hd_pad;        /* head_dim rounded up to 32 */
	uint32_t kv_base;       /* mem offsets of the caches */
	uint32_t k_stride, v_stride;
	uint32_t kcache, vcache, kscale, vscale;
	uint32_t mem_end;
	uint32_t pos;           /* tokens in the cache */
	/* sampling */
	uint32_t temp_q8;       /* temperature * 256; 0 = greedy */
	uint32_t topp_q16;      /* top-p * 65536 */
	uint32_t rng;
	/* last step's top-k, best first */
	uint32_t top_n;
	uint32_t top_id[MLM_TOPK];
	int64_t  top_acc[MLM_TOPK];
	int32_t  logit_exp;     /* real logit = top_acc * 2^logit_exp */
	/* statistics */
	uint32_t n_dot;
	uint64_t dot_rows;
} mlm_ctx_t;

/* 0 on success; sets up the caches after the image in memory. */
int mlm_init(mlm_ctx_t *m, uint32_t ctx);

/* Run one token at position m->pos; fills the top-k; pos++.
 * Returns 0, or -1 when the context is full. */
int mlm_step(mlm_ctx_t *m, uint32_t token);
/* The same; with logits = 0 the output head is skipped (a prompt token
 * whose successor is already known): about a quarter of the bytes of a
 * Qwen3-0.6B token, and the scan of the vocabulary. */
int mlm_step_ex(mlm_ctx_t *m, uint32_t token, int logits);

/* The GLU unit against the CPU on whatever the engine's banks 0 and 1
 * hold (bank 0 is overwritten, so is X); returns the differences. */
uint32_t mlm_glu_check(uint32_t n, int sg, int su, int sq);

/* The vector unit (rtl/mlm_vu.v): when set, the model's vector loops
 * call it instead of running on the CPU; the C loops in mlm_model.c are
 * its specification. Ops and their arguments: */
enum {
	MLM_VU_MAXABS = 1,      /* a, n -> max |a[i]| */
	MLM_VU_SQMUL = 2,       /* a, b = int16 g, d, n, p0 = s -> *sum */
	MLM_VU_SCALE15 = 3,     /* a, d, n, p0 = k */
	MLM_VU_RSHR = 4,        /* a, d, n, p0 = s */
	MLM_VU_RESID = 5,       /* a = h (= d), b = v, n, p0, p1 -> max |h| */
	MLM_VU_MULSHR30 = 6,    /* a, d, n, p0 = k */
	MLM_VU_KVQ8 = 8,        /* a, d, n, p0 = s, p1 = r */
	MLM_VU_ROPE = 9,        /* a = v (= d), b = rt, n = head dim, p0 = split */
	MLM_VU_ATTSC = 10,      /* a (= d) raw scores, b = K scales, n, p0 = k0 -> max (signed) */
	/* 7, 11-14: QPACK, EXPSUM, COEF, QPACKU, MXPACK, no longer in the unit */
	MLM_VU_EXPCOEF = 15,    /* a (= d), b = V scales, n, p0 = max score, p1 = max exponent -> max, *sum */
};
/* the same operations by the C loops, whatever mlm_vu is (tests) */
uint32_t mlm_vu_ref(int op, const void *a, const void *b, void *d, uint32_t n,
                    int32_t p0, int32_t p1, uint64_t *sum);
extern uint32_t (*mlm_vu)(int op, const void *a, const void *b, void *d, uint32_t n,
                          int32_t p0, int32_t p1, uint64_t *sum);

/* the forward pass's scratch (16 KB), for use between steps */
void *mlm_scratch(void);

/* Pick the next token from the last step's top-k. */
uint32_t mlm_sample(mlm_ctx_t *m);

/* Tokenizer. Returns the number of tokens written (<= max). */
int mlm_encode(mlm_ctx_t *m, const char *text, uint32_t *out, int max);
/* The bytes of a token; *len set. prev is the previous token (for the
 * SentencePiece leading-space rule). */
const uint8_t *mlm_decode(mlm_ctx_t *m, uint32_t prev, uint32_t tok, uint32_t *len);

/* exposed for tests */
uint32_t mlm_exp_neg_q16(uint32_t x_q16);
uint32_t mlm_isqrt64(uint64_t v);

#ifdef __cplusplus
}
#endif

#endif
