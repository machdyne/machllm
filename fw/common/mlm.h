/*
 * machllm
 * Copyright (c) 2025 Lone Dynamics Corporation. All rights reserved.
 *
 * The .mlm model image: layout and constants. See docs/format.md.
 *
 * Shared by the firmware (fw/), the PC reference engine (ref/) and,
 * through tools/mlm_format.py, the converter. Every field is a 32-bit
 * little-endian word.
 */

#ifndef MLM_H
#define MLM_H

#include <stdint.h>

#define MLM_MAGIC   0x4d4c4c4du   /* "MLLM" */
#define MLM_VERSION 1

#define MLM_ARCH_LLAMA 1          /* dense Llama-family decoder */

/* header flags */
#define MLM_F_QK_NORM   (1u << 0) /* RMSNorm on q and k per head (Qwen3) */
#define MLM_F_TIED      (1u << 1) /* output head is the embedding table */
#define MLM_F_BOS       (1u << 2) /* put BOS before the prompt */
#define MLM_F_CHAT_QWEN (1u << 3) /* Qwen chat template */
#define MLM_F_ROPE_HALF (1u << 4) /* RoPE pairs (i, i+hd/2) (HF); else (2i, 2i+1) */
#define MLM_F_TOK_BPE   (1u << 5) /* byte-level BPE vocabulary (Qwen) */

/* A quantised matrix. Rows are row_stride bytes apart (a multiple of
 * 16). A row is a run of chunks; a chunk is one header word holding
 * four u16 group multipliers (group 0 in bits 15:0) followed by the
 * data words of up to four groups of 32 weights: 4 words a group at
 * 8 bits, 2 at 4 bits. Weight k of a data word is byte k (Q8) or
 * nibble k, low nibble first (Q4). Real weight = q * m * 2^exp. */
typedef struct {
	uint32_t off;        /* byte offset in the image */
	uint32_t rows;
	uint32_t cols;       /* multiple of 32 */
	uint32_t qbits;      /* 4 or 8 */
	uint32_t row_stride; /* bytes */
	int32_t  exp;
	uint32_t groups;     /* cols / 32 */
	uint32_t pad;
} mlm_tensor_t;

/* A norm weight vector: n int16, real = g * 2^exp. */
typedef struct {
	uint32_t off;
	uint32_t n;
	int32_t  exp;
	uint32_t pad;
} mlm_norm_t;

typedef struct {
	mlm_norm_t   attn_norm;
	mlm_tensor_t wq, wk, wv, wo;
	mlm_norm_t   q_norm, k_norm;    /* used with MLM_F_QK_NORM */
	mlm_norm_t   ffn_norm;
	mlm_tensor_t w1, w3, w2;        /* gate, up, down */
} mlm_layer_t;

typedef struct {
	uint32_t magic;
	uint32_t version;
	uint32_t arch;
	uint32_t dim;
	uint32_t hidden_dim;
	uint32_t n_layers;
	uint32_t n_heads;
	uint32_t n_kv_heads;
	uint32_t head_dim;
	uint32_t vocab;
	uint32_t ctx_max;     /* rows in the RoPE table */
	uint32_t flags;
	int32_t  eps_m;       /* norm epsilon = eps_m * 2^eps_e */
	int32_t  eps_e;
	uint32_t attn_scale;  /* 1/sqrt(head_dim), Q30 */
	uint32_t tok_off;     /* tokenizer, see below */
	uint32_t tok_len;
	uint32_t rope_off;    /* ctx_max x head_dim/2 x {cos,sin} int16 Q14 */
	uint32_t layers_off;  /* n_layers x mlm_layer_t */
	uint32_t bos_id;
	uint32_t eos_id;
	uint32_t eos2_id;     /* a second end token (Qwen <|im_end|>), or ~0 */
	uint32_t file_size;
	uint32_t name_off;    /* NUL-terminated model name */
	mlm_norm_t   final_norm;
	mlm_tensor_t emb;     /* vocab x dim, one row per token */
	mlm_tensor_t head;    /* vocab x dim; same offset as emb when tied */
	uint32_t im_start_id, im_end_id; /* Qwen chat specials, or ~0 */
	uint32_t think_id, think_end_id; /* Qwen3 <think> </think>, or ~0 */
	uint32_t reserved[14];
} mlm_header_t;

/* Tokenizer block at tok_off:
 *   u32 n_tokens, u32 max_len,
 *   n_tokens x { i32 score, u32 len, len bytes }   (packed, no padding)
 * Scores rank merges (higher merges first); for BPE vocabularies the
 * converter writes -rank. Bytes are raw UTF-8 / raw bytes. Tokens
 * "<0xNN>" stand for single bytes in SentencePiece vocabularies. */

#define MLM_GROUP 32

/* words in one row of a tensor, as the engine reads it */
static inline uint32_t mlm_row_words(uint32_t groups, uint32_t qbits)
{
	uint32_t wpg = (qbits == 4) ? 2 : 4;
	uint32_t full = groups / 4, rem = groups % 4;
	return full * (1 + 4 * wpg) + (rem ? 1 + rem * wpg : 0);
}

#endif
