# The .mlm model image

One file holds everything the board needs: the architecture, the
quantised weights laid out in the order the engine reads them, norm
weights, the RoPE table and the tokenizer. The firmware copies it to
main memory at offset 0 and uses it in place. The C definition is
`fw/common/mlm.h`; `tools/mlm_convert.py` writes it.

All fields are 32-bit little-endian. Offsets are from the start of the
file (= from the start of main memory).

## Layout

| Offset | Contents |
|---|---|
| 0 | header (`mlm_header_t`, 248 bytes; the first 4 KB are reserved for it) |
| | model name (NUL-terminated) |
| | tokenizer block |
| | RoPE table: `ctx_max` x `head_dim/2` x {cos, sin}, int16 Q14 |
| | final norm, embedding/head tensor, per-layer norms and tensors |
| | layer table: `n_layers` x `mlm_layer_t` (288 bytes each) |

Tensors start on 64-byte boundaries, everything else on 16.

## Header

`magic` "MLLM", `version` 1, `arch` 1 (dense Llama-family decoder),
`dim`, `hidden_dim`, `n_layers`, `n_heads`, `n_kv_heads`, `head_dim`,
`vocab`, `ctx_max` (rows in the RoPE table), `flags`, `eps_m`/`eps_e`
(norm epsilon = eps_m x 2^eps_e), `attn_scale` (1/sqrt(head_dim), Q30),
`tok_off`/`tok_len`, `rope_off`, `layers_off`, `bos_id`, `eos_id`,
`eos2_id`, `file_size`, `name_off`, `final_norm`, `emb`, `head`,
`im_start_id`, `im_end_id`, `think_id`, `think_end_id`.

Flags:

| Bit | Name | Meaning |
|---|---|---|
| 0 | `QK_NORM` | RMSNorm on q and k, per head, before RoPE (Qwen3) |
| 1 | `TIED` | the output head is the embedding table (informational: `head` then has the same offset as `emb`) |
| 2 | `BOS` | put BOS before a prompt (Llama, llama2.c) |
| 3 | `CHAT_QWEN` | use the Qwen chat template (`<|im_start|>` ... ) |
| 4 | `ROPE_HALF` | RoPE pairs dimension i with i + head_dim/2 (Hugging Face); otherwise 2i with 2i+1 (llama2.c) |
| 5 | `TOK_BPE` | byte-level BPE vocabulary (no SentencePiece dummy prefix or `<0xNN>` byte tokens) |

## Quantised tensors (1 bit)

The engine runs **1-bit weights** (Bonsai 1.7B, GGUF `Q1_0`); 4- and
8-bit model weights are no longer supported (8-bit rows remain for the
KV cache, which the firmware writes; see [engine.md](engine.md)).

A tensor descriptor (`mlm_tensor_t`) gives `off`, `rows`, `cols`
(a multiple of 128; the converter pads), `qbits` (1), `row_stride`
(bytes, a multiple of 16), `exp` and `groups` (= cols/128).

Weights come in **groups of 128** along a row. Each group has an
unsigned 16-bit multiplier `m`; the tensor has one exponent `exp`:

```
real weight = (bit ? +m : -m) * 2^exp
```

A row is a run of **chunks**. A chunk is one 64-bit header word holding
four multipliers (group 0 in bits 15:0) followed by the bits of up to
four groups, 16 bytes each: weight k of a group is bit k & 7 of byte
k >> 3, least significant bit first (the GGUF Q1_0 order). A row of G
groups is

```
bytes = (G/4) x 72 + (G%4 ? 8 + (G%4) x 16 : 0)
```

padded to the row stride: 1.125 bits a weight. A GGUF Q1_0 block (an
FP16 scale and 128 sign bits) maps to one group exactly; the converter
turns the FP16 scales into the 16-bit multipliers and one exponent per
tensor (a negative scale flips its group's bits). Float checkpoints
(llama2.c, Hugging Face) are reduced to 1 bit (sign, mean |w| per
group) -- only useful for tests.

RoPE tables include YaRN scaling when the GGUF asks for it (Bonsai:
factor 4 over 8192 positions), exactly as llama.cpp computes it
(frequencies ramped between interpolated and original, magnitude
1 + 0.1 ln 4 folded into the Q14 table).

## Norm vectors

`mlm_norm_t`: `off`, `n`, `exp`: n int16 values, real = g x 2^exp.

## Tokenizer block

```
u32 n_tokens, u32 max_len, u32 entries, u32 offsets, u32 sorted     (offsets relative to the block)
entries:  per token  i32 score, u32 len, len bytes   (packed)
offsets:  n_tokens x u32   offset of each token's entry
sorted:   n_tokens x u32   token ids sorted by their bytes (for lookup)
```

Scores order merges (higher first). SentencePiece vocabularies keep
their scores (x1000, as integers); for byte-level BPE the converter
writes `-rank` of the merge that makes the token. Special tokens are
left out of the sorted table so text can never become them.

The encoder is llama2.c's: characters (or bytes, for BPE) to tokens,
then repeatedly merge the adjacent pair whose concatenation has the best
score. For SentencePiece models that is exact. **For BPE (Qwen) it is an
approximation:** real BPE merges only listed pairs and splits the text
with a regular expression first; differences show up as different but
valid token sequences for unusual text. Decoding is exact for both.

## Runtime memory beyond the image

`mlm_init()` puts the caches after the image, from the next 64 KB
boundary:

| Region | Size | Layout |
|---|---|---|
| K cache | layers x kv_heads x ctx x k_stride | one Q8 row per position (head_dim padded to 32) |
| V cache | layers x kv_heads x head_dim x v_stride | **transposed**: one Q8 row per dimension, one byte per position |
| K scales, V scales | layers x kv_heads x ctx x 4 | per position and head: 16-bit mantissa, 16-bit exponent |

Bonsai 1.7B with 512 positions: image 246 MB + caches ~33 MB.
