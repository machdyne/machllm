# The arithmetic

machllm computes in integers only. The specification is code:
`fw/common/mlm_model.c` (everything but matrix-vector products) and
`fw/common/eng_spec.h` (the engine). The firmware and the PC reference
`ref/mlm_run` compile the same files, and every simulation compares the
RTL against `eng_dot_spec()` call by call, so the board, the simulations
and the PC produce **the same tokens, bit for bit**.

## Vectors: block floating point

Every vector is int32 with one exponent for the whole vector:
`real = v x 2^e`. Operations choose the output exponent from the data
(the largest magnitude is kept below 2^30), so there is no fixed-point
range to tune per model and nothing overflows.

## Matrix x vector (the engine)

The input vector is packed to the **MX format** (`to_x()`, the vector
unit's MXPACK): int8 values in blocks of 32, each block with a 4-bit
offset d below the vector's exponent. Weights are **1-bit**: a sign per
weight and a 16-bit multiplier m per group of 128. The engine returns,
per row, the exact sum

```
acc = sum over groups g:  m_g x sum over blocks b in g:  (sum over k: s_k x q_bk) << (15 - d_b)     (int64)
      with s_k = +1 for a set bit, -1 for a clear one
real result = acc x 2^(tensor exp + input exp)
```

The 8-bit mode (the KV cache) is the same with int8 weights and no
multiplier.

It also reports the largest |acc| of the call (MAXABS) and serves each
result shifted, `(acc + 2^(s-1)) >> s` (OUT32), so the firmware picks
`s` from MAXABS and reads one 32-bit word per row.

## Norms

RMSNorm: the input is shifted to 15 bits, squares summed in 64 bits,
`mean x 2^16` divided once, epsilon added in the same units, an integer
square root and one reciprocal; then each element is (input x weight)
to 15 bits times the reciprocal to 15 bits, one 32-bit multiply.

Where a product only feeds an int8 engine input or an int8 cache
entry, both factors are taken to 15 bits so the CPU needs one 32 x 32
multiply instead of a 64-bit one (RMSNorm, RoPE, SiLU, the SwiGLU
product, the attention output). That costs a little accuracy (below)
and saves about a third of the CPU's time.

## Attention

- q and k: per-head RMSNorm when the model has it (Qwen3), then RoPE
  from the int16 Q14 table (the position's row is copied to block RAM
  once per token).
- The k and v of the new position are quantised to int8 per head
  (`kv_quant()`: scale = max/127, kept as a 16-bit mantissa and an
  exponent) and written to the caches.
- Scores: q is multiplied by 1/sqrt(head_dim) (Q30; with QK-norm the
  norm's 1/rms is folded into the same multiply) and packed to MX int8;
  the engine dots it with every cached K row; each score is multiplied
  by its K scale and becomes Q16.
- Softmax: max subtracted, `exp(-x)` from a 257-entry table of
  2^(-i/256) with linear interpolation (`mlm_exp_neg_q16()`).
- The weights times each position's V scale become int8 MX coefficients;
  the engine dots them with the **transposed** V cache (one row per
  dimension), and one reciprocal of the softmax sum scales the result.

## Feed-forward

SwiGLU: `silu(gate) x up`. Sigmoid comes from a table of 1025 values
(|x| < 16 in steps of 1/64, Q16, `mlm_sigtab.h`) with linear
interpolation, mirrored for negative x. Both factors of the product are
taken to 15 bits.

## Output and sampling

The head's rows are compared at full scale (each chunk read through the
result mirror, shifted back; a 32-bit threshold rejects most rows in one
compare) and the best 20 kept. Greedy takes the first. With a
temperature, weights `exp(-(best - logit)/T)` over the 20, top-p cut,
xorshift32 random number.

## Accuracy

Bonsai 1.7B, the PC reference (`ref/mlm_run`, this arithmetic) against
`tools/gguf_ref.py` (a float forward pass of the same GGUF weights):
the same top-5 next tokens in order, logits within about 1%. The
weights are imported exactly (one sign per weight, one scale per 128),
so the difference is the integer arithmetic alone.
