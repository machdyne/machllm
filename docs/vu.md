# Vector unit

`rtl/mlm_vu.v` runs the model's per-element loops (the work between the
engine's matrix products) on the CPU's own block RAM.

## How it works

The CPU writes the operands (addresses of arrays in its RAM, a length,
two parameters) to the unit's registers at `0x7300_0000`, then writes
the operation number. That last write is not acknowledged until the
operation has finished, so the CPU is stalled: it can't even fetch an
instruction. The RAM's single port belongs to the unit in the
meantime. That needs no arbitration, no second RAM port and no clock
crossing, and the arrays stay where the C code already keeps them.

| Offset | Register | |
|---|---|---|
| 0x00, 0x04, 0x08 | A, B, D | byte addresses in the RAM (word aligned) |
| 0x0c | N | elements |
| 0x10, 0x14 | P0, P1 | parameters (signed) |
| 0x18 | OP | write: run (returns when done) |
| 0x1c | R0 | max \|.\| (MAXABS, RESID) |
| 0x20, 0x24 | SUMLO, SUMHI | sum of squares (SQMUL) |
| 0x28 | ID | `VU01` |

## Operations

The specification is the `v_*` functions in `fw/common/mlm_model.c`:
when `mlm_vu` is set they call the unit, otherwise they run their own
loop. The unit must give exactly the same result.

| Op | | Used by |
|---|---|---|
| 1 MAXABS | max \|A[i]\|, one element a cycle | every normalisation and quantisation |
| 2 SQMUL | u = A[i] >> P0, D[i] = u g[i], sum of u² | RMSNorm |
| 3 SCALE15 | D[i] = (A[i] >> 15) P0 | RMSNorm |
| 4 RSHR | D[i] = A[i] >> P0 (left for P0 < 0) | residual renormalisation |
| 5 RESID | A[i] = (A[i] >> P0) + (B[i] >> P1), max | residual add |
| 6 MULSHR30 | D[i] = (A[i] P0) >> 30, 64-bit product | attention's q scale |
| 8 KVQ8 | int8 for the KV cache, four to a word | KV-cache stores |
| 9 ROPE | rotate pairs by a table of {sin, cos} | RoPE (both layouts) |
| 10 ATTSC | raw scores times each position's K scale, max | attention scores |
| 15 EXPCOEF | EXPSUM and COEF in one pass: exp(score − max), their sum, times each V scale, max | softmax, attention × V |

Numbers 7 and 11-14 (QPACK, EXPSUM, COEF, QPACKU, MXPACK) are gone:
EXPCOEF does EXPSUM and COEF in one pass, the engine's packer does
MXPACK, and the old int16 input format is no longer used.

`>>` is the model's rounded shift (`rshr32`). Products wrap to 32 bits,
except MULSHR30's and KVQ8's, as in C.

The unit works on the CPU's RAM while the CPU waits. Most operations
are **streamed**: reads go out at fixed phases of a short period and
each result is written a fixed number of cycles after its reads, so
the RAM's single port is busy every cycle:

| | Period | Per element |
|---|---|---|
| SCALE15, RSHR, MULSHR30 (S_FS) | 2 cycles: A[j], a write | 2.1 |
| SQMUL (S_FS) | 3: A[j], B[j/2], a write | 3.1 |
| ATTSC, EXPCOEF, RESID (S_AS) | 3: A[j], B[j], the write 14 cycles after A[j] | 3.1 |
| ROPE (S_P5) | 5 per pair: A[ia], A[ib], the rt word, two writes | 2.6 |
| KVQ8 (S_P5) | 5 per four: four reads, the word | 1.3 |

MAXABS reads an element a cycle. `sim/vu` prints each operation's
cycles an element; the CPU's own loops took 20-30 (profile: `make -C fw ... CFLAGS_EXTRA=-DVU_PROF`, then
`:v`).

## Testing

- `sim/vu`: the unit with a C++ RAM, 3000 random operations of the ten
  kinds and of every size against `mlm_vu_ref()` (the C loops). A planted fault
  makes about 1 in 9 of them fail.
- `sim/soc`, one clock and two: the whole firmware with the unit on
  must still generate exactly what `mlm_run` does.
- On the board: `vector unit self-test: 150 operations, 0 wrong` after
  each model load (the unit against the C loops on random data, within
  each operation's range). `:u` switches the unit off and on.

## Next

- The QK-norms of all heads in fewer calls (each call's set-up and the
  CPU's square root are most of a head's cost now).
