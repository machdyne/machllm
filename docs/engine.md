# The engine

`rtl/mlm_engine.v` computes matrix x vector products for quantised
matrices in main memory. Its specification is `eng_dot_spec()` in
`fw/common/eng_spec.h`; every simulation checks every call against it.

## What it does

Two modes, both against an **MX-packed input** x (up to 3072 int8 values
in blocks of 32, each block with a 4-bit exponent offset d):

- **1-bit** (QBITS 1; the model's weights, Bonsai 1.7B / GGUF Q1_0):
  groups of 128 weights, one 16-bit multiplier m each. For each block
  of 32 in a group, S = sum (bit ? q : -q), and

  ```
  acc += (S x m) << (15 - d)
  ```

  64 weights a cycle: one 64-bit word, two blocks.
- **8-bit** (QBITS 8; the KV cache, for attention): int8 weights in
  groups of 32 (one block each), `acc += (sum w x q) << (15 - d)`
  (the chunk multipliers are ignored: the cache writes 1). 8 weights a
  cycle.

exactly in 64 bits, one result per row into the output memory. The CPU
sets the matrix's address, row stride, rows (up to 4096), groups per
row and the mode, and starts it. It records the run's cycles, the
cycles it waited for data, and the largest |acc|.

The packing (`eng_mx_pack` in eng_spec.h, also the GLU unit's output):
per block s_b = max(0, bitlen(max |v|) - 7), s = max s_b, d = min(15,
s - s_b), q = clamp(rshr(v, s - d), +-127). The firmware's `to_x` does it
with the vector unit (MAXABS and KVQ8 per block).

## CPU window (0x8000_0000 in the SOC)

| Offset | Register | |
|---|---|---|
| 0x0000-0x0bff | X | int8 x[i] at byte i (16 or 32-bit writes; reads back when idle) |
| 0x1000-0x117f | XD | block b's exponent offset at 0x1000 + 4b (bits 3:0) |
| 0x4000 | ADDR | byte offset of row 0 in main memory, 16-byte aligned |
| 0x4004 | STRIDE | bytes between rows, a multiple of 16 |
| 0x4008 | ROWS | 1..4096 |
| 0x400c | GROUPS | 1-bit: groups of 128 (1..24); 8-bit: groups of 32 |
| 0x4010 | QBITS | 1 or 8 |
| 0x4014 | CTRL | write: start; read bit 0: busy |
| 0x4018 | CYCLES | cycles of the last run |
| 0x401c | WAITS | of those, cycles with no data |
| 0x4020, 0x4024 | MAXABS0 | largest \|acc\| written to bank 0 by its last run |
| 0x4028 | SHIFT | for OUT32 |
| 0x402c | OBANK | the output bank the next run writes (0, 1) |
| 0x4030, 0x4034 | MAXABS1 | the same for bank 1 |
| 0x4038 | GLU_N | elements for the GLU unit (1..3072, a multiple of 32) |
| 0x403c | GLU_P | bits 5:0 sg, 13:8 su (bank 0 / bank 1 results to 32 bits), 23:16 sq (signed: the sigmoid's input scale) |
| 0x4040 | GLU_GO | write: start the GLU unit; CTRL reads busy until it is done |
| 0x4044 | GLU_RES | read: psh (bits 7:0) and xsh (15:8), the exponent bookkeeping the CPU needs |
| 0x4048 | CO_GO | copy-out: n (1..2048) rows' OUT32 values into the CPU's RAM from word CO_DST on; CTRL bit 1 busy |
| 0x404c | CO_ROW | bank << 12, first row |
| 0x4050 | PK_GO | the packer: n (bits 12:0) values, blocks of 32 (22:16); CTRL bit 2 busy |
| 0x4054 | CO_DST | RAM word address of the copy-out's first row |
| 0x4058 | PK_SRC | RAM word address of the packer's int32 v[0] |
| 0x4060 | PK_S | read: the packing's s (the input's exponent is s − 15) |
| 0x8000-0xffff | OUT bank 0 | row i's acc at 0x8000 + 8i (low word, high word) |
| 0x18000-0x1ffff | OUT bank 1 | |
| 0x10000-0x13fff | OUT32 bank 0 | row i at 0x10000 + 4i: (acc + 2^(SHIFT-1)) >> SHIFT, low 32 bits |
| 0x14000-0x17fff | OUT32 bank 1 | |

**Two output banks** let the CPU work on one run's results while the
engine computes the next into the other bank. The CPU must not write X
during a run.

**The packer and the copy-out** work on the CPU's RAM through its second
port, on the memory clock, a word a cycle. The packer is
`eng_mx_pack()` (`fw/common/eng_spec.h`) in hardware: pass 1 reads v
for each block of 32's largest |v| (its s_b), pass 2 reads it again and
writes the int8 values four to an X word and each block's offset; a
vector of n values takes about 2n memory cycles. The copy-out writes
OUT32 values straight into the RAM (a run into the other bank may go
on). The CPU starts either and polls CTRL.

Bonsai's FFN (6144 hidden units) runs in two pieces of 3072: gate and
up for a piece into the two banks, the GLU unit packs SiLU(gate) x up
into X, and down runs over that piece's 24 groups of columns (starting
6 chunks into each row); the CPU adds the two pieces' results.

## The GLU unit

For SwiGLU feed-forward layers the engine leaves gate in bank 0 and up
in bank 1. The GLU unit then computes `silu(gate) x up` and MX-packs it
straight into X, ready for the down projection (pass 2 keeps each block
of 32's maximum, pass 3 quantises with the block's shift). The CPU only
writes GLU_N and GLU_P, starts it and reads two shifts back. Its
specification is `glu_spec()` in `fw/common/eng_spec.h` (exactly the
integer code the CPU used to run: the sigmoid table of
[arithmetic.md](arithmetic.md), 15-bit products, three passes: SiLU,
product, quantise). Every simulation checks every call bit for bit:
the shifts and the whole X vector read back.

It is one element at a time, about 25 cycles each, with every step
registered so it doesn't limit the clock (the first, less pipelined
version made the design fail timing at 29 MHz). It shares the OUT RAMs
with the engine and uses one block RAM for the sigmoid table. Since it
pipelines one element a cycle, Bonsai's 2 x 3,072 FFN values a layer
take it about 0.5M cycles a token, under the up projection's stream.

## Datapath

```
stream (128-bit blocks) -> FIFO (32 blocks, LUT RAM) -> head register
  -> word walker: header words carry four multipliers; data words go on
  -> X RAM read: one row of 64 int8 (16 BRAM banks x 64 x 32 bit) and
     the row's two block offsets
1-bit: bit ? q : ~q (64 lanes) -> adds in fours (two stages) -> per
       block sum + count of 0 bits = S -> S x m (two DSPs, registered)
8-bit: the word's 8 weights x their 8 x (eight DSPs) -> sum -> group sum
  -> << (15 - d) per block -> word total -> row accumulator (64 bits)
  -> OUT RAM
```

One 64-bit word per cycle: what the memory delivers at its best. The
multipliers' operands and products are registered in the fabric
(`rtl/mlm_mul18s.v`; the DSP blocks' own registers gave zeros on the
board), so a route into or out of a block can be a long one.

## Requests and credits

Rows are requested in pieces of at most 16 blocks, and only while the
FIFO has room for the whole piece (credits), so the memory never needs a
ready signal on its data. Rows longer than the FIFO are fine; a piece
can start any time. Memory answers requests in order.

## Measured

| Simulation | Words per busy cycle |
|---|---|
| ideal memory, no gaps (sim/engine) | 1.00 |
| ideal memory, 20-30% of slots missing, random latency | 0.88 |
| through the DDR3 controller and DRAM model, refresh included (sim/mem) | 0.95 |

On the board (DDR3 at 180, memory side 90 MHz): the stream test reads
648.5 MB/s with 90% of cycles busy. The first Bonsai bitstreams'
multipliers gave zeros on the board (DSP blocks instantiated with their
own registers; fixed 2026-11-27, see [architecture.md](architecture.md)); the `:e` console
command prints the engine's results against the CPU's for simple inputs.

## Not done yet

- Batched prefill (several input vectors per pass over the weights),
  so a prompt costs less than one weight pass per token.
- A hardware top-k for the output head; the CPU scans the logits
  (one 32-bit compare per row, under the next chunk's stream).
- Attention reads each KV head's cache once per query head (Bonsai: two
  query heads a KV head); both at once would need a second input
  vector, as batched prefill does.
