# Main memory: the DDR3 streaming controller

Sergei ML2 and Mozart ML2 carry one MT41K256M16TW-107:P (DDR3L, x16,
4 Gb = 512 MB). machllm uses Zeitlos's DDR3 PHY, clocking and training
(`ext/zeitlos/rtl/mem/ddr3_phy_ecp5.v`, `ddr3_phy_init.v`; see Zeitlos's
`docs/ddr3.md`), its own read assembly (`rtl/mem/mlm_ddr3_rdasm.v`) and
its own fork of the controller, `rtl/mem/mlm_ddr3_ctrl.v`, wrapped by
`rtl/mem/mlm_ddr3.v`.

**On the board** (`make MEM_MHZ=90`, DDR3 at 180 MT/s): training and
memtest pass, and the engine streams **648 MB/s** with 90% of cycles
busy (the x16 bus peaks at 720 MB/s at this clock).

A token reads every weight once, so this rate sets the floor on the
time per token. Zeitlos's controller serves a CPU cache, one 16-byte
access at a time; the fork adds a stream port that keeps the bus busy.

## The controller

1. **Address map: bank lowest.** A word address is
   `{row[14:0], column block[6:0], bank[2:0], word[1:0]}`, so
   consecutive 16-byte blocks rotate through all eight banks.
2. **Stream port** (`sq_*`: block address, length 1..256; `sd_*`:
   128-bit data out, in order). ACTIVATE and READ-with-auto-precharge
   alternate, a burst back to back; with tRCD of two cycles (75 MHz and
   up) the ACTIVATE for the next block goes out before the READ of this
   one. No open-row tracking: each block's bank was last used eight
   blocks before, longer than tRC. The next request is taken as the last
   READ of the current one issues, so pieces join without a gap, unless
   the CPU is waiting: it gets the memory between two pieces.
3. **Refresh inside a stream**: the stream pauses, the banks close, the
   refresh runs, the stream continues.
4. **Read data routing**: while stream reads are outstanding, every
   returning burst belongs to the stream; CPU accesses wait for them.
5. **Block-write port** (`wq_*`): sixteen bytes written whole, with no
   read-modify-write, for the SD loader.
6. **The CPU's word path**: the controller keeps the last 16-byte line
   read, and the SOC keeps a copy on the CPU's side of the clock crossing
   (SYSCTL+12, `:k`): a sequential copy loop costs ~12 CPU cycles a word
   instead of ~32.

Timings follow `CLK_MHZ` (tRCD, tRP and tRC from hundredths of a
nanosecond). Initialisation, mode registers, read-modify-write for CPU
stores and the training hooks are Zeitlos's. Write combining stays off.

## The DMA unit

`rtl/mlm_dma.v` moves data between main memory and the CPU's RAM on the
memory clock, through the RAM's second port, so the CPU does not copy
word by word across the clock crossing (~12 CPU cycles a word). The
firmware's `plat_mem_rd`, `plat_mem_wblk`, `plat_mem_wcol` and
`plat_dma_wait` (`fw/main.c`) drive it.

| Offset (0x8100_0000) | Register | |
|---|---|---|
| 0x00 | SRC | RD: main-memory offset (16-byte aligned); WBLK, COL: RAM byte address |
| 0x04 | DST | RD: RAM byte address; WBLK, COL: main-memory offset |
| 0x08 | N | RD: words; WBLK: blocks a row; COL: bytes a column |
| 0x0c | STRIDE | COL: bytes between a column's bytes |
| 0x10 | OUTER | WBLK, COL: rows / columns |
| 0x14 | OSTRIDE | WBLK, COL: bytes between rows / columns |
| 0x18 | GO | write 1 RD, 2 WBLK, 3 COL; read bit 0: busy |

- **RD** uses the stream port (at most 16 blocks outstanding, a word a
  cycle into the RAM), so only while the engine is not running; the
  firmware falls back to a CPU copy otherwise. The cache scales, norm
  weights, RoPE rows and layer descriptors come this way.
- **WBLK** writes 16-byte blocks through the block-write port (shared
  with the SD loader: whoever has one waiting keeps the port until it
  is taken). Each position's K row goes this way.
- **COL** stores single bytes through the controller's word port
  (read-modify-write there, as for the CPU; masked writes are not used).
  The transposed V cache gets a byte per dimension per position this
  way, ~30 memory cycles a byte. While it owns the word port the CPU's
  main-memory accesses wait.

A write operation returns at once: the firmware quantises the next KV
head while the DMA unit writes the previous one. The CPU's copy of the
last line read is dropped whenever a DMA operation starts.

## Clocks

The DRAM runs far below its rating (DDR3-1866), as Zeitlos and LiteX run
this board. ECP5 x2 gearing puts the controller at half the DRAM clock.
`rtl/mem/mlm_pll.v` makes the edge clock (128, 150, 180 or 200 MHz) and
the CPU's clock from the 48 MHz oscillator; `rtl/mem/mlm_ddr3_clk.v` is
Zeitlos's `ddr3_clk` with the clock as a parameter.

| Build | DRAM | controller, engine | CPU |
|---|---|---|---|
| `make MEM_MHZ=90` | 180 MT/s | 90 MHz | 54 MHz (540 / 10) |
| `make MEM_MHZ=75` | 150 MT/s | 75 MHz | 50 MHz |
| `make` | 96 MT/s | 48 MHz, one clock domain | |

At 90 MHz only some placement seeds meet timing (with the screen: seed
6, 92.5 MHz); `MEM_MHZ=100` reaches 92-95 MHz and does not. The
controller keeps a registered "free" flag per bank next to each
bank's busy counter (`bfree`, always equal to `bank_cnt == 0`, which
simulation checks every cycle), so the stream's next-command decision
does not wait for a 4-bit compare; that took the 90 MHz build's worst
path from 80-86 MHz to 85-92.

`:gap N` on the console (DDR3 CTRL[5:4]) puts N idle cycles after each
stream READ, a fallback if back-to-back reads ever misbehave on a board.

## Simulation

`sim/mem` runs the engine through the controller and
`sim/mem/mlm_dram_model.v`, a command-level DRAM model that fails the
run on any bank-state or timing violation (tRCD, tRP, tRC, tRAS, tRRD,
tFAW, tCCD, tWTR, tRFC, refresh with a bank open, the refresh interval)
with the MT41K256M16TW-107's timings. `make -C sim/mem test-fast` runs
it at 64, 75 and 100 MHz (0.95, 0.86 and 0.86 words a cycle).
