# SD card

`rtl/sd/mlm_sd.v` drives the card in **native mode** with a 4-bit data
bus and writes what it reads straight into DDR3. `fw/common/sd.c` has
the protocol and a FAT16/FAT32 reader. Zeitlos, Mach86 and machboot use
the card in SPI mode (kboot loads at about 0.5 MB/s); a 300 MB model
would take ten minutes that way.

## Pins (Sergei ML2)

| Signal | Ball | SPI name in Zeitlos |
|---|---|---|
| CLK | L1 | SD_SCK |
| CMD | L4 | SD_MOSI |
| DAT0 | L3 | SD_MISO |
| DAT1 | T2 | SD_D1 |
| DAT2 | T3 | SD_D2 |
| DAT3 | L2 | SD_SS |

All pulled up in the pin file. DAT3 is never driven (driven low at CMD0 it
would select SPI mode). Mozart ML2's DAT1/DAT2 balls are not in
Zeitlos's pin file yet.

## The host

Two engines, driven by the CPU:

- **CMD**: sends a command with its CRC7; receives a 48-bit (CRC7
  checked) or 136-bit response; optionally waits for busy on DAT0.
- **DAT**: armed before CMD17/CMD18; receives N blocks on DAT0 or
  DAT0-3, checks each line's CRC16, assembles 16-byte blocks and queues
  them for the DDR3 controller's block-write port. The clock stops while
  the queue is full (the card waits).

SCK = 48 MHz / (2 (DIV + 1)): 400 kHz for identification, 24 MHz for
data. The host changes CMD after a falling edge and samples at rising
edges (default-speed mode).

Registers (0x7200_0000, word offsets): 0 DIV, 1 ARG, 2 CMD (start:
index, response type, busy, no-CRC), 3 STATUS, 4-7 RESP, 8 DADDR,
9 DBLOCKS (arm: count, 4-bit), 10 DDONE, 11 IDLE (N clocks with CMD
high).

## The driver

`sd_init()`: 80 idle clocks, CMD0, CMD8, ACMD41 until ready (SDHC when
CCS), CMD2, CMD3, CMD7, ACMD6 (4-bit), CMD16 for SDSC, then 24 MHz.
`sd_read()`: CMD17 or CMD18 + CMD12 per run of up to 65535 blocks.
`fat_mount()` finds the first FAT partition (or a superfloppy),
`fat_list()` walks the root directory (8.3 names), `fat_load()` follows
the cluster chain and reads **contiguous runs** with one multi-block
read each.

## Measured (simulation)

C++ card model (`sim/sd/sdcard_model.h`: SDHC, native mode, CRCs both
ways, R1b busy after the response) with a FAT32 image made by
`mkfs.fat` and `mcopy`:

| | |
|---|---|
| 2.2 MB file through the SD host and the DDR3 block-write port (sim/sd) | 11.67 MB/s, file intact |
| 2.5 MB model loaded by the firmware in the whole-SOC simulation | 10.75 MB/s |

On the board (90 MHz build, 24 MHz SD clock): Bonsai's 246 MB load in
about 21 s, 11.9 MB/s. If reads fail, the firmware retries the piece
and drops the SD clock (to 12 MHz); one 75 MHz build did so.

## Not done yet

- **High Speed** (CMD6 switch, 50 MHz class): needs SCK at the system
  clock (an ODDR clock output) and sampling with less than a cycle of
  margin; about 2x. 
- Card writes (not needed to load models).
- Long file names (use 8.3 names; `tools/mkcard.sh` makes them).
