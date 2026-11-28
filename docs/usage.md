# Using machllm

## Build

Tools: yosys, nextpnr-ecp5 and ecppack (oss-cad-suite, or `pip install
yowasp-yosys yowasp-nextpnr-ecp5`), `riscv64-unknown-elf-gcc` with the
rv32im multilib (Debian/Ubuntu: `gcc-riscv64-unknown-elf`), Python 3
with numpy; Verilator for the simulations.

```
git submodule update --init          # ext/zeitlos (pinned)
make                                 # firmware + build/sergei_ml2/machllm.bit
make YOSYS=yowasp-yosys NEXTPNR=yowasp-nextpnr-ecp5 ECPPACK=yowasp-ecppack
make SEED=2                          # another placement
make MEM_MHZ=75                      # DDR3 at 150, CPU at 50 -> build/sergei_ml2_mem75/
make MEM_MHZ=90                      # DDR3 at 180, CPU at 54    -> build/sergei_ml2_mem90/ (see below)
make MEM_MHZ=90 fwbit                # only the firmware changed: into the placed design, seconds
make BOARD=mozart_ml2 MEM_MHZ=90     # Mozart ML2 -> build/mozart_ml2_mem90/
```

**Mozart ML2** carries the same ML2 module, so the same pins, but its
microSD slot wires only a 1-bit bus (no DAT1/DAT2): the firmware stays
on one data line and loading Bonsai takes about four times as long
(~1.5 minutes instead of ~21 s); the console says `1-bit`. Everything
else is as on Sergei ML2. Its placement has not been tried on a board
yet.

The block RAM is synthesised with random contents (`fw_rand.hex`) and
the firmware put in when packing (ecpbram), so `fwbit` needs only the
placed `soc.config` and `fw_rand.hex` of the build directory (both in
the release zips). With yowasp: `ECPBRAM="python3 tools/ecpbram.py"
ECPPACK=yowasp-ecppack` (yowasp's own ecpbram lacks its database).
`tools/timing_iter.sh 90 1 2` places seeds side by side (`soc.config`
is then the chosen `sN.config`). At 90 MHz only some seeds meet timing;
the Makefile's defaults are the ones that did last: seed 3 and
`--placer-heap-timingweight 30 --placer-heap-critexp 4` (memory side
90.1 MHz, CPU 56.7). The margin is small and any change to the design
can move it: try the seeds again (6 passed before). Another nextpnr
version can place differently, so check `pnr.log`'s clocks.

Timing is checked for speed grade 6; the log is `build/sergei_ml2/pnr.log`
("Max frequency for clock '$glbnet$clk'" must be at least 48 MHz).

## Load the bitstream

As any Sergei ML2 bitstream, e.g. into SRAM for a test:

```
openFPGALoader -c dirtyJtag build/sergei_ml2/machllm.bit
```

(`-f` writes it to flash; with machboot, install it as another system.)

## Bonsai 1.7B

The model the system is built for (1-bit weights; see
[models.md](models.md)). Convert its GGUF (the one Ollama downloads for
`hf.co/prism-ml/Bonsai-1.7B-gguf:Q1_0`; in `~/.ollama/models/blobs`, the
~248 MB file) once, on a PC:

```
python3 tools/mlm_convert.py gguf Bonsai-1.7B-Q1_0.gguf build/models/bonsai.mlm   # 230 MB, a few minutes
ref/mlm_run build/models/bonsai.mlm -i "The capital of France is" -n 16          # the same arithmetic on the PC
tools/mkcard.sh /media/you/CARD build/models/bonsai.mlm                            # -> BONSAI.MLM
```

Use the `make MEM_MHZ=90` bitstream (DDR3 at 180). The firmware samples
as Bonsai recommends (temperature 0.5, top-p 0.85, top-k 20; `:temp 0`
for greedy). `:think` (or F2 on the keyboard) turns on Qwen3's thinking
mode: the model reasons step by step, shown dimmed, before it answers,
for at most 256 tokens (`:think N`), then `</think>` is put in for it.
Bonsai 1.7B closes an empty reasoning block at once, so the firmware
starts it with "Okay, let me think about this carefully." Reasoning
catches some mistakes, not all, and costs minutes per answer
(`ref/mlm_run -think` tries it on the PC). `tools/gguf_ref.py model.gguf "ids"` is a float forward pass
of the GGUF, to check the integer path against. `:v` prints the time per
token and the stack's high-water mark (the board firmware leaves out the
per-section split for the stack's sake: `make MEM_MHZ=90 PROFILE=1 fw
fwbit` builds one with it); `:e` runs the engine on simple inputs and prints its
results against the CPU's (the board-side check of the multipliers);
`:k` turns the CPU's main-memory read line off and on.
After a restart, the first lines say whether the FPGA was reloaded or
only the CPU restarted, and where generation stopped (a black box in
the last 32 bytes of the RAM).

### Checking a new bitstream

1. At power-up: DDR3 training (`ddr3 tune ... L0:C L1:C`), `memtest
   ok`, the stream read (~650 MB/s at 90 MHz, ~540 at 75).
2. After the model loads: `engine self-test: 256 rows, 0 wrong`, `glu
   self-test: ... 0 wrong`, `vector unit self-test: 150 operations, 0
   wrong`.
3. `:e`: no `DIFF` lines.
4. `:temp 0` and a short question; `:v` for the time per token.
5. If something misbehaves: `:k`, `:u`, `:g`, `:o`, `:gap 1`, one at a
   time, the same prompt each time; each turns one hardware aid off.

## The card

FAT32 (or FAT16), models in the **root directory** with **8.3 names**
ending in `.MLM`. The firmware lists them at power-up and loads the
first.

```
tools/mkcard.sh /media/you/CARD build/models/stories42M.mlm   # -> STORIES4.MLM
tools/mkcard.sh -i card.img 512 a.mlm b.mlm                   # an image to dd
```

## The screen and keyboard

With the screen and keyboard built in (the default; `UI=0` leaves them
out), a monitor on the HDMI connector shows 80 x 25 characters
(640x480, DVI) and a USB keyboard goes in the first USB host port
(a plain keyboard, no hub).

- Type a question and press **Enter**. Long questions wrap; the input
  grows to six lines (up to 511 characters). While you type, the model
  already reads each finished word, and when you pause (0.4 s) the rest
  of the line and the chat template's end too, all but the last token:
  if Enter follows, the answer starts at once. Echo stays immediate: the
  keyboard is served between the model's layers. Typing on or editing
  takes back what was read ahead, exactly; the answer is the same as
  without type-ahead.
- **Esc** clears what you typed; while your question is read or the
  answer is written, it stops it (a single short press: the keyboard is
  checked between the model's layers, every few tens of ms).
- **F1** starts a new conversation.
- **F2** turns thinking on or off (the status line says which): the
  model then reasons step by step before it answers, shown dimmed, for
  up to 256 tokens (a few minutes). It catches some mistakes, not all.
- **F10** asks "Exit to machboot?": **Enter** returns to machboot (the
  FPGA reloads from flash address 0), **Esc** stays.
- The bottom line shows what is happening: "Reading your question"
  with a bar, then "Writing...".

The keyboard layout is US. The screen shows ASCII: the model's
typographic quotes and dashes appear as plain ones. The serial console
keeps working, and what is typed there also shows on the screen.

## The console

UART0 at 1,000,000 baud 8N1 (machboot's USB serial bridge). At power-up:

```
machllm <build> (zeitlos32ps, Sergei ML2)
ddr3 dd300083
ddr3 tune 08800334 L0:C L1:C          read-path training, as Zeitlos's BIOS
memtest ok
stream read: 32640 KB in ... = ... MB/s at 48 MHz (..% of cycles busy)
cpu word loads: 256 KB in ... = ... MB/s
sd card: ok, SDHC, 4-bit, 24 MHz
 0: STORY42.MLM 42 MB
loading STORY42.MLM (43210 KB) ... done in 3.70 s, 11.40 MB/s
stories42M.bin: dim 512, layers 8, heads 8/8, vocab 32000, context 512
> Once upon a time
```

A stories model continues the prompt (each prompt starts over). A Qwen
model chats: each line is a user turn in Qwen3's non-thinking template,
and the conversation stays in the KV cache until the context is full.
After each answer: prompt tokens and time, generated tokens and tok/s,
how busy the engine was, and the position in the context. Any key stops
a generation.

| Command | |
|---|---|
| `:models` | list the card's models |
| `:load N` | load model N |
| `:ctx N` | context length (default 512; up to the image's ctx_max and 1024) |
| `:temp T` | temperature x100 (default 50, Bonsai's recommendation, with top-p 0.85 and top-k 20; 0 = greedy) |
| `:think [N]` | thinking on/off; with N: on, at most N tokens of reasoning (default 256) |
| `:n N` | most tokens per answer (default 256) |
| `:reset` | forget the conversation |
| `:quit` | back to machboot: PROGRAMN (pin M8) pulled low, the FPGA reloads from flash address 0 |
| `:bw` | bandwidth test again |
| `:gap N` | DDR3 stream fallback: N idle cycles per burst (see below) |
| `:v` | per token: cycles, engine cycles, the stack's peak use (and the time per section of the model in a `PROFILE=1` firmware) |
| `:a` | type-ahead on/off (reading the question while it is typed; on by default) |
| `:s` | self-tests: the engine against the CPU's own computation on the loaded model's rows, and the GLU unit against the CPU |
| `:o` | CPU/engine overlap on/off: off makes every engine run finish before the CPU goes on (debugging, measurements) |
| `:g` | SwiGLU in the engine's GLU unit (default) or on the CPU (debugging) |
| `:x` | bus timing test (OUT32 reads, block RAM loops, with the engine idle and running) |

## When something is wrong

| Console | Meaning |
|---|---|
| `ddr3 ... init timeout`, `training FAILED` | as Zeitlos (its docs/ddr3.md, Troubleshooting) |
| `memtest FAILED` | the CPU path to DDR3 under machllm's address map |
| stream read far below 366 MB/s, or wrong answers from a model that works in `ref/mlm_run` | back-to-back reads through the PHY: try `:gap 1` (253 MB/s in simulation), then `:bw` and the prompt again. Please report which. |
| `engine self-test: ... WRONG` | the engine's datapath or its memory stream: `:gap 1` and `:s` again; send the log |
| `glu self-test: ... WRONG` | the GLU unit: `:g` moves SwiGLU to the CPU (slower, same results); send the log |
| text differs from `mlm_run` but the self-tests pass | `:o` (no overlap), then `:g`, then `:gap 1`, the prompt again after each; please report which helped |
| `sd card: error N` | N: 10-18 initialisation step, 20-32 a read (30 timeout, 31 CRC) |
| `no FAT file system` | the first partition is not FAT16/32 |
| `not a usable model image` | wrong file, or a format version mismatch: reconvert |
