# Testing

`make test` runs the first three; `sim/soc` takes longer.

| | What runs | Checks |
|---|---|---|
| `ref/` | `mlm_run`: the model code + `eng_dot_spec()` on the PC | against llama2.c `run.c` (`ref/check/l2c_logits.c`) and a float Qwen3/Llama (`ref/check/hf_float.py`); `ref/check/cmp_logits.py` compares dumps |
| `sim/engine` | `rtl/mlm_engine.v` + C++ memory with random latency and missing slots, and a model of the CPU's RAM on the engine's second-port pins; the whole model code runs with the RTL as its engine | every call against `eng_dot_spec()` (results, MAXABS, OUT32; 1-bit weights and the 8-bit cache), every GLU run against `glu_spec()`, every packing (the packer) against `eng_mx_pack()` and every copy-out against the rows; the token ids equal `mlm_run -q`. Models (`tools/test_models.sh`, all 1-bit): stories15M shape, Qwen3-tiny, Bonsai-tiny (Bonsai's FFN width, two pieces) |
| `sim/vu` | `rtl/mlm_vu.v` + RAM | 3,000 random runs of the 10 operations against `mlm_vu_ref()` (bit-exact), and each one's cycles an element |
| `sim/ui` | `rtl/mlm_ui.v`: cells written through the CPU port, a whole frame run | the TMDS symbols decoded back to pixels (`out.ppm`), sync widths, the visible pixel count, read-back of cells and registers |
| `sim/rdasm` | the DDR3 read assembly (`rtl/mem/mlm_ddr3_rdasm.v`) | against Zeitlos's, cycle by cycle |
| `sim/mem` | engine + `rtl/mem/mlm_ddr3_ctrl.v` + `sim/mem/mlm_dram_model.v` (DRAM protocol and timing checker, storage in C++) | CPU path: 600 stores (words, halves, bytes through read-modify-write) and 1024 loads under the new address map; the whole model through the stream port; token ids equal `mlm_run`; again with stream gap 1 |
| `sim/sd` | `rtl/sd/mlm_sd.v` + controller block writes + DRAM model; C++ SD card with a FAT32 image | `fw/common/sd.c` initialises the card (4-bit), mounts FAT32, lists, loads a 2.2 MB file; it must arrive intact |
| `sim/soc` | `rtl/mlm_soc.v` (with the DMA unit and the RAM's second port) and zeitlos32ps running the real firmware (built with `-DSIM`: no DDR3 training, fast UART), DRAM model, SD card model, UART decoded in C++ | boots, tests (engine and GLU self-tests), loads TINY.MLM from the card, takes a typed prompt; the printed text must equal `mlm_run`'s. `TB_CMD2=:g` or `:o` runs the same with SwiGLU on the CPU or without overlap: the text must still be equal |

In simulation the SOC also stops if the engine and the DMA unit ever
want the RAM's second port or the memory stream at the same time (the
firmware never lets them).

`make -C sim/soc test-2c` is the same with two clock domains (CPU 50 MHz,
memory side 75 MHz, as `make MEM_MHZ=75`); `make -C sim/soc test-board`
with the boards' clocks (CPU 54 MHz, memory side 90 MHz); `make -C
sim/mem test-fast` runs the controller and DRAM model at 64, 75 and 100
MHz. `TB_SD_DIV=0x81 make -C sim/sd test` loads through the SD host with
a three-cycle SD clock period (the boards' 18 MHz).

Type-ahead: `TB_TYPE_GAP=N` makes the testbench type the question with N
cycles between characters (`~` in it is a backspace), so the firmware
reads it ahead while it is typed; `TB_TYPE_PAUSE=N` adds a pause of N
cycles before Enter and at each `|` in the question (not typed), where
the firmware reads the rest of the line and the template's end ahead.
With `TB_CMD=":temp 0"` the answer must be byte-identical to the one
typed at full speed:

```
cd sim/soc && export TB_CMD=":temp 0"
obj_2c/Vmlm_soc @../../build/models/b1l.mlm "Tell me about dogs and cats please." 3000000000 12
TB_TYPE_GAP=1500000 obj_2c/Vmlm_soc @../../build/models/b1l.mlm "Tell me about birds and~~~~~~~~~dogs and cats please." 3000000000 12
TB_TYPE_GAP=1500000 TB_TYPE_PAUSE=40000000 obj_2c/Vmlm_soc @../../build/models/b1l.mlm "Tell me about birds|~~~~~dogs and c|ats please." 5000000000 12
```

(b1l: 54 prompt tokens in 1.00 s after Enter typed at once, 0.40 s typed
with gaps, 0.02 s with the pauses; the same 12 tokens each time.) `:v`
also prints the stack's high-water mark.

The stack is what the 96 KB block RAM leaves after code and variables:
about 2.4 KB in the board firmware (profiler off, the console's code
compiled for size), more in the simulation builds. Bonsai with typing
during read-ahead peaks at 984 bytes in `make -C sim/soc test-board`'s
firmware (`TB_CMD2=":v"`) and 1,060 on the board. If it ever comes within 256 bytes of the
variables, the prompt says so.

`sim/soc` also takes `@model.mlm` instead of a card image: the model is
put in memory directly (no card). The whole Bonsai model runs that way
in about 15 minutes:

```
cd sim/soc && TB_CMD=":v" obj_2c/Vmlm_soc @../../build/models/bonsai.mlm "Hi" 3000000000 3
```

`obj_ui/Vmlm_soc` (`make -C sim/soc obj_ui/Vmlm_soc`) is the SOC with
the screen: at the end the cells go to `ui_screen.txt`. `tools/ui_host.c`
runs the screen firmware (`fw/ui.c`) on the PC against an array, which
is quicker for its layout and wrapping.

`build/b1l_card.img` (one Bonsai-shaped layer, `tools/mlm_synth.py
hf bonsai-1l`) is the quick per-layer profile.

## Not simulated

The DDR3 PHY (ECP5 DQS primitives), and therefore read-path training
and back-to-back reads through the real PHY; the SD pins' timing; the
UART pins. And **the vendor primitives' configuration**: the DSP
multipliers simulate as a behavioural model (`ifndef SYNTHESIS`), so
how nextpnr configures the real MULT18X18D is not checked. That is how
the first Bonsai bitstreams computed zeros on the board while every
simulation passed (MULT18X18D instantiated with its own registers; see
[architecture.md](architecture.md)). The multipliers are now plain
Verilog, so the simulated model and what Yosys maps are the same
description. The firmware's `:e` command and
the engine self-test at load are the board-side checks.

## Synthetic models

```
python3 tools/mlm_synth.py llama2c tiny|stories15M|stories42M|stories110M out.bin
python3 tools/mlm_synth.py hf qwen3-tiny|qwen3-0.6b|smollm2-135m|bonsai-tiny|bonsai-1l outdir
```

Random weights with the real shapes; the hf form also writes a
byte-level BPE `tokenizer.json` with random merges. They exercise every
path but say nothing about quality.
