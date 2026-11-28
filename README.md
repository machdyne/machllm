# machllm

**A language model computer on an FPGA.** machllm is a system-on-chip
for Machdyne FPGA computers that does one thing: run a language model,
entirely in our own open gateware and firmware, offline.

![machllm architecture](docs/img/architecture.svg)

- **Board:** Sergei ML2 (LFE5U-45F, 512 MB DDR3L); Mozart ML2, the
  same chip and memory, is planned.
- **Model:** [Bonsai 1.7B](https://huggingface.co/prism-ml/Bonsai-1.7B-gguf)
  by PrismML: 1-bit weights in the Qwen3-1.7B layout, converted from its
  GGUF (`Q1_0`) to a 246 MB `.mlm` image. The engine is built for 1-bit
  weights; 4- and 8-bit weight models are no longer supported (8-bit
  stays for the attention cache).
- **Interface:** a screen (640x480 over the HDMI connector, DVI) and a
  USB keyboard: type a question, read the answer. The serial console
  (1,000,000 8N1) works alongside. `make UI=0` leaves the screen and
  keyboard out.

## How it works

A generated token reads every weight once: about 242 MB for Bonsai.
The **engine** streams them from DDR3 and multiplies 64 one-bit weights
per cycle against an int8 input in blocks of 32 with shared exponents
(MX format), keeping exact 64-bit sums. A **GLU unit** next to it turns
the FFN's gate and up outputs into the next input without the CPU.

The **CPU** (Zeitlos's zeitlos32ps, RV32IM) runs the model in integer
arithmetic: the same C code as the PC reference (`ref/mlm_run`). A
**vector unit** beside it does the elementwise work (norms, RoPE,
residuals, input packing, KV-cache quantisation). The memory side runs
at 90 MHz (DDR3-180), the CPU at 49 MHz; bulk data crosses between them
in dual-clock RAMs.

## Status

Measured on a Sergei ML2:

| | |
|---|---|
| DDR3 | trained at 180 MT/s; the engine streams 648 MB/s |
| loading Bonsai from SD | 246 MB in ~21 s (4-bit bus, 24 MHz) |
| engine self-test, `:e` | the engine's results equal the CPU's |
| generating (before the data-path work below) | coherent answers, **1.24 tokens/s** over a 256-token answer (0.83 s a token at position ~290, 45% of it with the engine idle) |

The firmware and the PC reference compute the same integer arithmetic,
and every simulation (`docs/testing.md`) is bit-exact against it.

The current design moves the bulk data on the memory clock (the RAM's
second port: the engine's packer and copy-out, a DMA unit), streams the
vector unit's operations and overlaps attention's heads with the
engine. In simulation (one Bonsai layer, position 337) that cuts a
token's CPU cycles by 30% (attention by 57%), the engine's own time
unchanged; on the board that should come to roughly 1.8 tokens/s, to
be measured.

Known limits:

- Context of 512 positions per conversation (`:reset` starts over);
  answers stop at the model's end token or after `:n` tokens (256).
- Attention grows with the context (it reads the whole KV cache, once
  per query head).
- Prompts cost one weight pass per token (no batched prefill yet).
- The firmware does not yet refuse a model image with 4- or 8-bit
  weights (an old one loads and produces nonsense).
- Builds for Sergei ML2 only; Mozart ML2 pins are not done. At 90 MHz
  only some placement seeds meet timing (the Makefile's default does).

## Quick start

```
git clone --recursive <this repository>
make ref                                        # the PC reference engine
python3 tools/mlm_convert.py gguf Bonsai-1.7B-Q1_0.gguf build/models/bonsai.mlm
./ref/mlm_run build/models/bonsai.mlm -i "The capital of France is" -n 16 -t 0
tools/mkcard.sh /media/you/CARD build/models/bonsai.mlm     # FAT32 -> BONSAI.MLM
openFPGALoader -c dirtyJtag build/sergei_ml2_mem90/machllm.bit
```

The GGUF is the file Ollama downloads for
`hf.co/prism-ml/Bonsai-1.7B-gguf:Q1_0`. Bitstreams come in the release
zips; building one is in [docs/usage.md](docs/usage.md) (synthesis and
place and route take about 30 minutes; a new firmware goes into a
placed design in seconds with `make fwbit`).

## Documents

| | |
|---|---|
| [docs/usage.md](docs/usage.md) | build, flash, the card, the console |
| [docs/models.md](docs/models.md) | converting and checking models |
| [docs/architecture.md](docs/architecture.md) | the SOC, memory map, clocks, resources |
| [docs/format.md](docs/format.md) | the `.mlm` image |
| [docs/arithmetic.md](docs/arithmetic.md) | the integer arithmetic (the specification) |
| [docs/engine.md](docs/engine.md) | the matrix-vector engine and GLU unit |
| [docs/vu.md](docs/vu.md) | the vector unit |
| [docs/memory.md](docs/memory.md) | the DDR3 streaming controller |
| [docs/sd.md](docs/sd.md) | the native SD host and FAT loader |
| [docs/testing.md](docs/testing.md) | the simulations and what each proves |

## Licence

Same as Zeitlos: see [LICENSE.md](LICENSE.md). Zeitlos is a pinned
submodule (`ext/zeitlos`); files forked from it say so in their header.
Model weights keep their own licences (Bonsai 1.7B: Apache-2.0).
