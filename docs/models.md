# Models

The system is built for **1-bit weights**: Bonsai 1.7B by PrismML
(Apache-2.0), a Qwen3-1.7B-layout model trained for 1-bit weights and
published as GGUF `Q1_0`. 4- and 8-bit weight models were dropped on
2026-11-26 for it (the engine's 8-bit mode remains, for the KV cache).

## Bonsai 1.7B

```
python3 tools/mlm_convert.py gguf Bonsai-1.7B-Q1_0.gguf build/models/bonsai.mlm   # 230 MB, a few minutes
./ref/mlm_run build/models/bonsai.mlm -i "The capital of France is" -n 16 -t 0
python3 tools/gguf_ref.py Bonsai-1.7B-Q1_0.gguf "ids"      # float forward pass of the GGUF
tools/mkcard.sh /media/you/CARD build/models/bonsai.mlm
```

The GGUF is the ~248 MB file Ollama downloads for
`hf.co/prism-ml/Bonsai-1.7B-gguf:Q1_0` (in `~/.ollama/models/blobs`).
The import is exact: GGUF `Q1_0` keeps one FP16 scale per 128 weights
and a sign bit per weight, and the `.mlm` format stores the same signs
with the scale as a 16-bit multiplier and one exponent per tensor
([format.md](format.md)). YaRN RoPE (factor 4 over 8,192 positions) is
folded into the RoPE table as llama.cpp does.

`mlm_run` produces **exactly** what the board computes (same code, same
engine specification), at a few tokens a second on a PC. If a model is
wrong there, it will be wrong on the board; if it is right there and
wrong on the board, it is a hardware problem.

| | |
|---|---|
| layout | dim 2048, FFN 6144, 28 layers, 16 query / 8 KV heads of 128, QK-norm, tied head, vocabulary 151,669 |
| vocabulary in the image | 101,064 of 151,669 tokens (below) |
| image | 230 MB (weights ~229 MB read per token) + caches for 512 positions |
| check | same top-5 next tokens as the float pass of the GGUF, logits within ~1% |
| sampling (firmware default) | temperature 0.5, top-k 20, top-p 0.85, as PrismML recommends; `:temp 0` for greedy |

## The vocabulary, cut to what the screen shows

By default the converter keeps only the tokens machllm can show:
every special token, the 256 single bytes, every token whose text is
ASCII, Latin (to U+024F), punctuation, currency, arrows or emoji, and
every token a BPE merge needs on the way to one of those (so any such
text encodes exactly as with the whole vocabulary). For Bonsai that is
101,064 of 151,669: the rest is CJK (21%), other scripts (9%) and Greek
and Cyrillic (3%). The output head (39 of the 242 MB read
per token) is a third shorter: about 13 MB less read per token (~5%),
and its top-k scan a third shorter. The model can no longer
pick a token it could not show (the screen printed those as "?"). On
five test questions, greedy, four English answers were identical to the
whole vocabulary's; asked to translate into Japanese, the whole
vocabulary writes Japanese (on the screen: "???") and the cut one,
without those tokens, repeats an English phrase instead.
`--vocab all` keeps every token (the serial console passes UTF-8 on, so
a terminal there could show the rest).

## Other models

`tools/mlm_convert.py` also reads llama2.c checkpoints (`llama2c`) and
Hugging Face directories (`hf`, model_type llama, mistral without
sliding windows, or qwen3) and makes their weights 1-bit (sign x mean
|w| per group of 128). That keeps every path testable with small
models (the simulations use it), but a model **not trained** for 1-bit
weights loses most of its quality that way. Useful models for this
system are ones published with 1-bit (`Q1_0`) GGUF weights in a
supported layout.

The firmware checks the model's sizes when loading (dim <= 2048,
heads <= 32, head_dim <= 256, the FFN in at most two pieces of 3,072,
context <= 512); it does not yet refuse a 4- or 8-bit image, which then
loads and produces nonsense (the old `QWEN06ML.MLM`, for one).

Larger 1-bit Bonsai models exist (4B, 8B), but the 4B one does not
leave room in 512 MB for the image plus its caches at a useful context,
and per token it would read about twice the data: ~0.5 tokens/s at
best on this board.

## Quality checks done so far

- Bonsai 1.7B, PC reference against `tools/gguf_ref.py` (float, the
  GGUF's own weights): same top-5 in order, logits within ~1%; greedy
  "The capital of France is" -> "Paris, and the capital of Spain is
  Madrid. ..."
- The whole Bonsai model on the simulated SOC (`sim/soc`, the image
  loaded directly): "Hi" in the chat template -> "Hello! I ...".
- Earlier (4/8-bit, synthetic weights): stories15M shape against
  llama2.c `run.c`, Qwen3 shape against transformers' semantics; see
  [arithmetic.md](arithmetic.md).
