#!/bin/sh
# machllm: the synthetic 1-bit test models the simulations use
# (build/models/*.mlm). Weights are random and reduced to 1 bit: the
# text is noise; the tests compare the hardware with ref/mlm_run.
set -e
cd "$(dirname "$0")/.."
M=build/models
TOK=${TOK:-../src/llama2.c/tokenizer.bin}
mkdir -p $M
[ -f $M/s15.bin ] || python3 tools/mlm_synth.py llama2c stories15M $M/s15.bin
python3 tools/mlm_convert.py llama2c $M/s15.bin $TOK $M/s15.mlm
python3 tools/mlm_synth.py llama2c tiny $M/tiny.bin
python3 tools/mlm_convert.py llama2c $M/tiny.bin $TOK $M/tiny.mlm
[ -d $M/qtiny ] || python3 tools/mlm_synth.py hf qwen3-tiny $M/qtiny
python3 tools/mlm_convert.py hf $M/qtiny $M/qtiny.mlm
[ -d $M/btiny ] || python3 tools/mlm_synth.py hf bonsai-tiny $M/btiny
python3 tools/mlm_convert.py hf $M/btiny $M/btiny.mlm
