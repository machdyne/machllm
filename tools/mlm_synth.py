#!/usr/bin/env python3
# machllm
# Copyright (c) 2025 Lone Dynamics Corporation. All rights reserved.
#
# mlm_synth: models with real shapes and random weights, for testing the
# converter, the reference and the hardware where real weights cannot be
# downloaded. Not for quality: random weights say nothing about that.
#
#   mlm_synth.py llama2c SHAPE out.bin [--seed N]
#       SHAPE: tiny | stories15M | stories42M | stories110M
#   mlm_synth.py hf SHAPE outdir [--seed N]
#       SHAPE: qwen3-0.6b | qwen3-tiny | smollm2-135m
#
# The hf form writes config.json, model.safetensors (bf16) and a
# byte-level BPE tokenizer.json with random merges up to the vocabulary.

import argparse, json, os, struct
import numpy as np

L2C = {
    'tiny':        dict(dim=64,  hid=256,  L=5,  nh=8,  nkv=4,  vocab=32000, seq=512),
    'stories15M':  dict(dim=288, hid=768,  L=6,  nh=6,  nkv=6,  vocab=32000, seq=256),
    'stories42M':  dict(dim=512, hid=1376, L=8,  nh=8,  nkv=8,  vocab=32000, seq=1024),
    'stories110M': dict(dim=768, hid=2048, L=12, nh=12, nkv=12, vocab=32000, seq=1024),
}
HF = {
    'qwen3-0.6b': dict(model_type='qwen3', hidden_size=1024, intermediate_size=3072,
                       num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=8,
                       head_dim=128, vocab_size=151936, rope_theta=1000000.0, rms_norm_eps=1e-6,
                       tie_word_embeddings=True, max_position_embeddings=40960, qk=True),
    'qwen3-tiny': dict(model_type='qwen3', hidden_size=256, intermediate_size=768,
                       num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                       head_dim=64, vocab_size=1024, rope_theta=1000000.0, rms_norm_eps=1e-6,
                       tie_word_embeddings=True, max_position_embeddings=4096, qk=True),
    # qwen3-tiny's width at Qwen3-0.6B's depth: what 56 residual adds do
    'qwen3-deep': dict(model_type='qwen3', hidden_size=256, intermediate_size=768,
                       num_hidden_layers=28, num_attention_heads=4, num_key_value_heads=2,
                       head_dim=64, vocab_size=1024, rope_theta=1000000.0, rms_norm_eps=1e-6,
                       tie_word_embeddings=True, max_position_embeddings=4096, qk=True),
    # Bonsai 1.7B's FFN width (two engine pieces of 3072) at qwen3-tiny's
    # other sizes
    'bonsai-tiny': dict(model_type='qwen3', hidden_size=256, intermediate_size=6144,
                        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                        head_dim=64, vocab_size=1024, rope_theta=1000000.0, rms_norm_eps=1e-6,
                        tie_word_embeddings=True, max_position_embeddings=4096, qk=True),
    # one Bonsai 1.7B layer, a small vocabulary: per-layer timing in the
    # SOC simulation
    'bonsai-1l': dict(model_type='qwen3', hidden_size=2048, intermediate_size=6144,
                      num_hidden_layers=1, num_attention_heads=16, num_key_value_heads=8,
                      head_dim=128, vocab_size=2048, rope_theta=1000000.0, rms_norm_eps=1e-6,
                      tie_word_embeddings=True, max_position_embeddings=4096, qk=True),
    'smollm2-135m': dict(model_type='llama', hidden_size=576, intermediate_size=1536,
                         num_hidden_layers=30, num_attention_heads=9, num_key_value_heads=3,
                         head_dim=64, vocab_size=49152, rope_theta=100000.0, rms_norm_eps=1e-5,
                         tie_word_embeddings=True, max_position_embeddings=8192, qk=False),
}

def llama2c(shape, out, rng):
    s = L2C[shape]
    dim, hid, L, nh, nkv, V, seq = s['dim'], s['hid'], s['L'], s['nh'], s['nkv'], s['vocab'], s['seq']
    hd = dim // nh
    with open(out, 'wb') as f:
        f.write(struct.pack('<7i', dim, hid, L, nh, nkv, V, seq))
        def w(*shape, std=0.02, base=0.0):
            f.write((rng.standard_normal(shape, dtype=np.float32) * std + base).astype(np.float32).tobytes())
        w(V, dim, std=0.1)
        w(L, dim, std=0.1, base=1.0)
        w(L, nh * hd, dim, std=0.08); w(L, nkv * hd, dim, std=0.08); w(L, nkv * hd, dim, std=0.08)
        w(L, dim, nh * hd, std=0.04)
        w(L, dim, std=0.1, base=1.0)
        w(L, hid, dim, std=0.06); w(L, dim, hid, std=0.04); w(L, hid, dim, std=0.06)
        w(dim, std=0.1, base=1.0)
        f.write(np.zeros(seq * hd, np.float32).tobytes())   # freq_cis, unused

def bf16(a):
    u = np.asarray(a, np.float32).view(np.uint32)
    return ((u + 0x7fff + ((u >> 16) & 1)) >> 16).astype(np.uint16)

def unicode_bytes():
    bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + \
         list(range(ord('®'), ord('ÿ') + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return [chr(cs[bs.index(b)]) for b in range(256)]

def hf(shape, outdir, rng):
    c = dict(HF[shape])
    qk = c.pop('qk')
    os.makedirs(outdir, exist_ok=True)
    dim, hid, L = c['hidden_size'], c['intermediate_size'], c['num_hidden_layers']
    nh, nkv, hd, V = c['num_attention_heads'], c['num_key_value_heads'], c['head_dim'], c['vocab_size']
    n_special = 3 if c['model_type'] == 'qwen3' else 2
    cfg = dict(c, architectures=['Qwen3ForCausalLM' if qk else 'LlamaForCausalLM'],
               hidden_act='silu', attention_bias=False, torch_dtype='bfloat16',
               bos_token_id=V - n_special, eos_token_id=V - n_special + 1)
    json.dump(cfg, open(os.path.join(outdir, 'config.json'), 'w'), indent=1)
    tensors = []
    def t(name, shape, std, base=0.0):
        tensors.append((name, shape, std, base))
    t('model.embed_tokens.weight', (V, dim), 0.1)
    for l in range(L):
        p = 'model.layers.%d.' % l
        t(p + 'input_layernorm.weight', (dim,), 0.1, 1.0)
        t(p + 'self_attn.q_proj.weight', (nh * hd, dim), 0.06)
        t(p + 'self_attn.k_proj.weight', (nkv * hd, dim), 0.06)
        t(p + 'self_attn.v_proj.weight', (nkv * hd, dim), 0.06)
        t(p + 'self_attn.o_proj.weight', (dim, nh * hd), 0.03)
        if qk:
            t(p + 'self_attn.q_norm.weight', (hd,), 0.1, 1.0)
            t(p + 'self_attn.k_norm.weight', (hd,), 0.1, 1.0)
        t(p + 'post_attention_layernorm.weight', (dim,), 0.1, 1.0)
        t(p + 'mlp.gate_proj.weight', (hid, dim), 0.05)
        t(p + 'mlp.up_proj.weight', (hid, dim), 0.05)
        t(p + 'mlp.down_proj.weight', (dim, hid), 0.03)
    t('model.norm.weight', (dim,), 0.1, 1.0)
    meta, off = {}, 0
    for name, shape, std, base in tensors:
        n = int(np.prod(shape)) * 2
        meta[name] = dict(dtype='BF16', shape=list(shape), data_offsets=[off, off + n])
        off += n
    hj = json.dumps(meta).encode()
    hj += b' ' * ((8 - len(hj) % 8) % 8)
    with open(os.path.join(outdir, 'model.safetensors'), 'wb') as f:
        f.write(struct.pack('<Q', len(hj)) + hj)
        for name, shape, std, base in tensors:
            a = rng.standard_normal(shape, dtype=np.float32) * std + base
            f.write(bf16(a).tobytes())
    # tokenizer: 256 byte tokens, random merges, specials at the end
    ub = unicode_bytes()
    vocab = {ub[b]: b for b in range(256)}
    toks = list(ub)
    merges = []
    n_normal = V - n_special
    while len(toks) < n_normal:
        a = toks[rng.integers(0, len(toks))]; b = toks[rng.integers(0, min(len(toks), 2000))]
        s = a + b
        if s in vocab or len(s) > 16: continue
        vocab[s] = len(toks); toks.append(s); merges.append(a + ' ' + b)
    specials = ['<|endoftext|>', '<|im_start|>', '<|im_end|>'] if n_special == 3 else ['<s>', '</s>']
    added = []
    for i, s in enumerate(specials):
        added.append(dict(id=n_normal + i, content=s, special=True))
    tj = dict(version='1.0', added_tokens=added,
              pre_tokenizer=dict(type='ByteLevel', add_prefix_space=False),
              decoder=dict(type='ByteLevel'),
              model=dict(type='BPE', vocab=vocab, merges=merges))
    json.dump(tj, open(os.path.join(outdir, 'tokenizer.json'), 'w'))
    if n_special == 3:
        cfg['bos_token_id'] = n_normal; cfg['eos_token_id'] = n_normal + 2
        json.dump(cfg, open(os.path.join(outdir, 'config.json'), 'w'), indent=1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('kind', choices=['llama2c', 'hf'])
    ap.add_argument('shape')
    ap.add_argument('out')
    ap.add_argument('--seed', type=int, default=1)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    (llama2c if a.kind == 'llama2c' else hf)(a.shape, a.out, rng)

if __name__ == '__main__':
    main()
