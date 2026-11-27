#!/usr/bin/env python3
# machllm
# Copyright (c) 2025 Lone Dynamics Corporation. All rights reserved.
#
# gguf_ref: a float reference forward pass of a qwen3 GGUF (Bonsai), in
# numpy, from the GGUF's own (dequantised) weights: to tell a problem in
# reading the model from a problem in machllm's integer arithmetic.
#
#   gguf_ref.py model.gguf "id id id ..." [--yarn 0|1]
#
# prints the top-5 next tokens (id:logit) after each position, as
# `mlm_run -d` does.

import argparse, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_read import GGUF


def rms(x, w, eps):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('gguf')
    ap.add_argument('ids')
    ap.add_argument('--yarn', type=int, default=1)
    a = ap.parse_args()
    g = GGUF(a.gguf)
    M = g.meta
    k = lambda n, d=None: M.get('qwen3.' + n, d)
    dim, L = int(k('embedding_length')), int(k('block_count'))
    nh, nkv, hd = int(k('attention.head_count')), int(k('attention.head_count_kv')), int(k('attention.key_length'))
    eps = float(k('attention.layer_norm_rms_epsilon'))
    base = float(k('rope.freq_base'))
    ids = [int(t) for t in a.ids.replace(',', ' ').split()]
    T = len(ids)
    W = lambda n: g.float(n)

    # RoPE (NEOX: pairs i, i + hd/2), YaRN as llama.cpp
    half = hd // 2
    i = np.arange(half, dtype=np.float64)
    inv = base ** (-i * 2 / hd)
    msc = 1.0
    if a.yarn and k('rope.scaling.type') == 'yarn':
        f = float(k('rope.scaling.factor')); n0 = float(k('rope.scaling.original_context_length'))
        cd = lambda nrot: hd * math.log(n0 / (nrot * 2 * math.pi)) / (2 * math.log(base))
        low, high = max(0.0, math.floor(cd(32.0))), min(hd - 1.0, math.ceil(cd(1.0)))
        ramp = 1 - np.clip((i - low) / max(0.001, high - low), 0, 1)
        inv = (inv / f) * (1 - ramp) + inv * ramp
        msc = 1.0 + 0.1 * math.log(f)
    ang = np.arange(T)[:, None] * inv[None, :]
    cos, sin = (np.cos(ang) * msc).astype(np.float32), (np.sin(ang) * msc).astype(np.float32)

    def rope(x):                      # x [T, heads, hd]
        x1, x2 = x[..., :half], x[..., half:]
        c, s = cos[:, None, :], sin[:, None, :]
        return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], -1)

    emb_d, emb_b = g.q1_0('token_embd.weight')
    def emb_row(t):
        return np.where(emb_b[t], 1.0, -1.0).astype(np.float32) * np.repeat(emb_d[t], 128)
    x = np.stack([emb_row(t) for t in ids])
    D = lambda w, v: print('%s: rms %.5f  %s' % (w, float(np.sqrt((v[0] ** 2).mean())), ' '.join('%.5f' % z for z in v[0][:6])))
    D('emb', x)
    mask = np.triu(np.full((T, T), -np.inf, np.float32), 1)
    for l in range(L):
        p = 'blk.%d.' % l
        h = rms(x, W(p + 'attn_norm.weight'), eps)
        if l == 0: D('norm', h)
        q = (h @ W(p + 'attn_q.weight').T).reshape(T, nh, hd)
        if l == 0: D('q', q.reshape(T, -1))
        kk = (h @ W(p + 'attn_k.weight').T).reshape(T, nkv, hd)
        v = (h @ W(p + 'attn_v.weight').T).reshape(T, nkv, hd)
        if l == 0: D('v', v.reshape(T, -1))
        q = rope(rms(q, W(p + 'attn_q_norm.weight'), eps))
        kk = rope(rms(kk, W(p + 'attn_k_norm.weight'), eps))
        rep = nh // nkv
        o = np.empty((T, nh, hd), np.float32)
        for hh in range(nh):
            kv = hh // rep
            sc = q[:, hh, :] @ kk[:, kv, :].T / math.sqrt(hd) + mask
            sc = np.exp(sc - sc.max(-1, keepdims=True))
            sc /= sc.sum(-1, keepdims=True)
            o[:, hh, :] = sc @ v[:, kv, :]
        if l == 0: D('o', o.reshape(T, -1))
        x = x + o.reshape(T, nh * hd) @ W(p + 'attn_output.weight').T
        if l < 2: D('attn', x)
        h = rms(x, W(p + 'ffn_norm.weight'), eps)
        gt = h @ W(p + 'ffn_gate.weight').T
        up = h @ W(p + 'ffn_up.weight').T
        x = x + ((gt / (1 + np.exp(-gt))) * up) @ W(p + 'ffn_down.weight').T
        if l < 2 or l == L - 1: D('ffn', x)
        sys.stderr.write('\rlayer %d/%d' % (l + 1, L)); sys.stderr.flush()
    sys.stderr.write('\n')
    h = rms(x, W('output_norm.weight'), eps)
    # logits in pieces (the head is the 1-bit embedding)
    V = emb_b.shape[0]
    logits = np.empty((T, V), np.float32)
    for r0 in range(0, V, 8192):
        r1 = min(V, r0 + 8192)
        blk = np.where(emb_b[r0:r1], 1.0, -1.0).astype(np.float32) * np.repeat(emb_d[r0:r1], 128, -1)
        logits[:, r0:r1] = h @ blk.T
    for t in range(T):
        top = np.argsort(-logits[t])[:5]
        print('pos %d in %d:' % (t, ids[t]), ' '.join('%d:%.4f' % (j, logits[t, j]) for j in top))


if __name__ == '__main__':
    main()
