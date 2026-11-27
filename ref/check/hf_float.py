#!/usr/bin/env python3
# machllm
# Copyright (c) 2025 Lone Dynamics Corporation. All rights reserved.
#
# A float NumPy forward pass of a Hugging Face Llama/Qwen3 directory,
# written from the transformers modelling code's semantics (rotate_half
# RoPE, per-head q_norm/k_norm before RoPE, GQA by repeating kv heads,
# SwiGLU, tied head). Independent of mlm_convert/mlm_run: used to check
# that they read a model the way transformers does.
#
#   hf_float.py hfdir id id id ...      prints "pos N in T: id:logit x5"
#   --requant BITS HEADBITS             use the weights as mlm_convert
#                                       quantises them

import sys, os, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'tools'))
import mlm_convert as mc

def main():
    args = sys.argv[1:]
    rq = None
    if args[0] == '--requant':
        rq = (int(args[1]), int(args[2])); args = args[3:]
    d, ids = args[0], [int(x) for x in args[1:]]
    cfg = json.load(open(os.path.join(d, 'config.json')))
    W = {}
    for fn in os.listdir(d):
        if fn.endswith('.safetensors'): W.update(mc.read_safetensors(os.path.join(d, fn)))
    q = lambda w, b: mc.requant_fast(w, b) if rq else np.asarray(w, np.float32)
    nv = lambda g: (lambda qq, e: qq.astype(np.float64) * 2.0 ** e)(*mc.norm_vec(g)) if rq else np.asarray(g, np.float64)
    dim, L = cfg['hidden_size'], cfg['num_hidden_layers']
    nh, nkv = cfg['num_attention_heads'], cfg.get('num_key_value_heads', cfg['num_attention_heads'])
    hd = cfg.get('head_dim') or dim // nh
    eps, theta = cfg.get('rms_norm_eps', 1e-6), cfg.get('rope_theta', 10000.0)
    bits, hbits = rq if rq else (0, 0)
    emb = q(W['model.embed_tokens.weight'], hbits)
    head = q(W['lm_head.weight'], hbits) if 'lm_head.weight' in W else emb
    def rms(x, g): return x / np.sqrt(np.mean(x * x, -1, keepdims=True) + eps) * g
    inv = theta ** (-np.arange(0, hd, 2, dtype=np.float64) / hd)
    def rope(x, pos):                                   # x [..., hd]
        ang = pos * inv
        c, s = np.cos(np.concatenate([ang, ang])), np.sin(np.concatenate([ang, ang]))
        rot = np.concatenate([-x[..., hd // 2:], x[..., :hd // 2]], -1)
        return x * c + rot * s
    P = 'model.layers.%d.'
    Ls = []
    for l in range(L):
        p = P % l
        g = lambda n: W[p + n]
        Ls.append(dict(an=nv(g('input_layernorm.weight')), fn=nv(g('post_attention_layernorm.weight')),
                       wq=q(g('self_attn.q_proj.weight'), bits), wk=q(g('self_attn.k_proj.weight'), bits),
                       wv=q(g('self_attn.v_proj.weight'), bits), wo=q(g('self_attn.o_proj.weight'), bits),
                       w1=q(g('mlp.gate_proj.weight'), bits), w3=q(g('mlp.up_proj.weight'), bits),
                       w2=q(g('mlp.down_proj.weight'), bits),
                       qn=nv(g('self_attn.q_norm.weight')) if p + 'self_attn.q_norm.weight' in W else None,
                       kn=nv(g('self_attn.k_norm.weight')) if p + 'self_attn.k_norm.weight' in W else None))
    fnorm = nv(W['model.norm.weight'])
    K = [[] for _ in range(L)]; V = [[] for _ in range(L)]
    for pos, tok in enumerate(ids):
        x = emb[tok].astype(np.float64)
        for l, Lw in enumerate(Ls):
            h = rms(x, Lw['an'])
            qv = (Lw['wq'] @ h).reshape(nh, hd); kv = (Lw['wk'] @ h).reshape(nkv, hd)
            vv = (Lw['wv'] @ h).reshape(nkv, hd)
            if Lw['qn'] is not None: qv = rms(qv, Lw['qn']); kv = rms(kv, Lw['kn'])
            qv, kv = rope(qv, pos), rope(kv, pos)
            K[l].append(kv); V[l].append(vv)
            Ks, Vs = np.stack(K[l], 1), np.stack(V[l], 1)          # [nkv, T, hd]
            o = np.zeros((nh, hd))
            for i in range(nh):
                j = i // (nh // nkv)
                s = Ks[j] @ qv[i] / np.sqrt(hd)
                s = np.exp(s - s.max()); s /= s.sum()
                o[i] = s @ Vs[j]
            x = x + Lw['wo'] @ o.reshape(-1)
            h = rms(x, Lw['fn'])
            a = Lw['w1'] @ h
            x = x + Lw['w2'] @ (a / (1 + np.exp(-a)) * (Lw['w3'] @ h))
        lg = head @ rms(x, fnorm)
        top = np.argsort(-lg)[:5]
        print('pos %d in %d:' % (pos, tok) + ''.join(' %d:%.4f' % (t, lg[t]) for t in top))
        sys.stdout.flush()

if __name__ == '__main__':
    main()
