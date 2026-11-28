#!/usr/bin/env python3
# machllm
# Copyright (c) 2025 Lone Dynamics Corporation. All rights reserved.
#
# mlm_convert: a model -> a .mlm image for machllm. See docs/format.md.
#
#   mlm_convert.py gguf     Bonsai-1.7B-Q1_0.gguf          out.mlm [opts]
#   mlm_convert.py llama2c  stories42M.bin tokenizer.bin out.mlm [opts]
#   mlm_convert.py hf       path/to/hf_dir                 out.mlm [opts]
#
# The engine runs 1-bit weights (groups of 128, one 16-bit multiplier
# each: the Bonsai / GGUF Q1_0 layout). A GGUF Q1_0 model (Bonsai) is
# imported exactly; float checkpoints (llama2c, hf) are reduced to 1 bit
# (sign, mean |w| per group) -- for tests: the text they make is noise.
#
# options:
#   --ctx N         longest context the image supports (RoPE table; 2048)
#   --name S        model name stored in the image
#
# Only numpy (and the standard library) is needed; safetensors files are
# read directly, bf16/f16/f32.

import argparse, json, math, os, struct, sys
import numpy as np

MAGIC, VERSION, ARCH_LLAMA = 0x4d4c4c4d, 1, 1
F_QK_NORM, F_TIED, F_BOS, F_CHAT_QWEN, F_ROPE_HALF, F_TOK_BPE = 1, 2, 4, 8, 16, 32
NONE = 0xffffffff
GROUP = 128                      # weights per group (1 bit each)

# ------------------------------------------------------------------
# quantisation

def pack1(m, bits, e):
    """multipliers m [rows, G] (uint16) and bits bool [rows, G*128] ->
    (bytes [rows, stride], exp, groups, stride). Row layout: chunks of up
    to four groups; a chunk is a u64 of four u16 multipliers then the
    groups' bits, 16 bytes each, least significant bit first."""
    rows, G = m.shape
    pb = np.packbits(bits.reshape(rows, G, GROUP), axis=-1, bitorder='little')   # [rows, G, 16]
    parts = []
    for c in range((G + 3) // 4):
        g0, g1 = 4 * c, min(G, 4 * c + 4)
        mm = np.zeros((rows, 4), np.uint16)
        mm[:, :g1 - g0] = m[:, g0:g1]
        parts.append(mm.view(np.uint8).reshape(rows, 8))
        parts.append(pb[:, g0:g1, :].reshape(rows, -1))
    data = np.concatenate(parts, 1)
    assert data.shape[1] == row_bytes(G)
    stride = (data.shape[1] + 15) & ~15
    if stride != data.shape[1]:
        data = np.concatenate([data, np.zeros((rows, stride - data.shape[1]), np.uint8)], 1)
    return np.ascontiguousarray(data), e, G, stride

def scales_to_m(d):
    """per-group scales (float, > 0) -> 16-bit multipliers and the tensor's exponent"""
    d = np.asarray(d, np.float64)
    dmax = float(d.max()) if d.size else 1.0
    if dmax <= 0: dmax = 1.0
    e = int(math.ceil(math.log2(dmax / 65535.0)))
    m = np.clip(np.round(d / 2.0 ** e), 1, 65535).astype(np.uint16)
    return m, e

def quantize(W):
    """float W [rows, cols] -> 1 bit: sign, and mean |w| per group of 128
    (cols padded with zeros; their x is 0, so their bit does not matter)"""
    W = np.asarray(W, dtype=np.float32)
    rows, cols = W.shape
    if cols % GROUP:
        W = np.concatenate([W, np.zeros((rows, GROUP - cols % GROUP), np.float32)], 1)
    G = W.shape[1] // GROUP
    d = np.abs(W.reshape(rows, G, GROUP)).mean(-1)
    d = np.maximum(d, 1e-12)
    m, e = scales_to_m(d)
    return pack1(m, W > 0, e)

def import_q1_0(d, bits):
    """a GGUF Q1_0 tensor (scales [rows, G], bits [rows, cols]) -> the same
    weights exactly, up to the multipliers' 16 bits (a negative scale
    flips its group's bits)"""
    d = np.asarray(d, np.float32)
    neg = d < 0
    if neg.any():
        bits = bits ^ np.repeat(neg, GROUP, -1)
        d = np.abs(d)
    m, e = scales_to_m(np.maximum(d, 1e-12))
    return pack1(m, bits, e)

def row_bytes(groups):
    full, rem = divmod(groups, 4)
    return full * 72 + ((8 + rem * 16) if rem else 0)

def dequantize_row(data, e, G):
    """inverse of pack1 for one row (tests)"""
    out, p = [], 0
    for c in range((G + 3) // 4):
        hdr = np.frombuffer(bytes(data[p:p + 8]), np.uint16); p += 8
        for j in range(min(4, G - 4 * c)):
            b = np.unpackbits(np.asarray(data[p:p + 16], np.uint8), bitorder='little'); p += 16
            out.append(np.where(b, 1.0, -1.0) * float(hdr[j]) * 2.0 ** e)
    return np.concatenate(out)

def norm_vec(g):
    g = np.asarray(g, np.float64)
    mx = float(np.abs(g).max()) or 1.0
    e = int(math.ceil(math.log2(mx / 32767.0)))
    return np.clip(np.round(g / 2.0 ** e), -32767, 32767).astype(np.int16), e

# ------------------------------------------------------------------
# image writer

class Image:
    def __init__(self):
        self.buf = bytearray(4096)          # header page
    def align(self, a):
        while len(self.buf) % a: self.buf.append(0)
    def put(self, b, align=16):
        self.align(align)
        off = len(self.buf)
        self.buf += b
        return off
    def tensor(self, W):
        """W: float [rows, cols], or ('q1_0', scales, bits) from a GGUF"""
        if isinstance(W, tuple):
            data, e, G, stride = import_q1_0(W[1], W[2])
        else:
            data, e, G, stride = quantize(W)
        off = self.put(data.tobytes(), 64)
        rows = data.shape[0]
        return struct.pack('<IIIIIiII', off, rows, G * GROUP, 1, stride, e, G, 0)
    def norm(self, g):
        q, e = norm_vec(g)
        off = self.put(q.tobytes())
        return struct.pack('<IIiI', off, len(q), e, 0)

def tok_block(tokens, scores, findable):
    """tokens: list of bytes; scores: list of int; findable: list of bool"""
    n = len(tokens)
    entries = bytearray(); offs = []
    hdr_size = 20
    for t, s in zip(tokens, scores):
        offs.append(len(entries))
        entries += struct.pack('<iI', int(s), len(t)) + t
    order = sorted([i for i in range(n) if findable[i]], key=lambda i: tokens[i])
    # de-duplicate identical byte strings: keep the lowest id
    seen, uniq = set(), []
    for i in sorted(order, key=lambda i: (tokens[i], i)):
        if tokens[i] in seen: continue
        seen.add(tokens[i]); uniq.append(i)
    uniq.sort(key=lambda i: tokens[i])
    ent_off = hdr_size
    offs_off = ent_off + len(entries)
    offs_off = (offs_off + 3) & ~3
    pad = offs_off - (ent_off + len(entries))
    sorted_off = offs_off + 4 * n
    b = struct.pack('<IIIII', n, max(len(t) for t in tokens), ent_off, offs_off, sorted_off)
    b += entries + b'\0' * pad
    b += struct.pack('<%dI' % n, *[ent_off + o for o in offs])
    # sorted table holds only findable tokens; pad the count to n with
    # the last entry is not allowed -- store count separately: we keep
    # n entries but the lookup searches len(uniq); write len(uniq) into
    # n_sorted by repeating... simpler: the firmware uses n for the
    # search range, so findable must cover every id or the table must
    # be padded. We pad with the first entry (duplicates are harmless
    # for a binary search that stops on equality).
    lst = uniq + [uniq[-1]] * (n - len(uniq))
    lst.sort(key=lambda i: tokens[i])
    b += struct.pack('<%dI' % n, *lst)
    return bytes(b)

def rope_table(ctx, hd, theta, yarn=None):
    """cos and sin per position and pair, Q14. yarn: dict(factor,
    orig_ctx[, beta_fast, beta_slow]) for YaRN scaling as llama.cpp does
    it (ggml rope_yarn: frequencies ramped between interpolated and
    original, magnitude 1 + 0.1 ln(factor))"""
    half = hd // 2
    i = np.arange(half, dtype=np.float64)
    inv = theta ** (-i * 2 / hd)
    mscale = 1.0
    if yarn:
        f = float(yarn['factor']); n0 = float(yarn['orig_ctx'])
        bf, bs = float(yarn.get('beta_fast', 32.0)), float(yarn.get('beta_slow', 1.0))
        cd = lambda nrot: hd * math.log(n0 / (nrot * 2 * math.pi)) / (2 * math.log(theta))
        low = max(0.0, math.floor(cd(bf)))
        high = min(hd - 1.0, math.ceil(cd(bs)))
        ramp = 1 - np.clip((i - low) / max(0.001, high - low), 0, 1)
        inv = (inv / f) * (1 - ramp) + inv * ramp
        mscale = 1.0 + 0.1 * math.log(f)
    ang = np.arange(ctx, dtype=np.float64)[:, None] * inv[None, :]
    t = np.empty((ctx, half, 2), np.int16)
    t[..., 0] = np.round(np.cos(ang) * 16384 * mscale)
    t[..., 1] = np.round(np.sin(ang) * 16384 * mscale)
    return t.tobytes()

def eps_pair(eps):
    e = math.floor(math.log2(eps)) - 23
    return int(round(eps / 2.0 ** e)), e

def write_image(path, cfg, get, tokens, scores, findable, name):
    """cfg: dict of the architecture; get(name) -> numpy array"""
    im = Image()
    dim, hid, L = cfg['dim'], cfg['hidden_dim'], cfg['n_layers']
    nh, nkv, hd = cfg['n_heads'], cfg['n_kv_heads'], cfg['head_dim']
    assert dim % 32 == 0, 'dim must be a multiple of 32'
    name_off = im.put(name.encode() + b'\0')
    tb = tok_block(tokens, scores, findable)
    tok_off = im.put(tb)
    rope_off = im.put(rope_table(cfg['ctx'], hd, cfg['rope_theta'], cfg.get('yarn')))
    final_norm = im.norm(get('final_norm'))
    emb = im.tensor(get('emb'))
    if cfg['tied']:
        head = emb
    else:
        head = im.tensor(get('head'))
    layers = bytearray()
    for l in range(L):
        sys.stderr.write('\rlayer %d/%d' % (l + 1, L)); sys.stderr.flush()
        lb = im.norm(get('attn_norm', l))
        for t in ('wq', 'wk', 'wv', 'wo'):
            lb += im.tensor(get(t, l))
        if cfg['qk_norm']:
            lb += im.norm(get('q_norm', l)) + im.norm(get('k_norm', l))
        else:
            lb += struct.pack('<IIiI', 0, 0, 0, 0) * 2
        lb += im.norm(get('ffn_norm', l))
        for t in ('w1', 'w3', 'w2'):
            lb += im.tensor(get(t, l))
        assert len(lb) == 288
        layers += lb
    sys.stderr.write('\n')
    layers_off = im.put(bytes(layers), 64)
    im.align(16)
    size = len(im.buf)
    flags = cfg['flags'] | (F_QK_NORM if cfg['qk_norm'] else 0) | (F_TIED if cfg['tied'] else 0)
    em, ee = eps_pair(cfg['eps'])
    hdr = struct.pack('<24I', MAGIC, VERSION, ARCH_LLAMA, dim, hid, L, nh, nkv, hd,
                      cfg['vocab'], cfg['ctx'], flags, em & 0xffffffff, ee & 0xffffffff,
                      int(round(2 ** 30 / math.sqrt(hd))), tok_off, len(tb), rope_off,
                      layers_off, cfg['bos'], cfg['eos'], cfg['eos2'], size, name_off)
    hdr += final_norm + emb + head + struct.pack('<IIII', cfg.get('im_start', NONE), cfg.get('im_end', NONE),
                                                  cfg.get('think', NONE), cfg.get('think_end', NONE))
    hdr += b'\0' * 56
    im.buf[0:len(hdr)] = hdr
    with open(path, 'wb') as f:
        f.write(im.buf)
    nparam = dim * hid * 3 * L + (dim * nh * hd * 2 + dim * nkv * hd * 2) * L
    sys.stderr.write('%s: %.1f MB, %d layers, %.1fM layer params, 1 bit\n' %
                     (path, size / 1e6, L, nparam / 1e6))

# ------------------------------------------------------------------
# llama2.c checkpoints

def load_llama2c(model_path, tok_path):
    raw = open(model_path, 'rb').read()
    dim, hid, L, nh, nkv, vocab, seq = struct.unpack('<7i', raw[:28])
    shared = vocab > 0
    vocab = abs(vocab)
    hd = dim // nh
    f = np.frombuffer(raw, np.float32, offset=28)
    p = [0]
    def take(*shape):
        n = int(np.prod(shape))
        a = f[p[0]:p[0] + n].reshape(shape); p[0] += n
        return a
    W = {}
    W['emb'] = take(vocab, dim)
    W['attn_norm'] = take(L, dim)
    W['wq'] = take(L, nh * hd, dim)
    W['wk'] = take(L, nkv * hd, dim)
    W['wv'] = take(L, nkv * hd, dim)
    W['wo'] = take(L, dim, nh * hd)
    W['ffn_norm'] = take(L, dim)
    W['w1'] = take(L, hid, dim)
    W['w2'] = take(L, dim, hid)
    W['w3'] = take(L, hid, dim)
    W['final_norm'] = take(dim)
    take(seq, hd // 2); take(seq, hd // 2)
    W['head'] = W['emb'] if shared else take(vocab, dim)
    get = lambda n, l=None: W[n] if l is None else W[n][l]
    # tokenizer.bin
    tb = open(tok_path, 'rb').read()
    maxlen = struct.unpack('<i', tb[:4])[0]
    q, tokens, scores = 4, [], []
    for i in range(vocab):
        sc, ln = struct.unpack('<fi', tb[q:q + 8]); q += 8
        tokens.append(tb[q:q + ln]); q += ln
        scores.append(int(max(-2**31 + 1, min(2**31 - 1, round(sc * 1000)))))
    cfg = dict(dim=dim, hidden_dim=hid, n_layers=L, n_heads=nh, n_kv_heads=nkv, head_dim=hd,
               vocab=vocab, ctx=seq, rope_theta=10000.0, eps=1e-5, tied=shared, qk_norm=False,
               flags=F_BOS, bos=1, eos=2, eos2=NONE)
    findable = [True] * vocab
    findable[0] = findable[1] = findable[2] = False   # <unk> <s> </s>
    return cfg, get, tokens, scores, findable

# ------------------------------------------------------------------
# Hugging Face directories (Llama, Qwen3, SmolLM2)

def read_safetensors(path):
    raw = open(path, 'rb').read()
    n = struct.unpack('<Q', raw[:8])[0]
    meta = json.loads(raw[8:8 + n])
    base = 8 + n
    out = {}
    for k, v in meta.items():
        if k == '__metadata__': continue
        a, b = v['data_offsets']
        buf = raw[base + a: base + b]
        dt = v['dtype']
        if dt == 'BF16':
            u = np.frombuffer(buf, np.uint16).astype(np.uint32) << 16
            arr = u.view(np.float32)
        elif dt == 'F16':
            arr = np.frombuffer(buf, np.float16).astype(np.float32)
        elif dt == 'F32':
            arr = np.frombuffer(buf, np.float32)
        else:
            raise SystemExit('unsupported dtype %s in %s' % (dt, k))
        out[k] = arr.reshape(v['shape'])
    return out

def bytes_to_unicode():
    bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + \
         list(range(ord('®'), ord('ÿ') + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}

def load_hf(d, ctx):
    cfgj = json.load(open(os.path.join(d, 'config.json')))
    W = {}
    for fn in sorted(os.listdir(d)):
        if fn.endswith('.safetensors'):
            W.update(read_safetensors(os.path.join(d, fn)))
    mt = cfgj.get('model_type', 'llama')
    if mt not in ('llama', 'qwen3', 'mistral'):
        raise SystemExit('model_type %s not supported yet (llama, qwen3)' % mt)
    if any(k.endswith('.bias') for k in W):
        raise SystemExit('models with biases are not supported yet')
    dim = cfgj['hidden_size']; nh = cfgj['num_attention_heads']
    nkv = cfgj.get('num_key_value_heads', nh)
    hd = cfgj.get('head_dim') or dim // nh
    L = cfgj['num_hidden_layers']
    tied = cfgj.get('tie_word_embeddings', False) or 'lm_head.weight' not in W
    qk = 'model.layers.0.self_attn.q_norm.weight' in W
    P = 'model.layers.%d.'
    names = {'attn_norm': 'input_layernorm.weight', 'wq': 'self_attn.q_proj.weight',
             'wk': 'self_attn.k_proj.weight', 'wv': 'self_attn.v_proj.weight',
             'wo': 'self_attn.o_proj.weight', 'q_norm': 'self_attn.q_norm.weight',
             'k_norm': 'self_attn.k_norm.weight', 'ffn_norm': 'post_attention_layernorm.weight',
             'w1': 'mlp.gate_proj.weight', 'w3': 'mlp.up_proj.weight', 'w2': 'mlp.down_proj.weight'}
    def get(n, l=None):
        if n == 'emb': return W['model.embed_tokens.weight']
        if n == 'head': return W['lm_head.weight']
        if n == 'final_norm': return W['model.norm.weight']
        return W[(P % l) + names[n]]
    # tokenizer
    tj = json.load(open(os.path.join(d, 'tokenizer.json')))
    model = tj['model']
    if model['type'] != 'BPE':
        raise SystemExit('tokenizer type %s not supported' % model['type'])
    vocab_n = W['model.embed_tokens.weight'].shape[0]
    b2u = bytes_to_unicode()
    tokens = [b''] * vocab_n; scores = [-(1 << 30)] * vocab_n; findable = [False] * vocab_n
    for s, i in model['vocab'].items():
        if i >= vocab_n: continue
        try: tokens[i] = bytes(b2u[c] for c in s)
        except KeyError: tokens[i] = s.encode()
        findable[i] = True
    for r, mg in enumerate(model.get('merges', [])):
        a, b = mg.split(' ', 1) if isinstance(mg, str) else mg
        t = model['vocab'].get(a + b)
        if t is not None and t < vocab_n and scores[t] == -(1 << 30):
            scores[t] = -r
    special = {}
    for at in tj.get('added_tokens', []):
        if at['id'] < vocab_n:
            tokens[at['id']] = at['content'].encode()
            findable[at['id']] = False
            special[at['content']] = at['id']
    for i in range(vocab_n):
        if not tokens[i]: tokens[i] = b'<unused%d>' % i
    flags = F_ROPE_HALF | F_TOK_BPE
    eos = cfgj.get('eos_token_id', NONE)
    if isinstance(eos, list): eos = eos[0]
    bos = cfgj.get('bos_token_id') or eos
    im_s, im_e = special.get('<|im_start|>', NONE), special.get('<|im_end|>', NONE)
    if im_s != NONE: flags |= F_CHAT_QWEN
    cfg = dict(dim=dim, hidden_dim=cfgj['intermediate_size'], n_layers=L, n_heads=nh,
               n_kv_heads=nkv, head_dim=hd, vocab=vocab_n,
               ctx=min(ctx, cfgj.get('max_position_embeddings', ctx)),
               rope_theta=float(cfgj.get('rope_theta', 10000.0)), eps=cfgj.get('rms_norm_eps', 1e-6),
               tied=tied, qk_norm=qk, flags=flags, bos=bos, eos=eos,
               eos2=im_e, im_start=im_s, im_end=im_e,
               think=special.get('<think>', NONE), think_end=special.get('</think>', NONE))
    if mt == 'llama' and cfgj.get('bos_token_id') is not None:
        cfg['flags'] |= F_BOS
    return cfg, get, tokens, scores, findable

def requant_fast(W):
    """W reduced to 1 bit and back to float (quantisation error alone)"""
    W = np.asarray(W, np.float32)
    rows, cols = W.shape
    pad = (-cols) % GROUP
    Wp = np.concatenate([W, np.zeros((rows, pad), np.float32)], 1) if pad else W
    G = Wp.shape[1] // GROUP
    d = np.maximum(np.abs(Wp.reshape(rows, G, GROUP)).mean(-1), 1e-12)
    m, e = scales_to_m(d)
    s2 = m.astype(np.float64) * 2.0 ** e
    q = np.where(Wp.reshape(rows, G, GROUP) > 0, 1.0, -1.0) * s2[..., None]
    return q.reshape(rows, -1)[:, :cols].astype(np.float32)

def write_dequant_l2c(path, src, cfg, get):
    raw = open(src, 'rb').read()
    hdr = raw[:28]
    L = cfg['n_layers']
    with open(path, 'wb') as f:
        f.write(hdr)
        nv = lambda g: (lambda q, e: (q.astype(np.float64) * 2.0 ** e).astype(np.float32))(*norm_vec(g))
        f.write(requant_fast(get('emb')).tobytes())
        for l in range(L): f.write(nv(get('attn_norm', l)).tobytes())
        for t in ('wq', 'wk', 'wv', 'wo'):
            for l in range(L): f.write(requant_fast(get(t, l)).tobytes())
        for l in range(L): f.write(nv(get('ffn_norm', l)).tobytes())
        for t in ('w1', 'w2', 'w3'):
            for l in range(L): f.write(requant_fast(get(t, l)).tobytes())
        f.write(nv(get('final_norm')).tobytes())
        hd = cfg['head_dim']
        f.write(np.zeros(cfg['ctx'] * hd, np.float32).tobytes())
        if not cfg['tied']:
            f.write(requant_fast(get('head')).tobytes())

# ------------------------------------------------------------------
# GGUF (Bonsai: qwen3, Q1_0 weights, F32 norms)

def load_gguf(path, ctx):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from gguf_read import GGUF, GGML_Q1_0
    g = GGUF(path)
    M = g.meta
    arch = M.get('general.architecture')
    if arch not in ('qwen3', 'llama'):
        raise SystemExit('GGUF architecture %s not supported (qwen3, llama)' % arch)
    k = lambda n, d=None: M.get('%s.%s' % (arch, n), d)
    dim, L = int(k('embedding_length')), int(k('block_count'))
    nh = int(k('attention.head_count')); nkv = int(k('attention.head_count_kv', nh))
    hd = int(k('attention.key_length', dim // nh))
    T = g.tensors
    tied = 'output.weight' not in T
    qk = 'blk.0.attn_q_norm.weight' in T
    names = {'attn_norm': 'attn_norm', 'wq': 'attn_q', 'wk': 'attn_k', 'wv': 'attn_v',
             'wo': 'attn_output', 'q_norm': 'attn_q_norm', 'k_norm': 'attn_k_norm',
             'ffn_norm': 'ffn_norm', 'w1': 'ffn_gate', 'w3': 'ffn_up', 'w2': 'ffn_down'}
    def get(n, l=None):
        t = {'emb': 'token_embd.weight', 'head': 'output.weight',
             'final_norm': 'output_norm.weight'}.get(n) or ('blk.%d.%s.weight' % (l, names[n]))
        if T[t][1] == GGML_Q1_0:
            d, bits = g.q1_0(t)
            return ('q1_0', d, bits)
        return g.float(t)
    # tokenizer: GPT-2 byte-level BPE (tokens, merges, token types)
    if M.get('tokenizer.ggml.model') != 'gpt2':
        raise SystemExit('GGUF tokenizer %s not supported (gpt2 BPE)' % M.get('tokenizer.ggml.model'))
    toks = M['tokenizer.ggml.tokens']
    types = M.get('tokenizer.ggml.token_type', [1] * len(toks))
    vocab_n = g.shape('token_embd.weight')[0]
    b2u = bytes_to_unicode()
    ids = {t: i for i, t in enumerate(toks)}
    tokens = [b''] * vocab_n; scores = [-(1 << 30)] * vocab_n; findable = [False] * vocab_n
    special = {}
    for i, t in enumerate(toks[:vocab_n]):
        if int(types[i]) in (3, 4):          # control, user-defined: specials
            tokens[i] = t.encode(); special[t] = i
            continue
        try: tokens[i] = bytes(b2u[c] for c in t)
        except KeyError: tokens[i] = t.encode()
        findable[i] = True
    parts = {}                               # id -> the merges that make it
    for r, mg in enumerate(M.get('tokenizer.ggml.merges', [])):
        a, b = mg.split(' ', 1)
        t = ids.get(a + b)
        if t is not None and t < vocab_n:
            if scores[t] == -(1 << 30): scores[t] = -r
            if a in ids and b in ids: parts.setdefault(t, []).append((ids[a], ids[b]))
    for i in range(vocab_n):
        if not tokens[i]: tokens[i] = b'<unused%d>' % i
    flags = F_ROPE_HALF | F_TOK_BPE if arch == 'qwen3' else F_TOK_BPE
    eos = int(M.get('tokenizer.ggml.eos_token_id', NONE))
    im_s, im_e = special.get('<|im_start|>', NONE), special.get('<|im_end|>', NONE)
    if im_s != NONE: flags |= F_CHAT_QWEN
    yarn = None
    if k('rope.scaling.type') == 'yarn':
        yarn = dict(factor=float(k('rope.scaling.factor')),
                    orig_ctx=float(k('rope.scaling.original_context_length')))
    cfg = dict(dim=dim, hidden_dim=int(k('feed_forward_length')), n_layers=L, n_heads=nh,
               n_kv_heads=nkv, head_dim=hd, vocab=vocab_n,
               ctx=min(ctx, int(k('context_length', ctx))),
               rope_theta=float(k('rope.freq_base', 10000.0)),
               eps=float(k('attention.layer_norm_rms_epsilon', 1e-6)),
               tied=tied, qk_norm=qk, flags=flags, bos=eos, eos=eos,
               eos2=im_e, im_start=im_s, im_end=im_e,
               think=special.get('<think>', NONE), think_end=special.get('</think>', NONE),
               yarn=yarn, merge_parts=parts, special_ids=set(special.values()))
    return cfg, get, tokens, scores, findable

# ------------------------------------------------------------------
# the vocabulary, cut to what machllm can show

def displayable(cp):
    """characters the screen shows or will (ASCII, Latin, punctuation,
    currency, emoji)"""
    return (cp < 0x250 or 0x2000 <= cp < 0x2070 or 0x20a0 <= cp < 0x20d0
            or cp in (0x2122, 0x2212, 0x221e, 0xfe0f)
            or 0x2190 <= cp < 0x2200 or 0x2600 <= cp < 0x27c0 or 0x1f000 <= cp < 0x1fb00)

def vocab_keep(cfg, tokens):
    """the ids to keep: the specials, every single byte, every token
    whose text is all displayable, and every token a merge needs on
    the way to one of those (so displayable text encodes exactly as
    with the whole vocabulary)"""
    keep = set(cfg.get('special_ids', ()))
    for i, t in enumerate(tokens):
        if len(t) == 1: keep.add(i); continue
        try: u = t.decode('utf-8')
        except UnicodeDecodeError: continue
        if all(displayable(ord(c)) for c in u): keep.add(i)
    parts, todo = cfg.get('merge_parts', {}), list(keep)
    while todo:
        for a, b in parts.get(todo.pop(), ()):
            for x in (a, b):
                if x not in keep: keep.add(x); todo.append(x)
    return sorted(keep)

def trim_vocab(cfg, get, tokens, scores, findable, keep):
    """the model with only the ids in keep (in order), renumbered: the
    embedding's and the head's rows and the tokenizer; a token the model
    can no longer write is one it could not show"""
    new = {o: n for n, o in enumerate(keep)}
    idx = np.asarray(keep)
    def sub(x):
        if isinstance(x, tuple): return (x[0], x[1][idx], x[2][idx])
        return np.asarray(x)[idx]
    def get2(n, l=None):
        x = get(n, l) if l is not None else get(n)
        return sub(x) if n in ('emb', 'head') else x
    cfg = dict(cfg, vocab=len(keep))
    for k in ('bos', 'eos', 'eos2', 'im_start', 'im_end', 'think', 'think_end'):
        if cfg.get(k, NONE) != NONE:
            assert cfg[k] in new, k
            cfg[k] = new[cfg[k]]
    return (cfg, get2, [tokens[i] for i in keep], [scores[i] for i in keep],
            [findable[i] for i in keep])

# ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('kind', choices=['gguf', 'llama2c', 'hf'])
    ap.add_argument('inputs', nargs='+')
    ap.add_argument('--ctx', type=int, default=2048)
    ap.add_argument('--name')
    ap.add_argument('--vocab', choices=['display', 'all'], default='display',
                    help='gguf: keep only the tokens machllm can show (display, '
                    'the default; their head rows are all the engine reads) or all')
    ap.add_argument('--dequant-bin', help='llama2c only: also write the 1-bit weights '
                    'back as a float llama2.c checkpoint (to separate quantisation error '
                    'from arithmetic error)')
    a = ap.parse_args()
    out = a.inputs[-1]
    if a.kind == 'gguf':
        cfg, get, tokens, scores, findable = load_gguf(a.inputs[0], a.ctx)
        name = a.name or os.path.splitext(os.path.basename(a.inputs[0]))[0]
        if a.vocab == 'display':
            keep = vocab_keep(cfg, tokens)
            sys.stderr.write('vocabulary: %d of %d tokens kept (--vocab all keeps every one)\n'
                             % (len(keep), len(tokens)))
            cfg, get, tokens, scores, findable = trim_vocab(cfg, get, tokens, scores, findable, keep)
    elif a.kind == 'llama2c':
        cfg, get, tokens, scores, findable = load_llama2c(a.inputs[0], a.inputs[1])
        cfg['ctx'] = min(cfg['ctx'], a.ctx)
        name = a.name or os.path.basename(a.inputs[0])
    else:
        cfg, get, tokens, scores, findable = load_hf(a.inputs[0], a.ctx)
        name = a.name or os.path.basename(os.path.normpath(a.inputs[0]))
    write_image(out, cfg, get, tokens, scores, findable, name)
    if a.dequant_bin:
        write_dequant_l2c(a.dequant_bin, a.inputs[0], cfg, get)

if __name__ == '__main__':
    main()
