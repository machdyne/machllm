#!/usr/bin/env python3
# machllm
# Copyright (c) 2025 Lone Dynamics Corporation. All rights reserved.
#
# gguf_read: a minimal GGUF (v2/v3) reader for mlm_convert: metadata and
# tensors, as numpy arrays or raw blocks. Only numpy and the standard
# library; the file is memory-mapped and never executed or evaluated.
#
#   python3 gguf_read.py model.gguf     prints the metadata and tensors

import struct, sys
import numpy as np

# value types
T_U8, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL, T_STR, T_ARR, T_U64, T_I64, T_F64 = range(13)
_SCALAR = {T_U8: '<B', T_I8: '<b', T_U16: '<H', T_I16: '<h', T_U32: '<I', T_I32: '<i',
           T_F32: '<f', T_BOOL: '<?', T_U64: '<Q', T_I64: '<q', T_F64: '<d'}

# tensor types we read (ggml numbering)
GGML_F32, GGML_F16, GGML_BF16, GGML_Q1_0 = 0, 1, 30, 41
QK1_0 = 128                     # Q1_0: fp16 scale, then 128 sign bits LSB first; 1 = +d, 0 = -d
Q1_0_BYTES = 2 + QK1_0 // 8


class GGUF:
    def __init__(self, path):
        self.mm = np.memmap(path, np.uint8, 'r')
        self.p = 0
        magic = bytes(self.mm[0:4])
        if magic != b'GGUF':
            raise ValueError('not a GGUF file')
        self.p = 4
        self.version = self._u('<I')
        if self.version not in (2, 3):
            raise ValueError('GGUF version %d not supported' % self.version)
        nt = self._u('<Q')
        nkv = self._u('<Q')
        self.meta = {}
        for _ in range(nkv):
            k = self._str()
            t = self._u('<I')
            self.meta[k] = self._val(t)
        self.tensors = {}
        for _ in range(nt):
            name = self._str()
            nd = self._u('<I')
            dims = [self._u('<Q') for _ in range(nd)]
            typ = self._u('<I')
            off = self._u('<Q')
            self.tensors[name] = (dims, typ, off)
        align = int(self.meta.get('general.alignment', 32))
        self.data0 = (self.p + align - 1) // align * align

    def _u(self, f):
        n = struct.calcsize(f)
        v = struct.unpack(f, bytes(self.mm[self.p:self.p + n]))[0]
        self.p += n
        return v

    def _str(self):
        n = self._u('<Q')
        s = bytes(self.mm[self.p:self.p + n]).decode('utf-8', 'replace')
        self.p += n
        return s

    def _val(self, t):
        if t in _SCALAR:
            return self._u(_SCALAR[t])
        if t == T_STR:
            return self._str()
        if t == T_ARR:
            et = self._u('<I')
            n = self._u('<Q')
            if et in _SCALAR and et != T_BOOL:
                dt = np.dtype(_SCALAR[et])
                a = np.frombuffer(bytes(self.mm[self.p:self.p + n * dt.itemsize]), dt).copy()
                self.p += n * dt.itemsize
                return a
            return [self._val(et) for _ in range(n)]
        raise ValueError('GGUF value type %d' % t)

    # ggml dims are innermost first: [cols, rows]
    def shape(self, name):
        dims = self.tensors[name][0]
        return tuple(int(d) for d in reversed(dims))

    def raw(self, name):
        dims, typ, off = self.tensors[name]
        n = 1
        for d in dims: n *= int(d)
        if typ == GGML_F32: size = 4 * n
        elif typ in (GGML_F16, GGML_BF16): size = 2 * n
        elif typ == GGML_Q1_0: size = n // QK1_0 * Q1_0_BYTES
        else: raise ValueError('%s: tensor type %d not supported' % (name, typ))
        a = self.data0 + off
        return typ, self.mm[a:a + size]

    def float(self, name):
        """the tensor as float32 [rows, cols] (or [n])"""
        typ, b = self.raw(name)
        shp = self.shape(name)
        if typ == GGML_F32: v = np.frombuffer(b, '<f4').astype(np.float32)
        elif typ == GGML_F16: v = np.frombuffer(b, '<f2').astype(np.float32)
        elif typ == GGML_BF16: v = (np.frombuffer(b, '<u2').astype(np.uint32) << 16).view(np.float32)
        else:
            d, bits = self.q1_0(name)
            v = np.where(bits, 1.0, -1.0).astype(np.float32) * np.repeat(d, QK1_0, -1)
        return v.reshape(shp)

    def q1_0(self, name):
        """a Q1_0 tensor as (scales float32 [rows, cols/128], bits bool [rows, cols])"""
        typ, b = self.raw(name)
        if typ != GGML_Q1_0:
            raise ValueError('%s is not Q1_0' % name)
        rows, cols = self.shape(name)
        blk = np.asarray(b).reshape(rows, cols // QK1_0, Q1_0_BYTES)
        d = blk[..., 0:2].copy().view('<f2')[..., 0].astype(np.float32)
        bits = np.unpackbits(blk[..., 2:], axis=-1, bitorder='little').astype(bool)
        return d, bits.reshape(rows, cols)


if __name__ == '__main__':
    g = GGUF(sys.argv[1])
    for k, v in g.meta.items():
        if isinstance(v, (list, np.ndarray)):
            print('%-44s [%d] %s' % (k, len(v), str(list(v[:4]))[:80]))
        else:
            print('%-44s %s' % (k, str(v)[:100]))
    for n, (dims, typ, off) in g.tensors.items():
        print('%-36s %-20s type %d' % (n, g.shape(n), typ))
