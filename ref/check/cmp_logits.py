#!/usr/bin/env python3
# Compare two "pos N in T[ next X]: id:logit ..." dumps (float reference vs
# mlm_run -d): top-1 agreement and logit error on shared top-5 entries.
import re, sys
def parse(path):
    out = []
    for l in open(path):
        if not l.startswith('pos'): continue
        body = l.split(':', 1)[1]
        out.append([(int(a), float(b)) for a, b in re.findall(r'(\d+):([-\d.]+)', body)])
    return out
f, i = parse(sys.argv[1]), parse(sys.argv[2])
agree = sum(a[0][0] == b[0][0] for a, b in zip(f, i))
md, n, sd = 0.0, 0, 0.0
for a, b in zip(f, i):
    fa = dict(a)
    for k, v in b:
        if k in fa: d = abs(fa[k] - v); md = max(md, d); sd += d; n += 1
print('positions %d, top-1 agree %d (%.0f%%), logit error on shared top-5: max %.4f mean %.4f'
      % (min(len(f), len(i)), agree, 100.0 * agree / max(1, min(len(f), len(i))), md, sd / max(n, 1)))
