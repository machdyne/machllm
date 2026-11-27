#!/usr/bin/env python3
# The text the firmware printed after the prompt must be the text
# mlm_run prints (both greedy, both stop at 24 new tokens or EOS).
import sys
got = open(sys.argv[1], errors='replace').read()
want = open(sys.argv[2], errors='replace').read()
prompt = sys.argv[3]
i = got.rfind('> ' + prompt)
g = got[i + 2 + len(prompt):] if i >= 0 else ''
g = g[g.find(prompt) + len(prompt):] if prompt in g else g
g = g.split('\n[')[0]
w = want[want.find(prompt) + len(prompt):].rstrip('\n')
# the firmware wraps lines at word boundaries (:w): compare the words
import re
g = re.sub(r'\s+', ' ', g); w = re.sub(r'\s+', ' ', w)
if g.strip() == w.strip() and w.strip():
    print('PASS: firmware on the SOC generates exactly what mlm_run generates (%d chars)' % len(w.strip()))
    sys.exit(0)
print('FAIL\n got: %r\nwant: %r' % (g[:300], w[:300]))
sys.exit(1)
