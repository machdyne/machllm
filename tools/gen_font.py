#!/usr/bin/env python3
# rtl/mlm_font8x16.hex from Zeitlos's 8x16 font (sw/data/font/font16.mem:
# 96 glyphs from ' ', 16 rows each, one pixel per number, MSB = left).
# Glyph 127 (index 95) becomes a solid block: the screen's cursor.
import sys
src = sys.argv[1] if len(sys.argv) > 1 else 'ext/zeitlos/sw/data/font/font16.mem'
rows = [l.split() for l in open(src) if l.strip()]
assert len(rows) == 96 * 16, len(rows)
out = []
for i, r in enumerate(rows):
    b = 0
    for p in r: b = (b << 1) | (1 if p != '0' else 0)
    if i // 16 == 95: b = 0xff if 1 <= i % 16 <= 14 else 0
    out.append('%02x' % b)
open('rtl/mlm_font8x16.hex', 'w').write('\n'.join(out) + '\n')
