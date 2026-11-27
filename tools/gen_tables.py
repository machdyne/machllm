#!/usr/bin/env python3
# Regenerates fw/common/mlm_exp2tab.h: 2^(-i/256) in Q16, i = 0..256.
vals = [round(65536 * 2 ** (-i / 256)) for i in range(257)]
rows = ["\t" + ", ".join(str(v) for v in vals[i:i + 8]) + "," for i in range(0, 257, 8)]
open("fw/common/mlm_exp2tab.h", "w").write(
    "/* 2^(-i/256) in Q16, i = 0..256; generated, see tools/gen_tables.py */\n"
    "static const uint32_t mlm_exp2tab[257] = {\n" + "\n".join(rows) + "\n};\n")

# fw/common/mlm_sigtab.h: sigmoid(i/64) in Q16, saturating at 65535, i = 0..1024
import math
vals = [min(65535, round(65536 / (1 + math.exp(-i / 64)))) for i in range(1025)]
rows = ["\t" + ", ".join(str(v) for v in vals[i:i + 12]) + "," for i in range(0, 1025, 12)]
open("fw/common/mlm_sigtab.h", "w").write(
    "/* sigmoid(i/64) in Q16 (max 65535), i = 0..1024; generated, see tools/gen_tables.py */\n"
    "static const uint16_t mlm_sigtab[1025] = {\n" + "\n".join(rows) + "\n};\n")

# rtl/mlm_sigtab.vh: the same table for the engine's GLU unit
lines = ["/* sigmoid(i/64), Q16, max 65535: the same table as fw/common/mlm_sigtab.h (tools/gen_tables.py) */",
         "initial begin"] + ["\tsigtab[%d] = 16'd%d;" % (i, v) for i, v in enumerate(vals)] + ["end"]
open("rtl/mlm_sigtab.vh", "w").write("\n".join(lines) + "\n")

# rtl/mlm_exp2tab.vh: the exp2 table for the vector unit's EXPSUM
vals = [round(65536 * 2 ** (-i / 256)) for i in range(257)]
lines = ["/* 2^(-i/256), Q16: the same table as fw/common/mlm_exp2tab.h (tools/gen_tables.py) */",
         "initial begin"] + ["\texp2tab[%d] = 17'd%d;" % (i, v) for i, v in enumerate(vals)] + ["end"]
open("rtl/mlm_exp2tab.vh", "w").write("\n".join(lines) + "\n")
