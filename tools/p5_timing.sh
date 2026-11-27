#!/bin/sh
# Phase 5 timing experiment: DDR3 controller + PHY (+ engine) with the
# system clock at EDGE/2. Usage: tools/p5_timing.sh EDGE_MHZ ENGINE(0|1)
# Needs yowasp-yosys / yowasp-nextpnr-ecp5 (or YOSYS / NEXTPNR set).
set -e
EDGE=${1:-150}; ENG=${2:-0}
YOSYS=${YOSYS:-yowasp-yosys}; NEXTPNR=${NEXTPNR:-yowasp-nextpnr-ecp5}
Z=ext/zeitlos/rtl
O=build/p5/e${EDGE}_g${ENG}
mkdir -p $O
SYS=$((EDGE / 2))
$YOSYS -q -l $O/synth.log -p "read_verilog -DOSC48 -I rtl boards/p5/p5_top.v rtl/mem/mlm_pll.v \
	rtl/mlm_engine.v rtl/mlm_mul16r.v rtl/mlm_mul18s.v rtl/mem/mlm_ddr3.v rtl/mem/mlm_ddr3_ctrl.v \
	$Z/mem/ddr3_phy_ecp5.v rtl/mem/mlm_ddr3_rdasm.v $Z/mem/ddr3_phy_init.v; \
	chparam -set EDGE_MHZ $EDGE -set ENGINE $ENG top; hierarchy -top top; \
	write_rtlil $O/elab.il; design -reset; read_rtlil $O/elab.il; \
	synth_ecp5 -abc9 -top top -json $O/p5.json"
$NEXTPNR --45k --package CABGA256 --speed 6 --json $O/p5.json --lpf boards/sergei_ml2/sergei_ml2.lpf \
	--lpf-allow-unconstrained --freq $SYS --seed 1 --textcfg $O/p5.config > $O/pnr.log 2>&1 || true
grep -E "Max frequency for clock|ERROR" $O/pnr.log | tail -4
