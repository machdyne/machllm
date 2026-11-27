#!/bin/sh
# timing closure helper: synthesise MEM_MHZ=$1 and place seeds $2.. in
# parallel; print each seed's clocks and the best seed's critical path.
M=${1:-100}; shift; SEEDS=${*:-1 2}
O=build/sergei_ml2_mem$M
rm -f $O/soc.json
# OVER=52: constrain the 48 MHz input to 52 MHz, i.e. ask the placer for
# 8% more on every derived clock (the reported MHz are then scaled back)
LPF=boards/sergei_ml2/sergei_ml2.lpf
SCALE=1
if [ -n "$OVER" ]; then
	sed "s/FREQUENCY PORT \"CLK_48\" [0-9.]* MH[Zz]/FREQUENCY PORT \"CLK_48\" $OVER MHZ/" $LPF > $O/over.lpf
	LPF=$O/over.lpf; SCALE=$(echo "scale=5; 48 / $OVER" | bc)
fi
make MEM_MHZ=$M fw > /dev/null 2>&1
make MEM_MHZ=$M YOSYS=yowasp-yosys ECPBRAM="python3 tools/ecpbram.py" $O/soc.json > /dev/null 2>&1 || { echo synth failed; exit 1; }
for s in $SEEDS; do
	yowasp-nextpnr-ecp5 --45k --package CABGA256 --speed 6 --lpf $LPF \
		--json $O/soc.json --textcfg $O/s$s.config -l $O/s$s.log --seed $s --timing-allow-fail --ignore-loops $NPR_ARGS > /dev/null 2>&1 &
done
wait
best=; bf=0
for s in $SEEDS; do
	f=$(grep -E "Max frequency for clock '\\\$glbnet\\\$clocks.sys_clk_o'" $O/s$s.log | tail -1 | sed 's/.*: \([0-9.]*\) MHz.*/\1/')
	c=$(grep -E "Max frequency for clock +'\\\$glbnet\\\$clk'" $O/s$s.log | tail -1 | sed 's/.*: \([0-9.]*\) MHz.*/\1/')
	f=$(echo "scale=2; $f * $SCALE / 1" | bc); c=$(echo "scale=2; $c * $SCALE / 1" | bc)
	echo "seed $s: memory $f MHz, cpu $c MHz"
	if [ "$(echo "$f > $bf" | bc)" = 1 ]; then bf=$f; best=$s; fi
done
echo "best seed $best"
awk "/Critical path report for clock '\\\$glbnet\\\$clocks.sys_clk_o'/,/^Info: [0-9.]+ ns logic/" $O/s$best.log | grep -E "Source|Sink" | \
	sed 's/_TRELLIS.*\|_CCU2C.*\|_LUT4.*\|_DPR16X4.*\|_MULT18.*\|_DP16KD.*\|_PFUMX.*\|_L6MUX.*//' | awk '{print $NF}' | uniq | head -10
