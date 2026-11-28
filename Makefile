# machllm
#
#   make                 firmware + bitstream for BOARD (default sergei_ml2)
#   make fw              firmware only
#   make test            all simulations (reference, engine, memory, SD)
#   make ref             the PC reference engine (ref/mlm_run)
#
# Tools: yosys + nextpnr-ecp5 + ecppack (or the yowasp-* packages:
# YOSYS=yowasp-yosys etc.), verilator, riscv64-unknown-elf-gcc.

BOARD ?= sergei_ml2
# placement seed and placer options. With the screen (UI=1): MEM_MHZ=75
# meets timing with seed 3; MEM_MHZ=90 with seed 3 and a stronger timing
# weight: Sergei ML2 memory side 93.0 MHz, CPU 57.6 (54 needed); Mozart
# ML2 91.9 and 56.2 (its earlier seed 6 now reaches 88.1, seed 7 misses
# the CPU's 54).
# Results depend on the nextpnr version (these: yowasp-nextpnr-ecp5).
SEED ?= 3
# the firmware's per-section profiler (:v); off on boards: its RAM is the
# stack's (make PROFILE=1 fw fwbit for a firmware with it)
PROFILE ?= 0
PNR_OPTS ?= $(if $(filter 90,$(MEM_MHZ)),--placer-heap-timingweight 30 --placer-heap-critexp 4,)
# memory-side clock in MHz; the DDR3 clock is twice this. 48 is
# Zeitlos's (DDR3 at 96), one clock domain. 75 or 100 (DDR3 at 150 or
# 200): two domains, the CPU on the PLL's 50 MHz.
MEM_MHZ ?= 48
# the CPU's clock: 48 in the one-domain build, else the PLL's CLKOS2
# (50; 49.09 with DDR3 at 180, whose VCO is 540)
SYS_MHZ = $(if $(filter 48,$(MEM_MHZ)),48,$(if $(filter 90,$(MEM_MHZ)),54,50))
# the screen (DVI, 80 x 25 text) and USB keyboard: on by default; they
# need the two-domain builds (MEM_MHZ 75 or 90), and UI=0 leaves them
# out (one block RAM and ~1,500 LUTs less). The firmware finds out at
# run time (rtl/mlm_ui.v's ID), so it is the same either way.
UI ?= 1
YOSYS ?= yosys
NEXTPNR ?= nextpnr-ecp5
ECPPACK ?= ecppack
OUT = build/$(BOARD)$(if $(filter 48,$(MEM_MHZ)),,_mem$(MEM_MHZ))
Z = ext/zeitlos/rtl

# The ML2 module on either motherboard: the same pins but for the
# microSD card's DAT1/DAT2, which only Sergei wires (Mozart: a 1-bit
# bus, so loading takes about four times as long). One top for both.
TOP = boards/sergei_ml2/top.v
SD_1BIT = 0
ifeq ($(BOARD),sergei_ml2)
DEVICE = 45k
PACKAGE = CABGA256
else ifeq ($(BOARD),mozart_ml2)
DEVICE = 45k
PACKAGE = CABGA256
SD_1BIT = 1
else
$(error BOARD: sergei_ml2 or mozart_ml2)
endif
FW_CFLAGS = $(if $(filter 1,$(SD_1BIT)),-DSD_1BIT,)

RTL_INC = rtl/mlm_sigtab.vh rtl/mlm_exp2tab.vh
RTL = $(TOP) rtl/mlm_soc.v rtl/mlm_dma.v rtl/mlm_vu.v rtl/mlm_engine.v rtl/mlm_mul16r.v rtl/mlm_mul18s.v rtl/mlm_uart.v rtl/sd/mlm_sd.v \
      rtl/mem/mlm_ddr3.v rtl/mem/mlm_ddr3_ctrl.v \
      $(Z)/mem/ddr3_phy_ecp5.v rtl/mem/mlm_ddr3_rdasm.v rtl/mem/mlm_ddr3_clk.v rtl/mem/mlm_pll.v $(Z)/mem/ddr3_phy_init.v \
      $(Z)/clk/pll2.v $(Z)/cpu/zeitlos32ps/zeitlos32ps.v $(Z)/cpu/zeitlos32ps/zeitlos32ps_alu.v \
      $(Z)/cpu/zeitlos32/zeitlos32_decode.v $(Z)/cpu/zeitlos32/zeitlos32_lsu.v \
      $(Z)/cpu/zeitlos32/zeitlos32_muldiv_lean.v \
      rtl/mlm_ui.v rtl/mlm_hid_rom.v rtl/mem/mlm_vpll.v $(Z)/ext/usb_hid_host/src/usb_hid_host.v
RTL_INC += rtl/mlm_font8x16.hex

all: $(OUT)/machllm.bit

fw:
	$(MAKE) -C fw OUT=../$(OUT) SYS_MHZ=$(SYS_MHZ) MEM_MHZ=$(MEM_MHZ) PROFILE=$(PROFILE) CFLAGS_EXTRA="$(FW_CFLAGS)"

$(OUT)/fw.hex: fw/*.c fw/*.h fw/*.S fw/common/*.c fw/common/*.h
	$(MAKE) -C fw OUT=../$(OUT) SYS_MHZ=$(SYS_MHZ) MEM_MHZ=$(MEM_MHZ) PROFILE=$(PROFILE) CFLAGS_EXTRA="$(FW_CFLAGS)"

# The block RAM is synthesised with random contents (ecpbram), which
# the firmware replaces in the placed design: a new firmware then needs
# only `make` (seconds), not synthesis and placement (an hour).
# (yowasp: ECPBRAM="python3 tools/ecpbram.py", which mounts its database)
ECPBRAM ?= ecpbram
RAM_WORDS = 24576
$(OUT)/fw_rand.hex:
	mkdir -p $(OUT)
	$(ECPBRAM) -g $@ -w 32 -d $(RAM_WORDS) -s 1

$(OUT)/fw_pad.hex: $(OUT)/fw.hex
	python3 -c "import sys; w=[l.strip() for l in open(sys.argv[1]) if l.strip()]; \
		assert len(w) <= $(RAM_WORDS); w += ['00000000'] * ($(RAM_WORDS) - len(w)); \
		open(sys.argv[2], 'w').write('\n'.join(w) + '\n')" $< $@

# the build's clock and (random) RAM contents, for boards/*/top.v
$(OUT)/cfg.vh: Makefile
	mkdir -p $(OUT)
	printf '`define SYS_MHZ %s\n`define MEM_MHZ %s\n`define FW_HEX "%s"\n`define UI %s\n' $(SYS_MHZ) $(MEM_MHZ) $(OUT)/fw_rand.hex $(UI) > $@
	$(if $(filter 1,$(SD_1BIT)),printf '`define SD_1BIT\n' >> $@,)

# The design goes through RTLIL before synth_ecp5 (as Zeitlos does for
# DDR3 boards): the PHY's parameterised DQSBUFM blackboxes otherwise make
# yosys re-elaborate the top late and fail.
$(OUT)/soc.json: $(RTL) $(RTL_INC) $(OUT)/fw_rand.hex $(OUT)/cfg.vh
	$(YOSYS) -q -l $(OUT)/synth.log -p "read_verilog -DOSC48 -DMLM_CFG -I rtl -I $(OUT) $(RTL); hierarchy -top top; \
		write_rtlil $(OUT)/elab.il; design -reset; read_rtlil $(OUT)/elab.il; \
		synth_ecp5 -abc9 -top top -json $@"

$(OUT)/soc.config: $(OUT)/soc.json boards/$(BOARD)/$(BOARD).lpf
	$(NEXTPNR) --$(DEVICE) --package $(PACKAGE) --speed 6 --lpf boards/$(BOARD)/$(BOARD).lpf \
		--json $< --textcfg $@ --report $(OUT)/report.txt -l $(OUT)/pnr.log \
		--seed $(SEED) --timing-allow-fail --ignore-loops $(PNR_OPTS)
	@grep -E "Max frequency|Info: +(TRELLIS_COMB|DP16KD|MULT18X18D|TRELLIS_FF):" $(OUT)/pnr.log | tail -6

$(OUT)/machllm.bit: $(OUT)/soc.config $(OUT)/fw_rand.hex $(OUT)/fw_pad.hex
	$(ECPBRAM) -i $< -o $(OUT)/soc_fw.config -f $(OUT)/fw_rand.hex -t $(OUT)/fw_pad.hex
	$(ECPPACK) --compress --freq 2.4 $(OUT)/soc_fw.config $@

# a new firmware into the placed design ($(OUT)/soc.config, synthesised
# with $(OUT)/fw_rand.hex): no synthesis or placement
fwbit: $(OUT)/fw_pad.hex
	test -f $(OUT)/soc.config && test -f $(OUT)/fw_rand.hex
	$(ECPBRAM) -i $(OUT)/soc.config -o $(OUT)/soc_fw.config -f $(OUT)/fw_rand.hex -t $(OUT)/fw_pad.hex
	$(ECPPACK) --compress --freq 2.4 $(OUT)/soc_fw.config $(OUT)/machllm.bit

ref:
	$(MAKE) -C ref

test: ref
	$(MAKE) -C sim/engine test
	$(MAKE) -C sim/mem test
	$(MAKE) -C sim/sd test
	$(MAKE) -C sim/vu test
	$(MAKE) -C sim/rdasm test
	$(MAKE) -C sim/soc test

.PHONY: all fw fwbit ref test
