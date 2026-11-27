# sim/rdasm: machllm's ddr3_rdasm against the Zeitlos original
ROOT = ../..
obj_dir/Vtb_rdasm: tb.cpp tb_rdasm.v $(ROOT)/rtl/mem/mlm_ddr3_rdasm.v $(ROOT)/ext/zeitlos/rtl/mem/ddr3_rdasm.v
	sed 's/module ddr3_rdasm/module ddr3_rdasm_zeitlos/' $(ROOT)/ext/zeitlos/rtl/mem/ddr3_rdasm.v > zeitlos_rdasm.v
	verilator --cc --exe --build -Wno-fatal -O2 tb_rdasm.v $(ROOT)/rtl/mem/mlm_ddr3_rdasm.v zeitlos_rdasm.v tb.cpp --top-module tb_rdasm > /dev/null
test: obj_dir/Vtb_rdasm
	obj_dir/Vtb_rdasm
.PHONY: test
