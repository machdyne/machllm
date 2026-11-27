# machllm: engine + DDR3 controller (stream port) + checking DRAM model
#   make test    CPU path self-test, then the whole model through DDR3,
#                token ids compared with mlm_run
ROOT = ../..
MODELS = $(ROOT)/build/models
VFLAGS = -O2 --cc --exe --build -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-WIDTHCONCAT -Wno-fatal \
         -CFLAGS "-O2 -I$(ROOT)/fw/common" -j 2 --top-module tb_mem_top
RTL = tb_mem_top.v mlm_dram_model.v $(ROOT)/rtl/mlm_engine.v $(ROOT)/rtl/mlm_mul16r.v $(ROOT)/rtl/mlm_mul18s.v $(ROOT)/rtl/mem/mlm_ddr3_ctrl.v

obj_dir/Vtb_mem_top: $(RTL) tb_mem.cpp $(ROOT)/fw/common/*.c $(ROOT)/fw/common/*.h
	verilator -I$(ROOT)/rtl $(VFLAGS) $(RTL) tb_mem.cpp $(ROOT)/fw/common/mlm_model.c

PROMPT = "Once upon a time"
define check
	$(ROOT)/ref/mlm_run $(1) -i $(PROMPT) -n $(2) -q 2>/dev/null > /tmp/want_m.txt
	obj_dir/Vtb_mem_top $(1) $(PROMPT) $(2) > /tmp/got_m.txt
	diff -w /tmp/want_m.txt /tmp/got_m.txt && echo "PASS $(1)"
endef

# the controller at a faster clock: same checks, DRAM model
# timed for DDR3 at 2 x MHZ
obj_%/Vtb_mem_top: $(RTL) tb_mem.cpp $(ROOT)/fw/common/*.c $(ROOT)/fw/common/*.h
	verilator -I$(ROOT)/rtl $(VFLAGS) -GCLK_MHZ=$* -CFLAGS -DTB_MHZ=$* -Mdir obj_$* $(RTL) tb_mem.cpp $(ROOT)/fw/common/mlm_model.c

test-fast: obj_64/Vtb_mem_top obj_75/Vtb_mem_top obj_100/Vtb_mem_top
	for f in 64 75 100; do \
	  $(ROOT)/ref/mlm_run $(MODELS)/btiny.mlm -i $(PROMPT) -n 8 -q 2>/dev/null > /tmp/want_f.txt; \
	  obj_$$f/Vtb_mem_top $(MODELS)/btiny.mlm $(PROMPT) 8 > /tmp/got_f$$f.txt; \
	  grep "words/cycle" /tmp/got_f$$f.txt; \
	  diff -w /tmp/want_f.txt /tmp/got_f$$f.txt && echo "PASS btiny, controller at $$f MHz" || exit 1; \
	done

test: obj_dir/Vtb_mem_top
	$(call check,$(MODELS)/s15.mlm,8)
	$(call check,$(MODELS)/btiny.mlm,8)
	ST_GAP=1 obj_dir/Vtb_mem_top $(MODELS)/btiny.mlm $(PROMPT) 8 > /tmp/got_g.txt
	diff -w /tmp/want_m.txt /tmp/got_g.txt && echo 'PASS btiny with stream gap 1'
