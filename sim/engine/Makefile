# machllm: the RTL engine running the whole model (Verilator)
#   make            build
#   make test       synthetic 1-bit models (tools/test_models.sh) against mlm_run
ROOT = ../..
MODELS = $(ROOT)/build/models
VFLAGS = -O2 --cc --exe --build -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-fatal \
         -CFLAGS "-O2 -I$(ROOT)/fw/common" -j 2

obj_dir/Vmlm_engine: $(ROOT)/rtl/mlm_engine.v $(ROOT)/rtl/mlm_mul16r.v $(ROOT)/rtl/mlm_mul18s.v tb_engine.cpp $(ROOT)/fw/common/*.c $(ROOT)/fw/common/*.h
	verilator -I$(ROOT)/rtl $(VFLAGS) $(ROOT)/rtl/mlm_engine.v $(ROOT)/rtl/mlm_mul16r.v $(ROOT)/rtl/mlm_mul18s.v tb_engine.cpp $(ROOT)/fw/common/mlm_model.c \
		--top-module mlm_engine

PROMPT = "Once upon a time"
define check
	$(ROOT)/ref/mlm_run $(1) -i $(PROMPT) -n $(2) -q 2>/dev/null > /tmp/want.txt
	obj_dir/Vmlm_engine $(1) $(PROMPT) $(2) $(3) > /tmp/got.txt
	diff -w /tmp/want.txt /tmp/got.txt && echo "PASS $(1) $(3)"
endef

test: obj_dir/Vmlm_engine
	$(call check,$(MODELS)/s15.mlm,12,14 10)
	$(call check,$(MODELS)/qtiny.mlm,12,10 0)
	$(call check,$(MODELS)/btiny.mlm,8,20 20)
