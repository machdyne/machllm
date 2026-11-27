ROOT = ../..
obj_dir/Vmlm_vu: $(ROOT)/rtl/mlm_vu.v $(ROOT)/rtl/mlm_exp2tab.vh tb_vu.cpp $(ROOT)/fw/common/mlm_model.c $(ROOT)/fw/common/*.h
	verilator --cc --exe --build -O2 -Wno-fatal -Wno-WIDTH -I$(ROOT)/rtl -CFLAGS "-I../$(ROOT)/fw/common -O2" \
		--top-module mlm_vu $(ROOT)/rtl/mlm_vu.v tb_vu.cpp $(ROOT)/fw/common/mlm_model.c -o Vmlm_vu > /dev/null

test: obj_dir/Vmlm_vu
	obj_dir/Vmlm_vu

clean:
	rm -rf obj_dir
