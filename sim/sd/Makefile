# machllm: SD card (C++ model, FAT32 image) -> SD host -> DDR3 -> memory
ROOT = ../..
VFLAGS = -O2 --cc --exe --build -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-WIDTHCONCAT -Wno-fatal \
         -CFLAGS "-O2 -I$(ROOT)/fw/common" -j 2 --top-module tb_sd_top
RTL = tb_sd_top.v $(ROOT)/rtl/sd/mlm_sd.v $(ROOT)/rtl/mem/mlm_ddr3_ctrl.v $(ROOT)/sim/mem/mlm_dram_model.v
IMG = $(ROOT)/build/card.img
MODEL = $(ROOT)/build/models/qtinyq4.mlm

obj_dir/Vtb_sd_top: $(RTL) tb_sd.cpp sdcard_model.h $(ROOT)/fw/common/sd.c $(ROOT)/fw/common/sd.h
	verilator $(VFLAGS) $(RTL) tb_sd.cpp $(ROOT)/fw/common/sd.c

$(IMG): $(MODEL)
	rm -f $@; truncate -s 64M $@
	mkfs.fat -F 32 -n MACHLLM $@ > /dev/null
	echo "machllm" > /tmp/readme.txt; mcopy -i $@ /tmp/readme.txt ::README.TXT
	mcopy -i $@ $(MODEL) ::QTINY.MLM

test: obj_dir/Vtb_sd_top $(IMG)
	obj_dir/Vtb_sd_top $(IMG) QTINY.MLM $(MODEL)
