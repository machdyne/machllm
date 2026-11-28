# machllm: the whole SOC running its firmware (Verilator)
#   make test   boot, load TINY.MLM from a FAT32 card image, type a prompt;
#               the generated text must equal mlm_run's
ROOT = ../..
Z = $(ROOT)/ext/zeitlos/rtl
RTL = $(ROOT)/rtl/mlm_soc.v $(ROOT)/rtl/mlm_dma.v $(ROOT)/rtl/mlm_ui.v $(ROOT)/rtl/mlm_vu.v $(ROOT)/rtl/mlm_engine.v $(ROOT)/rtl/mlm_mul16r.v $(ROOT)/rtl/mlm_mul18s.v $(ROOT)/rtl/mlm_uart.v $(ROOT)/rtl/sd/mlm_sd.v \
      $(ROOT)/rtl/mem/mlm_ddr3_ctrl.v $(ROOT)/sim/mem/mlm_dram_model.v \
      $(Z)/cpu/zeitlos32ps/zeitlos32ps.v $(Z)/cpu/zeitlos32ps/zeitlos32ps_alu.v \
      $(Z)/cpu/zeitlos32/zeitlos32_decode.v $(Z)/cpu/zeitlos32/zeitlos32_lsu.v \
      $(Z)/cpu/zeitlos32/zeitlos32_muldiv_lean.v
FW = $(ROOT)/build/sim/fw.hex
VFLAGS = -O2 --cc --exe --build -Wno-fatal -Wno-WIDTH -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-WIDTHCONCAT \
         -Wno-UNOPTFLAT -Wno-CASEINCOMPLETE -Wno-PINMISSING --top-module mlm_soc \
         -GSIM_DDR3=1 -GINIT_FILE='"$(abspath $(FW))"' -CFLAGS -O2 -j 2
IMG = $(ROOT)/build/soc_card.img
MODEL = $(ROOT)/build/models/tiny.mlm

obj_dir/Vmlm_soc: $(RTL) tb_soc.cpp ../sd/sdcard_model.h $(FW)
	verilator -I$(ROOT)/rtl $(VFLAGS) $(RTL) tb_soc.cpp

# two clock domains: CPU at 50 MHz, memory side at 75
FW2 = $(ROOT)/build/sim2c/fw.hex
obj_2c/Vmlm_soc: $(RTL) tb_soc.cpp ../sd/sdcard_model.h $(FW2)
	verilator -I$(ROOT)/rtl $(subst $(abspath $(FW)),$(abspath $(FW2)),$(VFLAGS)) -GTWO_CLK=1 -GSYS_MHZ=50 -GMEM_MHZ=75 \
		-CFLAGS "-DTB_TWO_CLK -DTB_CPU_MHZ=50 -DTB_MEM_MHZ=75 -DTB_UART_DIV=50" -Mdir obj_2c $(RTL) tb_soc.cpp

# the same with the screen and keyboard (UI = 1); at the end the screen's
# cells go to ui_screen.txt
obj_ui/Vmlm_soc: $(RTL) tb_soc.cpp ../sd/sdcard_model.h $(FW2) $(ROOT)/rtl/mlm_hid_rom.v
	verilator -I$(ROOT)/rtl $(subst $(abspath $(FW)),$(abspath $(FW2)),$(VFLAGS)) -GTWO_CLK=1 -GSYS_MHZ=50 -GMEM_MHZ=75 -GUI=1 \
		--public-flat-rw +define+MLM_FONT_FILE='"$(abspath $(ROOT))/rtl/mlm_font8x16.hex"' \
		+define+MLM_HID_ROM='"$(abspath $(Z))/ext/usb_hid_host/src/usb_hid_host_rom.hex"' \
		-CFLAGS "-DTB_TWO_CLK -DTB_CPU_MHZ=50 -DTB_MEM_MHZ=75 -DTB_UART_DIV=50 -DTB_UI" -Mdir obj_ui \
		$(RTL) $(ROOT)/rtl/mlm_hid_rom.v $(Z)/ext/usb_hid_host/src/usb_hid_host.v tb_soc.cpp

$(FW2): $(ROOT)/fw/*.c $(ROOT)/fw/*.h $(ROOT)/fw/common/*.c $(ROOT)/fw/common/*.h
	$(MAKE) -C $(ROOT)/fw sim2c

test-2c: obj_2c/Vmlm_soc $(IMG)
	obj_2c/Vmlm_soc $(IMG) "$(PROMPT)" | tee /tmp/soc_out2.txt
	$(ROOT)/ref/mlm_run $(MODEL) -i "$(PROMPT)" -n 24 > /tmp/soc_want.txt 2>/dev/null
	python3 compare.py /tmp/soc_out2.txt /tmp/soc_want.txt "$(PROMPT)"

$(FW): $(ROOT)/fw/*.c $(ROOT)/fw/*.h $(ROOT)/fw/common/*.c $(ROOT)/fw/common/*.h
	$(MAKE) -C $(ROOT)/fw sim

$(MODEL):
	python3 $(ROOT)/tools/mlm_synth.py llama2c tiny $(ROOT)/build/models/tiny.bin
	python3 $(ROOT)/tools/mlm_convert.py llama2c $(ROOT)/build/models/tiny.bin \
		$(ROOT)/../src/llama2.c/tokenizer.bin $@

$(IMG): $(MODEL)
	rm -f $@; truncate -s 40M $@
	mkfs.fat -F 32 -n MACHLLM $@ > /dev/null
	mcopy -i $@ $(MODEL) ::TINY.MLM

PROMPT = Once upon a time
test: obj_dir/Vmlm_soc $(IMG)
	obj_dir/Vmlm_soc $(IMG) "$(PROMPT)" | tee /tmp/soc_out.txt
	$(ROOT)/ref/mlm_run $(MODEL) -i "$(PROMPT)" -n 24 > /tmp/soc_want.txt 2>/dev/null
	python3 compare.py /tmp/soc_out.txt /tmp/soc_want.txt "$(PROMPT)"
