R = ../..
test: obj/Vmlm_ui
	cd obj && ./Vmlm_ui && cp out.ppm ..
obj/Vmlm_ui: tb_ui.cpp $(R)/rtl/mlm_ui.v $(R)/rtl/mlm_hid_rom.v
	verilator -O2 --cc --exe --build -Wno-fatal -Wno-WIDTH -Wno-UNOPTFLAT -Wno-CASEINCOMPLETE --public-flat-rw \
		--top-module mlm_ui -GFONT_FILE='"'$(abspath $(R))/rtl/mlm_font8x16.hex'"' \
		-Mdir obj $(R)/rtl/mlm_ui.v $(R)/rtl/mlm_hid_rom.v $(R)/ext/zeitlos/rtl/ext/usb_hid_host/src/usb_hid_host.v tb_ui.cpp
clean:
	rm -rf obj out.ppm
.PHONY: test clean
