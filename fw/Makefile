# machllm firmware (RV32IM, runs from block RAM)
CROSS ?= riscv64-unknown-elf-
CC = $(CROSS)gcc
BUILD ?= $(shell git -C .. describe --always --dirty 2>/dev/null || echo dev)
CFLAGS = -march=rv32im -mabi=ilp32 -O2 -flto -ffreestanding -fno-builtin-printf -nostdlib \
         -Wall -Wextra -Wno-unused-parameter -I. -Icommon \
         -DMLM_MAX_DIM=2048 -DMLM_MAX_SCRATCH=4096 -DMLM_MAX_HEADS=32 -DMLM_MAX_CTX=512 \
         -DMLM_BUILD_STR='"$(BUILD)"' -DMLM_PROFILE -DSYS_MHZ=$(SYS_MHZ) -DMEM_MHZ=$(MEM_MHZ) $(CFLAGS_EXTRA)
SRC = crt0.S main.c libc.c common/mlm_model.c common/sd.c
OUT ?= ../build/sergei_ml2
SYS_MHZ ?= 48
MEM_MHZ ?= $(SYS_MHZ)

all: $(OUT)/fw.hex

$(OUT)/fw.elf: $(SRC) common/*.h hw.h link.ld
	mkdir -p $(OUT)
	$(CC) $(CFLAGS) -T link.ld -o $@ $(SRC) -lgcc
	$(CROSS)size $@

$(OUT)/fw.bin: $(OUT)/fw.elf
	$(CROSS)objcopy -O binary $< $@

# one 32-bit word per line, the whole RAM (96 KB)
$(OUT)/fw.hex: $(OUT)/fw.bin
	python3 -c "import sys,struct; b=open('$<','rb').read(); b+=bytes((-len(b))%4); \
	w=struct.unpack('<%dI'%(len(b)//4),b); assert len(w)<=24576, 'firmware too big'; \
	open('$@','w').write('\n'.join('%08x'%x for x in w)+'\n')"

clean:
	rm -f $(OUT)/fw.*

# firmware for sim/soc: no DDR3 training (the model has no PHY), UART at the real divider
sim:
	$(MAKE) OUT=../build/sim CFLAGS_EXTRA="-DSIM -DSIM_UART_DIV=48" all

# the same for the two-clock SOC simulation (CPU 50 MHz, memory side 75)
sim2c:
	$(MAKE) OUT=../build/sim2c SYS_MHZ=50 MEM_MHZ=75 CFLAGS_EXTRA="-DSIM -DSIM_UART_DIV=50" all
