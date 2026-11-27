/*
 * machllm
 * Copyright (c) 2025 Lone Dynamics Corporation. All rights reserved.
 *
 * tb_soc: the whole system with its firmware. rtl/mlm_soc.v (SIM_DDR3:
 * the DDR3 controller against sim/mem/mlm_dram_model.v), the SD card
 * model holding a FAT32 image, and the serial port decoded here. The
 * firmware boots, loads the model from the card, and the testbench types
 * a prompt; the text that comes back is printed.
 *
 *   tb_soc card.img "prompt" [max cycles] [n tokens]
 *   tb_soc @model.mlm ...   no card: the image is put in memory directly
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include "Vmlm_soc.h"
#include "svdpi.h"
#include "verilated.h"
#include "../sd/sdcard_model.h"

#define MEM_SIZE (512u << 20)
/* cycles per bit: the real 1 Mbaud divider (48 at 48 MHz, 50 at 50) */
#ifndef TB_UART_DIV
#define TB_UART_DIV 48
#endif
#define DIV TB_UART_DIV
static uint8_t *mem;
static Vmlm_soc *top;
static SdCard card;
static uint64_t cycles;

extern "C" int dram_rd32(int blk, int word)
{ uint32_t v; memcpy(&v, mem + (uint64_t)(uint32_t)blk * 16 + 4 * word, 4); return (int)v; }
extern "C" void dram_wr32(int blk, int word, int val, int mask)
{
	uint8_t *p = mem + (uint64_t)(uint32_t)blk * 16 + 4 * word;
	for (int i = 0; i < 4; i++) if (!(mask & (1 << i))) p[i] = (uint8_t)(val >> (8 * i));
}
extern "C" void dram_fail(int code, long long t)
{ fprintf(stderr, "\nDRAM MODEL failure %d at %lld ps (cycle %llu)\n", code, t, (unsigned long long)cycles); exit(3); }

// UART receive (from the SOC) and transmit (to it)
static int rx_state = -1, rx_cnt, rx_byte;
static std::string out, typed;
static size_t tx_pos;
static int tx_bit = -1, tx_cnt;

static long framing_errors = 0;
static void uart_step()
{
	int line = top->uart_tx;
	if (rx_state < 0) {
		if (!line) { rx_state = 0; rx_cnt = DIV + DIV / 2; rx_byte = 0; }
	} else if (--rx_cnt == 0) {
		if (rx_state < 8) { rx_byte |= line << rx_state; rx_state++; rx_cnt = DIV; }
		else {
			rx_state = -1;
			/* the stop bit's centre: a real receiver needs it high */
			if (!line) framing_errors++;
			if (rx_byte != '\r') { putchar(rx_byte); fflush(stdout); out += (char)rx_byte; }
		}
	}
	// to the SOC
	if (tx_bit < 0 && tx_pos < typed.size()) { tx_bit = 0; tx_cnt = DIV; }
	if (tx_bit >= 0) {
		int c = (unsigned char)typed[tx_pos];
		top->uart_rx = tx_bit == 0 ? 0 : tx_bit <= 8 ? (c >> (tx_bit - 1)) & 1 : 1;
		if (--tx_cnt == 0) { tx_cnt = DIV; if (++tx_bit == 10) { tx_bit = -1; tx_pos++; } }
	} else top->uart_rx = 1;
}

static void sd_pins()
{
	int cmd_host = top->sd_cmd_oe ? top->sd_cmd_o : 1;
	card.step(top->sd_sck, cmd_host);
	top->sd_cmd_i = top->sd_cmd_oe ? top->sd_cmd_o : card.cmd_out;
	top->sd_dat_i = card.dat_out;
}

#ifndef TB_TWO_CLK
static void tick()
{
	sd_pins();
	uart_step();
	top->clk = 0; top->eval();
	top->clk = 1; top->eval();
	cycles++;
}
#else
/* two clocks: the CPU's clk at TB_CPU_MHZ, the memory side's mclk at
 * TB_MEM_MHZ (the SD host is on clk); one call = one clk cycle */
static const uint64_t HS = 500000 / TB_CPU_MHZ, HF = 500000 / TB_MEM_MHZ;   /* half periods, ps */
static uint64_t t_s = HS, t_f = HF;
static void tick()
{
	for (;;) {
		uint64_t t = t_s < t_f ? t_s : t_f;
		bool es = (t == t_s), ef = (t == t_f);
		if (ef) { t_f += HF; top->mclk = !top->mclk; }
		if (es) { t_s += HS; if (!top->clk) { uart_step(); sd_pins(); } top->clk = !top->clk; }
		top->mrst = top->rst;
		top->eval();
		if (es && top->clk) { cycles++; return; }
	}
}
#endif

int main(int argc, char **argv)
{
	Verilated::commandArgs(argc, argv);
	mem = (uint8_t *)calloc(1, MEM_SIZE);
	bool backdoor = argv[1][0] == '@';
	FILE *fi = fopen(argv[1] + (backdoor ? 1 : 0), "rb");
	if (!fi) { perror(argv[1]); return 1; }
	fseek(fi, 0, SEEK_END); long n = ftell(fi); fseek(fi, 0, SEEK_SET);
	if (backdoor) {
		if (fread(mem, 1, n, fi) != (size_t)n) return 1;
		card.img.resize(512);     /* a card that answers nothing useful */
		card.ccs = true;
	} else {
		card.img.resize(n);
		if (fread(card.img.data(), 1, n, fi) != (size_t)n) return 1;
	}
	fclose(fi);
	// lines to type, one at each "> ": a setting, then the prompt
	std::string lines[4] = { getenv("TB_CMD2") ? getenv("TB_CMD2") : ":help", std::string(":n ") + (argc > 4 ? argv[4] : "24"),
	                         getenv("TB_CMD") ? getenv("TB_CMD") : getenv("TB_VERBOSE") ? ":v" : ":n " + std::string(argc > 4 ? argv[4] : "24"),
	                         argc > 2 ? argv[2] : "Once upon a time" };
	uint64_t maxc = argc > 3 ? strtoull(argv[3], 0, 0) : 400000000ull;
	top = new Vmlm_soc;
	svSetScope(svGetScopeFromName("TOP.mlm_soc.sim_ddr3.dram"));
	top->rst = 1; top->uart_rx = 1;
	for (int i = 0; i < 8; i++) tick();
	top->rst = 0;
	int phase = 0;
	size_t mark = 0;
	while (cycles < maxc) {
		tick();
		if (phase < 4 && out.size() > mark + 1 && out.compare(out.size() - 2, 2, "> ") == 0) {
			typed = lines[phase] + "\r"; tx_pos = 0; phase++; mark = out.size();
		} else if (phase == 4 && out.size() > mark + 4 && out.compare(out.size() - 2, 2, "> ") == 0) {
			fprintf(stderr, "\n[tb_soc: done after %llu cycles, %llu card blocks]\n",
			        (unsigned long long)cycles, (unsigned long long)card.blocks);
			if (card.errors) {
				fprintf(stderr, "[tb_soc: SD CARD MODEL ERRORS: %d]\n", card.errors);
				return 5;
			}
			if (framing_errors) {
				fprintf(stderr, "[tb_soc: UART FRAMING ERRORS: %ld bytes without a stop bit]\n", framing_errors);
				return 4;
			}
			return 0;
		}
	}
	fprintf(stderr, "\n[tb_soc: stopped at %llu cycles]\n", (unsigned long long)cycles);
	return 2;
}
