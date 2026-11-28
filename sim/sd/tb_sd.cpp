/*
 * machllm
 * Copyright (c) 2025 Lone Dynamics Corporation. All rights reserved.
 *
 * tb_sd: card image -> SD host (4-bit) -> DDR3 block writes -> memory.
 * Runs fw/common/sd.c's init, mounts the FAT image, lists the root,
 * loads the named file and compares it with the file on disk.
 *
 *   tb_sd card.img NAME.MLM file.mlm
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "Vtb_sd_top.h"
#include "svdpi.h"
#include "verilated.h"
#include "sdcard_model.h"
extern "C" {
#include "../../fw/common/sd.h"
}

#define MEM_SIZE (512u << 20)
static uint8_t *mem;
static Vtb_sd_top *top;
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
{ fprintf(stderr, "DRAM MODEL failure %d at %lld ps\n", code, t); exit(3); }

static void tick()
{
	int cmd_host = top->sd_cmd_oe ? top->sd_cmd_o : 1;
	card.step(top->sd_sck, cmd_host);
	top->sd_cmd_i = top->sd_cmd_oe ? top->sd_cmd_o : card.cmd_out;
	top->sd_dat_i = card.dat_out;
	top->clk = 0; top->eval();
	top->clk = 1; top->eval();
	cycles++;
}
static uint32_t bus(bool we, int reg, uint32_t v)
{
	top->bus_req = 1; top->bus_we = we; top->bus_adr = reg; top->bus_wdat = v;
	do tick(); while (!top->bus_ack);
	uint32_t r = top->bus_rdat;
	top->bus_req = 0;
	tick();
	return r;
}
extern "C" uint32_t plat_sd_rd(int reg) { return bus(false, reg, 0); }
extern "C" void plat_sd_wr(int reg, uint32_t v) { bus(true, reg, v); }
extern "C" uint8_t *plat_mem(void) { return mem; }

struct find { const char *want; uint32_t clus, size; int found; };
static int cb(void *ctx, const char *n, uint32_t c, uint32_t sz)
{
	find *f = (find *)ctx;
	fprintf(stderr, "  %-12s %9u bytes, cluster %u\n", n, sz, c);
	if (!strcmp(n, f->want)) { f->clus = c; f->size = sz; f->found = 1; }
	return 0;
}

int main(int argc, char **argv)
{
	Verilated::commandArgs(argc, argv);
	FILE *fi = fopen(argv[1], "rb");
	fseek(fi, 0, SEEK_END); long n = ftell(fi); fseek(fi, 0, SEEK_SET);
	card.img.resize(n);
	if (fread(card.img.data(), 1, n, fi) != (size_t)n) return 1;
	fclose(fi);
	mem = (uint8_t *)calloc(1, MEM_SIZE);
	top = new Vtb_sd_top;
	svSetScope(svGetScopeFromName("TOP.tb_sd_top.dram"));
	top->rst = 1; for (int i = 0; i < 4; i++) tick(); top->rst = 0;
	while (!top->init_done) tick();

	sd_t sd; fat_t fat;
	/* TB_SD_DIV: the data clock divider (0x81: three cycles a period) */
	int rc = sd_init(&sd, getenv("TB_SD_DIV") ? strtoul(getenv("TB_SD_DIV"), 0, 0) : 0);
	fprintf(stderr, "sd_init %d: ccs %d rca %04x wide %d, %llu cycles\n", rc, sd.ccs, sd.rca, sd.wide,
	        (unsigned long long)cycles);
	if (rc) return 1;
	rc = fat_mount(&fat, &sd, 0x1ff00000, 0);
	fprintf(stderr, "fat_mount %d: fat32 %d spc %u\n", rc, fat.fat32, fat.spc);
	if (rc) return 1;
	find f = { argv[2], 0, 0, 0 };
	fat_list(&fat, cb, &f);
	if (!f.found) { fprintf(stderr, "%s not found\n", argv[2]); return 1; }
	uint64_t c0 = cycles;
	int32_t got = fat_load(&fat, f.clus, f.size, 0, 0);
	uint64_t dc = cycles - c0;
	fprintf(stderr, "fat_load %d bytes in %llu cycles: %.2f MB/s at 48 MHz; %llu blocks, %d card errors\n",
	        got, (unsigned long long)dc, got / (dc / 48e6) / 1e6, (unsigned long long)card.blocks, card.errors);
	FILE *fm = fopen(argv[3], "rb");
	std::vector<uint8_t> want(f.size);
	if (fread(want.data(), 1, f.size, fm) != f.size) return 1;
	fclose(fm);
	if (got != (int32_t)f.size || memcmp(want.data(), mem, f.size) || card.errors) {
		for (uint32_t i = 0; i < f.size; i++)
			if (want[i] != mem[i]) { fprintf(stderr, "first difference at %u\n", i); break; }
		fprintf(stderr, "FAIL\n");
		return 1;
	}
	fprintf(stderr, "PASS: file loaded intact\n");
	return 0;
}
