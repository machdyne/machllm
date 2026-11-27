/*
 * machllm
 * Copyright (c) 2025 Lone Dynamics Corporation. All rights reserved.
 *
 * SD card, native mode, through rtl/sd/mlm_sd.v; and a FAT reader that
 * loads a whole file into main memory in contiguous runs. Shared by the
 * firmware and the testbench (sim/sd), like the model code.
 */

#include "sd.h"

#define R_DIV 0
#ifndef SYS_MHZ
#define SYS_MHZ 48
#endif
#ifndef MEM_MHZ
#define MEM_MHZ SYS_MHZ
#endif
/* identification clock: SYS_MHZ / (2 (div + 1)) <= 400 kHz (59 at 48
 * MHz); the SD host runs on the CPU's clock */
#define SD_INIT_DIV ((SYS_MHZ * 1000 + 799) / 800 - 1)
#define R_ARG 1
#define R_CMD 2
#define R_STATUS 3
#define R_RESP0 4
#define R_DADDR 8
#define R_DBLOCKS 9
#define R_DDONE 10
#define R_IDLE 11

#define ST_CMD_BUSY  (1u << 0)
#define ST_CMD_TMO   (1u << 1)
#define ST_CMD_CRC   (1u << 2)
#define ST_DAT_BUSY  (1u << 3)
#define ST_DAT_CRC   (1u << 4)
#define ST_DAT_TMO   (1u << 5)
#define ST_Q_BUSY    (1u << 7)

#define RSP_NONE 0
#define RSP_48   (1u << 8)
#define RSP_136  (2u << 8)
#define RSP_BUSY (1u << 10)
#define RSP_NOCRC (1u << 11)

static int cmd(uint32_t idx, uint32_t arg, uint32_t flags, uint32_t *resp)
{
	plat_sd_wr(R_ARG, arg);
	plat_sd_wr(R_CMD, idx | flags);
	uint32_t st, n = 0;
	while ((st = plat_sd_rd(R_STATUS)) & ST_CMD_BUSY)
		if (++n > 20000000u) return -100;
	if (st & ST_CMD_TMO) return -1;
	if ((st & ST_CMD_CRC) && !(flags & RSP_NOCRC)) return -2;
	if (resp) *resp = plat_sd_rd(R_RESP0);
	return 0;
}

static void idle_clocks(uint32_t n)
{
	plat_sd_wr(R_IDLE, n);
	/* the clocks take a few microseconds; a status read is enough of a wait */
	for (uint32_t i = 0; i < n / 2 + 4; i++) (void)plat_sd_rd(R_STATUS);
}

int sd_init(sd_t *s, uint32_t fast_div)
{
	uint32_t r;
	s->ok = 0;
	plat_sd_wr(R_DIV, SD_INIT_DIV);         /* <= 400 kHz */
	idle_clocks(80);
	cmd(0, 0, RSP_NONE, 0);
	int v2 = (cmd(8, 0x1aa, RSP_48, &r) == 0) && ((r & 0xfff) == 0x1aa);
	uint32_t ocr = 0;
	for (int i = 0; i < 2000; i++) {
		if (cmd(55, 0, RSP_48, 0)) return -10;
		if (cmd(41, v2 ? 0x40ff8000u : 0x00ff8000u, RSP_48 | RSP_NOCRC, &ocr)) return -11;
		if (ocr & 0x80000000u) break;
	}
	if (!(ocr & 0x80000000u)) return -12;
	s->ccs = (ocr >> 30) & 1;
	if (cmd(2, 0, RSP_136 | RSP_NOCRC, 0)) return -13;
	if (cmd(3, 0, RSP_48, &r)) return -14;
	s->rca = r >> 16;
	if (cmd(7, s->rca << 16, RSP_48 | RSP_BUSY, 0)) return -15;
	if (cmd(55, s->rca << 16, RSP_48, 0)) return -16;
	if (cmd(6, 2, RSP_48, 0)) return -17;    /* 4-bit bus */
	s->wide = 1;
	if (!s->ccs && cmd(16, 512, RSP_48, 0)) return -18;
	plat_sd_wr(R_DIV, fast_div);
	s->div = fast_div;
	s->ok = 1;
	return 0;
}

int sd_last_err;          /* the last failed read: code, and the host's status bits */
uint32_t sd_last_status;

uint32_t sd_last_r1 = ~0u;     /* the card's status (CMD13) when a read last failed */

/* until the card is back in the transfer state (after CMD12, or an
 * error): CMD13 until its CURRENT_STATE is tran (4) */
static int wait_tran(sd_t *s)
{
	uint32_t r = 0;
	for (int i = 0; i < 2000; i++) {
		if (cmd(13, s->rca << 16, RSP_48, &r) == 0 && ((r >> 9) & 15) == 4) return 0;
		idle_clocks(8);
	}
	sd_last_r1 = r;
	return -50;
}

static int sd_read1(sd_t *s, uint32_t lba, uint32_t n, uint32_t mem_addr);
int sd_read(sd_t *s, uint32_t lba, uint32_t n, uint32_t mem_addr)
{
	int rc = sd_read1(s, lba, n, mem_addr);
	idle_clocks(16);        /* the card's 8 clocks after the transfer (and after an error) */
	if (rc) {
		sd_last_err = rc; sd_last_status = plat_sd_rd(R_STATUS);
		uint32_t r = ~0u;
		if (cmd(13, s->rca << 16, RSP_48, &r)) r = ~0u;
		sd_last_r1 = r;
	}
	return rc;
}

static int sd_read1(sd_t *s, uint32_t lba, uint32_t n, uint32_t mem_addr)
{
	while (n) {
		uint32_t k = n > 65535 ? 65535 : n;
		plat_sd_wr(R_DADDR, mem_addr);
		plat_sd_wr(R_DBLOCKS, k | (s->wide ? (1u << 16) : 0));
		int rc = cmd(k == 1 ? 17 : 18, s->ccs ? lba : lba * 512, RSP_48, 0);
		if (rc) return rc - 20;
		uint32_t st, w = 0;
		while ((st = plat_sd_rd(R_STATUS)) & ST_DAT_BUSY)
			if (++w > 400000000u) break;
		if (k > 1) {
			cmd(12, 0, RSP_48 | RSP_BUSY, 0);
			idle_clocks(16);
			int w = wait_tran(s);
			if (w) return w;
		}
		while (plat_sd_rd(R_STATUS) & ST_Q_BUSY) ;
		if (st & ST_DAT_TMO) return -30;
		if (st & ST_DAT_CRC) return -31;
		if (st & ST_DAT_BUSY) return -32;
		lba += k; n -= k; mem_addr += k * 512;
	}
	return 0;
}

/* ---------------------------------------------------------------- */
/* FAT16/FAT32                                                      */

static uint32_t le16(const uint8_t *p) { return p[0] | (p[1] << 8); }
static uint32_t le32(const uint8_t *p) { return le16(p) | (le16(p + 2) << 16); }

static int rd_sec(fat_t *f, uint32_t lba)
{
	if (f->cached == lba) return 0;
	f->cached = ~0u;
	int rc = sd_read(f->sd, lba, 1, f->scratch);
	if (!rc) f->cached = lba;
	return rc;
}

static const uint8_t *sec(fat_t *f) { return plat_mem() + f->scratch; }

int fat_mount(fat_t *f, sd_t *s, uint32_t scratch, int part)
{
	f->sd = s;
	f->scratch = scratch;
	f->cached = ~0u;
	if (rd_sec(f, 0)) return -1;
	const uint8_t *b = sec(f);
	uint32_t base = 0;
	if (part > 0 && (b[0] == 0xeb || b[0] == 0xe9) && le16(b + 11) == 512) return -2;
	if (!(b[0] == 0xeb || b[0] == 0xe9) || le16(b + 11) != 512) {
		/* MBR: the part-th FAT partition */
		for (int i = 0, k = 0; i < 4; i++) {
			const uint8_t *e = b + 446 + 16 * i;
			uint8_t t = e[4];
			if (t == 0x0b || t == 0x0c || t == 0x0e || t == 0x06 || t == 0x04) {
				if (k++ == part) { base = le32(e + 8); break; }
			}
		}
		if (!base) return -2;
		if (rd_sec(f, base)) return -3;
		b = sec(f);
	}
	if (le16(b + 11) != 512) return -4;
	f->spc = b[13];
	uint32_t rsvd = le16(b + 14), nfats = b[16], rootents = le16(b + 17);
	uint32_t fatsz = le16(b + 22) ? le16(b + 22) : le32(b + 36);
	uint32_t tot = le16(b + 19) ? le16(b + 19) : le32(b + 32);
	f->fat_lba = base + rsvd;
	uint32_t rootsecs = (rootents * 32 + 511) / 512;
	f->data_lba = f->fat_lba + nfats * fatsz + rootsecs;
	uint32_t clusters = (tot - (f->data_lba - base)) / f->spc;
	f->fat32 = clusters >= 65525;
	if (f->fat32) {
		f->root_clus = le32(b + 44);
		f->root_lba = 0;
		f->root_secs = 0;
	} else {
		f->root_clus = 0;
		f->root_lba = f->fat_lba + nfats * fatsz;
		f->root_secs = rootsecs;
	}
	return 0;
}

static uint32_t fat_next(fat_t *f, uint32_t c)
{
	uint32_t per = f->fat32 ? 128 : 256;
	if (rd_sec(f, f->fat_lba + c / per)) return 0x0fffffff;
	const uint8_t *b = sec(f);
	if (f->fat32) return le32(b + (c % per) * 4) & 0x0fffffff;
	uint32_t v = le16(b + (c % per) * 2);
	return v >= 0xfff8 ? 0x0fffffff : v;
}

static int is_end(fat_t *f, uint32_t c) { return c < 2 || c >= (f->fat32 ? 0x0ffffff8u : 0xfff8u); }

/* walk the root directory; cb returns non-zero to stop */
int fat_list(fat_t *f, int (*cb)(void *ctx, const char *name83, uint32_t clus, uint32_t size),
             void *ctx)
{
	uint32_t clus = f->root_clus, lba = f->root_lba, left = f->root_secs;
	for (;;) {
		uint32_t nsec = f->fat32 ? f->spc : left;
		uint32_t l0 = f->fat32 ? f->data_lba + (clus - 2) * f->spc : lba;
		for (uint32_t s = 0; s < nsec; s++) {
			if (rd_sec(f, l0 + s)) return -1;
			for (int e = 0; e < 16; e++) {
				const uint8_t *d = sec(f) + 32 * e;
				if (d[0] == 0) return 0;
				if (d[0] == 0xe5 || (d[11] & 0x0f) == 0x0f || (d[11] & 0x18)) continue;
				char nm[13];
				int n = 0;
				for (int i = 0; i < 8 && d[i] != ' '; i++) nm[n++] = (char)d[i];
				if (d[8] != ' ') {
					nm[n++] = '.';
					for (int i = 8; i < 11 && d[i] != ' '; i++) nm[n++] = (char)d[i];
				}
				nm[n] = 0;
				uint32_t c = le16(d + 26) | (f->fat32 ? (le16(d + 20) << 16) : 0);
				if (cb(ctx, nm, c, le32(d + 28))) return 1;
				/* the callback may have used the sector buffer */
				if (rd_sec(f, l0 + s)) return -1;
			}
		}
		if (!f->fat32) return 0;
		clus = fat_next(f, clus);
		if (is_end(f, clus)) return 0;
	}
}

/* load a file (first cluster, size) to mem_addr, in runs of
 * contiguous clusters; returns bytes or < 0 */
/* a failed transfer: stop whatever the card is doing (CMD12 may fail:
 * it is not in a transfer then) and give it clocks */
static void sd_recover(sd_t *s)
{
	(void)s;
	cmd(12, 0, RSP_48 | RSP_BUSY | RSP_NOCRC, 0);
	idle_clocks(64);
	(void)wait_tran(s);
}

uint32_t fat_retries;           /* chunks read again by the last fat_load */
uint32_t fat_reinits;           /* times the card had to be initialised again */
uint32_t fat_fail_lba;          /* where the last fat_load gave up */

/* the file in pieces of at most FAT_CHUNK sectors: a failed piece is
 * read again (up to 3 times); prog(done, size) after each piece */
#define FAT_CHUNK 4096u         /* 2 MB */
int32_t fat_load(fat_t *f, uint32_t clus, uint32_t size, uint32_t mem_addr,
                 void (*prog)(uint32_t done, uint32_t size))
{
	uint32_t csz = f->spc * 512, done = 0;
	fat_retries = fat_reinits = 0;
	while (done < size && !is_end(f, clus)) {
		uint32_t first = clus, n = 1;
		for (;;) {
			uint32_t nx = fat_next(f, clus);
			if (nx != clus + 1 || (done + n * csz) >= size) { clus = nx; break; }
			clus = nx;
			n++;
		}
		uint32_t secs = n * f->spc;
		uint32_t need = (size - done + 511) / 512;
		if (secs > need) secs = need;
		f->cached = ~0u;
		uint32_t lba = f->data_lba + (first - 2) * f->spc, got = 0;
		while (got < secs) {
			uint32_t k = secs - got > FAT_CHUNK ? FAT_CHUNK : secs - got;
			int rc, tries = 0;
			while ((rc = sd_read(f->sd, lba + got, k, mem_addr + done + got * 512)) != 0) {
				/* twice: stop the transfer and try again; then start
				 * the card from scratch (a card that has reset, for
				 * example on a supply dip, answers nothing else) */
				if (++tries > 6) { fat_fail_lba = lba + got; return rc; }
				fat_retries++;
				if (tries <= 2) sd_recover(f->sd);
				else {
					/* and each time one step slower (24 -> 12 -> 8 -> 6 MHz
					 * at 48): commands the card rejects with a CRC error
					 * at speed mean the line can't carry it */
					fat_reinits++;
					if (f->sd->div < 3) f->sd->div++;
					idle_clocks(80);
					if (sd_init(f->sd, f->sd->div)) { idle_clocks(800); (void)sd_init(f->sd, f->sd->div); }
				}
			}
			got += k;
			if (prog) prog(done + got * 512 < size ? done + got * 512 : size, size);
		}
		done += n * csz;
	}
	return done >= size ? (int32_t)size : -40;
}
