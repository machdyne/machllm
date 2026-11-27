/*
 * machllm
 * Copyright (c) 2025 Lone Dynamics Corporation. All rights reserved.
 *
 * A behavioural SD card (SDHC, native mode, 1- and 4-bit) for the
 * testbenches. Call step() once per host clock cycle with the pins; it
 * samples CMD on SCK rising edges and changes its outputs on falling
 * edges, as a default-speed card does. Checks the host's CRC7 on every
 * command and counts protocol errors.
 *
 * Commands: 0 2 3 6(app) 7 8 9 12 16 17 18 41(app) 55.
 */

#pragma once
#include <cstdint>
#include <cstdio>
#include <deque>
#include <vector>

struct SdCard {
	std::vector<uint8_t> img;
	int errors = 0;
	uint32_t rca = 0x4d4c;
	bool wide = false, app = false, ccs = true;
	int acmd41_polls = 0;
	// pins out
	int cmd_out = 1, dat_out = 0xf;
	// receive
	uint64_t rx = 0;
	int rx_bits = 0;
	bool rx_on = false;
	// transmit queues (one entry per SCK cycle)
	std::deque<int> cmdq, datq;
	// multi-block read
	bool reading = false;
	uint32_t next_lba = 0;
	int prev_sck = 0;
	uint64_t cmds = 0, blocks = 0;

	static uint8_t crc7(const uint8_t *b, int n) {
		uint8_t c = 0;
		for (int i = 0; i < n; i++)
			for (int k = 7; k >= 0; k--) {
				int f = ((c >> 6) & 1) ^ ((b[i] >> k) & 1);
				c = (uint8_t)((c << 1) & 0x7f);
				if (f) c ^= 0x09;
			}
		return c;
	}
	static uint16_t crc16_add(uint16_t c, int bit) {
		int f = ((c >> 15) & 1) ^ bit;
		c = (uint16_t)(c << 1);
		if (f) c ^= 0x1021;
		return c;
	}

	void push_bits(std::deque<int> &q, const uint8_t *b, int nbits) {
		for (int i = 0; i < nbits; i++) q.push_back((b[i / 8] >> (7 - i % 8)) & 1);
	}
	void r48(int idx, uint32_t v, bool crc = true) {
		uint8_t b[6] = { (uint8_t)(idx & 0x3f), (uint8_t)(v >> 24), (uint8_t)(v >> 16),
		                 (uint8_t)(v >> 8), (uint8_t)v, 0 };
		b[5] = (uint8_t)(((crc ? crc7(b, 5) : 0x7f) << 1) | 1);
		for (int i = 0; i < 2; i++) cmdq.push_back(1);      // NCR
		push_bits(cmdq, b, 48);
	}
	void r136(const uint8_t reg[16]) {
		uint8_t b[17];
		b[0] = 0x3f;
		for (int i = 0; i < 15; i++) b[1 + i] = reg[i];
		b[16] = (uint8_t)((crc7(reg, 15) << 1) | 1);
		for (int i = 0; i < 2; i++) cmdq.push_back(1);
		push_bits(cmdq, b, 136);
	}
	uint32_t status() { return (4u << 9) | (app ? (1u << 5) : 0); }   /* tran */

	void queue_block(uint32_t lba) {
		const uint8_t *p = &img[(size_t)lba * 512 % img.size()];
		for (int i = 0; i < 2; i++) datq.push_back(0xf);    // NAC
		datq.push_back(0x0);                                // start bit(s)
		if (wide) {
			uint16_t c[4] = { 0, 0, 0, 0 };
			for (int i = 0; i < 512; i++)
				for (int h = 1; h >= 0; h--) {
					int nb = (p[i] >> (4 * h)) & 0xf;
					datq.push_back(nb);
					for (int l = 0; l < 4; l++) c[l] = crc16_add(c[l], (nb >> l) & 1);
				}
			for (int k = 15; k >= 0; k--) {
				int nb = 0;
				for (int l = 0; l < 4; l++) nb |= ((c[l] >> k) & 1) << l;
				datq.push_back(nb);
			}
		} else {
			uint16_t c = 0;
			for (int i = 0; i < 512; i++)
				for (int k = 7; k >= 0; k--) {
					int bit = (p[i] >> k) & 1;
					datq.push_back(0xe | bit);
					c = crc16_add(c, bit);
				}
			for (int k = 15; k >= 0; k--) datq.push_back(0xe | ((c >> k) & 1));
		}
		datq.push_back(0xf);                                // end bit(s)
		blocks++;
	}

	// fault injection: ignore the n-th CMD18 (SD_DROP_CMD18=n), as a
	// card that missed it would
	int drop18 = getenv("SD_DROP_CMD18") ? atoi(getenv("SD_DROP_CMD18")) : 0, n18 = 0;
	// fault injection: on the n-th CMD18 the card resets (SD_RESET_CMD18=n):
	// from then on it ignores everything until CMD0, as a card that lost
	// power for a moment
	int reset18 = getenv("SD_RESET_CMD18") ? atoi(getenv("SD_RESET_CMD18")) : 0;
	bool idle_reset = false;
	void command(int idx, uint32_t arg) {
		if (idx == 18 && n18 + 1 == reset18) { n18++; idle_reset = true; reading = false; datq.clear(); fprintf(stderr, "sdcard: reset at CMD18 #%d (fault injection)\n", n18); return; }
		if (idle_reset && idx != 0) return;
		if (idx == 0) idle_reset = false;
		if (idx == 18 && ++n18 == drop18) { fprintf(stderr, "sdcard: ignoring CMD18 #%d (fault injection)\n", n18); return; }
		cmds++;
		bool was_app = app;
		app = false;
		if (was_app && idx == 41) {
			uint32_t ocr = 0x00ff8000u | (ccs ? 0x40000000u : 0);
			if (++acmd41_polls >= 3) ocr |= 0x80000000u;
			r48(0x3f, ocr, false);
			return;
		}
		if (was_app && idx == 6) { wide = (arg & 3) == 2; r48(6, status()); return; }
		switch (idx) {
		case 0: wide = false; reading = false; datq.clear(); break;
		case 8: r48(8, arg & 0xfff); break;
		case 55: app = true; r48(55, status() | (1u << 5)); break;
		case 2: { uint8_t cid[16] = { 0x03, 'M', 'L', 'M', 'L', 'L', 'M', 0x10, 1, 2, 3, 4, 0x01, 0x6a, 0 };
		          r136(cid); break; }
		case 9: { uint8_t csd[16] = { 0x40, 0x0e, 0x00, 0x32, 0x5b, 0x59, 0, 0, 0x3f, 0xff, 0x7f, 0x80, 0x0a, 0x40, 0 };
		          r136(csd); break; }
		case 3: r48(3, (rca << 16) | 0x0500); break;
		case 7: r48(7, status()); busy(8); break;
		case 16: r48(16, status()); break;
		case 13: r48(13, status()); break;
		case 17: r48(17, status()); datq.clear(); queue_block(arg); single = true; break;
		case 18: r48(18, status()); datq.clear(); reading = true; next_lba = arg; break;
		case 12: r48(12, status()); reading = false; datq.clear(); busy(20); break;
		default: fprintf(stderr, "sdcard: unexpected CMD%d\n", idx); errors++; break;
		}
	}
	// R1b: DAT0 low after the response's end bit (2 + 48 clocks away)
	void busy(int n) { for (int i = 0; i < 52 + n; i++) datq.push_back(i < 52 ? 0xf : 0xe); }

	// one host clock; cmd_in is what the host drives (1 when released)
	// a real card takes no command until 8 clocks after a single-block
	// read's end bit (the host has to keep SCK running for them)
	bool single = false;
	int deaf = 0;
	long missed = 0;
	void step(int sck, int cmd_in) {
		if (sck && !prev_sck) {
			// rising: sample CMD
			if (deaf > 0 || single) {
				if (deaf > 0) deaf--;
				if (!rx_on && cmd_in == 0) { missed++; fprintf(stderr, "sdcard: command sent before the card's 8 clocks after a read\n"); errors++; }
			} else if (cmdq.empty()) {
				if (!rx_on && cmd_in == 0) { rx_on = true; rx = 0; rx_bits = 0; }
				if (rx_on) {
					rx = (rx << 1) | (uint64_t)cmd_in;
					if (++rx_bits == 48) {
						rx_on = false;
						uint8_t b[6];
						for (int i = 0; i < 6; i++) b[i] = (uint8_t)(rx >> (40 - 8 * i));
						int idx = b[0] & 0x3f;
						uint32_t arg = ((uint32_t)b[1] << 24) | (b[2] << 16) | (b[3] << 8) | b[4];
						if ((b[0] & 0xc0) != 0x40 || (b[5] & 1) != 1 || (b[5] >> 1) != crc7(b, 5)) {
							fprintf(stderr, "sdcard: bad command frame %012llx\n", (unsigned long long)rx);
							errors++;
						} else command(idx, arg);
					}
				}
			}
		}
		if (!sck && prev_sck) {
			// falling: drive the next bits
			if (!cmdq.empty()) { cmd_out = cmdq.front(); cmdq.pop_front(); } else cmd_out = 1;
			if (datq.empty() && reading) queue_block(next_lba++);
			if (!datq.empty()) { dat_out = datq.front(); datq.pop_front(); } else dat_out = 0xf;
			if (single && datq.empty()) { single = false; deaf = 8; }
		}
		prev_sck = sck;
	}
};
