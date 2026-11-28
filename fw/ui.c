/*
 * machllm firmware: the screen and the keyboard (rtl/mlm_ui.v)
 * Copyright (c) 2026 Lone Dynamics Corporation. All rights reserved.
 *
 * 80 x 25 characters. Row 0: the title. Then the conversation, which
 * scrolls. At the bottom a bar: the question being typed (1 to 6
 * lines, growing upwards) and a status line. Nothing here is needed:
 * without the hardware (ui_init() finds no ID) every call returns at
 * once and the serial console is the interface, as before.
 */
#include <stdint.h>
#include "hw.h"
#include "ui.h"

#pragma GCC optimize ("Os")      /* not a hot path; the RAM is nearly full */

#ifndef UI_HOST
#define UI_BASE   0x76000000u
#define UI_CELL   ((volatile uint32_t *)UI_BASE)
#define UI_ID     REG(0x76002000)
#define UI_ROWBG  REG(0x76002004)
#define UI_KEYS   REG(0x76002008)
#define UI_KSTAT  REG(0x7600200c)
#else                                   /* tools/ui_host.c: on the PC */
extern uint32_t host_cells[2048], host_id, host_rowbg, host_keys, host_kstat;
#define UI_CELL   host_cells
#define UI_ID     host_id
#define UI_ROWBG  host_rowbg
#define UI_KEYS   host_keys
#define UI_KSTAT  host_kstat
#endif

#define COLS 80
#define ROWS 25
#define C_TEXT 0
#define C_USER 1
#define C_ACC  2
#define C_DIM  3
#define CURSOR 127
#define LEFT   1                       /* text from column 1 to 78 */
#define WIDTH  78
#define IN_W   76                      /* the question: columns 3 to 78 */
#define IN_MAX 6

int ui_on;
static int fresh;                       /* the welcome text is on screen */

#ifdef UI_HOST
static uint64_t now(void) { static uint64_t t; return t += 1000; }
#else
static uint64_t now(void)
{
	uint32_t hi, lo, hi2;
	do {
		__asm__ volatile ("rdcycleh %0" : "=r"(hi));
		__asm__ volatile ("rdcycle %0" : "=r"(lo));
		__asm__ volatile ("rdcycleh %0" : "=r"(hi2));
	} while (hi != hi2);
	return ((uint64_t)hi << 32) | lo;
}
#endif
#define MS(n) ((uint64_t)(n) * SYS_MHZ * 1000u)

static void cell(int r, int c, int ch, int col) { UI_CELL[r * COLS + c] = ((uint32_t)col << 7) | (uint32_t)(ch & 127); }
static uint32_t cell_rd(int r, int c) { return UI_CELL[r * COLS + c]; }
static void clear_rows(int r0, int r1) { for (int i = r0 * COLS; i < (r1 + 1) * COLS; i++) UI_CELL[i] = ' '; }
static void text(int r, int c, const char *s, int col) { while (*s && c < COLS) cell(r, c++, *s++, col); }

/* ---------------------------------------------------------------- */
/* the conversation                                                 */

static int bar_top = ROWS - 2;         /* first row of the bar */
static int crow = 2, ccol = LEFT;      /* where the next character goes */
static int ccolour = C_TEXT;
static int wstart = -1;                /* the current word's first column on crow, -1: none */
static int cursor_on;

static int conv_bottom(void) { return bar_top - 2; }      /* a blank row above the bar */

static void scroll(void)
{
	int bot = conv_bottom();
	for (int r = 2; r < bot; r++)
		for (int c = 0; c < COLS; c++) UI_CELL[r * COLS + c] = UI_CELL[(r + 1) * COLS + c];
	clear_rows(bot, bot);
}
static void newline(void)
{
	crow++; ccol = LEFT; wstart = -1;
	if (crow > conv_bottom()) { scroll(); crow = conv_bottom(); }
}
static void cursor(int on)
{
	if (on) cell(crow, ccol, CURSOR, C_ACC);
	else if (cursor_on) cell(crow, ccol, ' ', C_TEXT);
	cursor_on = on;
}
static void put(int ch)
{
	if (ch == '\n') { newline(); return; }
	if (ch == ' ') {
		wstart = -1;
		if (ccol >= LEFT + WIDTH) { newline(); return; }
		if (ccol == LEFT) return;             /* no spaces at a line's start */
		cell(crow, ccol++, ' ', ccolour);
		return;
	}
	if (wstart < 0) wstart = ccol;
	if (ccol >= LEFT + WIDTH) {
		/* the word goes on to the next line whole, unless it is a line long */
		int n = ccol - wstart;
		if (wstart > LEFT && n < WIDTH) {
			uint32_t w[WIDTH];
			for (int i = 0; i < n; i++) { w[i] = cell_rd(crow, wstart + i); cell(crow, wstart + i, ' ', C_TEXT); }
			newline();
			for (int i = 0; i < n; i++) UI_CELL[crow * COLS + LEFT + i] = w[i];
			ccol = LEFT + n;
			wstart = LEFT;
		} else {
			newline();
			wstart = LEFT;
		}
	}
	cell(crow, ccol++, ch, ccolour);
}
static void puts_c(const char *s) { while (*s) put(*s++); }

/* the model writes UTF-8: the common punctuation as ASCII, the rest '?' */
static uint32_t u8_cp;
static int u8_left;
static void put_cp(uint32_t cp)
{
	if (cp < 128) { if (cp >= 32 || cp == '\n') put((int)cp); else if (cp == '\t') put(' '); return; }
	switch (cp) {
	case 0x2018: case 0x2019: case 0x2032: put('\''); return;
	case 0x201c: case 0x201d: case 0x2033: put('"'); return;
	case 0x2013: case 0x2014: case 0x2212: put('-'); return;
	case 0x2026: puts_c("..."); return;
	case 0x00a0: put(' '); return;
	case 0x00b0: put('o'); return;          /* degrees */
	case 0x00d7: put('x'); return;
	case 0x2022: case 0x00b7: put('*'); return;
	case 0x00bd: puts_c("1/2"); return;
	case 0x00bc: puts_c("1/4"); return;
	case 0x00be: puts_c("3/4"); return;
	default: put('?');
	}
}
static void out_byte(uint8_t b)
{
	if (u8_left) {
		if ((b & 0xc0) == 0x80) {
			u8_cp = (u8_cp << 6) | (b & 0x3f);
			if (--u8_left == 0) put_cp(u8_cp);
			return;
		}
		u8_left = 0; put('?');
	}
	if (b < 0x80) put_cp(b);
	else if ((b & 0xe0) == 0xc0) { u8_cp = b & 0x1f; u8_left = 1; }
	else if ((b & 0xf0) == 0xe0) { u8_cp = b & 0x0f; u8_left = 2; }
	else if ((b & 0xf8) == 0xf0) { u8_cp = b & 0x07; u8_left = 3; }
}

/* ---------------------------------------------------------------- */
/* the bar: the question and the status line                        */

static void bar(int lines)
{
	int top = ROWS - 1 - lines;
	if (top != bar_top) {
		/* the conversation keeps its last rows in view */
		while (crow > top - 2) { scroll(); crow--; }
		if (top > bar_top) clear_rows(bar_top, top - 1);
		bar_top = top;
		UI_ROWBG = ((1u << ROWS) - 1) & ~((1u << top) - 1);
	}
	clear_rows(bar_top - 1, bar_top - 1);
}
void ui_status(const char *s)
{
	if (!ui_on) return;
	clear_rows(ROWS - 1, ROWS - 1);
	text(ROWS - 1, LEFT, s, C_DIM);
}
/* the question as typed: a whole redraw only when its line count
 * changes (or it is cleared); otherwise the one cell that changed and
 * the cursor. A full redraw is ~500 cell writes, longer than the serial
 * port's 16-byte queue lasts when text is pasted at 1 Mbaud. */
static int in_n = -1, in_lines;
void ui_input(const char *b, int n, int typing)
{
	if (!ui_on) return;
	if (typing && in_n >= 0 && n / IN_W + 1 == in_lines && in_lines <= IN_MAX && (n == in_n + 1 || n == in_n - 1)) {
		int r0 = bar_top + in_n / IN_W, c0 = 3 + in_n % IN_W;
		int r1 = bar_top + n / IN_W, c1 = 3 + n % IN_W;
		if (n > in_n) cell(r0, c0, b[in_n], C_TEXT);
		else cell(r0, c0, ' ', C_TEXT);
		cell(r1, c1, CURSOR, C_ACC);
		in_n = n;
		return;
	}
	in_n = typing ? n : -1;
	in_lines = n / IN_W + 1;
	int lines = n / IN_W + 1;
	int first = 0;                         /* the first line shown */
	if (lines > IN_MAX) { first = lines - IN_MAX; lines = IN_MAX; }
	bar(lines);
	clear_rows(bar_top, ROWS - 2);
	if (!typing) return;
	cell(bar_top, LEFT, '>', C_ACC);
	for (int l = 0; l < lines; l++) {
		int o = (first + l) * IN_W;
		for (int i = 0; i < IN_W && o + i < n; i++) cell(bar_top + l, 3 + i, b[o + i], C_TEXT);
	}
	int r = bar_top + (n / IN_W - first), c = 3 + n % IN_W;
	cell(r, c, CURSOR, C_ACC);
}
void ui_progress(int i, int n)
{
	if (!ui_on || n <= 0) return;
	char s[48];
	const char *t = "Reading your question  [";
	int k = 0;
	while (*t) s[k++] = *t++;
	int f = i * 20 / n;
	for (int j = 0; j < 20; j++) s[k++] = j < f ? '#' : '.';
	s[k++] = ']'; s[k] = 0;
	ui_status(s);
}

/* ---------------------------------------------------------------- */
/* a turn                                                           */

void ui_user(const char *line)
{
	if (!ui_on) return;
	ui_welcome_off();
	if (crow > 2 || ccol > LEFT) { if (ccol > LEFT) newline(); newline(); }
	ccolour = C_ACC; puts_c("You:"); newline();
	ccolour = C_USER;
	while (*line) { uint8_t c = (uint8_t)*line++; put(c < 128 ? c : '?'); }
}
void ui_answer_begin(void)
{
	if (!ui_on) return;
	newline(); newline();
	ccolour = C_ACC; puts_c("machllm:"); newline();
	ccolour = C_TEXT;
	u8_left = 0;
	cursor(1);
}
void ui_out(uint8_t b)
{
	if (!ui_on) return;
	cursor(0);
	out_byte(b);
	cursor(1);
}
void ui_answer_end(void)
{
	if (!ui_on) return;
	cursor(0);
}
void ui_clear(void)
{
	if (!ui_on) return;
	clear_rows(0, ROWS - 1);
	text(0, LEFT, "machllm", C_ACC);
	bar_top = ROWS - 2;
	UI_ROWBG = (1u << (ROWS - 2)) | (1u << (ROWS - 1));
	crow = 2; ccol = LEFT; wstart = -1; cursor_on = 0;
	text(11, 32, "Ask me anything.", C_TEXT);
	text(13, 24, "Type a question and press Enter.", C_DIM);
	crow = 2;
	fresh = 1;                         /* (cleared by the next question) */
}
static int think_on;
void ui_set_think(int on) { think_on = on; }
void ui_ready(void)
{
	if (!ui_on) return;
	ui_status(think_on ? "Enter: ask   Esc: clear   F1: new   F2: thinking (on)   F10: exit"
	                   : "Enter: ask   Esc: clear   F1: new   F2: thinking (off)   F10: exit");
}
/* the model's reasoning (thinking mode): dimmed, then the answer */
void ui_think_begin(void)
{
	if (!ui_on) return;
	cursor(0);
	ccolour = C_DIM;
	cursor(1);
}
void ui_think_end(void)
{
	if (!ui_on) return;
	cursor(0);
	ccolour = C_TEXT;
	if (ccol > LEFT) newline();
	newline();
	cursor(1);
}

/* ---------------------------------------------------------------- */
/* the keyboard: USB HID key codes (US layout)                      */

static const char k_lo[] = "abcdefghijklmnopqrstuvwxyz1234567890\n\033\b\t -=[]\\#;'`,./";
static const char k_hi[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()\n\033\b\t _+{}|~:\"~<>?";
static uint32_t k_seq, k_prev;
static int k_held, caps;
static uint64_t k_next;

static int to_char(int code, uint32_t mods)
{
	if (code == 0x3a) return UI_F1;
	if (code == 0x43) return UI_F10;
	if (code == 0x3b) return UI_F2;
	if (code < 0x04 || code > 0x38) return -1;
	int shift = (mods & 0x22) != 0;
	int ch = shift ? k_hi[code - 4] : k_lo[code - 4];
	if (caps && code <= 0x1d) ch = shift ? k_lo[code - 4] : k_hi[code - 4];
	if ((mods & 0x11) && code <= 0x1d) return -1;   /* Ctrl-letters: nothing */
	return ch;
}
static int in_report(uint32_t keys, int code)
{
	return code && (((keys >> 8) & 0xff) == (uint32_t)code || ((keys >> 16) & 0xff) == (uint32_t)code ||
	                ((keys >> 24) & 0xff) == (uint32_t)code);
}
int ui_key(void)
{
	if (!ui_on) return -1;
	uint32_t st = UI_KSTAT;
	uint32_t seq = st >> 16;
	uint32_t keys = UI_KEYS;
	if (seq != k_seq) {
		k_seq = seq;
		int fresh_key = 0;
		for (int i = 1; i <= 3; i++) {
			int code = (keys >> (8 * i)) & 0xff;
			if (code && !in_report(k_prev, code)) fresh_key = code;
		}
		k_prev = keys;
		if (fresh_key) {
			if (fresh_key == 0x39) { caps = !caps; return -1; }
			k_held = fresh_key;
			k_next = now() + MS(500);
			return to_char(fresh_key, keys & 0xff);
		}
		if (!in_report(keys, k_held)) k_held = 0;
		return -1;
	}
	/* repeat while a key stays down */
	if (k_held && in_report(keys, k_held) && now() >= k_next) {
		k_next = now() + MS(40);
		if (k_held == 0x29 || k_held == 0x28 || k_held == 0x3a || k_held == 0x3b || k_held == 0x43) return -1;   /* not Esc, Enter, F keys */
		return to_char(k_held, keys & 0xff);
	}
	return -1;
}
int ui_keyboard(void) { return ui_on && (UI_KSTAT & 3) == 1; }

int ui_init(void)
{
	ui_on = UI_ID == 0x55493031u;
	if (!ui_on) return 0;
	k_seq = UI_KSTAT >> 16;
	k_prev = UI_KEYS;
	ui_clear();
	fresh = 1;
	ui_input("", 0, 1);
	ui_ready();
	return 1;
}
void ui_welcome_off(void)
{
	if (!ui_on || !fresh) return;
	fresh = 0;
	clear_rows(2, bar_top - 1);
	crow = 2; ccol = LEFT;
}
