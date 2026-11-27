/* fw/ui.c on the PC: a conversation drawn into an array, printed as
 * text (cc -DUI_HOST -DSYS_MHZ=49 -Ifw tools/ui_host.c fw/ui.c) */
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "ui.h"
uint32_t host_cells[2048], host_id = 0x55493031u, host_rowbg, host_keys, host_kstat;
static void show(const char *what)
{
	printf("---- %s (ROWBG %07x)\n", what, host_rowbg);
	for (int r = 0; r < 25; r++) {
		putchar(host_rowbg >> r & 1 ? '|' : ' ');
		for (int c = 0; c < 80; c++) { uint32_t v = host_cells[r * 80 + c] & 127; putchar(v == 127 ? '@' : v < 32 ? ' ' : (int)v); }
		putchar('\n');
	}
}
static void answer(const char *s) { ui_answer_begin(); while (*s) ui_out((uint8_t)*s++); ui_answer_end(); ui_ready(); }
int main(void)
{
	ui_init(); show("start");
	char b[600] = "";
	const char *q = "How do I bake a pumpkin pie? Please explain every step carefully, with the oven temperature, the time, and what to look for when it is done, because I have never baked anything before.";
	int n = 0;
	for (; q[n]; n++) { b[n] = q[n]; b[n + 1] = 0; ui_input(b, n + 1, 1); }
	show("typing a long question");
	ui_input(b, 0, 0);
	ui_user(b);
	ui_progress(7, 20);
	show("reading");
	answer("Here is a simple way to bake a pumpkin pie " "\xe2\x80\x94" " you" "\xe2\x80\x99" "ll need a crust.\n\n1. Heat the oven to 220" "\xc2\xb0" "C.\n2. Mix the pumpkin, eggs, sugar, cream and cinnamon.\n3. Bake for 15 minutes, then 45 minutes at 175" "\xc2\xb0" "C, until the middle is set. Averyveryveryveryveryveryveryveryveryveryveryveryveryveryveryveryveryverylongword ends here.");
	show("the first answer");
	for (int i = 0; i < 4; i++) { ui_user("And another question?"); answer("Another answer, which is long enough to take a couple of lines on the screen so that the conversation scrolls up and the oldest lines go away at the top of the screen."); }
	ui_input("", 0, 1);
	show("after five turns");
	return 0;
}
