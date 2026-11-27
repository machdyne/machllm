/* machllm firmware: the screen and the keyboard (fw/ui.c) */
#ifndef UI_H
#define UI_H
#include <stdint.h>
#define UI_F1 0x101
extern int ui_on;
int ui_init(void);                  /* 1 if the hardware is there */
int ui_key(void);                   /* a character, '\n', '\b', 27 (Esc), UI_F1, or -1 */
int ui_keyboard(void);              /* a keyboard is connected */
void ui_input(const char *b, int n, int typing);
void ui_status(const char *s);
void ui_ready(void);
void ui_progress(int i, int n);
void ui_user(const char *line);
void ui_answer_begin(void);
void ui_out(uint8_t b);
void ui_answer_end(void);
void ui_clear(void);
void ui_welcome_off(void);
#endif
