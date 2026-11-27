/* the few library functions gcc may call */
#include <stddef.h>
#include <stdint.h>
__attribute__((used)) void *memset(void *d, int c, size_t n) { uint8_t *p = d; while (n--) *p++ = (uint8_t)c; return d; }
__attribute__((used)) void *memcpy(void *d, const void *s, size_t n) { uint8_t *p = d; const uint8_t *q = s; while (n--) *p++ = *q++; return d; }
void *memmove(void *d, const void *s, size_t n)
{
	uint8_t *p = d; const uint8_t *q = s;
	if (p < q) while (n--) *p++ = *q++; else { p += n; q += n; while (n--) *--p = *--q; }
	return d;
}
int memcmp(const void *a, const void *b, size_t n)
{
	const uint8_t *p = a, *q = b;
	for (; n; n--, p++, q++) if (*p != *q) return *p - *q;
	return 0;
}
size_t strlen(const char *s) { size_t n = 0; while (s[n]) n++; return n; }

/* 64-bit division, small (shift and subtract): libgcc's versions are
 * 2.2 KB of a nearly full RAM, and nothing hot divides 64-bit numbers
 * (statistics, printing, loading; rmsnorm only for sizes that are not
 * a power of two) */
__attribute__((used)) unsigned long long __udivdi3(unsigned long long n, unsigned long long d)
{
	if (!d) return ~0ull;
	unsigned long long q = 0, r = 0;
	for (int i = 63; i >= 0; i--) {
		r = (r << 1) | ((n >> i) & 1);
		if (r >= d) { r -= d; q |= 1ull << i; }
	}
	return q;
}
__attribute__((used)) long long __divdi3(long long n, long long d)
{
	int neg = (n < 0) != (d < 0);
	unsigned long long q = __udivdi3(n < 0 ? -(unsigned long long)n : (unsigned long long)n,
	                                 d < 0 ? -(unsigned long long)d : (unsigned long long)d);
	return neg ? -(long long)q : (long long)q;
}
