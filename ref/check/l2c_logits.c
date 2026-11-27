// Compare harness: llama2.c's float forward() on a token sequence; prints
// the top 5 (id:logit) per position, like mlm_run -d. Build against a
// llama2.c checkout:  cc -O2 -DTESTING -I<llama2.c> l2c_logits.c -lm
#include "run.c"
int main(int argc, char **argv) {
	Transformer t; build_transformer(&t, argv[1]);
	int n = argc - 2;
	for (int i = 0; i < n; i++) {
		int tok = atoi(argv[2 + i]);
		float *lg = forward(&t, tok, i);
		int V = t.config.vocab_size, best[5] = {0};
		for (int k = 0; k < 5; k++) {
			int b = -1;
			for (int j = 0; j < V; j++) {
				int used = 0; for (int q = 0; q < k; q++) if (best[q] == j) used = 1;
				if (!used && (b < 0 || lg[j] > lg[b])) b = j;
			}
			best[k] = b;
		}
		printf("pos %d in %d:", i, tok);
		for (int k = 0; k < 5; k++) printf(" %d:%.4f", best[k], lg[best[k]]);
		printf("\n");
	}
	return 0;
}
