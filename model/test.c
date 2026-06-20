/* gcc -O2 -std=c11 test.c -o test -lm && ./test  (run test.sh to re-export first) */

#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <time.h>
#include "model.c"

/* Logits buffer - static so it doesn't live on the stack (VOCAB=2048 q16_t). */
static q16_t g_logits[VOCAB];

static void run(const char *label, const int *seed, int n_seed,
                int max_tok, float temp)
{
    LMState s;
    lm_init(&s);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int i = 0; i < n_seed; i++)
        lm_forward(seed[i], &s, g_logits);

    Tokens out = {0};
    for (int i = 0; i < max_tok && out.len < MAX_TOKENS; i++) {
        int tok = lm_sample(g_logits, temp);
        if (tok == EOS_ID || tok == SEP_ID) break;
        out.ids[out.len++] = (int16_t)tok;
        lm_forward(tok, &s, g_logits);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) * 1e-6;

    char buf[4096];
    tok_decode(&out, buf, sizeof(buf));

    printf("── %s (temp=%.1f) ──────────────────────────────\n", label, temp);
    printf("%s\n", buf);
    printf("[%d tokens · %.0f ms · %.1f tok/s]\n\n",
           out.len, ms, out.len / (ms * 1e-3));
}

int main(void) {
    srand((unsigned)time(NULL));

    int bos[] = { BOS_ID };

    run("greedy",    bos, 1, 128, 0.0f);
    run("temp 0.8",  bos, 1, 128, 0.8f);
    run("temp 1.2",  bos, 1, 128, 1.2f);

    Tokens seed = {0};
    tok_encode("the researchers found that", &seed);
    int seed2[MAX_TOKENS + 1];
    seed2[0] = BOS_ID;
    for (int i = 0; i < seed.len; i++) seed2[i + 1] = seed.ids[i];
    run("seeded \"the researchers found that\"", seed2, seed.len + 1, 128, 0.8f);

    return 0;
}
