/* gcc -O2 -std=c11 chat.c -o chat -lm && ./chat  ("reset" clears memory) */

#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include "model.c"

#define CHAT_TEMP  0.8f
#define OUT_SIZE   1024
#define IN_SIZE    512

int main(int argc, char **argv) {
    srand((unsigned)time(NULL));

    float temp = CHAT_TEMP;
    if (argc >= 2) temp = strtof(argv[1], NULL);

    LMState state;
    lm_init(&state);

    char   input[IN_SIZE];
    char   output[OUT_SIZE];
    int    turn = 0;

    printf("esp-lm chat  (temp=%.2f)  - 'reset' clears memory, Ctrl+D quits\n", (double)temp);
    printf("──────────────────────────────────────────────────────────────────\n\n");

    for (;;) {
        printf("[you]: ");
        fflush(stdout);

        if (!fgets(input, sizeof(input), stdin)) break;

        int len = (int)strlen(input);
        while (len > 0 && (input[len-1] == '\n' || input[len-1] == '\r'))
            input[--len] = '\0';
        if (!len) continue;

        if (strcmp(input, "reset") == 0 || strcmp(input, "/reset") == 0) {
            lm_init(&state);
            turn = 0;
            printf("[memory cleared]\n\n");
            continue;
        }

        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        int n = lm_respond(input, &state, output, sizeof(output), temp);

        clock_gettime(CLOCK_MONOTONIC, &t1);
        double ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) * 1e-6;

        printf("[bot]: %s\n", output);
        printf("       [%d tok · %.0f ms · %.0f tok/s]\n\n",
               n, ms, ms > 0.0 ? n / (ms * 1e-3) : 0.0);

        turn++;
    }

    printf("\nbye.\n");
    return 0;
}
