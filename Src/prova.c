#include "../Header/suffix_arrays.h"
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void suffix_sort(const int *str, int n, int *pos, int *rank_arr) {
    int *cnt = calloc(n + 1, sizeof(int));
    int *next_bucket = calloc(n + 1, sizeof(int));
    char *bh = calloc(n + 1, sizeof(char));
    char *b2h = calloc(n + 1, sizeof(char));
    if (!cnt || !next_bucket || !bh || !b2h) { fprintf(stderr, "Alloc failed\n"); exit(1); }

    int *freq = calloc(ALPHABET_SIZE, sizeof(int));
    if (!freq) { fprintf(stderr, "Alloc failed\n"); exit(1); }

    // // Parallel counting con buffer locale per thread
    // #pragma omp parallel
    // {
    //     int local_freq[ALPHABET_SIZE] = {0};
    //     #pragma omp for nowait
    //     for (int i = 0; i < n; i++)
    //         local_freq[(unsigned char)str[i]]++;

    //     #pragma omp critical
    //     for (int i = 1; i < ALPHABET_SIZE; i++)
    //         freq[i] += local_freq[i - 1];
    // }

    // // // Cumulative sum
    // // for (int i = 1; i < ALPHABET_SIZE; i++)
    // //     freq[i] += freq[i - 1];

    #pragma omp parallel for reduction(+:freq[:ALPHABET_SIZE])
    for (int i = 0; i < n; i++)
        freq[(unsigned char)str[i]]++;

    //Cumulative sum calculation
    for (int i = 1; i < ALPHABET_SIZE; i++)
        freq[i] += freq[i - 1];

    // Sorting iniziale
    #pragma omp parallel for
    for (int i = n - 1; i >= 0; i--)
        pos[--freq[(unsigned char)str[i]]] = i;

    free(freq);

    // Initialize bh and b2h
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        bh[i] = (i == 0 || str[pos[i]] != str[pos[i - 1]]);
        b2h[i] = 0;
    }

    for (int h = 1; h < n; h <<= 1) {
        int buckets = 0;

        // sequential bucket calculation (necessario)
        for (int i = 0, j; i < n; i = j) {
            j = i + 1;
            while (j < n && !bh[j]) j++;
            next_bucket[i] = j;
            buckets++;
        }

        if (buckets == n) break;

        // Parallel ranking per bucket
        for (int i = 0; i < n; i = next_bucket[i]) {
            for (int j = i; j < next_bucket[i]; j++)
                rank_arr[pos[j]] = i;
        }

        cnt[rank_arr[n - h]]++;
        b2h[rank_arr[n - h]] = 1;

        // spostamento suffissi sequenziale
        for (int i = 0; i < n; i = next_bucket[i]) {
            for (int j = i; j < next_bucket[i]; j++) {
                int s = pos[j] - h;
                if (s >= 0) {
                    int head = rank_arr[s];
                    rank_arr[s] = head + cnt[head]++;
                    b2h[rank_arr[s]] = 1;
                }
            }

            // pulizia flag b2h sequenziale
            for (int j = i; j < next_bucket[i]; j++) {
                int s = pos[j] - h;
                if (s >= 0 && b2h[rank_arr[s]]) {
                    for (int k = rank_arr[s] + 1; k < n && !bh[k] && b2h[k]; k++)
                        b2h[k] = 0;
                }
            }
        }

        // aggiornamento finale parallel
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            pos[rank_arr[i]] = i;
            bh[i] |= b2h[i];
        }
    }

    // final rank parallel
    #pragma omp parallel for
    for (int i = 0; i < n; i++)
        rank_arr[pos[i]] = i;

    free(cnt);
    free(next_bucket);
    free(bh);
    free(b2h);
}
