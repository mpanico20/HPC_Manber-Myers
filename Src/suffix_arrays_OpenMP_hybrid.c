    /*
    * Course: High Performance Computing 2025/2026
    * 
    * Lecturer: Francesco Moscato fmoscato@unisa.it
    *
    * Student: 
    * Panico Marco  0622702416  m.panico20@studenti.unisa.it
    *
    * This file is part of Mamber-Myers.
    *
    * Copyright (C) 2024 - All Rights Reserved
    *
    * This program is free software: you can redistribute it and/or modify it under the terms of
    * the GNU General Public License as published by the Free Software Foundation, either version
    * 3 of the License, or (at your option) any later version.
    *
    * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
    * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
    * See the GNU General Public License for more details.
    *
    * You should have received a copy of the GNU General Public License along with ContestOMP.
    * If not, see <http://www.gnu.org/licenses/>.
    */
    #include "../Header/suffix_arrays.h"
    #include <stdio.h>
    #include <stdlib.h>

    static inline void shift_suffixes(
    int n, int h,
    int *pos,
    int *rank_arr,
    int *next_bucket,
    int *cnt,
    char *bh,
    char *b2h
);

static inline void shift_suffixes_parallel(
    int n, int h,
    int *pos,
    int *rank_arr,
    int *next_bucket,
    int *cnt,
    char *bh,
    char *b2h
);


    void suffix_sort(const int *str, int n, int *pos, int *rank_arr) {
        //Allocate memory for the variables
        int *cnt = calloc(n + 1, sizeof(int));
        int *next_bucket = calloc(n + 1, sizeof(int));
        char *bh = calloc(n + 1, sizeof(char));
        char *b2h = calloc(n + 1, sizeof(char));
        if (!cnt || !next_bucket || !bh || !b2h) { fprintf(stderr, "Alloc failed\n"); exit(1); }

        int *freq = calloc(ALPHABET_SIZE, sizeof(int));
        if (!freq) { fprintf(stderr, "Alloc failed\n"); exit(1); }

        // parallel counting
        #pragma omp parallel for reduction(+:freq[:ALPHABET_SIZE])
        for (int i = 0; i < n; i++)
            freq[(unsigned char)str[i]]++;

        //Cumulative sum calculation
        for (int i = 1; i < ALPHABET_SIZE; i++)
            freq[i] += freq[i - 1];

        //Final sorting
        for (int i = n - 1; i >= 0; i--)
            pos[--freq[(unsigned char)str[i]]] = i;

        free(freq);

        //
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            bh[i] = (i == 0) || (str[pos[i]] != str[pos[i - 1]]);
            b2h[i] = 0;
        }

        //
        for (int h = 1; h < n; h <<= 1) {
            int buckets = 0;

            // calcolo dei bucket (sequenziale!)
            for (int i = 0, j; i < n; i = j) {
                j = i + 1;
                while (j < n && !bh[j]) j++;
                next_bucket[i] = j;
                buckets++;
            }

            if (buckets == n) break;

            if (h > 4){
                for (int i = 0; i < n; i = next_bucket[i]) {
                    cnt[i] = 0;
                    for (int j = i; j < next_bucket[i]; j++)
                        rank_arr[pos[j]] = i;
                }
            } else {
                for (int i = 0; i < n; i = next_bucket[i]) {
                    cnt[i] = 0;
                    #pragma omp parallel for
                    for (int j = i; j < next_bucket[i]; j++)
                        rank_arr[pos[j]] = i;
                }
            }     

            cnt[rank_arr[n - h]]++;
            b2h[rank_arr[n - h]] = 1;

            // spostamento dei suffissi (sequenziale)
            if (h > 4) {shift_suffixes(n, h, pos, rank_arr, next_bucket, cnt, bh, b2h);}
            else {shift_suffixes_parallel(n, h, pos, rank_arr, next_bucket, cnt, bh, b2h);}

            // aggiornamento finale: parallelizzabile
            #pragma omp parallel for
            for (int i = 0; i < n; i++) {
                pos[rank_arr[i]] = i;
                bh[i] |= b2h[i];
            }
        }

        // Rank finale: parallelizzabile
        #pragma omp parallel for
        for (int i = 0; i < n; i++)
            rank_arr[pos[i]] = i;

        free(cnt);
        free(next_bucket);
        free(bh);
        free(b2h);
    }


static void shift_suffixes(
    int n, int h,
    int *pos,
    int *rank_arr,
    int *next_bucket,
    int *cnt,
    char *bh,
    char *b2h
) {
    for (int i = 0; i < n; i = next_bucket[i]) {

        // spostamento dei suffissi
        for (int j = i; j < next_bucket[i]; j++) {
            int s = pos[j] - h;
            if (s >= 0) {
                int head = rank_arr[s];
                rank_arr[s] = head + cnt[head]++;
                b2h[rank_arr[s]] = 1;
            }
        }

        // pulizia dei flag temporanei
        for (int j = i; j < next_bucket[i]; j++) {
            int s = pos[j] - h;
            if (s >= 0 && b2h[rank_arr[s]]) {
                for (int k = rank_arr[s] + 1;
                     k < n && !bh[k] && b2h[k];
                     k++)
                    b2h[k] = 0;
            }
        }
    }
}

static void shift_suffixes_parallel(
    int n, int h,
    int *pos,
    int *rank_arr,
    int *next_bucket,
    int *cnt,
    char *bh,
    char *b2h
) {
    for (int i = 0; i < n; i = next_bucket[i]) {
        #pragma omp parallel for
        for (int j = i; j < next_bucket[i]; j++) {
            int s = pos[j] - h;
            if (s >= 0) {
                int head = rank_arr[s];
                int new_rank;
                #pragma omp atomic capture
                new_rank = cnt[head]++;

                rank_arr[s] = head + new_rank;
                b2h[rank_arr[s]] = 1;
            }
        }

        //
        #pragma omp parallel for
        for (int j = i; j < next_bucket[i]; j++) {
            int s = pos[j] - h;
            if (s >= 0 && b2h[rank_arr[s]]) {
                for (int k = rank_arr[s] + 1; k < n && !bh[k] && b2h[k]; k++)
                    b2h[k] = 0;
            }
        }
    }
}
