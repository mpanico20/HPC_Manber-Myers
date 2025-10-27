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

void suffix_sort(const int *str, int n, int *pos, int *rank_arr, int *height) {
    int *cnt = calloc(n + 1, sizeof(int));
    int *next_bucket = calloc(n + 1, sizeof(int));
    char *bh = calloc(n + 1, sizeof(char));
    char *b2h = calloc(n + 1, sizeof(char));
    if (!cnt || !next_bucket || !bh || !b2h) { fprintf(stderr, "Alloc failed\n"); exit(1); }
    
    int *freq = calloc(ALPHABET_SIZE, sizeof(int));
    if(!freq) { fprintf(stderr, "Alloc failed\n"); exit(1); }
    
    for (int i = 0; i < n; i++) freq[str[i]]++;
    for (int i = 1; i < ALPHABET_SIZE; i++) freq[i] += freq[i - 1];
    for (int i = 0; i < n; i++) pos[--freq[str[i]]] = i;
    free(freq);

    // ---------- INIZIALIZZA BUCKET ----------
    for (int i = 0; i < n; i++) {
        bh[i] = (i == 0) || (str[pos[i]] != str[pos[i - 1]]);
        b2h[i] = 0;
    }

    // ---------- MANBER & MYERS ----------
    for (int h = 1; h < n; h <<= 1) {
        int buckets = 0;

        for (int i = 0, j; i < n; i = j) {
            j = i + 1;
            while (j < n && !bh[j]) j++;
            next_bucket[i] = j;
            buckets++;
        }

        if (buckets == n) break; // tutti distinti

        for (int i = 0; i < n; i = next_bucket[i]) {
            cnt[i] = 0;
            for (int j = i; j < next_bucket[i]; j++)
                rank_arr[pos[j]] = i;
        }

        cnt[rank_arr[n - h]]++;
        b2h[rank_arr[n - h]] = 1;

        for (int i = 0; i < n; i = next_bucket[i]) {
            for (int j = i; j < next_bucket[i]; j++) {
                int s = pos[j] - h;
                if (s >= 0) {
                    int head = rank_arr[s];
                    rank_arr[s] = head + cnt[head]++;
                    b2h[rank_arr[s]] = 1;
                }
            }

            for (int j = i; j < next_bucket[i]; j++) {
                int s = pos[j] - h;
                if (s >= 0 && b2h[rank_arr[s]]) {
                    for (int k = rank_arr[s] + 1; !bh[k] && b2h[k]; k++)
                        b2h[k] = 0;
                }
            }
        }

        for (int i = 0; i < n; i++) {
            pos[rank_arr[i]] = i;
            bh[i] |= b2h[i];
        }
    }

    // Rank finale
    for (int i = 0; i < n; i++)
        rank_arr[pos[i]] = i;

    free(cnt);
    free(next_bucket);
    free(bh);
    free(b2h);
}

void build_lcp(const int *str, int n, int *pos, int *rank_arr, int *height) {
    for (int i = 0; i < n; i++)
        rank_arr[pos[i]] = i;

    int h = 0;
    height[0] = 0;
    for (int i = 0; i < n; i++) {
        if (rank_arr[i] > 0) {
            int j = pos[rank_arr[i] - 1];
            while (i + h < n && j + h < n && str[i + h] == str[j + h]) h++;
            height[rank_arr[i]] = h;
            if (h > 0) h--;
        }
    }
}

char *load_string_from_file(const char *filename, int *n) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
        perror("Errore apertura file");
        exit(1);
    }

    // Vai alla fine per scoprire la dimensione
    if (fseek(f, 0, SEEK_END) != 0) {
        perror("Errore fseek");
        fclose(f);
        exit(1);
    }

    long file_size = ftell(f);
    if (file_size < 0) {
        perror("Errore ftell");
        fclose(f);
        exit(1);
    }
    rewind(f);

    // Alloca memoria (aggiungiamo 2 byte per '$' e '\0')
    char *input = malloc(file_size + 2);
    if (!input) {
        fprintf(stderr, "Errore allocazione memoria (%.2f MB richiesti)\n", file_size / (1024.0 * 1024.0));
        fclose(f);
        exit(1);
    }

    // Legge il file
    size_t read_bytes = fread(input, 1, file_size, f);
    fclose(f);

    // Rimuove eventuali newline finali
    while (read_bytes > 0 && (input[read_bytes - 1] == '\n' || input[read_bytes - 1] == '\r'))
        read_bytes--;

    // Aggiunge il terminatore del suffisso array
    input[read_bytes++] = '$';
    input[read_bytes] = '\0';

    *n = (int)read_bytes;
    printf("Letti %d caratteri (%.2f MB)\n", *n, *n / (1024.0 * 1024.0));

    return input;
}