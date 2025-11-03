
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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "../Header/suffix_arrays.h"

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Uso: %s <nomefile>\n", argv[0]);
        return 1;
    }

    int n;
    char *input = load_string_from_file(argv[1], &n);

    // Copia in array di int
    int *str = malloc(n * sizeof(int));
    for (int i = 0; i < n; i++)
        str[i] = (unsigned char)input[i];

    int *pos = malloc(n * sizeof(int));
    int *rank_arr = malloc(n * sizeof(int));
    int *height = malloc(n * sizeof(int));

    double start_par = omp_get_wtime();
    suffix_sort(str, n, pos, rank_arr);
    double end_par = omp_get_wtime();
    build_lcp(str, n, pos, rank_arr, height);

    double sequential_time = end_par - start_par;
    printf("Suffix Array (pos):\n");
    for (int i = 0; i < 5; i++)
        printf("%2d:\n", pos[i]);

    printf("\nLCP array:\n");
    for (int i = 0; i < 5; i++)
        printf("%d ", height[i]);
    printf("\n");

    int max_len = 0;
    int start_index = 0;

    for (int i = 1; i < n; i++) {
        if (height[i] > max_len) {
            max_len = height[i];
            start_index = pos[i-1];  // puoi anche scegliere pos[i-1], sono equivalenti
        }
    }

    printf("Max substring length: %d\n", max_len);
    printf("Substring: ");
    printf("Pos: %d\n", start_index);
    for (int j = 0; j < max_len; j++)
        putchar(str[start_index + j]);
    printf("\n");
    printf("Sequantian time: %f\n", sequential_time);

    free(str);
    free(pos);
    free(rank_arr);
    free(height);
    free(input);

    return 0;
}
