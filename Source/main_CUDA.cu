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
#include "../Header/suffix_arrays.h"

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Uso: %s <nomefile> <optimization_level> <version to compare>\n", argv[0]);
        return 1;
    }

    int n;
    char *input = load_string_from_file(argv[1], &n);


    // Copia in array di int
    int *str = (int *)malloc(n * sizeof(int));
    if (!str) { fprintf(stderr, "Malloc failed\n"); exit(1); }

    for (int i = 0; i < n; i++)
        str[i] = (unsigned char)input[i];

    int *pos = (int *)malloc(n * sizeof(int));
    int *rank_arr = (int *)malloc(n * sizeof(int));
    if (!pos || !rank_arr ) { fprintf(stderr, "Malloc failed\n"); exit(1); }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    suffix_sort(str, n, pos, rank_arr);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_time_float = 0.0f;
    cudaEventElapsedTime(&elapsed_time_float, start, stop);
    double elapsed_time = (double)elapsed_time_float / 1000.0; // Convert to second

    printf("Suffix Array (pos):\n");
    for (int i = 0; i < 5; i++){
        printf("rank %d: %d\n",i , rank_arr[i]);
        printf("%2d:\n", pos[i]);
    }
    

    printf("Time elapsed: %f\n", elapsed_time);

    free(str);
    free(pos);
    free(rank_arr);
    free(input);

    return 0;
}