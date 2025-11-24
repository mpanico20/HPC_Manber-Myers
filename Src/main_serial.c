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
#include <time.h>
#include <ctype.h>
#include "../Header/suffix_arrays.h"

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Uso: %s <nomefile> <optimization_level> <version to compare>\n", argv[0]);
        return 1;
    }

    int n;
    char *input = load_string_from_file(argv[1], &n);

    // Copia in array di int
    int *str = malloc(n * sizeof(int));
    if (!str) { fprintf(stderr, "Malloc failed\n"); exit(1); }

    for (int i = 0; i < n; i++)
        str[i] = (unsigned char)input[i];

    int *pos = malloc(n * sizeof(int));
    int *rank_arr = malloc(n * sizeof(int));
    int *height = malloc(n * sizeof(int));
    if (!pos || !rank_arr || !height) { fprintf(stderr, "Malloc failed\n"); exit(1); }

    clock_t start_seq = clock();
    suffix_sort(str, n, pos, rank_arr);
    clock_t end_seq = clock();
    build_lcp(str, n, pos, rank_arr, height);

    double sequential_time = ((double)(end_seq - start_seq)) / CLOCKS_PER_SEC;

    double speedup = (calculateSpeedup(sequential_time, sequential_time));
    double efficiency = (calculateEfficiency(speedup,1));

    char filename[250];
    int mb = extractMB(argv[1]);
    sprintf(filename, "../Measures/%s/%d/times_%s.csv",argv[3],mb,argv[2]);

    FILE *csv;
    csv = fopen(filename, "a");

    if (csv == NULL) {
        printf("Error!");
        exit(1);
    }

    if (ftell(csv) == 0){
        fprintf(csv, "Version;Num of thread;Elapsed Time (s);Speedup;Efficency\n");
    }

    fprintf(csv, "Serial;1;%f;%f;%f%%\n", sequential_time, speedup,efficiency);
    fclose(csv);

    free(str);
    free(pos);
    free(rank_arr);
    free(height);
    free(input);

    return 0;
}