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
        printf("Uso: %s <nomefile> <optimization level>\n", argv[0]);
        return 1;
    }

    int n;
    char *input = load_string_from_file(argv[1], &n);

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

    float cuda_time = 0;
    cudaEventElapsedTime(&cuda_time, start, stop);
    cuda_time = cuda_time / 1000.0f;

    int mb = extractMB(argv[1]);
    char filename[250];
    sprintf(filename, "Measures/CUDA/%d/times_%s.csv", mb, argv[2]);
    FILE *csv_serial = fopen(filename, "r");
    double sequential_time;
    char line[256];

    if (fgets(line, sizeof(line), csv_serial) == NULL) {
        fprintf(stderr, "Error in reading the file!\n");
        exit(1);
    }

    if (fscanf(csv_serial, "%*[^;];%lf", &sequential_time) != 1){
        fprintf(stderr, "Error in reading the time in csv file!\n");
        exit(1);
    }

    fclose(csv_serial);

    double speedup = calculateSpeedup(sequential_time, cuda_time);

    char filenameCsv[250];
    sprintf(filenameCsv, "Measures/CUDA/%d/times_%s.csv",mb,argv[2]);

    FILE *csv;
    csv = fopen(filenameCsv, "a");

    if (csv == NULL) {
        printf("Error!");
        exit(1);
    }

    if (ftell(csv) == 0){
        fprintf(csv, "Version;Elapsed Time (s);Speedup\n");
    }
    fprintf(csv, "CUDA;%f;%f\n", cuda_time, speedup);
    fclose(csv);


    free(str);
    free(pos);
    free(rank_arr);
    free(input);
    return 0;
}
