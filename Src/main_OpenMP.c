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
    if (argc < 4) {
        printf("Uso: %s <nomefile> <optimization level> <number of thread> <parallel version>\n", argv[0]);
        return 1;
    }

    int num_thread = atoi(argv[3]);
    if (num_thread > omp_get_max_threads()){ printf("Number of thread not supported!\n"); exit(1);}
    omp_set_num_threads(num_thread);

    int version_p = atoi(argv[4]);

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

    double start_par = omp_get_wtime();
    suffix_sort(str, n, pos, rank_arr);
    double end_par = omp_get_wtime();
    build_lcp(str, n, pos, rank_arr, height);

    double parallel_time = end_par - start_par;
    int mb = extractMB(argv[1]);
    char filename[250];
    sprintf(filename, "../Measures/OpenMP/%d/times_%s.csv", mb, argv[2]);
    FILE *csv_serial = fopen(filename, "r");
    double sequential_time;
    char line[256];

    if (fgets(line, sizeof(line), csv_serial) == NULL) {
        fprintf(stderr, "Error in reading the file!\n");
        exit(1);
    }

    if (fscanf(csv_serial, "%*[^;];%*[^;];%lf", &sequential_time) != 1){
        fprintf(stderr, "Error in reading the time in csv file!\n");
        exit(1);
    }

    fclose(csv_serial);

    double speedup = calculateSpeedup(sequential_time, parallel_time);
    double efficiency = calculateEfficiency(speedup, num_thread);

    char filenameCsv[250];
    sprintf(filenameCsv, "../Measures/OpenMP/%d/times_%s.csv",mb,argv[2]);

    FILE *csv;
    csv = fopen(filenameCsv, "a");

    if (csv == NULL) {
        printf("Error!");
        exit(1);
    }

    if (ftell(csv) == 0){
        fprintf(csv, "Version;Num of thread;Elapsed Time (s);Speedup;Efficency\n");
    }
    fprintf(csv, "OpenMP_v%d;%d;%f;%f;%f\n",version_p,num_thread, parallel_time, speedup,efficiency);
    fclose(csv);


    free(str);
    free(pos);
    free(rank_arr);
    free(height);
    free(input);

    return 0;
}
