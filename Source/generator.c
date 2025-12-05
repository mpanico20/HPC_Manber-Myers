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
#include <time.h>

int main() {
    //Desired total size
    size_t total_sizes[] = {1024*1024, 50*1024*1024, 100*1024*1024, 200*1024*1024, 500*1024*1024};
    size_t n_sizes = sizeof(total_sizes)/sizeof(total_sizes[0]);
    size_t n_arrays = 8;
    const char charset[] = "abcdefghijklmnopqrstuvwxyz";

    srand((unsigned int)time(NULL));

    for (size_t s = 0; s < n_sizes; s++) {
        size_t n_chars = total_sizes[s] / n_arrays;

        char filename[64];
        sprintf(filename, "../Data/string_%zuMB.txt", total_sizes[s]/1024/1024);

        FILE *f = fopen(filename, "w");
        if (!f) { perror("fopen"); return 1; }

        for (size_t i = 0; i < n_chars; i++)
            fputc(charset[rand() % 26], f);

        fclose(f);
    }

    return 0;
}
