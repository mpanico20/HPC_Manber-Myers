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