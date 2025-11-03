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

__global__ void calculate_freq(int *c_str, int n, int *c_freq){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        atomicAdd(&c_freq[c_str[idx]], 1);
    }
}

__global__ void assign_pos(const int *d_str, int *d_pos, int *d_freq, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int val = d_str[idx];
        int pos_idx = atomicSub(&d_freq[val], 1) - 1;
        d_pos[pos_idx] = idx;
    }
}

__global__ void init_buckets(const int *d_str, int *d_pos, char *d_bh, char *d_b2h, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        d_bh[i] = (i == 0) || (d_str[d_pos[i]] != d_str[d_pos[i - 1]]);
        d_b2h[i] = 0;
    }
}

__global__ void assign_rank(int *d_pos, int *d_rank_arr, int d_i, int d_j) {
    int j = blockIdx.x * blockDim.x + threadIdx.x + d_i;
    if (j < d_j) {
        d_rank_arr[d_pos[j]] = d_i;
    }
}

__global__ void update_rank_offset(int *d_pos, int *d_rank_arr, int *d_cnt, char *d_b2h, int h, int start, int end) {
    int j = blockIdx.x * blockDim.x + threadIdx.x + start;
    if (j < end) {
        int s = d_pos[j] - h;
        if (s >= 0) {
            int head = d_rank_arr[s];
            int offset = atomicAdd(&d_cnt[head], 1);
            int new_rank = head + offset;
            d_rank_arr[s] = new_rank;
            d_b2h[new_rank] = 1;
        }
    }
}

__global__ void clean_b2h(int *d_pos, int *d_rank_arr, char *d_bh, char *d_b2h, int h, int start, int end, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x + start;
    if (j < end) {
        int s = d_pos[j] - h;
        if (s >= 0 && d_b2h[d_rank_arr[s]]) {
            for (int k = d_rank_arr[s] + 1; k < n && !d_bh[k] && d_b2h[k]; k++) {
                d_b2h[k] = 0;
            }
        }
    }
}

__global__ void update_pos_bh(int *d_pos, int *d_rank_arr, char *d_bh, char *d_b2h, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        d_pos[d_rank_arr[i]] = i;
        d_bh[i] |= d_b2h[i];
    }
}

__global__ void final_rank(int *d_pos, int *d_rank_arr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        d_rank_arr[d_pos[i]] = i;
    }
}