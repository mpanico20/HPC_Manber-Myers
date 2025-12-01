
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

__global__ void calculate_freq(int *c_str, int n, int *c_freq){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        atomicAdd(&c_freq[c_str[idx]], 1);
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
                d_b2h[k] = 0; // Scrittura diretta, no atomic
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


void suffix_sort(const int *str, int n, int *pos, int *rank_arr) {

    int *cnt = calloc(n+1, sizeof(int));
    int *next_bucket = calloc(n+1, sizeof(int));
    char *bh = calloc(n+1, sizeof(char));
    char *b2h = calloc(n+1, sizeof(char));
    int *freq = calloc(ALPHABET_SIZE, sizeof(int));

    int *d_str, *d_freq, *d_pos, *d_rank_arr, *d_cnt;
    char *d_bh, *d_b2h;

    cudaMalloc(&d_str, n*sizeof(int));
    cudaMalloc(&d_freq, ALPHABET_SIZE*sizeof(int));
    cudaMalloc(&d_pos, n*sizeof(int));
    cudaMalloc(&d_rank_arr, n*sizeof(int));
    cudaMalloc(&d_cnt, (n+1)*sizeof(int));
    cudaMalloc(&d_bh, (n+1)*sizeof(char));
    cudaMalloc(&d_b2h, (n+1)*sizeof(char));

    cudaMemcpy(d_str, str, n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_freq, 0, ALPHABET_SIZE*sizeof(int));

    int block = 256;
    int grid = (n + block - 1)/block;

    // ------------------ COUNTING PARALLEL ------------------
    calculate_freq<<<grid, block>>>(d_str, n, d_freq);
    cudaMemcpy(freq, d_freq, ALPHABET_SIZE*sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 1; i < ALPHABET_SIZE; i++)
        freq[i] += freq[i-1];

    for (int i = n-1; i >= 0; i--)
        pos[--freq[str[i]]] = i;

    cudaMemcpy(d_pos, pos, n*sizeof(int), cudaMemcpyHostToDevice);

    // ------------------ INIT BUCKETS PARALLEL ------------------
    init_buckets<<<grid, block>>>(d_str, d_pos, d_bh, d_b2h, n);

    cudaMemcpy(bh, d_bh, (n+1)*sizeof(char), cudaMemcpyDeviceToHost);
    cudaMemcpy(b2h, d_b2h, (n+1)*sizeof(char), cudaMemcpyDeviceToHost);

    // ------------------ MAIN LOOP ------------------
    for (int h = 1; h < n; h <<= 1) {

        // bucket boundaries (CPU)
        int buckets = 0;
        for (int i = 0, j; i < n; i = j) {
            j = i+1;
            while (j < n && !bh[j]) j++;
            next_bucket[i] = j;
            buckets++;
        }
        if (buckets == n) break;

        // ------------------ ASSIGN RANK ------------------
        if (h <= 4) {
            // PARALLEL
            cudaMemcpy(d_rank_arr, rank_arr, n*sizeof(int), cudaMemcpyHostToDevice);

            for (int i = 0; i < n; i = next_bucket[i]) {
                cudaMemset(d_cnt + i, 0, sizeof(int));

                int size = next_bucket[i] - i;
                int g = (size + block - 1) / block;

                assign_rank_parallel<<<g, block>>>(d_pos, d_rank_arr, i, next_bucket[i]);
            }

            cudaMemcpy(rank_arr, d_rank_arr, n*sizeof(int), cudaMemcpyDeviceToHost);
        }
        else {
            // SEQUENTIAL
            for (int i = 0; i < n; i = next_bucket[i]) {
                cnt[i] = 0;
                for (int j = i; j < next_bucket[i]; j++)
                    rank_arr[pos[j]] = i;
            }
        }

        cnt[rank_arr[n-h]]++;
        b2h[rank_arr[n-h]] = 1;

        // ------------------ SHIFT SUFFIXES ------------------
        if (h <= 4) {
            // PARALLEL VERSION
            cudaMemcpy(d_rank_arr, rank_arr, n*sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_bh, bh, n*sizeof(char), cudaMemcpyHostToDevice);
            cudaMemcpy(d_b2h, b2h, n*sizeof(char), cudaMemcpyHostToDevice);
            cudaMemcpy(d_cnt, cnt, (n+1)*sizeof(int), cudaMemcpyHostToDevice);

            for (int i = 0; i < n; i = next_bucket[i]) {
                int size = next_bucket[i] - i;
                int g = (size + block - 1) / block;

                shift_suffixes_first_pass<<<g, block>>>(
                    d_pos, d_rank_arr, d_cnt, d_b2h,
                    i, next_bucket[i], h
                );

                shift_suffixes_second_pass<<<g, block>>>(
                    d_pos, d_rank_arr, d_bh, d_b2h,
                    i, next_bucket[i], h, n
                );
            }

            cudaMemcpy(rank_arr, d_rank_arr, n*sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(b2h, d_b2h, n*sizeof(char), cudaMemcpyDeviceToHost);
        }
        else {
            // SEQUENTIAL VERSION
            shift_suffixes(n, h, pos, rank_arr, next_bucket, cnt, bh, b2h);
        }

        // ------------------ UPDATE POS + BH (parallel always) ------------------
        cudaMemcpy(d_rank_arr, rank_arr, n*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b2h, b2h, n*sizeof(char), cudaMemcpyHostToDevice);
        cudaMemcpy(d_bh, bh, n*sizeof(char), cudaMemcpyHostToDevice);

        update_pos_bh<<<grid, block>>>(d_pos, d_rank_arr, d_bh, d_b2h, n);

        cudaMemcpy(pos, d_pos, n*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(bh, d_bh, n*sizeof(char), cudaMemcpyDeviceToHost);
    }

    // ------------------ FINAL RANK ------------------
    cudaMemcpy(d_pos, pos, n*sizeof(int), cudaMemcpyHostToDevice);
    final_rank<<<grid, block>>>(d_pos, d_rank_arr, n);
    cudaMemcpy(rank_arr, d_rank_arr, n*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_str); cudaFree(d_freq);
    cudaFree(d_pos); cudaFree(d_rank_arr);
    cudaFree(d_cnt); cudaFree(d_bh); cudaFree(d_b2h);

    free(cnt); free(next_bucket); free(bh); free(b2h); free(freq);
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