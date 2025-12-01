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

__global__ void calculate_freq(const int *c_str, int n, int *c_freq) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(&c_freq[c_str[idx]], 1);
    }
}

__global__ void init_buckets(const int *d_str, const int *d_pos, char *d_bh, char *d_b2h, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        d_bh[i] = (i == 0) || (d_str[d_pos[i]] != d_str[d_pos[i - 1]]);
        d_b2h[i] = 0;
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

    int *cnt = (int *)calloc(n + 1, sizeof(int));
    int *next_bucket = (int *)calloc(n + 1, sizeof(int));
    char *bh = (char *)calloc(n + 1, sizeof(char));
    char *b2h = (char *)calloc(n + 1, sizeof(char));
    int *freq = (int *)calloc(ALPHABET_SIZE, sizeof(int));

    int *d_str, *d_freq, *d_pos, *d_rank_arr;
    char *d_bh, *d_b2h;

    cudaMalloc(&d_str, n * sizeof(int));
    cudaMalloc(&d_freq, ALPHABET_SIZE * sizeof(int));
    cudaMalloc(&d_pos, n * sizeof(int));
    cudaMalloc(&d_rank_arr, n * sizeof(int));
    cudaMalloc(&d_bh, (n + 1) * sizeof(char));
    cudaMalloc(&d_b2h, (n + 1) * sizeof(char));

    cudaMemcpy(d_str, str, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_freq, 0, ALPHABET_SIZE * sizeof(int));

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    // ---------------------- PARALLEL COUNTING ----------------------
    calculate_freq<<<gridSize, blockSize>>>(d_str, n, d_freq);
    cudaMemcpy(freq, d_freq, ALPHABET_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 1; i < ALPHABET_SIZE; i++)
        freq[i] += freq[i - 1];

    for (int i = n - 1; i >= 0; i--)
        pos[--freq[str[i]]] = i;

    cudaMemcpy(d_pos, pos, n * sizeof(int), cudaMemcpyHostToDevice);

    // ---------------------- PARALLEL INIT BUCKET FLAGS ----------------------
    cudaMemset(d_bh, 0, (n + 1) * sizeof(char));
    cudaMemset(d_b2h, 0, (n + 1) * sizeof(char));
    init_buckets<<<gridSize, blockSize>>>(d_str, d_pos, d_bh, d_b2h, n);

    cudaMemcpy(bh, d_bh, (n + 1) * sizeof(char), cudaMemcpyDeviceToHost);
    cudaMemcpy(b2h, d_b2h, (n + 1) * sizeof(char), cudaMemcpyDeviceToHost);

    // ---------------------- SEQUENTIAL MAIN LOOP (LIKE OpenMP) ----------------------
    for (int h = 1; h < n; h <<= 1) {
        int buckets = 0;

        // IDENTICAL TO OpenMP: sequential bucket calculation
        for (int i = 0, j; i < n; i = j) {
            j = i + 1;
            while (j < n && !bh[j]) j++;
            next_bucket[i] = j;
            buckets++;
        }

        if (buckets == n) break;

        // sequential rank assignment (OpenMP does NOT parallelize this)
        for (int i = 0; i < n; i = next_bucket[i]) {
            cnt[i] = 0;
            for (int j = i; j < next_bucket[i]; j++)
                rank_arr[pos[j]] = i;
        }

        cnt[rank_arr[n - h]]++;
        b2h[rank_arr[n - h]] = 1;

        // sequential update of ranks and cleanup
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
                    for (int k = rank_arr[s] + 1; k < n && !bh[k] && b2h[k]; k++)
                        b2h[k] = 0;
                }
            }
        }

        // ---------------------- PARALLEL final update ----------------------
        cudaMemcpy(d_rank_arr, rank_arr, n * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b2h, b2h, (n + 1) * sizeof(char), cudaMemcpyHostToDevice);
        cudaMemcpy(d_bh, bh, (n + 1) * sizeof(char), cudaMemcpyHostToDevice);

        update_pos_bh<<<gridSize, blockSize>>>(d_pos, d_rank_arr, d_bh, d_b2h, n);

        cudaMemcpy(pos, d_pos, n * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(bh, d_bh, (n + 1) * sizeof(char), cudaMemcpyDeviceToHost);
    }

    // ---------------------- PARALLEL FINAL RANK ----------------------
    cudaMemcpy(d_pos, pos, n * sizeof(int), cudaMemcpyHostToDevice);
    final_rank<<<gridSize, blockSize>>>(d_pos, d_rank_arr, n);
    cudaMemcpy(rank_arr, d_rank_arr, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_str);
    cudaFree(d_freq);
    cudaFree(d_pos);
    cudaFree(d_rank_arr);
    cudaFree(d_bh);
    cudaFree(d_b2h);

    free(cnt);
    free(next_bucket);
    free(bh);
    free(b2h);
    free(freq);
}
