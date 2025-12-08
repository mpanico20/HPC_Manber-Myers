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
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/scan.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include "../Header/suffix_arrays.h"

#define CHECK_CUDA(call) do {                                    \
    cudaError_t e = (call);                                      \
    if (e != cudaSuccess) {                                      \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
      exit(1);                                                   \
    } } while(0)


struct make_key_functor {
    uint32_t* rank_ptr;
    int n;
    int hstep;

    make_key_functor(uint32_t* r, int n_, int h)
        : rank_ptr(r), n(n_), hstep(h) {}

    __host__ __device__
    uint64_t operator()(int i) const {
        uint64_t a = (uint32_t)rank_ptr[i];
        uint64_t b = (uint32_t)((i + hstep < n) ? rank_ptr[i + hstep] : 0xFFFFFFFFu);
        return (a << 32) | b;
    }
};

__global__ void key_diff_flags(const uint64_t* keys_sorted, int n, int* flags) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    if (tid == 0) {
        flags[0] = 1;
    } else {
        flags[tid] = (keys_sorted[tid] != keys_sorted[tid - 1]);
    }
}


__global__ void scatter_ranks(const int *idx_sorted, const int *scanned, int n,
                              uint32_t *rank_by_pos, int *pos_out)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    int orig_idx = idx_sorted[tid];
    int r = scanned[tid];

    rank_by_pos[orig_idx] = r;
    pos_out[r] = orig_idx;
}


void suffix_sort(const int *h_str, int n, int *h_pos, int *h_rank)
{
    thrust::device_vector<int> d_str(h_str, h_str + n);
    thrust::device_vector<uint32_t> d_rank = d_str;
    thrust::device_vector<uint32_t> d_new_rank(n);
    thrust::device_vector<int> d_pos(n), d_pos_tmp(n);

    thrust::device_vector<uint64_t> d_key(n);
    thrust::device_vector<uint64_t> d_keys_sorted(n);
    thrust::device_vector<int> d_idx(n), d_idx_sorted(n);
    thrust::sequence(d_idx.begin(), d_idx.end());

    thrust::device_vector<int> d_flags(n);
    thrust::device_vector<int> d_scanned(n);

    const int BLOCK = 256;
    int hstep = 1;

    while (hstep < n)
    {
        thrust::transform(
            thrust::make_counting_iterator<int>(0),
            thrust::make_counting_iterator<int>(n),
            d_key.begin(),
            make_key_functor(thrust::raw_pointer_cast(d_rank.data()), n, hstep)
        );

        thrust::copy(d_key.begin(), d_key.end(), d_keys_sorted.begin());
        thrust::sequence(d_idx.begin(), d_idx.end());
        thrust::copy(d_idx.begin(), d_idx.end(), d_idx_sorted.begin());

        thrust::sort_by_key(d_keys_sorted.begin(), d_keys_sorted.end(), d_idx_sorted.begin());

        int grid = (n + BLOCK - 1) / BLOCK;
        key_diff_flags<<<grid, BLOCK>>>(
            thrust::raw_pointer_cast(d_keys_sorted.data()),
            n,
            thrust::raw_pointer_cast(d_flags.data())
        );
        CHECK_CUDA(cudaGetLastError());

        thrust::inclusive_scan(d_flags.begin(), d_flags.end(), d_scanned.begin());
        thrust::transform(d_scanned.begin(), d_scanned.end(),
                          d_scanned.begin(),
                          thrust::placeholders::_1 - 1);

        scatter_ranks<<<grid, BLOCK>>>(
            thrust::raw_pointer_cast(d_idx_sorted.data()),
            thrust::raw_pointer_cast(d_scanned.data()),
            n,
            thrust::raw_pointer_cast(d_new_rank.data()),
            thrust::raw_pointer_cast(d_pos_tmp.data())
        );
        CHECK_CUDA(cudaGetLastError());

        d_rank.swap(d_new_rank);
        d_pos.swap(d_pos_tmp);

        int last_group;
        CHECK_CUDA(cudaMemcpy(&last_group,
            thrust::raw_pointer_cast(d_scanned.data()) + (n - 1),
            sizeof(int),
            cudaMemcpyDeviceToHost));

        if (last_group + 1 == n)
            break;

        hstep <<= 1;
    }

    thrust::copy(d_pos.begin(), d_pos.end(), h_pos);
    thrust::copy(d_rank.begin(), d_rank.end(), h_rank);
}

