#!/bin/bash

for size in 1MB 50MB 100MB 200MB 500MB
do
    ./CUDA_O0 ../Data/string_$size.txt O0
    ./CUDA_O1 ../Data/string_$size.txt O1
    ./CUDA_O2 ../Data/string_$size.txt O2
    ./CUDA_O3 ../Data/string_$size.txt O3

done
