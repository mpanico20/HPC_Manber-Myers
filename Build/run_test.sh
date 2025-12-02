#!/bin/bash

for size in 1MB 50MB 100MB 200MB 500MB
do
    for threads in 1 2 4 8
    do
        ./OpenMP_v2_O0 ../Data/string_$size.txt O0 $threads 2
        ./OpenMP_v2_O1 ../Data/string_$size.txt O1 $threads 2
        ./OpenMP_v2_O2 ../Data/string_$size.txt O2 $threads 2
        ./OpenMP_v2_O3 ../Data/string_$size.txt O3 $threads 2
    done
done
