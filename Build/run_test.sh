#!/bin/bash

for size in 1MB 50MB 100MB 200MB 500MB
do
    for threads in 1 2 4 8
    do
        ./OpenMP_v1_O3 ../Data/string_$size.txt O3 $threads 1
    done
done
