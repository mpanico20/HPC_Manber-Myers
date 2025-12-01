#!/bin/bash

for size in 1MB 50MB 100MB 200MB 500MB
do
    ./OpenMP_v1_O0 ../Data/string_$size.txt O0 8 1
done
