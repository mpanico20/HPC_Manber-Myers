#!/bin/bash

for size in 1MB 50MB 100MB 200MB 500MB
do
    ./sequential_O1 ../Data/string_$size.txt O1
    ./sequential_O2 ../Data/string_$size.txt O2
    ./sequential_O3 ../Data/string_$size.txt O3

done
