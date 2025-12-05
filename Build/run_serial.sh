#!/bin/bash

for size in 1MB 50MB 100MB 200MB 500MB
do
    ./sequential_O0 ../Data/string_$size.txt O0

done
