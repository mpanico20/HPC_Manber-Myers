#!/bin/bash

# Valori da sostituire: 1, 50, 100, 200, 500
values=(1 50 100 200 500)

for v in "${values[@]}"; do
    ./sequential_O1 "../Data/string_${v}MB.txt" O1 OpenMP
done

