#!/bin/bash

folders=(1 50 100 200 500)

opts=(O0 O1 O2 O3)

# Ciclo principale
for folder in "${folders[@]}"; do
    for opt in "${opts[@]}"; do
        file_path="Measures/CUDA/${folder}/times_${opt}.csv"
        size_label="${folder}MB"
        cmd="python3 graph_CUDA.py ${file_path} ${opt} ${size_label}"
        echo "Eseguo: $cmd"
        $cmd
    done
done
