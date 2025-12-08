#!/bin/bash

# ---------------------- Configurazioni ----------------------
folders=(1 50 100 200 500)
versions=(O0 O1 O2 O3)
sizes=(1 50 100 200 500)
parallel_name="OpenMP_v2"
script="grapg.py"

# ---------------------- Loop su tutte le combinazioni ----------------------
for folder in "${folders[@]}"; do
    for serial in "${versions[@]}"; do
        for size in "${sizes[@]}"; do
            csv_file="Measures/OpenMP/${folder}/times_${serial}.csv"
            final_size="${size}MB"

            # Controllo se il file CSV esiste
            if [ ! -f "$csv_file" ]; then
                echo "[WARNING] File not found: $csv_file"
                continue
            fi

            python3 "$script" "$csv_file" "$serial" "$parallel_name" "$final_size"
        done
    done
done
