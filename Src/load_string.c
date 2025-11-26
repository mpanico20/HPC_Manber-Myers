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
#include "../Header/suffix_arrays.h"
#include <stdio.h>
#include <stdlib.h>

char *load_string_from_file(const char *filename, int *n) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
        perror("Errore apertura file");
        exit(1);
    }

    // Vai alla fine per scoprire la dimensione
    if (fseek(f, 0, SEEK_END) != 0) {
        perror("Errore fseek");
        fclose(f);
        exit(1);
    }

    long file_size = ftell(f);
    if (file_size < 0) {
        perror("Errore ftell");
        fclose(f);
        exit(1);
    }
    rewind(f);

    // Alloca memoria (aggiungiamo 2 byte per '$' e '\0')
    char *input = malloc(file_size + 2);
    if (!input) {
        fprintf(stderr, "Errore allocazione memoria (%.2f MB richiesti)\n", file_size / (1024.0 * 1024.0));
        fclose(f);
        exit(1);
    }

    // Legge il file
    size_t read_bytes = fread(input, 1, file_size, f);
    fclose(f);

    // Rimuove eventuali newline finali
    while (read_bytes > 0 && (input[read_bytes - 1] == '\n' || input[read_bytes - 1] == '\r'))
        read_bytes--;

    // Aggiunge il terminatore del suffisso array
    input[read_bytes++] = '$';
    input[read_bytes] = '\0';

    *n = (int)read_bytes;
    return input;
}