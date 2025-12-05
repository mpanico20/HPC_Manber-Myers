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
        perror("Error opening file");
        exit(1);
    }

    //Go to the end to find out the size
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

    //Allocate memory (add 2 bytes for '$' and '\0')
    char *input = malloc(file_size + 2);
    if (!input) {
        fprintf(stderr, "Memory allocation error (%.2f MB required)\n", file_size / (1024.0 * 1024.0));
        fclose(f);
        exit(1);
    }

    //Read the file
    size_t read_bytes = fread(input, 1, file_size, f);
    fclose(f);

    //Removes any trailing newlines
    while (read_bytes > 0 && (input[read_bytes - 1] == '\n' || input[read_bytes - 1] == '\r'))
        read_bytes--;

    //Adds array suffix terminator
    input[read_bytes++] = '$';
    input[read_bytes] = '\0';

    *n = (int)read_bytes;
    return input;
}