
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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <ctype.h>
#include "../Header/suffix_arrays.h"

int extractMB(char *filepath) {
    // Estrai solo il nome del file senza percorso
    char *last_slash = strrchr(filepath, '/');
    char *filename = (last_slash) ? last_slash + 1 : filepath;

    // Rimuovi estensione, se c'è
    char *dot = strrchr(filename, '.');
    if (dot) *dot = '\0';

    // Trova l'ultimo underscore
    char *underscore = strrchr(filename, '_');
    if (!underscore) {
        fprintf(stderr, "Formato del file non valido (nessun '_')\n");
        return -1;
    }

    underscore++; // punta al numero prima di "MB"

    // Copia solo la parte numerica fino a "MB"
    char num_str[20] = {0};
    int i = 0;
    while (isdigit(underscore[i])) {
        num_str[i] = underscore[i];
        i++;
    }

    if (i == 0) {
        fprintf(stderr, "Nessun numero trovato nel nome del file\n");
        return -1;
    }

    return atoi(num_str);
}