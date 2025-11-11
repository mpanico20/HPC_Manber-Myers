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