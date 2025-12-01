#include <stdio.h>
#include <stdlib.h>
#include <time.h>

const char charset[] = "abcdefghijklmnopqrstuvwxyz";

// Genera un array di dimensione "size_in_bytes" in memoria
char *generate_subarray(size_t size_in_bytes) {
    char *str = calloc(size_in_bytes + 1, sizeof(char)); // +1 per terminatore
    if (!str) {
        fprintf(stderr, "Errore calloc: impossibile allocare %.2f MB\n",
                size_in_bytes / (1024.0 * 1024.0));
        exit(1);
    }

    for (size_t i = 0; i < size_in_bytes; i++) {
        str[i] = charset[rand() % 26];
    }

    str[size_in_bytes] = '$'; // terminatore per suffix array
    return str;
}

int main() {
    srand((unsigned int)time(NULL));

    size_t target_sizes[] = {
        1 * 1024 * 1024,   // 1 MB
        50 * 1024 * 1024,  // 50 MB
        100 * 1024 * 1024, // 100 MB
        200 * 1024 * 1024, // 200 MB
        500 * 1024 * 1024  // 500 MB
    };

    size_t n_sizes = sizeof(target_sizes) / sizeof(target_sizes[0]);
    size_t n_arrays = 9; // numero di array che userai

    for (size_t i = 0; i < n_sizes; i++) {
        size_t total_size = target_sizes[i];
        size_t array_size = total_size / n_arrays;

        printf("Generazione %zu array da %.2f MB ciascuno (totale %.2f MB)...\n",
               n_arrays, array_size / (1024.0*1024.0), total_size / (1024.0*1024.0));

        char *arrays[9];
        for (size_t j = 0; j < n_arrays; j++) {
            arrays[j] = generate_subarray(array_size);
        }

        printf("Tutti gli array generati in memoria.\n");

        // esempio: scrittura su file opzionale
        char filename[64];
        sprintf(filename, "string_%zuMB.txt", total_size / (1024*1024));
        FILE *f = fopen(filename, "wb");
        if (!f) {
            perror("Errore apertura file");
            exit(1);
        }
        for (size_t j = 0; j < n_arrays; j++) {
            fwrite(arrays[j], 1, array_size, f);
            free(arrays[j]);
        }
        fclose(f);
        printf("File %s creato con dimensione totale %.2f MB\n\n",
               filename, total_size / (1024.0*1024.0));
    }

    return 0;
}
