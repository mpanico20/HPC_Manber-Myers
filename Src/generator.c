#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
    // Dimensione totale desiderata
    size_t total_sizes[] = {1024*1024, 50*1024*1024, 100*1024*1024, 200*1024*1024, 500*1024*1024};
    size_t n_sizes = sizeof(total_sizes)/sizeof(total_sizes[0]);
    size_t n_arrays = 9; // userà 9 array in memoria
    const char charset[] = "abcdefghijklmnopqrstuvwxyz";

    srand((unsigned int)time(NULL));

    for (size_t s = 0; s < n_sizes; s++) {
        // Calcola dimensione di ciascun file (1/9 della dimensione totale)
        size_t n_chars = total_sizes[s] / n_arrays;

        char filename[64];
        sprintf(filename, "../Data/string_%zuMB.txt", total_sizes[s]/1024/1024);

        FILE *f = fopen(filename, "w");
        if (!f) { perror("fopen"); return 1; }

        for (size_t i = 0; i < n_chars; i++)
            fputc(charset[rand() % 26], f);

        fclose(f);
        printf("File %s generato con %zu caratteri (~%.2f MB)\n", 
               filename, n_chars, n_chars/(1024.0*1024.0));
    }

    return 0;
}
