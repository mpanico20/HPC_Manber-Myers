#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
    size_t sizes_in_bytes[] = {1024*1024, 50*1024*1024, 100*1024*1024, 200*1024*1024, 500*1024*1024};
    size_t n_sizes = sizeof(sizes_in_bytes)/sizeof(sizes_in_bytes[0]);
    const char charset[] = "abcdefghijklmnopqrstuvwxyz";

    srand(time(NULL));

    for (size_t s = 0; s < n_sizes; s++) {
        size_t n_chars = sizes_in_bytes[s] / 4; // 4 byte per int
        char filename[64];
        sprintf(filename, "string_%zuMB.txt", sizes_in_bytes[s]/1024/1024);

        FILE *f = fopen(filename, "w");
        if (!f) { perror("fopen"); return 1; }

        for (size_t i = 0; i < n_chars; i++)
            fputc(charset[rand() % 26], f);

        fclose(f);
        printf("File %s generato con %zu caratteri\n", filename, n_chars);
    }
    return 0;
}
