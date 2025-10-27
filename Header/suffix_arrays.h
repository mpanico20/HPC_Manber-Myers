#ifndef SUFFIX_ARRAYS_H
#define SUFFIX_ARRAYS_H

#define ALPHABET_SIZE 256

void suffix_sort(const int *str, int n, int *pos, int *rank_arr, int *height);
void build_lcp(const int *str, int n, int *pos, int *rank_arr, int *height);
char *load_string_from_file(const char *filename, int *n);
double calculateSpeedup(double sequentialTime, double parallelTime);
double calculateEfficiency(double speedup, int numProcessors);

#endif