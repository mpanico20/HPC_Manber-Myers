CC=gcc
NV=nvcc
ARC=50

BUILD_DIR=Build
SOURCE_DIR=Source
MEASURES_OpenMP_DIR=Measures/OpenMP
MEASURES_CUDA_DIR=Measures/CUDA
DATA_DIR=Data
GRAPH_DIR=Graphs

all: clean generate exe_serial exe_omp_v1 exe_omp_v2 exe_cuda run run_omp_v1 run_omp_v2 run_cuda run_graphs_omp_v1 run_graphs_omp_v2 run_graphs_cuda

generate: clean_string generator run_generator

exe_serial: sequential_O0 sequential_O1 sequential_O2 sequential_O3

exe_omp_v1: OpenMP_v1_O0 OpenMP_v1_O1 OpenMP_v1_O2 OpenMP_v1_O3

exe_omp_v2: OpenMP_v2_O0 OpenMP_v2_O1 OpenMP_v2_O2 OpenMP_v2_O3

exe_cuda: CUDA_O0 CUDA_O1 CUDA_O2 CUDA_O3


generator: $(SOURCE_DIR)/generator.c
	$(CC) -o $(BUILD_DIR)/generator $(SOURCE_DIR)/generator.c

sequential_O0: $(SOURCE_DIR)/main_serial.c $(SOURCE_DIR)/time.c $(SOURCE_DIR)/suffix_arrays.c $(SOURCE_DIR)/load_string.c $(SOURCE_DIR)/extractmb.c
	$(CC) -o $(BUILD_DIR)/sequential_O0 $(SOURCE_DIR)/main_serial.c $(SOURCE_DIR)/time.c $(SOURCE_DIR)/suffix_arrays.c $(SOURCE_DIR)/load_string.c $(SOURCE_DIR)/extractmb.c

sequential_O1: $(SOURCE_DIR)/main_serial.c $(SOURCE_DIR)/time.c $(SOURCE_DIR)/suffix_arrays.c $(SOURCE_DIR)/load_string.c $(SOURCE_DIR)/extractmb.c
	$(CC) -O1 -o $(BUILD_DIR)/sequential_O1 $(SOURCE_DIR)/main_serial.c $(SOURCE_DIR)/time.c $(SOURCE_DIR)/suffix_arrays.c $(SOURCE_DIR)/load_string.c $(SOURCE_DIR)/extractmb.c

sequential_O2: $(SOURCE_DIR)/main_serial.c $(SOURCE_DIR)/time.c $(SOURCE_DIR)/suffix_arrays.c $(SOURCE_DIR)/load_string.c $(SOURCE_DIR)/extractmb.c
	$(CC) -O2 -o $(BUILD_DIR)/sequential_O2 $(SOURCE_DIR)/main_serial.c $(SOURCE_DIR)/time.c $(SOURCE_DIR)/suffix_arrays.c $(SOURCE_DIR)/load_string.c $(SOURCE_DIR)/extractmb.c

sequential_O3: $(SOURCE_DIR)/main_serial.c $(SOURCE_DIR)/time.c $(SOURCE_DIR)/suffix_arrays.c $(SOURCE_DIR)/load_string.c $(SOURCE_DIR)/extractmb.c
	$(CC) -O3 -o $(BUILD_DIR)/sequential_O3 $(SOURCE_DIR)/main_serial.c $(SOURCE_DIR)/time.c $(SOURCE_DIR)/suffix_arrays.c $(SOURCE_DIR)/load_string.c $(SOURCE_DIR)/extractmb.c

OpenMP_v1_O0: $(SOURCE_DIR)/main_OpenMP.c $(SOURCE_DIR)/time.c $(SOURCE_DIR)/suffix_arrays_OpenMP_v1.c $(SOURCE_DIR)/load_string.c $(SOURCE_DIR)/extractmb.c
	$(CC) -fopenmp -o $(BUILD_DIR)/OpenMP_v1_O0 $(SOURCE_DIR)/main_OpenMP.c $(SOURCE_DIR)/time.c $(SOURCE_DIR)/suffix_arrays_OpenMP_v1.c $(SOURCE_DIR)/load_string.c $(SOURCE_DIR)/extractmb.c

OpenMP_v1_O1: $(SOURCE_DIR)/main_OpenMP.c $(SOURCE_DIR)/time.c $(SOURCE_DIR)/suffix_arrays_OpenMP_v1.c $(SOURCE_DIR)/load_string.c $(SOURCE_DIR)/extractmb.c
	$(CC) -fopenmp -O1 -o $(BUILD_DIR)/OpenMP_v1_O1 $(SOURCE_DIR)/main_OpenMP.c $(SOURCE_DIR)/time.c $(SOURCE_DIR)/suffix_arrays_OpenMP_v1.c $(SOURCE_DIR)/load_string.c $(SOURCE_DIR)/extractmb.c

OpenMP_v1_O2: $(SOURCE_DIR)/main_OpenMP.c $(SOURCE_DIR)/time.c $(SOURCE_DIR)/suffix_arrays_OpenMP_v1.c $(SOURCE_DIR)/load_string.c $(SOURCE_DIR)/extractmb.c
	$(CC) -fopenmp -O2 -o $(BUILD_DIR)/OpenMP_v1_O2 $(SOURCE_DIR)/main_OpenMP.c $(SOURCE_DIR)/time.c $(SOURCE_DIR)/suffix_arrays_OpenMP_v1.c $(SOURCE_DIR)/load_string.c $(SOURCE_DIR)/extractmb.c

OpenMP_v1_O3: $(SOURCE_DIR)/main_OpenMP.c $(SOURCE_DIR)/time.c $(SOURCE_DIR)/suffix_arrays_OpenMP_v1.c $(SOURCE_DIR)/load_string.c $(SOURCE_DIR)/extractmb.c
	$(CC) -fopenmp -O3 -o $(BUILD_DIR)/OpenMP_v1_O3 $(SOURCE_DIR)/main_OpenMP.c $(SOURCE_DIR)/time.c $(SOURCE_DIR)/suffix_arrays_OpenMP_v1.c $(SOURCE_DIR)/load_string.c $(SOURCE_DIR)/extractmb.c

OpenMP_v2_O0: $(SOURCE_DIR)/main_OpenMP.c $(SOURCE_DIR)/time.c $(SOURCE_DIR)/suffix_arrays_OpenMP_v2.c $(SOURCE_DIR)/load_string.c $(SOURCE_DIR)/extractmb.c
	$(CC) -fopenmp -o $(BUILD_DIR)/OpenMP_v2_O0 $(SOURCE_DIR)/main_OpenMP.c $(SOURCE_DIR)/time.c $(SOURCE_DIR)/suffix_arrays_OpenMP_v2.c $(SOURCE_DIR)/load_string.c $(SOURCE_DIR)/extractmb.c

OpenMP_v2_O1: $(SOURCE_DIR)/main_OpenMP.c $(SOURCE_DIR)/time.c $(SOURCE_DIR)/suffix_arrays_OpenMP_v2.c $(SOURCE_DIR)/load_string.c $(SOURCE_DIR)/extractmb.c
	$(CC) -fopenmp -O1 -o $(BUILD_DIR)/OpenMP_v2_O1 $(SOURCE_DIR)/main_OpenMP.c $(SOURCE_DIR)/time.c $(SOURCE_DIR)/suffix_arrays_OpenMP_v2.c $(SOURCE_DIR)/load_string.c $(SOURCE_DIR)/extractmb.c

OpenMP_v2_O2: $(SOURCE_DIR)/main_OpenMP.c $(SOURCE_DIR)/time.c $(SOURCE_DIR)/suffix_arrays_OpenMP_v2.c $(SOURCE_DIR)/load_string.c $(SOURCE_DIR)/extractmb.c
	$(CC) -fopenmp -O2 -o $(BUILD_DIR)/OpenMP_v2_O2 $(SOURCE_DIR)/main_OpenMP.c $(SOURCE_DIR)/time.c $(SOURCE_DIR)/suffix_arrays_OpenMP_v2.c $(SOURCE_DIR)/load_string.c $(SOURCE_DIR)/extractmb.c

OpenMP_v2_O3: $(SOURCE_DIR)/main_OpenMP.c $(SOURCE_DIR)/time.c $(SOURCE_DIR)/suffix_arrays_OpenMP_v2.c $(SOURCE_DIR)/load_string.c $(SOURCE_DIR)/extractmb.c
	$(CC) -fopenmp -O3 -o $(BUILD_DIR)/OpenMP_v2_O3 $(SOURCE_DIR)/main_OpenMP.c $(SOURCE_DIR)/time.c $(SOURCE_DIR)/suffix_arrays_OpenMP_v2.c $(SOURCE_DIR)/load_string.c $(SOURCE_DIR)/extractmb.c

CUDA_O0: $(SOURCE_DIR)/main_CUDA.cu $(SOURCE_DIR)/time.c $(SOURCE_DIR)/suffix_arrays_CUDA.cu $(SOURCE_DIR)/load_string.c $(SOURCE_DIR)/extractmb.c
	$(NV) -o $(BUILD_DIR)/CUDA_O0 -arch=sm_$(ARC) --extended-lambda $(SOURCE_DIR)/suffix_arrays_CUDA.cu $(SOURCE_DIR)/main_CUDA.cu $(SOURCE_DIR)/time.c $(SOURCE_DIR)/load_string.c $(SOURCE_DIR)/extractmb.c

CUDA_O1: $(SOURCE_DIR)/main_CUDA.cu $(SOURCE_DIR)/time.c $(SOURCE_DIR)/suffix_arrays_CUDA.cu $(SOURCE_DIR)/load_string.c $(SOURCE_DIR)/extractmb.c
	$(NV) -O1 -o $(BUILD_DIR)/CUDA_O1 -arch=sm_$(ARC) --extended-lambda $(SOURCE_DIR)/suffix_arrays_CUDA.cu $(SOURCE_DIR)/main_CUDA.cu $(SOURCE_DIR)/time.c $(SOURCE_DIR)/load_string.c $(SOURCE_DIR)/extractmb.c

CUDA_O2: $(SOURCE_DIR)/main_CUDA.cu $(SOURCE_DIR)/time.c $(SOURCE_DIR)/suffix_arrays_CUDA.cu $(SOURCE_DIR)/load_string.c $(SOURCE_DIR)/extractmb.c
	$(NV) -O2 -o $(BUILD_DIR)/CUDA_O2 -arch=sm_$(ARC) --extended-lambda $(SOURCE_DIR)/suffix_arrays_CUDA.cu $(SOURCE_DIR)/main_CUDA.cu $(SOURCE_DIR)/time.c $(SOURCE_DIR)/load_string.c $(SOURCE_DIR)/extractmb.c

CUDA_O3: $(SOURCE_DIR)/main_CUDA.cu $(SOURCE_DIR)/time.c $(SOURCE_DIR)/suffix_arrays_CUDA.cu $(SOURCE_DIR)/load_string.c $(SOURCE_DIR)/extractmb.c
	$(NV) -O3 -o $(BUILD_DIR)/CUDA_O3 -arch=sm_$(ARC) --extended-lambda $(SOURCE_DIR)/suffix_arrays_CUDA.cu $(SOURCE_DIR)/main_CUDA.cu $(SOURCE_DIR)/time.c $(SOURCE_DIR)/load_string.c $(SOURCE_DIR)/extractmb.c

run: run_serialO0 run_serialO1 run_serialO2 run_serialO3

run_omp_v1: run_omp_v1_O0 run_omp_v1_O1 run_omp_v1_O2 run_omp_v1_O3

run_omp_v2: run_omp_v2_O0 run_omp_v2_O1 run_omp_v2_O2 run_omp_v2_O3

run_cuda: run_cuda_O0 run_cuda_O1 run_cuda_O2 run_cuda_O3

run_generator: generator
	$(BUILD_DIR)/generator

run_serialO0: sequential_O0
	$(BUILD_DIR)/sequential_O0 $(DATA_DIR)/string_1MB.txt O0
	$(BUILD_DIR)/sequential_O0 $(DATA_DIR)/string_50MB.txt O0
	$(BUILD_DIR)/sequential_O0 $(DATA_DIR)/string_100MB.txt O0
	$(BUILD_DIR)/sequential_O0 $(DATA_DIR)/string_200MB.txt O0
	$(BUILD_DIR)/sequential_O0 $(DATA_DIR)/string_500MB.txt O0

run_serialO1: sequential_O1
	$(BUILD_DIR)/sequential_O1 $(DATA_DIR)/string_1MB.txt O1
	$(BUILD_DIR)/sequential_O1 $(DATA_DIR)/string_50MB.txt O1
	$(BUILD_DIR)/sequential_O1 $(DATA_DIR)/string_100MB.txt O1
	$(BUILD_DIR)/sequential_O1 $(DATA_DIR)/string_200MB.txt O1
	$(BUILD_DIR)/sequential_O1 $(DATA_DIR)/string_500MB.txt O1

run_serialO2: sequential_O2
	$(BUILD_DIR)/sequential_O2 $(DATA_DIR)/string_1MB.txt O2
	$(BUILD_DIR)/sequential_O2 $(DATA_DIR)/string_50MB.txt O2
	$(BUILD_DIR)/sequential_O2 $(DATA_DIR)/string_100MB.txt O2
	$(BUILD_DIR)/sequential_O2 $(DATA_DIR)/string_200MB.txt O2
	$(BUILD_DIR)/sequential_O2 $(DATA_DIR)/string_500MB.txt O2

run_serialO3: sequential_O3
	$(BUILD_DIR)/sequential_O3 $(DATA_DIR)/string_1MB.txt O3
	$(BUILD_DIR)/sequential_O3 $(DATA_DIR)/string_50MB.txt O3
	$(BUILD_DIR)/sequential_O3 $(DATA_DIR)/string_100MB.txt O3
	$(BUILD_DIR)/sequential_O3 $(DATA_DIR)/string_200MB.txt O3
	$(BUILD_DIR)/sequential_O3 $(DATA_DIR)/string_500MB.txt O3

run_omp_v1_O0: OpenMP_v1_O0
	$(BUILD_DIR)/OpenMP_v1_O0 $(DATA_DIR)/string_1MB.txt O0 1 1
	$(BUILD_DIR)/OpenMP_v1_O0 $(DATA_DIR)/string_1MB.txt O0 2 1
	$(BUILD_DIR)/OpenMP_v1_O0 $(DATA_DIR)/string_1MB.txt O0 4 1
	$(BUILD_DIR)/OpenMP_v1_O0 $(DATA_DIR)/string_1MB.txt O0 8 1
	$(BUILD_DIR)/OpenMP_v1_O0 $(DATA_DIR)/string_50MB.txt O0 1 1
	$(BUILD_DIR)/OpenMP_v1_O0 $(DATA_DIR)/string_50MB.txt O0 2 1
	$(BUILD_DIR)/OpenMP_v1_O0 $(DATA_DIR)/string_50MB.txt O0 4 1
	$(BUILD_DIR)/OpenMP_v1_O0 $(DATA_DIR)/string_50MB.txt O0 8 1
	$(BUILD_DIR)/OpenMP_v1_O0 $(DATA_DIR)/string_100MB.txt O0 1 1
	$(BUILD_DIR)/OpenMP_v1_O0 $(DATA_DIR)/string_100MB.txt O0 2 1
	$(BUILD_DIR)/OpenMP_v1_O0 $(DATA_DIR)/string_100MB.txt O0 4 1
	$(BUILD_DIR)/OpenMP_v1_O0 $(DATA_DIR)/string_100MB.txt O0 8 1
	$(BUILD_DIR)/OpenMP_v1_O0 $(DATA_DIR)/string_200MB.txt O0 1 1
	$(BUILD_DIR)/OpenMP_v1_O0 $(DATA_DIR)/string_200MB.txt O0 2 1
	$(BUILD_DIR)/OpenMP_v1_O0 $(DATA_DIR)/string_200MB.txt O0 4 1
	$(BUILD_DIR)/OpenMP_v1_O0 $(DATA_DIR)/string_200MB.txt O0 8 1
	$(BUILD_DIR)/OpenMP_v1_O0 $(DATA_DIR)/string_500MB.txt O0 1 1
	$(BUILD_DIR)/OpenMP_v1_O0 $(DATA_DIR)/string_500MB.txt O0 2 1
	$(BUILD_DIR)/OpenMP_v1_O0 $(DATA_DIR)/string_500MB.txt O0 4 1
	$(BUILD_DIR)/OpenMP_v1_O0 $(DATA_DIR)/string_500MB.txt O0 8 1

run_omp_v1_O1: OpenMP_v1_O1
	$(BUILD_DIR)/OpenMP_v1_O1 $(DATA_DIR)/string_1MB.txt O1 1 1
	$(BUILD_DIR)/OpenMP_v1_O1 $(DATA_DIR)/string_1MB.txt O1 2 1
	$(BUILD_DIR)/OpenMP_v1_O1 $(DATA_DIR)/string_1MB.txt O1 4 1
	$(BUILD_DIR)/OpenMP_v1_O1 $(DATA_DIR)/string_1MB.txt O1 8 1
	$(BUILD_DIR)/OpenMP_v1_O1 $(DATA_DIR)/string_50MB.txt O1 1 1
	$(BUILD_DIR)/OpenMP_v1_O1 $(DATA_DIR)/string_50MB.txt O1 2 1
	$(BUILD_DIR)/OpenMP_v1_O1 $(DATA_DIR)/string_50MB.txt O1 4 1
	$(BUILD_DIR)/OpenMP_v1_O1 $(DATA_DIR)/string_50MB.txt O1 8 1
	$(BUILD_DIR)/OpenMP_v1_O1 $(DATA_DIR)/string_100MB.txt O1 1 1
	$(BUILD_DIR)/OpenMP_v1_O1 $(DATA_DIR)/string_100MB.txt O1 2 1
	$(BUILD_DIR)/OpenMP_v1_O1 $(DATA_DIR)/string_100MB.txt O1 4 1
	$(BUILD_DIR)/OpenMP_v1_O1 $(DATA_DIR)/string_100MB.txt O1 8 1
	$(BUILD_DIR)/OpenMP_v1_O1 $(DATA_DIR)/string_200MB.txt O1 1 1
	$(BUILD_DIR)/OpenMP_v1_O1 $(DATA_DIR)/string_200MB.txt O1 2 1
	$(BUILD_DIR)/OpenMP_v1_O1 $(DATA_DIR)/string_200MB.txt O1 4 1
	$(BUILD_DIR)/OpenMP_v1_O1 $(DATA_DIR)/string_200MB.txt O1 8 1
	$(BUILD_DIR)/OpenMP_v1_O1 $(DATA_DIR)/string_500MB.txt O1 1 1
	$(BUILD_DIR)/OpenMP_v1_O1 $(DATA_DIR)/string_500MB.txt O1 2 1
	$(BUILD_DIR)/OpenMP_v1_O1 $(DATA_DIR)/string_500MB.txt O1 4 1
	$(BUILD_DIR)/OpenMP_v1_O1 $(DATA_DIR)/string_500MB.txt O1 8 1

run_omp_v1_O2: OpenMP_v1_O2
	$(BUILD_DIR)/OpenMP_v1_O2 $(DATA_DIR)/string_1MB.txt O2 1 1
	$(BUILD_DIR)/OpenMP_v1_O2 $(DATA_DIR)/string_1MB.txt O2 2 1
	$(BUILD_DIR)/OpenMP_v1_O2 $(DATA_DIR)/string_1MB.txt O2 4 1
	$(BUILD_DIR)/OpenMP_v1_O2 $(DATA_DIR)/string_1MB.txt O2 8 1
	$(BUILD_DIR)/OpenMP_v1_O2 $(DATA_DIR)/string_50MB.txt O2 1 1
	$(BUILD_DIR)/OpenMP_v1_O2 $(DATA_DIR)/string_50MB.txt O2 2 1
	$(BUILD_DIR)/OpenMP_v1_O2 $(DATA_DIR)/string_50MB.txt O2 4 1
	$(BUILD_DIR)/OpenMP_v1_O2 $(DATA_DIR)/string_50MB.txt O2 8 1
	$(BUILD_DIR)/OpenMP_v1_O2 $(DATA_DIR)/string_100MB.txt O2 1 1
	$(BUILD_DIR)/OpenMP_v1_O2 $(DATA_DIR)/string_100MB.txt O2 2 1
	$(BUILD_DIR)/OpenMP_v1_O2 $(DATA_DIR)/string_100MB.txt O2 4 1
	$(BUILD_DIR)/OpenMP_v1_O2 $(DATA_DIR)/string_100MB.txt O2 8 1
	$(BUILD_DIR)/OpenMP_v1_O2 $(DATA_DIR)/string_200MB.txt O2 1 1
	$(BUILD_DIR)/OpenMP_v1_O2 $(DATA_DIR)/string_200MB.txt O2 2 1
	$(BUILD_DIR)/OpenMP_v1_O2 $(DATA_DIR)/string_200MB.txt O2 4 1
	$(BUILD_DIR)/OpenMP_v1_O2 $(DATA_DIR)/string_200MB.txt O2 8 1
	$(BUILD_DIR)/OpenMP_v1_O2 $(DATA_DIR)/string_500MB.txt O2 1 1
	$(BUILD_DIR)/OpenMP_v1_O2 $(DATA_DIR)/string_500MB.txt O2 2 1
	$(BUILD_DIR)/OpenMP_v1_O2 $(DATA_DIR)/string_500MB.txt O2 4 1
	$(BUILD_DIR)/OpenMP_v1_O2 $(DATA_DIR)/string_500MB.txt O2 8 1

run_omp_v1_O3: OpenMP_v1_O3
	$(BUILD_DIR)/OpenMP_v1_O3 $(DATA_DIR)/string_1MB.txt O3 1 1
	$(BUILD_DIR)/OpenMP_v1_O3 $(DATA_DIR)/string_1MB.txt O3 2 1
	$(BUILD_DIR)/OpenMP_v1_O3 $(DATA_DIR)/string_1MB.txt O3 4 1
	$(BUILD_DIR)/OpenMP_v1_O3 $(DATA_DIR)/string_1MB.txt O3 8 1
	$(BUILD_DIR)/OpenMP_v1_O3 $(DATA_DIR)/string_50MB.txt O3 1 1
	$(BUILD_DIR)/OpenMP_v1_O3 $(DATA_DIR)/string_50MB.txt O3 2 1
	$(BUILD_DIR)/OpenMP_v1_O3 $(DATA_DIR)/string_50MB.txt O3 4 1
	$(BUILD_DIR)/OpenMP_v1_O3 $(DATA_DIR)/string_50MB.txt O3 8 1
	$(BUILD_DIR)/OpenMP_v1_O3 $(DATA_DIR)/string_100MB.txt O3 1 1
	$(BUILD_DIR)/OpenMP_v1_O3 $(DATA_DIR)/string_100MB.txt O3 2 1
	$(BUILD_DIR)/OpenMP_v1_O3 $(DATA_DIR)/string_100MB.txt O3 4 1
	$(BUILD_DIR)/OpenMP_v1_O3 $(DATA_DIR)/string_100MB.txt O3 8 1
	$(BUILD_DIR)/OpenMP_v1_O3 $(DATA_DIR)/string_200MB.txt O3 1 1
	$(BUILD_DIR)/OpenMP_v1_O3 $(DATA_DIR)/string_200MB.txt O3 2 1
	$(BUILD_DIR)/OpenMP_v1_O3 $(DATA_DIR)/string_200MB.txt O3 4 1
	$(BUILD_DIR)/OpenMP_v1_O3 $(DATA_DIR)/string_200MB.txt O3 8 1
	$(BUILD_DIR)/OpenMP_v1_O3 $(DATA_DIR)/string_500MB.txt O3 1 1
	$(BUILD_DIR)/OpenMP_v1_O3 $(DATA_DIR)/string_500MB.txt O3 2 1
	$(BUILD_DIR)/OpenMP_v1_O3 $(DATA_DIR)/string_500MB.txt O3 4 1
	$(BUILD_DIR)/OpenMP_v1_O3 $(DATA_DIR)/string_500MB.txt O3 8 1

run_omp_v2_O0: OpenMP_v2_O0
	$(BUILD_DIR)/OpenMP_v2_O0 $(DATA_DIR)/string_1MB.txt O0 1 1
	$(BUILD_DIR)/OpenMP_v2_O0 $(DATA_DIR)/string_1MB.txt O0 2 1
	$(BUILD_DIR)/OpenMP_v2_O0 $(DATA_DIR)/string_1MB.txt O0 4 1
	$(BUILD_DIR)/OpenMP_v2_O0 $(DATA_DIR)/string_1MB.txt O0 8 1
	$(BUILD_DIR)/OpenMP_v2_O0 $(DATA_DIR)/string_50MB.txt O0 1 1
	$(BUILD_DIR)/OpenMP_v2_O0 $(DATA_DIR)/string_50MB.txt O0 2 1
	$(BUILD_DIR)/OpenMP_v2_O0 $(DATA_DIR)/string_50MB.txt O0 4 1
	$(BUILD_DIR)/OpenMP_v2_O0 $(DATA_DIR)/string_50MB.txt O0 8 1
	$(BUILD_DIR)/OpenMP_v2_O0 $(DATA_DIR)/string_100MB.txt O0 1 1
	$(BUILD_DIR)/OpenMP_v2_O0 $(DATA_DIR)/string_100MB.txt O0 2 1
	$(BUILD_DIR)/OpenMP_v2_O0 $(DATA_DIR)/string_100MB.txt O0 4 1
	$(BUILD_DIR)/OpenMP_v2_O0 $(DATA_DIR)/string_100MB.txt O0 8 1
	$(BUILD_DIR)/OpenMP_v2_O0 $(DATA_DIR)/string_200MB.txt O0 1 1
	$(BUILD_DIR)/OpenMP_v2_O0 $(DATA_DIR)/string_200MB.txt O0 2 1
	$(BUILD_DIR)/OpenMP_v2_O0 $(DATA_DIR)/string_200MB.txt O0 4 1
	$(BUILD_DIR)/OpenMP_v2_O0 $(DATA_DIR)/string_200MB.txt O0 8 1
	$(BUILD_DIR)/OpenMP_v2_O0 $(DATA_DIR)/string_500MB.txt O0 1 1
	$(BUILD_DIR)/OpenMP_v2_O0 $(DATA_DIR)/string_500MB.txt O0 2 1
	$(BUILD_DIR)/OpenMP_v2_O0 $(DATA_DIR)/string_500MB.txt O0 4 1
	$(BUILD_DIR)/OpenMP_v2_O0 $(DATA_DIR)/string_500MB.txt O0 8 1

run_omp_v2_O1: OpenMP_v2_O1
	$(BUILD_DIR)/OpenMP_v2_O1 $(DATA_DIR)/string_1MB.txt O1 1 1
	$(BUILD_DIR)/OpenMP_v2_O1 $(DATA_DIR)/string_1MB.txt O1 2 1
	$(BUILD_DIR)/OpenMP_v2_O1 $(DATA_DIR)/string_1MB.txt O1 4 1
	$(BUILD_DIR)/OpenMP_v2_O1 $(DATA_DIR)/string_1MB.txt O1 8 1
	$(BUILD_DIR)/OpenMP_v2_O1 $(DATA_DIR)/string_50MB.txt O1 1 1
	$(BUILD_DIR)/OpenMP_v2_O1 $(DATA_DIR)/string_50MB.txt O1 2 1
	$(BUILD_DIR)/OpenMP_v2_O1 $(DATA_DIR)/string_50MB.txt O1 4 1
	$(BUILD_DIR)/OpenMP_v2_O1 $(DATA_DIR)/string_50MB.txt O1 8 1
	$(BUILD_DIR)/OpenMP_v2_O1 $(DATA_DIR)/string_100MB.txt O1 1 1
	$(BUILD_DIR)/OpenMP_v2_O1 $(DATA_DIR)/string_100MB.txt O1 2 1
	$(BUILD_DIR)/OpenMP_v2_O1 $(DATA_DIR)/string_100MB.txt O1 4 1
	$(BUILD_DIR)/OpenMP_v2_O1 $(DATA_DIR)/string_100MB.txt O1 8 1
	$(BUILD_DIR)/OpenMP_v2_O1 $(DATA_DIR)/string_200MB.txt O1 1 1
	$(BUILD_DIR)/OpenMP_v2_O1 $(DATA_DIR)/string_200MB.txt O1 2 1
	$(BUILD_DIR)/OpenMP_v2_O1 $(DATA_DIR)/string_200MB.txt O1 4 1
	$(BUILD_DIR)/OpenMP_v2_O1 $(DATA_DIR)/string_200MB.txt O1 8 1
	$(BUILD_DIR)/OpenMP_v2_O1 $(DATA_DIR)/string_500MB.txt O1 1 1
	$(BUILD_DIR)/OpenMP_v2_O1 $(DATA_DIR)/string_500MB.txt O1 2 1
	$(BUILD_DIR)/OpenMP_v2_O1 $(DATA_DIR)/string_500MB.txt O1 4 1
	$(BUILD_DIR)/OpenMP_v2_O1 $(DATA_DIR)/string_500MB.txt O1 8 1

run_omp_v2_O2: OpenMP_v2_O2
	$(BUILD_DIR)/OpenMP_v2_O2 $(DATA_DIR)/string_1MB.txt O2 1 1
	$(BUILD_DIR)/OpenMP_v2_O2 $(DATA_DIR)/string_1MB.txt O2 2 1
	$(BUILD_DIR)/OpenMP_v2_O2 $(DATA_DIR)/string_1MB.txt O2 4 1
	$(BUILD_DIR)/OpenMP_v2_O2 $(DATA_DIR)/string_1MB.txt O2 8 1
	$(BUILD_DIR)/OpenMP_v2_O2 $(DATA_DIR)/string_50MB.txt O2 1 1
	$(BUILD_DIR)/OpenMP_v2_O2 $(DATA_DIR)/string_50MB.txt O2 2 1
	$(BUILD_DIR)/OpenMP_v2_O2 $(DATA_DIR)/string_50MB.txt O2 4 1
	$(BUILD_DIR)/OpenMP_v2_O2 $(DATA_DIR)/string_50MB.txt O2 8 1
	$(BUILD_DIR)/OpenMP_v2_O2 $(DATA_DIR)/string_100MB.txt O2 1 1
	$(BUILD_DIR)/OpenMP_v2_O2 $(DATA_DIR)/string_100MB.txt O2 2 1
	$(BUILD_DIR)/OpenMP_v2_O2 $(DATA_DIR)/string_100MB.txt O2 4 1
	$(BUILD_DIR)/OpenMP_v2_O2 $(DATA_DIR)/string_100MB.txt O2 8 1
	$(BUILD_DIR)/OpenMP_v2_O2 $(DATA_DIR)/string_200MB.txt O2 1 1
	$(BUILD_DIR)/OpenMP_v2_O2 $(DATA_DIR)/string_200MB.txt O2 2 1
	$(BUILD_DIR)/OpenMP_v2_O2 $(DATA_DIR)/string_200MB.txt O2 4 1
	$(BUILD_DIR)/OpenMP_v2_O2 $(DATA_DIR)/string_200MB.txt O2 8 1
	$(BUILD_DIR)/OpenMP_v2_O2 $(DATA_DIR)/string_500MB.txt O2 1 1
	$(BUILD_DIR)/OpenMP_v2_O2 $(DATA_DIR)/string_500MB.txt O2 2 1
	$(BUILD_DIR)/OpenMP_v2_O2 $(DATA_DIR)/string_500MB.txt O2 4 1
	$(BUILD_DIR)/OpenMP_v2_O2 $(DATA_DIR)/string_500MB.txt O2 8 1

run_omp_v2_O3: OpenMP_v2_O3
	$(BUILD_DIR)/OpenMP_v2_O3 $(DATA_DIR)/string_1MB.txt O3 1 1
	$(BUILD_DIR)/OpenMP_v2_O3 $(DATA_DIR)/string_1MB.txt O3 2 1
	$(BUILD_DIR)/OpenMP_v2_O3 $(DATA_DIR)/string_1MB.txt O3 4 1
	$(BUILD_DIR)/OpenMP_v2_O3 $(DATA_DIR)/string_1MB.txt O3 8 1
	$(BUILD_DIR)/OpenMP_v2_O3 $(DATA_DIR)/string_50MB.txt O3 1 1
	$(BUILD_DIR)/OpenMP_v2_O3 $(DATA_DIR)/string_50MB.txt O3 2 1
	$(BUILD_DIR)/OpenMP_v2_O3 $(DATA_DIR)/string_50MB.txt O3 4 1
	$(BUILD_DIR)/OpenMP_v2_O3 $(DATA_DIR)/string_50MB.txt O3 8 1
	$(BUILD_DIR)/OpenMP_v2_O3 $(DATA_DIR)/string_100MB.txt O3 1 1
	$(BUILD_DIR)/OpenMP_v2_O3 $(DATA_DIR)/string_100MB.txt O3 2 1
	$(BUILD_DIR)/OpenMP_v2_O3 $(DATA_DIR)/string_100MB.txt O3 4 1
	$(BUILD_DIR)/OpenMP_v2_O3 $(DATA_DIR)/string_100MB.txt O3 8 1
	$(BUILD_DIR)/OpenMP_v2_O3 $(DATA_DIR)/string_200MB.txt O3 1 1
	$(BUILD_DIR)/OpenMP_v2_O3 $(DATA_DIR)/string_200MB.txt O3 2 1
	$(BUILD_DIR)/OpenMP_v2_O3 $(DATA_DIR)/string_200MB.txt O3 4 1
	$(BUILD_DIR)/OpenMP_v2_O3 $(DATA_DIR)/string_200MB.txt O3 8 1
	$(BUILD_DIR)/OpenMP_v2_O3 $(DATA_DIR)/string_500MB.txt O3 1 1
	$(BUILD_DIR)/OpenMP_v2_O3 $(DATA_DIR)/string_500MB.txt O3 2 1
	$(BUILD_DIR)/OpenMP_v2_O3 $(DATA_DIR)/string_500MB.txt O3 4 1
	$(BUILD_DIR)/OpenMP_v2_O3 $(DATA_DIR)/string_500MB.txt O3 8 1

run_cuda_O0: CUDA_O0
	$(BUILD_DIR)/CUDA_O0 $(DATA_DIR)/string_1MB.txt O0
	$(BUILD_DIR)/CUDA_O0 $(DATA_DIR)/string_50MB.txt O0
	$(BUILD_DIR)/CUDA_O0 $(DATA_DIR)/string_100MB.txt O0
	$(BUILD_DIR)/CUDA_O0 $(DATA_DIR)/string_200MB.txt O0
	$(BUILD_DIR)/CUDA_O0 $(DATA_DIR)/string_500MB.txt O0

run_cuda_O1: CUDA_O1
	$(BUILD_DIR)/CUDA_O1 $(DATA_DIR)/string_1MB.txt O1
	$(BUILD_DIR)/CUDA_O1 $(DATA_DIR)/string_50MB.txt O1
	$(BUILD_DIR)/CUDA_O1 $(DATA_DIR)/string_100MB.txt O1
	$(BUILD_DIR)/CUDA_O1 $(DATA_DIR)/string_200MB.txt O1
	$(BUILD_DIR)/CUDA_O1 $(DATA_DIR)/string_500MB.txt O1

run_cuda_O2: CUDA_O2
	$(BUILD_DIR)/CUDA_O2 $(DATA_DIR)/string_1MB.txt O2
	$(BUILD_DIR)/CUDA_O2 $(DATA_DIR)/string_50MB.txt O2
	$(BUILD_DIR)/CUDA_O2 $(DATA_DIR)/string_100MB.txt O2
	$(BUILD_DIR)/CUDA_O2 $(DATA_DIR)/string_200MB.txt O2
	$(BUILD_DIR)/CUDA_O2 $(DATA_DIR)/string_500MB.txt O2

run_cuda_O3: CUDA_O3
	$(BUILD_DIR)/CUDA_O3 $(DATA_DIR)/string_1MB.txt O3
	$(BUILD_DIR)/CUDA_O3 $(DATA_DIR)/string_50MB.txt O3
	$(BUILD_DIR)/CUDA_O3 $(DATA_DIR)/string_100MB.txt O3
	$(BUILD_DIR)/CUDA_O3 $(DATA_DIR)/string_200MB.txt O3
	$(BUILD_DIR)/CUDA_O3 $(DATA_DIR)/string_500MB.txt O3

run_graphs_omp_v1:
	bash run_omp_v1.sh

run_graphs_omp_v2:
	bash run_omp_v2.sh

run_graphs_cuda:
	bash cuda.sh

clean_string:
	rm -f $(DATA_DIR)/*

clean:
	rm -f $(BUILD_DIR)/* $(GRAPH_DIR)/* $(MEASURES_CUDA_DIR)/1/* $(MEASURES_CUDA_DIR)/50/* $(MEASURES_CUDA_DIR)/100/* $(MEASURES_CUDA_DIR)/200/* $(MEASURES_CUDA_DIR)/500/* $(MEASURES_OpenMP_DIR)/1/* $(MEASURES_OpenMP_DIR)/50/* $(MEASURES_OpenMP_DIR)/100/* $(MEASURES_OpenMP_DIR)/200/* $(MEASURES_OpenMP_DIR)/500/* $(DATA_DIR)/*