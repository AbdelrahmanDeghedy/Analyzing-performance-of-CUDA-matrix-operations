#include <stdio.h>
#include <stdlib.h>
#include <ctime>	

#define N 2

void cpuMatrixOperation(int* A, int* B, int* C) {
    // Allocate memory for intermediate matrices
    int* temp1 = new int[N * N];
    int* temp2 = new int[N * N];

    // Compute A + B and store in temp1
    for (int i = 0; i < N * N; i++) {
        temp1[i] = A[i] + B[i];
    }

    // Compute A * (A + B) and store in temp2
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            temp2[i * N + j] = 0;
            for (int k = 0; k < N; k++) {
                temp2[i * N + j] += A[i * N + k] * temp1[k * N + j];
            }
        }
    }

    // Compute A * (A + B) + C and store in C
    for (int i = 0; i < N * N; i++) {
        C[i] = temp2[i] + C[i];
    }

    // Free memory for intermediate matrices
    delete[] temp1;
    delete[] temp2;
}


__global__ void matrix_sum(int* A, int* B, int* C) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;

    if (row < N && col < N) {
        int index = row * N + col;
        C[index] = A[index] + B[index];
    }
}

__global__ void matrix_mul(int *a, int *b, int *c) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int sum = 0;
    for (int k = 0; k < N; k++) {
        sum += a[row * N + k] * b[k * N + col];
    }

    c[row * N + col] = sum;
}

void print_matrix(int *matrix) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", matrix[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void debug_matrices_results (int* a, int* b, int* c) {
    // Print input and output matrices
    printf("Input Matrix 1:\n");
    print_matrix(a);

    printf("Input Matrix 2:\n");
    print_matrix(b);

    printf("Output Matrix:\n");
    print_matrix(c);
}

int main() {
    int *a, *b, *c, *d, *e, *f;
    int *dev_a, *dev_b, *dev_c, *dev_d, *dev_e, *dev_f;

    int size = N * N * sizeof(int);

    // Change the seed using the time at each run
    srand(time(NULL));
    
    int max_rand_element = 5;

    // Allocate memory on host
    a = (int*)malloc(size);
    b = (int*)malloc(size);
    c = (int*)malloc(size);
    d = (int*)malloc(size);
    e = (int*)malloc(size);
    f = (int*)malloc(size);

    // Initialize matrices with random values
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[i * N + j] = rand() % max_rand_element;
            b[i * N + j] = rand() % max_rand_element;
            c[i * N + j] = rand() % max_rand_element;
        }
    }

    // Allocate memory on device GPU
    cudaMalloc((void**)&dev_a, size);
    cudaMalloc((void**)&dev_b, size);
    cudaMalloc((void**)&dev_c, size);
    cudaMalloc((void**)&dev_d, size);
    cudaMalloc((void**)&dev_e, size);
    cudaMalloc((void**)&dev_f, size);

    // Copy matrices from host to device
    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, c, size, cudaMemcpyHostToDevice);

    // Set up kernel parameters
    dim3 threadsPerBlock(N, N);
    dim3 numBlocks(1, 1);

    std::clock_t start_cpu = std::clock(); // start timer
    cpuMatrixOperation(a, b, c);
    std::clock_t end_cpu = std::clock();
    auto time_cpu = double(end_cpu - start_cpu) / CLOCKS_PER_SEC;
    // printf("Time CPU: %f \n", time_cpu);
    
    // CPU Result
    printf("CPU Result:\n");
    debug_matrices_results (a, b, c);
    
    std::clock_t start_gpu = std::clock(); // start timer
    // Launch kernel
    matrix_sum<<<numBlocks, threadsPerBlock>>>(dev_a, dev_b, dev_d);    
    matrix_mul<<<numBlocks, threadsPerBlock>>>(dev_a, dev_d, dev_e);
    matrix_sum<<<numBlocks, threadsPerBlock>>>(dev_e, dev_c, dev_f);
    cudaDeviceSynchronize();
    std::clock_t end_gpu = std::clock();
    auto time_gpu = double(end_gpu - start_gpu) / CLOCKS_PER_SEC;
    // printf("Time GPU: %f \n", time_gpu);
    
    // Copy result matrix from device to host
    cudaMemcpy(d, dev_d, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(e, dev_e, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(f, dev_f, size, cudaMemcpyDeviceToHost);
   
    // D = A + B
    //printf("A + B:\n\n");
    //debug_matrices_results (a, b, d);
    //printf("-----------------------------\n");
    
    
    // E = A * (A + B) = A * D
   // printf("A * (A + B):\n\n");
    //debug_matrices_results (a, d, e);
  //  printf("-----------------------------\n");
    
    // F = A * (A + B) + C = E + C
//    printf("A * (A + B) + C:\n\n");

    printf("GPU Result:\n");
    // GPU Result
    debug_matrices_results (a, b, f);
    
    // Copy the result into Matrix A (A = F)
    cudaMemcpy(a, dev_f, size, cudaMemcpyDeviceToHost);

    // Freeing memory
    free(a);
    free(b);
    free(c);
    free(d);
    free(e);
    free(f);
    
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    cudaFree(dev_d);
    cudaFree(dev_e);
    cudaFree(dev_f);

    return 0;
}

