#include <stdio.h>
#include "shared.h"

#define BLOCK_SIZE 1024 // max 1024 on current system

void cuda_last_error_check (const char *message);

__global__ void add_rows_gpu_kernel(float* mat, float* out, int n, int m);
void add_rows_gpu(float* rowsum, float* mat1d, int n, int m, dim3* dimBlock, dim3* dimGrid);

extern struct Options options; // Global config var

void perform_gpu_operations() {
  int n = options.rows;
  int m = options.cols;

  // Host: alloc & initialize
  float* mat1d = (float*) malloc(n*m*sizeof(float));  // matrix as linear n x m array in host memory
  float** mat = (float**) malloc(n*sizeof(float*));   // pointers to host memory
  for (int i = 0; i < n; i++) {
    mat[i] = &(mat1d[i*m]);
  }

  initialize_matrix_values(mat, n, m); // Sets mat, and consequently mat1d, values

  // Compute execution GPU config
  dim3 dimBlock(BLOCK_SIZE, 1);
  int blocks_in_grid = (int) ceil((double) n / BLOCK_SIZE);
  dim3 dimGrid(blocks_in_grid, 1);

  // Rowsum
  float* rowsum = (float*) malloc(n*sizeof(float));
  add_rows_gpu(rowsum, mat1d, n, m, &dimBlock, &dimGrid);

  // Print results
  if (n < 5 && m < 5) print_matrix(mat, n, m);
  if (n < 5) print_vector(rowsum, n, (char*) "rowsum_GPU");
  printf("Rowsum sum GPU: %f \n", reduce_vector(rowsum, n));

  // Free memory
  free(mat1d);
  free(mat);
  free(rowsum);
}

void add_rows_gpu(float* rowsum, float* mat1d, int n, int m, dim3* dimBlock, dim3* dimGrid) {
  // Device: alloc
  float* mat1d_GPU;
  float* rowsum_GPU;
  cudaMalloc((void**) &mat1d_GPU, n*m*sizeof(float));
  cudaMalloc((void**) &rowsum_GPU, n*sizeof(float));

  // Host->Device copy
  cudaMemcpy(mat1d_GPU, mat1d, n*m*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(rowsum_GPU, rowsum, n*sizeof(float), cudaMemcpyHostToDevice);

  // Device: execution + timing
  clock_t start = clock();
  add_rows_gpu_kernel<<<*dimGrid, *dimBlock>>>(mat1d_GPU, rowsum_GPU, n, m);
  clock_t end = clock();
  cuda_last_error_check("add_rows_gpu");
  print_elapsed_time((char*) "add_rows_gpu", start, end);

  // Device->Host copy
  cudaMemcpy(mat1d, mat1d_GPU, n*m*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(rowsum, rowsum_GPU, n*sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(mat1d_GPU);
  cudaFree(rowsum_GPU);
}

__global__ void add_rows_gpu_kernel(float* mat, float* out, int n, int m) {
       int tx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
       int ty = threadIdx.y;
       if (tx < n && ty == 0) { // Only 0th thread in the y dimension is used
	 out[tx] = 0;
	 for (int i = 0; i < m; i++) {
	   out[tx] += mat[i+(tx*m)];
	 }
       }
}

// Cuda error check util
void cuda_last_error_check (const char *message) {
	cudaError_t err = cudaGetLastError();
	if(cudaSuccess != err) {
		printf("[CUDA] [ERROR] %s: %s\n", message, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}
