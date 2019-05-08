#include <stdio.h>
#include "shared.h"

#define BLOCK_SIZE 1024 // max 1024 on current system

void cuda_last_error_check (const char *message);

// Add rows kernel & related operations
__global__ void add_rows_gpu_kernel(float* mat, float* out, int n, int m);
void add_rows_gpu(float* rowsum, float* mat1d, int n, int m, struct Timer* timer);

// Add columns kernel & related operations
__global__ void add_cols_gpu_kernel(float* mat, float* out, int n, int m);
void add_columns_gpu(float* rowsum, float* mat1d, int n, int m, struct Timer* timer);

// Reduce vector kernel & related operations
__global__ void reduce_vector_gpu_kernel(float* vec, float* result, int n);
void reduce_vector_gpu(float* vec, float* result, int n, struct Timer* timer);

extern struct Options options; // Global config var

void perform_gpu_operations(float* mat1d, struct Stats* stats) {
  int n = options.rows;
  int m = options.cols;

  float* rowsum = (float*) malloc(n*sizeof(float));
  add_rows_gpu(rowsum, mat1d, n, m, &(stats->add_rows));

  float* colsum = (float*) malloc(n*sizeof(float));
  add_columns_gpu(colsum, mat1d, n, m, &(stats->add_columns));

  float rowsum_reduced;
  reduce_vector_gpu(rowsum, &rowsum_reduced, n, &(stats->reduce_vector_rows));

  float colsum_reduced;
  reduce_vector_gpu(colsum, &colsum_reduced, m, &(stats->reduce_vector_cols));

  print_compute_results((char*) "GPU Results:", rowsum, colsum, rowsum_reduced, colsum_reduced, n, m);

  // Free memory
  free(rowsum);
  free(colsum);
}

void add_rows_gpu(float* rowsum, float* mat1d, int n, int m, struct Timer* timer) {
  // Compute execution GPU config
  dim3 dimBlock(BLOCK_SIZE, 1);
  int blocks_in_grid = (int) ceil((double) n / BLOCK_SIZE);
  dim3 dimGrid(blocks_in_grid, 1);

  // Device: alloc
  float* mat1d_GPU;
  float* rowsum_GPU;
  cudaMalloc((void**) &mat1d_GPU, n*m*sizeof(float));
  cudaMalloc((void**) &rowsum_GPU, n*sizeof(float));

  // Host->Device copy
  cudaMemcpy(mat1d_GPU, mat1d, n*m*sizeof(float), cudaMemcpyHostToDevice);

  // Device: execution + timing
  start_timer(timer);
  add_rows_gpu_kernel<<<dimGrid, dimBlock>>>(mat1d_GPU, rowsum_GPU, n, m);
  end_timer(timer);

  cuda_last_error_check("add_rows_gpu");

  // Device->Host copy
  cudaMemcpy(mat1d, mat1d_GPU, n*m*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(rowsum, rowsum_GPU, n*sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(mat1d_GPU);
  cudaFree(rowsum_GPU);
}

void add_columns_gpu(float* colsum, float* mat1d, int n, int m, struct Timer* timer) {
  // Compute execution GPU config
  dim3 dimBlock(1, BLOCK_SIZE);
  int blocks_in_grid = (int) ceil((double) n / BLOCK_SIZE);
  dim3 dimGrid(blocks_in_grid, 1);

  // Device: alloc
  float* mat1d_GPU;
  float* colsum_GPU;
  cudaMalloc((void**) &mat1d_GPU, n*m*sizeof(float));
  cudaMalloc((void**) &colsum_GPU, m*sizeof(float));

  // Host->Device copy
  cudaMemcpy(mat1d_GPU, mat1d, n*m*sizeof(float), cudaMemcpyHostToDevice);

  // Device: execution + timing
  start_timer(timer);
  add_cols_gpu_kernel<<<dimGrid, dimBlock>>>(mat1d_GPU, colsum_GPU, n, m);
  end_timer(timer);

  cuda_last_error_check("add_columns_gpu");

  // Device->Host copy
  cudaMemcpy(mat1d, mat1d_GPU, n*m*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(colsum, colsum_GPU, m*sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(mat1d_GPU);
  cudaFree(colsum_GPU);
}

void reduce_vector_gpu(float* vec, float* result, int n, struct Timer* timer) {
  // Compute execution GPU config
  dim3 dimBlock(1, 1);
  int blocks_in_grid = (int) ceil((double) n / BLOCK_SIZE);
  dim3 dimGrid(blocks_in_grid, 1);

  // Device: alloc
  float* vec_GPU;
  float* result_GPU;
  cudaMalloc((void**) &vec_GPU, n*sizeof(float));
  cudaMalloc((void**) &result_GPU, sizeof(float));

  // Host->Device copy
  cudaMemcpy(vec_GPU, vec, n*sizeof(float), cudaMemcpyHostToDevice);

  // Device: execution + timing
  start_timer(timer);
  reduce_vector_gpu_kernel<<<dimGrid, dimBlock>>>(vec_GPU, result_GPU, n);
  end_timer(timer);

  cuda_last_error_check("reduce_vector_gpu");

  // Device->Host copy
  cudaMemcpy(vec, vec_GPU, n*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(result, result_GPU, sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(result_GPU);
}


// Kernels

__global__ void add_rows_gpu_kernel(float* mat, float* out, int n, int m) {
       int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
       int y = threadIdx.y;
       if (x < n && y == 0) { // Only 0th thread in the y dimension is used
	 out[x] = 0;
	 for (int i = 0; i < m; i++) {
	   out[x] += mat[i+(x*m)];
	 }
       }
}

__global__ void add_cols_gpu_kernel(float* mat, float* out, int n, int m) {
       int x = threadIdx.x;
       int y = blockIdx.x * BLOCK_SIZE + threadIdx.y;
       if (y < m && x == 0) { // Only 0th thread in the x dimension is used
	 out[y] = 0;
	 for (int i = 0; i < n; i++) {
	   out[y] += mat[(i*n)+y];
	 }
       }
}

__global__ void reduce_vector_gpu_kernel(float* vec, float* result, int n) {
       int x = threadIdx.x;
       int y = threadIdx.y;
       if (x == 0 && y == 0) { // Only 1 thread used
	 *result = 0;
	 for (int i = 0; i < n; i++) {
	   *result += vec[i];
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
