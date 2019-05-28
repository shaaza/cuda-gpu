#include <stdio.h>
#include "shared.h"
#include "block_size.h"

void cuda_last_error_check (const char *message);

// Add rows kernel & related operations
__global__ void evolve_gpu_kernel(float* mat, float* out, int n, int m);
__global__ void copy_matrix(float* mat, float* out, int n, int m);
__global__ void row_avg(float* rowavg, float* mat, int n, int m);
void evolve_gpu(float* rowsum, float* mat1d, int n, int m, int iters, struct DeviceStats* stats);

void set_duration(Timer* timer, cudaEvent_t* start, cudaEvent_t* stop);

extern struct Options options; // Global config var

void perform_gpu_evolution(float* out_mat1d, float* mat1d, struct DeviceStats* stats) {
  int n = options.rows;
  int m = options.cols;
  int iters = options.iterations;

  evolve_gpu(out_mat1d, mat1d, n, m, iters, stats);

  //print_compute_results((char*) "GPU Results:", rowsum, n, m);
}

void evolve_gpu(float* out_mat1d, float* mat1d, int n, int m, int iters, struct DeviceStats* stats) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Compute execution GPU config
  int x_blocks_in_grid = (int) ceil((double) n / BLOCK_SIZE);
  int y_blocks_in_grid = (int) ceil((double) m / BLOCK_SIZE);

  printf("Block size: %d*%d = %d, x_blocks per grid: %d, y_blocks per grid: %d\n",
	 BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE*BLOCK_SIZE,
	 x_blocks_in_grid, y_blocks_in_grid);
  // Host: alloc
  float* rowavg = (float*) malloc(n*sizeof(float));
  // Device: alloc
  float* mat1d_GPU;
  float* out_mat1d_GPU;
  float* rowavg_GPU;

  cudaEventRecord(start);
  cudaMalloc((void**) &mat1d_GPU, n*m*sizeof(float));
  cudaMalloc((void**) &out_mat1d_GPU, n*m*sizeof(float));
  cudaMalloc((void**) &rowavg_GPU, n*sizeof(float));
  cudaEventRecord(stop);
  set_duration(&(stats->allocation), &start, &stop);

  // Host->Device copy
  cudaEventRecord(start);
  cudaMemcpy(mat1d_GPU, mat1d, n*m*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(rowavg_GPU, rowavg, n*sizeof(float), cudaMemcpyHostToDevice);
  cudaEventRecord(stop);
  set_duration(&(stats->to_gpu_transfer), &start, &stop);

  // Device: execution + timing
  dim3 dimBlock(BLOCK_SIZE*BLOCK_SIZE, 1);
  dim3 dimGrid(x_blocks_in_grid, y_blocks_in_grid); // TODO: check n != m

  cudaEventRecord(start);
  for (int i = 0; i < iters; i++) {
    evolve_gpu_kernel<<<dimGrid, dimBlock>>>(mat1d_GPU, out_mat1d_GPU, n, m);
    copy_matrix<<<dimGrid, dimBlock>>>(mat1d_GPU, out_mat1d_GPU, n, m);
  }
  cudaEventRecord(stop);
  set_duration(&(stats->gpu_compute), &start, &stop);

  cuda_last_error_check("evolve_gpu");

  // Print row average
  cudaEventRecord(start);
  if (options.show_average != 0) {
    dim3 dimBlock(BLOCK_SIZE, 1);
    dim3 dimGrid((int) ceil((double) n / BLOCK_SIZE), 1);
    row_avg<<<dimGrid, dimBlock>>>(rowavg_GPU, out_mat1d_GPU, n, m);
    cudaMemcpy(rowavg, rowavg_GPU, n*sizeof(float), cudaMemcpyDeviceToHost);
    print_row_avg(rowavg, n, 0);
  }
  cudaEventRecord(stop);
  set_duration(&(stats->row_avg), &start, &stop);
  cuda_last_error_check("row_average_gpu");

  // Device->Host copy
  cudaEventRecord(start);
  cudaMemcpy(out_mat1d, out_mat1d_GPU, n*m*sizeof(float), cudaMemcpyDeviceToHost);
  cudaEventRecord(stop);
  set_duration(&(stats->to_cpu_transfer), &start, &stop);

  cudaFree(mat1d_GPU);
  cudaFree(out_mat1d_GPU);
}


// Kernels
__global__ void evolve_gpu_kernel(float* mat, float* out, int n, int m) {
  // __shared__ float mat_local[BLOCK_SIZE+4]; // Tile width = block size
  int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

  // Load tile into shared and wait for threads
  //__syncthreads();

  // Left-most columns 0 and 1
  if (x >= 0 && x < n && y >= 0 && y <= 1) {
    out[x*m + y] = mat[x*m + y];
  }

  // Other columns
  if (x >= 0 && x < n && y > 1 && y < m) {
    out[x*m + y] = ((1.9*mat[x*m + y-2]) +
		    (1.5*mat[x*m + y-1]) +
		    mat[x*m + y] +
		    (0.5*mat[x*m + (y+1)%m]) +
		    (0.1*mat[x*m + (y+2)%m])) / (float) 5;
  }
}

__global__ void copy_matrix(float* mat, float* out, int n, int m) {
       int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
       int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;
       if (x >= 0 && x < n && y > 1 && y < m) {
	 mat[x*m + y] = out[x*m + y];
       }
}

__global__ void row_avg(float* rowavg, float* mat1d, int n, int m) {
  int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  int y = threadIdx.y;

  if (y == 0 && x < n && x >= 0) {
    double sum = 0;
    for (int i = 0; i < m; i++) {
      sum += mat1d[x*m + i];
    }
    double avg = sum / (double) m;
    rowavg[x] = (float) avg;
  }

}

// GPU specific util fns: for cuda error check and duration calc from cuda events
void cuda_last_error_check (const char *message) {
	cudaError_t err = cudaGetLastError();
	if(cudaSuccess != err) {
		printf("[CUDA] [ERROR] %s: %s\n", message, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void set_duration(Timer* timer, cudaEvent_t* start, cudaEvent_t* stop) {
  float milliseconds = 0;
  cudaEventSynchronize(*stop);
  cudaEventElapsedTime(&milliseconds, *start, *stop);
  timer->duration_ms = milliseconds;
}
