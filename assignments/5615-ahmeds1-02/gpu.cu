#include <stdio.h>
#include "shared.h"
#include "constants.h"

void cuda_last_error_check (const char *message);

// Add rows kernel & related operations
__global__ void evolve_gpu_kernel(float* mat, float* out, int n, int m, int iters);
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
  int x_blocks_in_grid = (int) ceil((double) n / BLOCK_WIDTH);
  int y_blocks_in_grid = (int) ceil((double) m / BLOCK_WIDTH);

  printf("Block size: %d*%d = %d, x_blocks per grid: %d, y_blocks per grid: %d\n",
	 BLOCK_WIDTH, BLOCK_WIDTH, BLOCK_SIZE,
	 x_blocks_in_grid, y_blocks_in_grid);
  // Host: alloc
  float* rowavg = (float*) malloc(n*sizeof(float));
  // Device: alloc
  float* mat1d_GPU;
  float* rowavg_GPU;

  cudaEventRecord(start);
  cudaMalloc((void**) &mat1d_GPU, n*m*sizeof(float));
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
  dim3 dimBlock(BLOCK_SIZE, 1);
  dim3 dimGrid(n, 1); // TODO: check n != m

  cudaEventRecord(start);
  evolve_gpu_kernel<<<dimGrid, dimBlock>>>(mat1d_GPU, mat1d_GPU, n, m, iters);
  cudaEventRecord(stop);
  set_duration(&(stats->gpu_compute), &start, &stop);

  cuda_last_error_check("evolve_gpu");

  // Print row average

  if (options.show_average != 0) {
    dim3 dimBlock(BLOCK_WIDTH, 1);
    dim3 dimGrid((int) ceil((double) n / BLOCK_WIDTH), 1);

    cudaEventRecord(start);
    row_avg<<<dimGrid, dimBlock>>>(rowavg_GPU, mat1d_GPU, n, m);
    cudaEventRecord(stop);
    set_duration(&(stats->row_avg), &start, &stop);

    cuda_last_error_check("row_average_gpu");

    cudaMemcpy(rowavg, rowavg_GPU, n*sizeof(float), cudaMemcpyDeviceToHost);
    print_row_avg(rowavg, n, 0);
  }

  // Device->Host copy
  cudaEventRecord(start);
  cudaMemcpy(out_mat1d, mat1d_GPU, n*m*sizeof(float), cudaMemcpyDeviceToHost);
  cudaEventRecord(stop);
  set_duration(&(stats->to_cpu_transfer), &start, &stop);

  cudaFree(mat1d_GPU);
}


// Kernels
__global__ void evolve_gpu_kernel(float* mat, float* out, int n, int m, int iters) {
  int dx = threadIdx.x; // Starting column
  int row_no = blockIdx.x; // Row number of matrix
  int elems_per_thread = (int) ceil((float) m / (float) BLOCK_SIZE);
  __shared__ float row[MAX_COLS]; // One row per block, elems_per_thread cells per thread
  __shared__ float out_row[MAX_COLS]; // One row per thread, elems_per_thread cells per thread

  // Load row: global -> shared & barrier sync
  int y;
  for (int i = 0; i < elems_per_thread; i++) {
    y = dx + i*BLOCK_SIZE;
    if (y < m) {
      row[y] = mat[row_no*m + y];
    }
  }
  __syncthreads();

  // Left-most columns 0 and 1
  if (dx >= 0 && dx <= 1) {
    out_row[dx] = row[dx];
  }

  // Evolve iterations
  if (dx > 1) {
    for (int i = 0; i < iters; i++) {
      // Propagate row
      for (int i = 0; i < elems_per_thread; i++) {
  	y = dx + i*BLOCK_SIZE;
  	if (y < m) {
  	  out_row[y] = ((1.9*row[y-2]) +
  			(1.5*row[y-1]) +
  			row[y] +
  			(0.5*row[(y+1)%m]) +
  			(0.1*row[(y+2)%m])) / (float) 5;
  	}
      }
      __syncthreads();
      // Copy over row
      for (int i = 0; i < elems_per_thread; i++) {
  	y = dx + (i*BLOCK_SIZE);
  	if (y < m)
  	  row[y] = out_row[y];
      }
      __syncthreads();
    }
  }

  // Store row: shared -> global
  for (int i = 0; i < elems_per_thread; i++) {
    y = dx + i*BLOCK_SIZE;
    if (y < m)
      mat[row_no*m + y] = row[y];
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
