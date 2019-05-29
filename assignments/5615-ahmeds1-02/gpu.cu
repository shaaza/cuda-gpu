#include <stdio.h>
#include "shared.h"
#include "constants.h"

void cuda_last_error_check (const char *message);

// Add rows kernel & related operations
__global__ void evolve_gpu_kernel(float* mat, int n, int m, int iters, int* iteration_counters);
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
  int elems_per_thread = (int) ceil((float) m / (float) BLOCK_SIZE);
  printf("Block size: %d*%d = %d, x_blocks per grid: %d, y_blocks per grid: %d\n",
	 BLOCK_SIZE, 1, BLOCK_SIZE, n, elems_per_thread);

  // Host: alloc
  float* rowavg = (float*) malloc(n*sizeof(float));

  int iteration_counters[n*elems_per_thread];
  for (int i = 0; i < n*elems_per_thread; i++) {
    iteration_counters[i] = 0;
  }

  // Device: alloc
  float* mat1d_GPU;
  float* rowavg_GPU;
  int* iteration_counters_GPU;

  cudaEventRecord(start);
  cudaMalloc((void**) &mat1d_GPU, n*m*sizeof(float));
  cudaMalloc((void**) &rowavg_GPU, n*sizeof(float));
  cudaMalloc((void**) &iteration_counters_GPU, n*elems_per_thread*sizeof(int));
  cudaEventRecord(stop);
  set_duration(&(stats->allocation), &start, &stop);

  // Host->Device copy
  cudaEventRecord(start);
  cudaMemcpy(mat1d_GPU, mat1d, n*m*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(rowavg_GPU, rowavg, n*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(iteration_counters_GPU, iteration_counters, n*elems_per_thread*sizeof(int), cudaMemcpyHostToDevice);
  cudaEventRecord(stop);
  set_duration(&(stats->to_gpu_transfer), &start, &stop);

  // Device: execution + timing
  dim3 dimBlock(BLOCK_SIZE, 1);
  dim3 dimGrid(n, elems_per_thread); // TODO: check n != m

  cudaEventRecord(start);
  evolve_gpu_kernel<<<dimGrid, dimBlock>>>(mat1d_GPU, n, m, iters, iteration_counters_GPU);
  cudaEventRecord(stop);
  set_duration(&(stats->gpu_compute), &start, &stop);

  cuda_last_error_check("evolve_gpu");

  // Print row average

  if (options.show_average != 0) {
    dim3 dimBlock(1, 1);
    dim3 dimGrid(n, 1);

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
__global__ void evolve_gpu_kernel(float* mat, int n, int m, int iters, int* iteration_counters) {
  int dx = threadIdx.x; // Column within a tile
  int row_no = blockIdx.x; // Row number of matrix
  int i = blockIdx.y; // Starting column of tile

  int tiles_per_row = (int) ceil((float) m / (float) BLOCK_SIZE);
  int cols = (i != tiles_per_row) ? m : m % BLOCK_SIZE;
  int iter_counter_index = row_no*tiles_per_row + blockIdx.y;

  __shared__ float row[BLOCK_SIZE]; // One row per block
  __shared__ float out_row[BLOCK_SIZE]; // One row per thread

  // Load row: global -> shared & barrier sync
  if (dx < cols) {
    row[dx] = mat[row_no*m + i*BLOCK_SIZE + dx];
  }
  __syncthreads();

  // Left-most tile's columns 0 and 1
  if (i == 0 && dx >= 0 && dx <= 1) {
    out_row[dx] = row[dx];
  }

  // Evolve iterations
  if (dx > 1) {
    for (int iter = 0; iter < iters; iter++) {
      // Propagate row
      if (dx < cols) {
	out_row[dx] = ((1.9*row[dx-2]) +
		      (1.5*row[dx-1]) +
		      row[dx] +
		      (0.5*row[(dx+1)%BLOCK_SIZE]) +
		      (0.1*row[(dx+2)%BLOCK_SIZE])) / (float) 5;
      }
      __syncthreads();

      // Copy over row
      if (dx < cols)
	row[dx] = out_row[dx];

      if (tiles_per_row > 1 && dx == 0) { // Only barrier sync inter-block when there is more than 1 tile per row
	atomicAdd(&(iteration_counters[iter_counter_index]), 1); // Atomic incremenet iteration counter

	if (blockIdx.y == 0) { // Leftmost tile
	  while (iteration_counters[iter_counter_index+1] < iter) {
	    // Block until right neighbour in sync
	  }
	} else if (blockIdx.y == tiles_per_row-1) { // Right most tile
	  while (iteration_counters[iter_counter_index-1] < iter) {
	    // Block until left neighbour in sync
	  }
	} else {
	  while (iteration_counters[iter_counter_index-1] < iter && iteration_counters[iter_counter_index+1] < iter) {
	    // Block until left and right neighbours are in sync
	  }
	}
      }

      __syncthreads();
    }
  }

  // Store row: shared -> global
  if (dx < cols)
    mat[row_no*m + i*BLOCK_SIZE + dx] = row[dx];

}

__global__ void row_avg(float* rowavg, float* mat1d, int n, int m) {
  int x = blockIdx.x;
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
