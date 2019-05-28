#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <getopt.h>
#include "shared.h"

void parse_options_with_defaults(int argc, char** argv, struct Options* options);
void print_timing_report(struct HostStats host_stats, struct DeviceStats device_stats);
float** allocate_matrix(int n, int m);
void initialize_matrix_values(float** mat, float* mat1d, int n, int m);

struct Options options; // Global config var

int main(int argc, char* argv[]) {
  // Parse options
  parse_options_with_defaults(argc, argv, &options);
  int n = options.rows;
  int m = options.cols;
  int iters = options.iterations;
  printf("Iters %d, Rows %d, Cols %d\n", iters, n, m);

  // Allocation
  float* mat1d = (float*) malloc(n*m*sizeof(float));
  float* out_mat1d = (float*) malloc(n*m*sizeof(float));

  float** old_mat = allocate_matrix(n, m);   // pointers to host memory
  float** new_mat = allocate_matrix(n, m);

  // CPU version
  printf("Performing CPU computations...\n");
  initialize_matrix_values(old_mat, mat1d, n, m);
  initialize_matrix_values(new_mat, out_mat1d, n, m);
  printf("Initialized matrix values...\n");
  struct HostStats host_stats;
  perform_cpu_evolution(old_mat, new_mat, &host_stats);
  //  print_matrix(new_mat, n, m); // Uncomment to print output matrix

  // GPU version
  printf("Performing GPU computations...\n");
  initialize_matrix_values(old_mat, mat1d, n, m);
  initialize_matrix_values(new_mat, out_mat1d, n, m);

  struct DeviceStats device_stats;
  perform_gpu_evolution(out_mat1d, mat1d, &device_stats);
  //  print_matrix(new_mat, n, m);  // Uncomment to print output matrix

  // Timings
  print_timing_report(host_stats, device_stats);

  // Free memory
  free(old_mat);
  free(new_mat);
  free(mat1d);
  free(out_mat1d);

  return 0;



}

float** allocate_matrix(int n, int m) {
  float** mat = (float**) malloc(n*sizeof(float*));
  for (int i = 0; i < m; i++) {
    mat[i] = (float*) malloc(m*sizeof(float));
  }

  return mat;
}


void parse_options_with_defaults(int argc, char** argv, struct Options* options) {
  // Set defaults
  options->rows = 10;
  options->cols = 10;
  options->iterations = 1;
  options->show_average = 0;

  int option_index = 0;
  while (( option_index = getopt(argc, argv, "n:m:p:a")) != -1) {
    switch (option_index) {
    case 'n':
      options->rows = atoi(optarg);
      if (options->rows == 0) {
	printf("Invalid matrix size\n");
	exit(EXIT_FAILURE);
      }
      break;

    case 'm':
      options->cols = atoi(optarg);
      if (options->cols == 0) {
	printf("Invalid matrix size\n");
	exit(EXIT_FAILURE);
      }
      break;

    case 'p':
      options->iterations = atoi(optarg);
      if (options->iterations == 0) {
	printf("Invalid iteration count\n");
	exit(EXIT_FAILURE);
      }
      break;

    case 'a':
      options->show_average = 1;
      break;

    default:
      printf("Incorrect options provided.\n");
      exit(EXIT_FAILURE);
    }
  }
}

void print_timing_report(struct HostStats cpu, struct DeviceStats gpu) {
  printf("\nTiming Report:\n");
  print_elapsed_time((char*) "[CPU] Compute", cpu.cpu_compute.start, cpu.cpu_compute.end);
  print_elapsed_time((char*) "[CPU] Row Average", cpu.row_avg.start, cpu.row_avg.end);
  printf("\n");
  printf("[GPU] Allocation: %fms\n", gpu.allocation.duration_ms);
  printf("[GPU] Host -> GPU Transfer: %fms\n", gpu.to_gpu_transfer.duration_ms);
  printf("[GPU] compute: %fms\n", gpu.gpu_compute.duration_ms);
  printf("[GPU] GPU -> Host Transfer: %fms\n", gpu.to_cpu_transfer.duration_ms);
  printf("[GPU] Row Average: %fms\n", gpu.row_avg.duration_ms);
}

void initialize_matrix_values(float** mat, float* mat1d, int n, int m) {
  for (int i = 0; i < n; i++) {
    mat[i] = &(mat1d[i*m]); // map row-beginnings in 1d to mat
  }

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      mat[i][j] = 0;
    }
  }


  for (int i = 0; i < n; i++) {
    mat[i][0] = 1.00 * (float) (i+1) / (float) n;
    mat[i][1] = 0.80 * (float) (i+1) / (float) n;
  }

  if (options.print_vectors)
    print_matrix(mat, n, m);

}
