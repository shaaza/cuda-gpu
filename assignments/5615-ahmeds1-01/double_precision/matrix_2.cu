#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <getopt.h>
#include "shared.h"

// Double precision experiments
// Write-up of code

void parse_options_with_defaults(int argc, char** argv, struct Options* options);
void alloc_matrix_host(double** mat, double* mat1d, int n, int m);
void print_timing_report(struct Stats host_stats, struct Stats device_stats);

struct Options options; // Global config var

int main(int argc, char* argv[]) {
  parse_options_with_defaults(argc, argv, &options);

  int n = options.rows;
  int m = options.cols;

  // Host: alloc & initialize
  double* mat1d = (double*) malloc(n*m*sizeof(double));  // matrix as linear n x m array in host memory
  double** mat = (double**) malloc(n*sizeof(double*));   // pointers to host memory
  initialize_matrix_values(mat, mat1d, n, m);         // Sets mat values, and consequently mat1d values

  struct Stats host_stats;
  struct Stats device_stats;
  perform_cpu_operations(mat, &host_stats);
  perform_gpu_operations(mat1d, &device_stats);

  print_timing_report(host_stats, device_stats);

  free(mat);
  free(mat1d);
  return 0;
}

void parse_options_with_defaults(int argc, char** argv, struct Options* options) {
  // Set defaults
  options->rows = 10;
  options->cols = 10;
  options->seed_milliseconds = 0;
  options->timing = 0;
  options->print_vectors = 0;
  options->disp_time_adjacent = 0;

  int option_index = 0;
  while (( option_index = getopt(argc, argv, "n:m:rtpd")) != -1) {
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

    case 'r':
      options->seed_milliseconds = 1;
      break;

    case 't':
      options->timing = 1;
      break;

    case 'p':
      options->print_vectors = 1;
      break;

    case 'd':
      options->disp_time_adjacent = 1;
      break;

    default:
      printf("Incorrect options provided.\n");
      exit(EXIT_FAILURE);
    }
  }
}

void print_timing_report(struct Stats cpu, struct Stats gpu) {
  if (options.timing && !options.disp_time_adjacent) {
    print_elapsed_time((char*) "add_rows CPU", cpu.add_rows.start, cpu.add_rows.end);
    print_elapsed_time((char*) "add_columns CPU", cpu.add_columns.start, cpu.add_columns.end);
    print_elapsed_time((char*) "reduce_vector rows CPU", cpu.reduce_vector_rows.start, cpu.reduce_vector_rows.end);
    print_elapsed_time((char*) "reduce_vector cols CPU", cpu.reduce_vector_cols.start, cpu.reduce_vector_cols.end);

    print_elapsed_time((char*) "add_rows_gpu", gpu.add_rows.start, gpu.add_rows.end);
    print_elapsed_time((char*) "add_columns_gpu", gpu.add_columns.start, gpu.add_columns.end);
    print_elapsed_time((char*) "reduce_vector rows GPU", gpu.reduce_vector_rows.start, gpu.reduce_vector_rows.end);
    print_elapsed_time((char*) "reduce_vector cols GPU", gpu.reduce_vector_cols.start, gpu.reduce_vector_cols.end);

  }

  // Display times next to each other if both -t and -d flags are specified
  if (options.timing && options.disp_time_adjacent) {
    printf("[FN NAME]: [CPU TIME], [GPU TIME]\n");
    printf("add_rows: %fms, %fms\n",
	   elapsed_time(cpu.add_rows.start, cpu.add_rows.end),
	   elapsed_time(gpu.add_rows.start, gpu.add_rows.end));
    printf("add_cols: %fms, %fms\n",
	   elapsed_time(cpu.add_columns.start, cpu.add_columns.end),
	   elapsed_time(gpu.add_columns.start, gpu.add_columns.end));
    printf("reduce_vector_rows: %fms, %fms\n",
	   elapsed_time(cpu.reduce_vector_rows.start, cpu.reduce_vector_rows.end),
	   elapsed_time(gpu.reduce_vector_rows.start, gpu.reduce_vector_rows.end));
    printf("reduce_vector_cols: %fms, %fms\n",
	   elapsed_time(cpu.reduce_vector_cols.start, cpu.reduce_vector_cols.end),
	   elapsed_time(gpu.reduce_vector_cols.start, gpu.reduce_vector_cols.end));

  }
}