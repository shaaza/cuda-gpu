#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <getopt.h>
#include "shared.h"

// Add GPU impls for add_column, reduce_vector

void parse_options_with_defaults(int argc, char** argv, struct Options* options);
void alloc_matrix_host(float** mat, float* mat1d, int n, int m);
void print_timing_report(struct Stats host_stats, struct Stats device_stats);

struct Options options; // Global config var

int main(int argc, char* argv[]) {
  parse_options_with_defaults(argc, argv, &options);

  int n = options.rows;
  int m = options.cols;

  // Host: alloc & initialize
  float* mat1d = (float*) malloc(n*m*sizeof(float));  // matrix as linear n x m array in host memory
  float** mat = (float**) malloc(n*sizeof(float*));   // pointers to host memory
  initialize_matrix_values(mat, mat1d, n, m);         // Sets mat values, and consequently mat1d values

  struct Stats host_stats;
  struct Stats device_stats;
  perform_cpu_operations(mat, &host_stats);
  perform_gpu_operations(mat1d, &device_stats);

  if (n < 5 && m < 5) print_matrix(mat, n, m);
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

  int option_index = 0;
  while (( option_index = getopt(argc, argv, "n:m:rt")) != -1) {
    switch (option_index) {
    case 'n':
      options->rows = atoi(optarg);
      if (options->rows == 0) {
	printf("Invalid matrix size");
	exit(EXIT_FAILURE);
      }
      break;

    case 'm':
      options->cols = atoi(optarg);
      if (options->cols == 0) {
	printf("Invalid matrix size");
	exit(EXIT_FAILURE);
      }
      break;

    case 'r':
      options->seed_milliseconds = 1;
      break;

    case 't':
      options->timing = 1;
      break;

    default:
      printf("Incorrect options provided.");
      exit(EXIT_FAILURE);
    }
  }
}

void print_timing_report(struct Stats cpu, struct Stats gpu) {
  if (options.timing) {
    print_elapsed_time((char*) "add_rows CPU", cpu.add_rows.start, cpu.add_rows.end);
    print_elapsed_time((char*) "add_columns CPU", cpu.add_columns.start, cpu.add_columns.end);
    print_elapsed_time((char*) "reduce_vector rows CPU", cpu.reduce_vector_rows.start, cpu.reduce_vector_rows.end);
    print_elapsed_time((char*) "reduce_vector cols CPU", cpu.reduce_vector_cols.start, cpu.reduce_vector_cols.end);

    print_elapsed_time((char*) "add_rows_gpu", gpu.add_rows.start, gpu.add_rows.end);
    print_elapsed_time((char*) "add_columns_gpu", gpu.add_columns.start, gpu.add_columns.end);
    print_elapsed_time((char*) "reduce_vector rows GPU", gpu.reduce_vector_rows.start, gpu.reduce_vector_rows.end);
    print_elapsed_time((char*) "reduce_vector cols GPU", gpu.reduce_vector_cols.start, gpu.reduce_vector_cols.end);

  }
}