#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <getopt.h>
#include "shared.h"

// Pull timing out and print reports separately

// Add GPU impls for add_column, reduce_vector

void parse_options_with_defaults(int argc, char** argv, struct Options* options);

struct Options options; // Global config var

// MAIN

int main(int argc, char* argv[]) {
  parse_options_with_defaults(argc, argv, &options);

  struct Stats host_stats;
  struct Stats device_stats;
  perform_cpu_operations(&host_stats);
  perform_gpu_operations(&device_stats);

  if (options.timing) {
    print_elapsed_time((char*) "add_rows CPU", host_stats.add_rows.start, host_stats.add_rows.end);
    print_elapsed_time((char*) "add_columns CPU", host_stats.add_columns.start, host_stats.add_columns.end);
    print_elapsed_time((char*) "reduce_vector rows CPU", host_stats.reduce_vector_rows.start, host_stats.reduce_vector_rows.end);
    print_elapsed_time((char*) "reduce_vector cols CPU", host_stats.reduce_vector_cols.start, host_stats.reduce_vector_cols.end);

    print_elapsed_time((char*) "add_rows_gpu", device_stats.add_rows.start, device_stats.add_rows.end);
  }

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
