#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <getopt.h>
#include "shared.h"

// Pull timing out

// Timing stats
struct Timing {
  clock_t start;
  clock_t end;
};

struct Stats {
  struct Timing add_rows;
  struct Timing add_columns;
  struct Timing reduce_vector;
};

void parse_options_with_defaults(int argc, char** argv, struct Options* options);

struct Options options; // Global config var

// MAIN

int main(int argc, char* argv[]) {
  parse_options_with_defaults(argc, argv, &options);
  perform_cpu_operations();
  perform_gpu_operations();

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
