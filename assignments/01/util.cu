#include <time.h>
#include <stdio.h>

#include "shared.h"

extern struct Options options; // Global config var

// Print a matrix n x m size.
void print_matrix(float** mat, int n, int m) {
  printf("(matrix)\n");
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      printf("%f, ", mat[i][j]);
    }
    printf("\n");
  }
  printf("\n");
};

// Print a vector of length n
void print_vector(float* vector, int n, char* name) {
  printf("(vector: %s) ", name);
  for (int i = 0; i < n; i++) {
    printf("%f,", vector[i]);
  }
  printf("\n");
}

// Print elapsed time given start and end
void print_elapsed_time(char* fn_name, clock_t start, clock_t end) {
  double time_spent_ms = (double)(end - start) / (CLOCKS_PER_SEC/1000);
  printf("(timing) %s: %fms \n", fn_name, time_spent_ms);
};

// Initialize values of matrix randomized using
// milliseconds since epoch as seed
long int milliseconds_since_epoch_now() {
  struct timespec spec;
  clock_gettime(CLOCK_REALTIME, &spec);
  long int ms = round(spec.tv_nsec / 1.0e6); // Convert nanoseconds to milliseconds
  return ms;
}

void initialize_matrix_values(float** matrix, int n, int m) {
  long int seed = options.seed_milliseconds ? milliseconds_since_epoch_now() : 123456; // if -r flag
  srand48(seed);

  // Initialize floating point matrix
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      matrix[i][j] = (float) (drand48()*2.0) - 1.0;
    }
  }
}

// Record timing
void start_timer(struct Timer* timer) {
  timer->start = clock();
}

void end_timer(struct Timer* timer) {
  timer->end = clock();
}