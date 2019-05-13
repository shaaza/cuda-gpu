#include <time.h>
#include <stdio.h>

#include "shared.h"

extern struct Options options; // Global config var

// Print a matrix n x m size.
void print_matrix(double** mat, int n, int m) {
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
void print_vector(double* vector, int n, char* name) {
  printf("(vector: %s) ", name);
  for (int i = 0; i < n; i++) {
    printf("%f,", vector[i]);
  }
  printf("\n");
}

void print_compute_results(char* title, double* rowvec, double* colvec, double rowsum, double colsum, int n, int m) {
  printf("%s\n", title);
  if (options.print_vectors) { // Print result vectors if command-line flag enabled
    print_vector(rowvec, n, (char*) "Rowsum Vector");
    print_vector(colvec, m, (char*) "Colsum Vector");
  }

  printf("Rowsum sum: %f \n", rowsum);
  printf("Colsum sum: %f \n", colsum);
  printf("\n");
}

// Print elapsed time given start and end
double elapsed_time(clock_t start, clock_t end) {
  double time_spent_ms = (double)(end - start) / (CLOCKS_PER_SEC/1000);
  return time_spent_ms;
}
void print_elapsed_time(char* fn_name, clock_t start, clock_t end) {
  printf("(timing) %s: %fms \n", fn_name, elapsed_time(start, end));
};

// Initialize values of matrix randomized using
// milliseconds since epoch as seed
long int milliseconds_since_epoch_now() {
  struct timespec spec;
  clock_gettime(CLOCK_REALTIME, &spec);
  long int ms = round(spec.tv_nsec / 1.0e6); // Convert nanoseconds to milliseconds
  return ms;
}

void initialize_matrix_values(double** matrix, double* mat1d, int n, int m) {
  for (int i = 0; i < n; i++) {
    matrix[i] = &(mat1d[i*m]); // map row-beginnings in 1d to mat
  }

  long int seed = options.seed_milliseconds ? milliseconds_since_epoch_now() : 123456; // if -r flag
  srand48(seed);

  // Initialize doubleing point matrix
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      matrix[i][j] = (double) (drand48()*2.0) - 1.0;
    }
  }

  if (options.print_vectors)
    print_matrix(matrix, n, m);
}

// Record timing
void start_timer(struct Timer* timer) {
  timer->start = clock();
}

void end_timer(struct Timer* timer) {
  timer->end = clock();
}