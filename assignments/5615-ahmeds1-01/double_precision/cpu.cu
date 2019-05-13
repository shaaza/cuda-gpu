#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "shared.h"

double* add_rows(double** mat, int n, int m, struct Timer* timer);
double* add_columns(double** mat, int n, int m, struct Timer* timer);
double reduce_vector(double* vec, int n, struct Timer* timer);

extern struct Options options; // Global config var

void perform_cpu_operations(double** matrix, struct Stats* stats) {
  int n = options.rows;
  int m = options.cols;

  // Compute row-sum, col-sum and their reduced values.
  double* rowsum = add_rows(matrix, n, m, &(stats->add_rows));
  double* colsum = add_columns(matrix, n, m, &(stats->add_columns));
  double rowsum_reduced = reduce_vector(rowsum, n, &(stats->reduce_vector_rows));
  double colsum_reduced = reduce_vector(colsum, m, &(stats->reduce_vector_cols));

  print_compute_results((char*) "CPU Results:", rowsum, colsum, rowsum_reduced, colsum_reduced, n, m);

  // Free matrix and vectors
  free(rowsum);
  free(colsum);
}

// Sum up n rows of a matrix into a vector of size n
double* add_rows(double** mat, int n, int m, struct Timer* timer) {
  start_timer(timer);

  double* output = (double*) malloc(n*sizeof(double));
  for (int i = 0; i < n; i++) {
    double sum = 0;
    for (int j = 0; j < m; j++) {
      sum += mat[i][j];
    }
    output[i] = sum;
  }

  end_timer(timer);
  return output;
}
// Sum up m columns of a matrix into a vector of size m
double* add_columns(double** mat, int n, int m, struct Timer* timer) {
  start_timer(timer);

  double* output = (double*) malloc(m*sizeof(double));
  for (int j = 0; j < m; j++) {
    double sum = 0;
    for (int i = 0; i < n; i++) {
      sum += mat[i][j];
    }
    output[j] = sum;
  }

  end_timer(timer);
  return output;
}

// Sum up n elements of a vector
double reduce_vector(double* vec, int n, struct Timer* timer) {
  start_timer(timer);

  double sum = 0;
  for (int i = 0; i < n; i++) {
    sum += vec[i];
  }

  end_timer(timer);
  return sum;
}
