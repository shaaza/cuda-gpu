#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "shared.h"

float* add_rows(float** mat, int n, int m, struct Timer* timer);
float* add_columns(float** mat, int n, int m, struct Timer* timer);
float reduce_vector(float* vec, int n, struct Timer* timer);

extern struct Options options; // Global config var

void perform_cpu_operations(float** matrix, struct Stats* stats) {
  int n = options.rows;
  int m = options.cols;

  // Compute row-sum, col-sum and their reduced values.
  float* rowsum = add_rows(matrix, n, m, &(stats->add_rows));
  float* colsum = add_columns(matrix, n, m, &(stats->add_columns));
  float rowsum_reduced = reduce_vector(rowsum, n, &(stats->reduce_vector_rows));
  float colsum_reduced = reduce_vector(colsum, n, &(stats->reduce_vector_cols));

  print_compute_results((char*) "CPU Results:", rowsum, colsum, rowsum_reduced, colsum_reduced, n, m);

  // Free matrix and vectors
  free(rowsum);
  free(colsum);
}

// Sum up n rows of a matrix into a vector of size n
float* add_rows(float** mat, int n, int m, struct Timer* timer) {
  start_timer(timer);

  float* output = (float*) malloc(n*sizeof(float));
  for (int i = 0; i < n; i++) {
    float sum = 0;
    for (int j = 0; j < m; j++) {
      sum += mat[i][j];
    }
    output[i] = sum;
  }

  end_timer(timer);
  return output;
}
// Sum up m columns of a matrix into a vector of size m
float* add_columns(float** mat, int n, int m, struct Timer* timer) {
  start_timer(timer);

  float* output = (float*) malloc(m*sizeof(float));
  for (int j = 0; j < m; j++) {
    float sum = 0;
    for (int i = 0; i < n; i++) {
      sum += mat[i][j];
    }
    output[j] = sum;
  }

  end_timer(timer);
  return output;
}

// Sum up n elements of a vector
float reduce_vector(float* vec, int n, struct Timer* timer) {
  start_timer(timer);

  float sum = 0;
  for (int i = 0; i < n; i++) {
    sum += vec[i];
  }

  end_timer(timer);
  return sum;
}
