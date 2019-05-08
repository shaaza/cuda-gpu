#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "shared.h"

// Matrix memory allocation and init
float** alloc_matrix(int n, int m);
void free_matrix(float** mat, int n);

// Computations
float* add_rows(float** mat, int n, int m, struct Timer* timer);
float* add_columns(float** mat, int n, int m, struct Timer* timer);
float reduce_vector(float* vec, int n, struct Timer* timer);

extern struct Options options; // Global config var

void perform_cpu_operations(struct Stats* host_stats) {
  int n = options.rows;
  int m = options.cols;

  // Allocate, initialize floating point matrix (n/m)
  float** matrix = alloc_matrix(n, m);
  initialize_matrix_values(matrix, n, m);

  // Compute row-sum, col-sum and their reduced values.
  float* rowsum_vec = add_rows(matrix, n, m, &(host_stats->add_rows));
  float* colsum_vec = add_columns(matrix, n, m, &(host_stats->add_columns));

  // Print matrix and vectors if small
  if (n < 5 && m < 5) print_matrix(matrix, n, m);
  if (n < 5) print_vector(rowsum_vec, n, (char*) "rowsum");
  if (m < 5) print_vector(colsum_vec, n, (char*) "colsum");

  printf("Rowsum sum: %f \n", reduce_vector(rowsum_vec, n, &(host_stats->reduce_vector_rows)));
  printf("Colsum sum: %f \n", reduce_vector(colsum_vec, m, &(host_stats->reduce_vector_cols)));

  // Free matrix and vectors
  free(rowsum_vec);
  free(colsum_vec);
  free_matrix(matrix, n);
}

float** alloc_matrix(int n, int m) {
  float** matrix = (float**) malloc(n*sizeof(float*));
  for (int i = 0; i < n; i++) {
    matrix[i] = (float*) malloc(m*sizeof(float));
  }
  return matrix;
}

void free_matrix(float** mat, int n) {
   for (int i = 0; i < n; i++) {
    free(mat[i]);
  }
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
