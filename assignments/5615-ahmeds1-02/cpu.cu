#include <stdio.h>
#include <stdlib.h>

#include "shared.h"

extern struct Options options; // Global config var

float** allocate_matrix(int n, int m);
void copy_matrix(float** src, float** dest, int n, int m);
void initialize_matrix_values(float** mat, int n, int m);

void evolve(float** new_mat, float** old_mat, int n, int m, int iters);
void propagate(float** new_mat, float** old_mat, int n, int m);
void propagate_row(float** new_mat, float** old_mat, int row, int m);
float next_value(float** old_mat, int ui, int uj, int m);

void perform_cpu_evolution(struct Stats* host_stats) {
  int n = options.rows;
  int m = options.cols;
  int iters = options.iterations;

  printf("Iters %d, Rows %d, Cols %d\n", iters, n, m);

  // Host: alloc & initialize
  float* mat1d = (float*) malloc(n*m*sizeof(float));
  float** old_mat = allocate_matrix(n, m);   // pointers to host memory
  float** new_mat = allocate_matrix(n, m);
  initialize_matrix_values(new_mat, n, m);

  printf("Initialized matrices...\n");

  evolve(new_mat, old_mat, n, m, iters);


  free(old_mat);
  free(new_mat);
  free(mat1d);

};

float** allocate_matrix(int n, int m) {
  float** mat = (float**) malloc(n*sizeof(float*));
  for (int i = 0; i < m; i++) {
    mat[i] = (float*) malloc(m*sizeof(float));
  }

  return mat;
}

float next_value(float** old_mat, int ui, int uj, int m) {
  return ((1.9*old_mat[ui][uj-2]) +
	  (1.5*old_mat[ui][uj-1]) +
	  old_mat[ui][uj] +
	  (0.5*old_mat[ui][(uj+1)%m]) +
	  (0.1*old_mat[ui][(uj+2)%m])) / (float) 5;
};

void propagate_row(float** new_mat, float** old_mat, int row, int m) {
  int start_index = 2; // Skip columns 0 and 1
  for (int i = start_index; i < m; i++)
    new_mat[row][i] = next_value(old_mat, row, i, m);
}

void propagate(float** new_mat, float** old_mat, int n, int m) {
  for (int i = 0; i < n; i++)
    propagate_row(new_mat, old_mat, i, m);
}

void copy_matrix(float** src, float** dest, int n, int m) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      dest[i][j] = src[i][j];
    }
  }
}

void initialize_matrix_values(float** mat, int n, int m) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      mat[i][j] = 0;
    }
  }

  for (int i = 0; i < n; i++) {
    mat[i][0] = 1.00 * (float) (i+1) / (float) n;
    mat[i][1] = 0.80 * (float) (i+1) / (float) n;
  }
}

void print_row_avg(float** mat, int n, int m, int iter) {
  if (options.show_average == 0)
    return;
  if ((iter+1) % options.show_average != 0)
    return;

  printf("Row Average Temp\n");
  for (int i = 0; i < n; i++) {
    double sum = 0;
    for (int j = 0; j < m; j++) {
      sum += mat[i][j];
    }

    double avg = sum / (double) m;
    printf("%f\n", avg);
  }
  printf("\n");
}

void evolve(float** new_mat, float** old_mat, int n, int m, int iters) {
  for (int i = 0; i < iters; i++) {
    copy_matrix(new_mat, old_mat, n, m);
    print_matrix(new_mat, n, m);
    propagate(new_mat, old_mat, n, m);

    print_row_avg(new_mat, n, m, i);
  }
}