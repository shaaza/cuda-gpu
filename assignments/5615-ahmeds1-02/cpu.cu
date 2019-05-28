#include <stdio.h>
#include <stdlib.h>

#include "shared.h"

extern struct Options options; // Global config var

void copy_matrix(float** src, float** dest, int n, int m);
void row_avg(float* rowavg, float** mat, int n, int m);


void evolve(float** new_mat, float** old_mat, int n, int m, int iters);
void propagate(float** new_mat, float** old_mat, int n, int m);
void propagate_row(float** new_mat, float** old_mat, int row, int m);
float next_value(float** old_mat, int ui, int uj, int m);

void perform_cpu_evolution(float** old_mat, float** new_mat, struct HostStats* stats) {
  int n = options.rows;
  int m = options.cols;
  int iters = options.iterations;

  // Time evolution
  start_timer(&(stats->cpu_compute));
  evolve(new_mat, old_mat, n, m, iters);
  end_timer(&(stats->cpu_compute));

  // Row avg
  float* rowavg = (float*) malloc(n*sizeof(float));
  start_timer(&(stats->row_avg));
  row_avg(rowavg, new_mat, n, m);
  end_timer(&(stats->row_avg));

  print_row_avg(rowavg, n, 0);


};

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

void row_avg(float* rowavg, float** mat, int n, int m) {
  for (int i = 0; i < n; i++) {
    double sum = 0;
    for (int j = 0; j < m; j++) {
      sum += mat[i][j];
    }

    double avg = sum / (double) m;
    rowavg[i] = avg;
  }
}

void evolve(float** new_mat, float** old_mat, int n, int m, int iters) {
  for (int i = 0; i < iters; i++) {
    copy_matrix(new_mat, old_mat, n, m);
    propagate(new_mat, old_mat, n, m);
  }
}
