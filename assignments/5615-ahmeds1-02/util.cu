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

void print_compute_results(char* title, float* rowvec, int n, int m) {
  printf("%s\n", title);
  if (options.print_vectors) { // Print result vectors if command-line flag enabled
    print_vector(rowvec, n, (char*) "Rowsum Vector");
  }

  printf("\n");
}

void print_row_avg(float* rowavg, int n, int iter) {
  if (options.show_average == 0)
    return;
  if ((iter+1) % options.show_average != 0)
    return;

  printf("Row Average Temperatures:\n");
  for (int i = 0; i < n; i++) {
    printf("%f, ", rowavg[i]);
  }
  printf("\n");
}

// Print elapsed time given start and end
double elapsed_time(clock_t start, clock_t end) {
  double time_spent_ms = (double)(end - start) / (CLOCKS_PER_SEC/1000);
  return time_spent_ms;
}
void print_elapsed_time(char* fn_name, clock_t start, clock_t end) {
  printf("%s: %fms \n", fn_name, elapsed_time(start, end));
};


// Record timing
void start_timer(struct Timer* timer) {
  timer->start = clock();
}

void end_timer(struct Timer* timer) {
  timer->end = clock();
}