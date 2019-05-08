#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <getopt.h>

// Parsing options
struct Options {
  int seed_milliseconds;
  int timing;
  int rows;
  int cols;
};
void parse_options_with_defaults(int argc, char** argv, struct Options* options);

// Printing functions
void print_matrix(float** mat, int n, int m);
void print_vector(float* vector, int n);

// Matrix memory allocation and init
float** alloc_matrix(int n, int m);
void initialize_matrix_values(float** matrix, int n, int m);
void free_matrix(float** mat, int n);

// Computations
float* add_rows(float** mat, int n, int m);
float* add_columns(float** mat, int n, int m);
float reduce_vector(float* vec, int n);

struct Options options; // global var

int main(int argc, char* argv[]) {
  parse_options_with_defaults(argc, argv, &options);

  int n = options.rows;
  int m = options.cols;

  // Allocate, initialize floating point matrix (n/m)
  float** matrix = alloc_matrix(n, m);
  initialize_matrix_values(matrix, n, m);

  // Compute row-sum, col-sum and their reduced values.
  float* rowsum_vec = add_rows(matrix, n, m);
  float* colsum_vec = add_columns(matrix, n, m);

  // Print matrix and vectors if small
  if (n < 5 && m < 5) print_matrix(matrix, n, m);
  if (n < 5) print_vector(rowsum_vec, n);
  if (m < 5) print_vector(colsum_vec, n);

  printf("Rowsum sum: %f \n", reduce_vector(rowsum_vec, n));
  printf("Colsum sum: %f \n", reduce_vector(colsum_vec, m));

  // Free matrix and vectors
  free(rowsum_vec);
  free(colsum_vec);
  free_matrix(matrix, n);
}

float** alloc_matrix(int n, int m) {
  float** matrix = malloc(n*sizeof(float*));
  for (int i = 0; i < n; i++) {
    matrix[i] = malloc(m*sizeof(float));
  }
  return matrix;
}

void free_matrix(float** mat, int n) {
   for (int i = 0; i < n; i++) {
    free(mat[i]);
  }
}

long int milliseconds_since_epoch_now() {
  struct timespec spec;
  clock_gettime(CLOCK_REALTIME, &spec);
  long int ms = round(spec.tv_nsec / 1.0e6); // Convert nanoseconds to milliseconds
  return ms;
}

void initialize_matrix_values(float** matrix, int n, int m) {
  long int seed = options.seed_milliseconds ? milliseconds_since_epoch_now() : 123456; // Use default if -r not specified
  srand48(seed);

  // Initialize floating point matrix
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      matrix[i][j] = (float) (drand48()*2.0) - 1.0;
    }
  }
}

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
void print_vector(float* vector, int n) {
  printf("(vector) ");
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

// Sum up n rows of a matrix into a vector of size n
float* add_rows(float** mat, int n, int m) {
  clock_t start, end;
  start = clock();

  float* output = malloc(n*sizeof(float));
  for (int i = 0; i < n; i++) {
    float sum = 0;
    for (int j = 0; j < m; j++) {
      sum += mat[i][j];
    }
    output[i] = sum;
  }

  end = clock();
  if (options.timing) print_elapsed_time("add_rows", start, end);

  return output;
}
// Sum up m columns of a matrix into a vector of size m
float* add_columns(float** mat, int n, int m) {
  clock_t start, end;
  start = clock();

  float* output = malloc(m*sizeof(float));
  for (int j = 0; j < m; j++) {
    float sum = 0;
    for (int i = 0; i < n; i++) {
      sum += mat[i][j];
    }
    output[j] = sum;
  }

  end = clock();
  if (options.timing) print_elapsed_time("add_columns", start, end);

  return output;
}

// Sum up n elements of a vector
float reduce_vector(float* vec, int n) {
  clock_t start, end;
  start = clock();

  float sum = 0;
  for (int i = 0; i < n; i++) {
    sum += vec[i];
  }

  end = clock();
  if (options.timing) print_elapsed_time("reduce_vector", start, end);

  return sum;
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
