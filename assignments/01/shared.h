
// Options struct
struct Options {
  int seed_milliseconds;
  int timing;
  int rows;
  int cols;
};

// extern struct Options options;

// CPU versions
void perform_cpu_operations();

// GPU versions
void perform_gpu_operations();

// Util fns shared across CPU & GPU
void print_matrix(float** mat, int n, int m);
void print_vector(float* vector, int n, char* name);
void print_elapsed_time(char* fn_name, clock_t start, clock_t end);
void initialize_matrix_values(float** matrix, int n, int m);

// TEmporarily
float reduce_vector(float* vec, int n);
