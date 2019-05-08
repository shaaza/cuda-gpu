
// Options struct
struct Options {
  int seed_milliseconds;  // -r
  int timing;             // -t
  int rows;               // -n NUMBER
  int cols;               // -m NUMBER
};

// Timer
struct Timer {
  clock_t start;
  clock_t end;
};

struct Stats {
  struct Timer add_rows;
  struct Timer add_columns;
  struct Timer reduce_vector_rows;
  struct Timer reduce_vector_cols;
};

// CPU versions
void perform_cpu_operations(struct Stats* host_stats);

// GPU versions
void perform_gpu_operations(struct Stats* device_stats);

// Util fns shared across CPU & GPU
void print_matrix(float** mat, int n, int m);
void print_vector(float* vector, int n, char* name);
void print_elapsed_time(char* fn_name, clock_t start, clock_t end);
void initialize_matrix_values(float** matrix, int n, int m);
void start_timer(struct Timer* timing);
void end_timer(struct Timer* timing);

// Temporarily
float reduce_vector(float* vec, int n, struct Timer* timing);
