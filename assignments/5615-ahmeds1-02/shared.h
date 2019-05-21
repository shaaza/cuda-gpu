
// Options (command-line)
struct Options {
  int rows;               // -n NUMBER
  int cols;               // -m NUMBER
  int iterations;         // -p NUMBER
  int show_average;       // -a NUMBER

  int print_vectors;
  int disp_time_adjacent; // -d
  int seed_milliseconds;  // -r
  int timing;             // -t

};

// Timing
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
void perform_cpu_evolution(struct Stats* host_stats);

// GPU versions
void perform_gpu_operations(float* mat1d, struct Stats* device_stats);

// Util fns shared across CPU & GPU
void print_matrix(float** mat, int n, int m);
void print_compute_results(char* title, float* rowvec, float* colvec, float rowsum, float colsum, int n, int m);
void print_elapsed_time(char* fn_name, clock_t start, clock_t end);

double elapsed_time(clock_t start, clock_t end);
void start_timer(struct Timer* timing);
void end_timer(struct Timer* timing);
