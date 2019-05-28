
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
  float duration_ms;
};

struct HostStats {
  struct Timer cpu_compute;
  struct Timer row_avg;
};

struct DeviceStats {
  struct Timer allocation;
  struct Timer to_gpu_transfer;
  struct Timer gpu_compute;
  struct Timer row_avg;
  struct Timer to_cpu_transfer;
};
// CPU versions
void perform_cpu_evolution(float** old_mat, float** new_mat, struct HostStats* host_stats);

// GPU versions
void perform_gpu_evolution(float* out_mat1d, float* mat1d, struct DeviceStats* stats);

// Util fns shared across CPU & GPU
void print_matrix(float** mat, int n, int m);
void print_compute_results(char* title, float* rowvec, int n, int m);
void print_row_avg(float* rowavg, int n, int iter);
void print_elapsed_time(char* fn_name, clock_t start, clock_t end);

double elapsed_time(clock_t start, clock_t end);
void start_timer(struct Timer* timing);
void end_timer(struct Timer* timing);
