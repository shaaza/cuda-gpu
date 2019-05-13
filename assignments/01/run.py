#!/usr/bin/python
import subprocess
import matplotlib.pyplot as plt
import sys

matrix_sizes = [1000,5000,10000,30000]
block_sizes = [4,8,16,32,64,128,256,512,1024]

if len(sys.argv) < 2:
    print("Please specify block size for filename")
    sys.exit()

filename_block_size = sys.argv[1]

def run_computation(n: int, m: int, threads_per_block: int) -> str:
    result = subprocess.run(['./matrix_2.exec', '-n', str(n), '-m', str(m), '-t', '-d'], stdout=subprocess.PIPE)
    return str(result.stdout)

def timing(timings_array):
    t = {}
    t["add_rows"] = timings_array[0]
    t["add_cols"] = timings_array[1]
    t["reduce_vector_rows"] = timings_array[2]
    t["reduce_vector_cols"] = timings_array[3]
    return t

def parse_output(output: str):
    lines = output.split("\\n")
    non_empty_lines = filter(lambda x: len(x) > 0, lines)
    timing_lines = lines[9:-1] # Drop first 8 lines and last line
    split_lines = list(map(lambda x: x.split(": "), timing_lines))
    flattened = [elem for arr in split_lines for elem in arr]
    timings_array = map(lambda x: {"cpu": float(x.split(", ")[0][:-2]),
                                   "gpu": float(x.split(", ")[1][:-2]),
                                   "speedup": float(x.split(", ")[0][:-2])/float(x.split(", ")[1][:-2])},
                        flattened[1::2])
    return timing(list(timings_array))

timing_results = []
for n in matrix_sizes:
    out = run_computation(n, n, 4)
    timing_results.append(parse_output(out));

def plot_speedup_graph(timing_results):
    plt.plot(matrix_sizes, list(map(lambda x: x["add_rows"]["speedup"], timing_results)), linestyle="--", marker='o', color='b')
    plt.plot(matrix_sizes, list(map(lambda x: x["add_cols"]["speedup"], timing_results)), linestyle="--", marker='o', color='g')
    plt.plot(matrix_sizes, list(map(lambda x: x["reduce_vector_rows"]["speedup"], timing_results)), linestyle="--", marker='o', color='r')
    plt.plot(matrix_sizes, list(map(lambda x: x["reduce_vector_cols"]["speedup"], timing_results)), linestyle="--", marker='o', color='black')
    plt.xlabel("Matrix size")
    plt.ylabel("Speedup")
    plt.savefig("speedup_"+filename_block_size+".png")
    plt.clf()

plot_speedup_graph(timing_results)

print(timing_results)
