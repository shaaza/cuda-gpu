#!/usr/bin/python
import subprocess
import matplotlib.pyplot as plt
import sys
import re
from tempfile import mkstemp
from shutil import move
import pwd
import grp
from os import fdopen, remove, chmod, chown, getcwd

matrix_sizes = [1000,5000,10000,30000]
block_sizes = [4,8,16,32,64,128,256,512,1024]

def run_computation(n: int, m: int) -> str:
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

def set_block_size_in_header(file_path, block_size):
    fh, abs_path = mkstemp()      # Create temp file
    with fdopen(fh,'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                new_file.write(re.sub(r"\d+", str(block_size), line))

    remove(file_path)             # Remove original file
    move(abs_path, file_path)     # Move new file

    chmod(file_path, 0o755)
    uid = pwd.getpwnam("shaaza").pw_uid
    gid = grp.getgrnam("shaaza").gr_gid
    chown(file_path, uid, gid)

    subprocess.run(['make'])      # Run make to recompile

def plot_speedup_graph(timing_results, block_size):
    add_rows, = plt.plot(matrix_sizes, list(map(lambda x: x["add_rows"]["speedup"], timing_results)), linestyle="--", marker='o', color='b')
    add_cols, = plt.plot(matrix_sizes, list(map(lambda x: x["add_cols"]["speedup"], timing_results)), linestyle="--", marker='o', color='g')
    reduce_vector_rows, = plt.plot(matrix_sizes, list(map(lambda x: x["reduce_vector_rows"]["speedup"], timing_results)), linestyle="--", marker='o', color='r')
    reduce_vector_cols, = plt.plot(matrix_sizes, list(map(lambda x: x["reduce_vector_cols"]["speedup"], timing_results)), linestyle="--", marker='o', color='black')

    plt.legend([add_rows, add_cols, reduce_vector_rows, reduce_vector_cols], ['Add rows', 'Add columns', 'Reduce vector (rows)', 'Reduce vector (cols)'])
    plt.title("Block Size: "+ str(block_size))
    plt.xlabel("Matrix size")
    plt.ylabel("Speedup")
    plt.savefig("speedup_"+ str(block_size) +".png")
    plt.clf()

def plot_time_graph(timing_results, block_size):
    add_rows, = plt.plot(matrix_sizes, list(map(lambda x: x["add_rows"]["cpu"], timing_results)), linestyle="--", marker='o', color='b')
    add_cols, = plt.plot(matrix_sizes, list(map(lambda x: x["add_cols"]["cpu"], timing_results)), linestyle="--", marker='o', color='g')
    reduce_vector_rows, = plt.plot(matrix_sizes, list(map(lambda x: x["reduce_vector_rows"]["cpu"], timing_results)), linestyle="--", marker='o', color='r')
    reduce_vector_cols, = plt.plot(matrix_sizes, list(map(lambda x: x["reduce_vector_cols"]["cpu"], timing_results)), linestyle="--", marker='o', color='black')

    plt.legend([add_rows, add_cols, reduce_vector_rows, reduce_vector_cols], ['Add rows', 'Add columns', 'Reduce vector (rows)', 'Reduce vector (cols)'])
    plt.title("Block Size: "+ str(block_size))
    plt.xlabel("Matrix size")
    plt.ylabel("CPU Time (ms)")
    plt.savefig("time_cpu_"+ str(block_size) +".png")
    plt.clf()

    add_rows, = plt.plot(matrix_sizes, list(map(lambda x: x["add_rows"]["gpu"], timing_results)), linestyle="--", marker='o', color='b')
    add_cols, = plt.plot(matrix_sizes, list(map(lambda x: x["add_cols"]["gpu"], timing_results)), linestyle="--", marker='o', color='g')
    reduce_vector_rows, = plt.plot(matrix_sizes, list(map(lambda x: x["reduce_vector_rows"]["gpu"], timing_results)), linestyle="--", marker='o', color='r')
    reduce_vector_cols, = plt.plot(matrix_sizes, list(map(lambda x: x["reduce_vector_cols"]["gpu"], timing_results)), linestyle="--", marker='o', color='black')

    plt.legend([add_rows, add_cols, reduce_vector_rows, reduce_vector_cols], ['Add rows', 'Add columns', 'Reduce vector (rows)', 'Reduce vector (cols)'])
    plt.title("Block Size: "+ str(block_size))
    plt.xlabel("Matrix size")
    plt.ylabel("GPU Time (ms)")
    plt.savefig("time_gpu_"+ str(block_size) +".png")
    plt.clf()


for b in block_sizes:
    timing_results = []
    print("Block size: "+str(b))
    for n in matrix_sizes:
        set_block_size_in_header(str(getcwd()) + "/block_size.h", b)
        out = run_computation(n, n)
        timing_results.append(parse_output(out));
    plot_speedup_graph(timing_results, b)
    # plot_time_graph(timing_results, b)
    print(timing_results)
