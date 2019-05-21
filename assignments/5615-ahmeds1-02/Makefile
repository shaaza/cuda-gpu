CC=nvcc
CFLAGS=
BINARY_NAME=radiator.exec
OBJS=gpu.o cpu.o util.o
HEADERS=shared.h

all: $(BINARY_NAME) clean

$(BINARY_NAME): $(OBJS) $(HEADERS)
	$(CC) $(CFLAGS) radiator.cu $(OBJS) -o $(BINARY_NAME)

gpu.o:  cpu.o util.o
	$(CC) $(CFLAGS) -c gpu.cu

cpu.o:  util.o
	$(CC) $(CFLAGS) -c cpu.cu

util.o:
	$(CC) $(CFLAGS) -c util.cu

clean:
	rm -rf *.o
	rm -rf *~

generate-graphs:
	./run.py
