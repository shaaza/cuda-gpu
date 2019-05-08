//=============================================================================================
// Name        		: matrixMultiplicationCommented.cu
// Author      		: Jose Refojo
// Version     		:	06-02-2018
// Creation date	:	22-09-2010
// Copyright		: Copyright belongs to Trinity Centre for High Performance Computing
// Description		: This program will multiply two matrices into a third one (AB=C)
//					  This file will describe matmulGPU, which allocates and transfers the matrices in the global memory of the gpu, and then sets up the kernel and runs it.
//					  The kernel uses a 2d grid (so it spawns a 2d set of threads), one thread per each element of the matrix C
//					  Each particular thread multiplies its row in A by its column in B and stores the obtained value in its position in C
//=============================================================================================

extern int block_size_x;
extern int block_size_y;

extern int MATRIX_SIZE_N;	// Those are the default values of N and M
extern int MATRIX_SIZE_M;
extern int verbose;
extern int skipCpuTest;		// Since the CPU test might take quite a long time, we give an option to skip it

#include "cudaUtils.h"
#include "matrixMultiplicationCommented.h"

#include "stdio.h"
#include "time.h"
#include <getopt.h>

// computeMatMulGPU is the kernel that will run the compute in the GPU: It is run by each and every thread of the grid
// It is run by every thread in a 2d grid (so each thread has an id in the first and second dimensions).
__global__ void computeMatMulGPU (int N,int M,float *A1dGPU,float *B1dGPU,float *C1dGPU) {
	int idx=blockIdx.x*blockDim.x+threadIdx.x;	// The global id of the thread in the first dimension
	int idy=blockIdx.y*blockDim.y+threadIdx.y;	// The global id of the thread in the second dimension
	int k;

	if (idx<N) {	// We do this check to make sure that we do not go past the boundaries of the matrix in the first dimension
		if (idy<N) {	// We do this check to make sure that we do not go past the boundaries of the matrix in the second dimension
			C1dGPU[idx+idy*N]=0.0f;	// Start at zero, add up from there
			for (k=0;k<M;k++) {	// Add the product of the row of A with the column of B
				C1dGPU[idx+idy*N]+=A1dGPU[k+idy*M]*B1dGPU[idx+k*N];
			}
		}
	}
}

// This function serves as a bridge between the main and the GPU code -  we can call it from C or C++ code, and it fires up the CUDA code
void	matmulGPU	(int N,int M,float *A1d,float *B1d,float *C1d) {
	//int i,j;
	cudaError_t err;	// We can use this variable to check the return error of many cuda functions

	float *A1dGPU,*B1dGPU,*C1dGPU;

	// Allocate and transfer matrix A in the GPU
	// There are two ways we can catch errors with cudaMalloc and other cuda functions that aren't kernel - we can request the last error by calling cudaLastErrorCheck, or we can do it this way:
	err = cudaMalloc ((void **) &A1dGPU, sizeof(float)*(N*M));
	if( cudaSuccess != err) {
		printf("(Cuda error %s): %s\n","(matrixMultiplication::cudaMalloc A1dGPU)",cudaGetErrorString( err) );
		exit(EXIT_FAILURE);
	}
	err = cudaMemcpy(A1dGPU, A1d, sizeof(float)*(N*M), cudaMemcpyHostToDevice);
	if( cudaSuccess != err) {
		printf("(Cuda error %s): %s\n","(matrixMultiplication::cudaMemcpy A1dGPU)",cudaGetErrorString( err) );
		exit(EXIT_FAILURE);
	}

	err = cudaMalloc ((void **) &B1dGPU, sizeof(float)*(M*N));
	if( cudaSuccess != err) {
		printf("(Cuda error %s): %s\n","(matrixMultiplication::cudaMalloc B1dGPU)",cudaGetErrorString( err) );
		exit(EXIT_FAILURE);
	}
	err = cudaMemcpy(B1dGPU, B1d, sizeof(float)*(M*N), cudaMemcpyHostToDevice);
	if( cudaSuccess != err) {
		printf("(Cuda error %s): %s\n","(matrixMultiplication::cudaMemcpy B1dGPU)",cudaGetErrorString( err) );
		exit(EXIT_FAILURE);
	}
	err = cudaMemcpy(B1dGPU, B1d, sizeof(float)*(N*M), cudaMemcpyHostToDevice);
	if( cudaSuccess != err) {	// Check for error values
		printf("(Cuda error %s): %s\n","(matmulGPU::cudaMemcpy B1dGPU)",cudaGetErrorString( err) );
		exit(EXIT_FAILURE);
	}

	err = cudaMalloc ((void **) &C1dGPU, sizeof(float)*(N*N));
	if( cudaSuccess != err) {
		printf("(Cuda error %s): %s\n","(matrixMultiplication::cudaMalloc C1dGPU)",cudaGetErrorString( err) );
		exit(EXIT_FAILURE);
	}

	// Set the grid
	dim3 dimBlock(block_size_x,block_size_y);	// Set the number of threads per block
	dim3 dimGrid ( (N/dimBlock.x) + (!(N%dimBlock.x)?0:1),(N/dimBlock.y) + (!(N%dimBlock.y)?0:1) );	// Set the number of blocks in the grid (There are at least as many threads as elements in the result matrix, maybe even more)

	// Test block and grid
	cudaTestBlockInformation (dimBlock);	// Check that we have a legal amount of threads (in most cards, no more than 1024)
	cudaLastErrorCheck("(Cuda error cudaTestBlockInformation)");
	cudaTestGridInformation (dimGrid);		// Check that we have a legal amount of blocks
	cudaLastErrorCheck("(Cuda error cudaTestGridInformation)");

	// Print the size of the grid
	printf("Block size test (2d): %dx%d\n",dimBlock.x,dimBlock.y);
	printf("Grid size in each dimension: %dx%dx%d\n",dimGrid.x,dimGrid.y,dimGrid.z);

	// Call the kernel
	computeMatMulGPU <<<dimGrid,dimBlock>>> (N,M,A1dGPU,B1dGPU,C1dGPU);
	// Fetch the last error state, just in case something went wrong
	cudaLastErrorCheck("(Cuda error in computeMatMulGPU)");

	// Copy the result matrix back into the RAM memory
	err = cudaMemcpy(&(C1d[0]), C1dGPU, sizeof(float)*N*N, cudaMemcpyDeviceToHost);
	if( cudaSuccess != err) {
		printf("(Cuda error %s): %s\n","(matmulGPU::cudaMemcpy C1dGPU)",cudaGetErrorString( err) );
		exit(EXIT_FAILURE);
	}

	// Free the global memory used
	cudaFree(A1dGPU);
	cudaFree(B1dGPU);
	cudaFree(C1dGPU);
}

