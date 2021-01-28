#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <time.h>
#include <random>
#include <curand.h>
#include <math.h>
#include <curand_kernel.h>

__global__ void setup_kernel(curandState* state);
__global__ void monte_carlo_pi_kernel(curandState* state, int* count, int m);

__global__ void setup_kernel(curandState* state)
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	curand_init(123, index, 0, &state[index]);
}




__global__ void monte_carlo_pi_kernel(curandState* state, int* count, int m)
{
	unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;

	__shared__ int memory[256];
	memory[threadIdx.x] = 0;
	__syncthreads();


	unsigned int temp = 0;
	while (temp < m) {
		float x = curand_uniform(&state[index]);
		float y = curand_uniform(&state[index]);
		float r = x * x + y * y;

		if (r <= 1) {
			memory[threadIdx.x]++;
			
		}
		temp++;
	}
	__syncthreads();
	// reduction
	int i = blockDim.x / 2;
	while (i != 0) {
		if (threadIdx.x < i) {
			memory[threadIdx.x] += memory[threadIdx.x + i];
		}

		i /= 2;
		__syncthreads();
	}


	// update to our global variable count
	if (threadIdx.x == 0) {
		atomicAdd(count, memory[0]);
	}
}





int main()
{
	unsigned int n = 256* 256;
	unsigned int m = 1000;
	int* h_count;
	int* d_count;
	curandState* d_state;
	float pi;


	// allocate memory
	h_count = (int*)malloc(n * sizeof(int));
	cudaMalloc((void**)&d_count, n * sizeof(int));
	cudaMalloc((void**)&d_state, n * sizeof(curandState));
	cudaMemset(d_count, 0, sizeof(int));


	// set up timing stuff
	float gpu_elapsed_time;
	cudaEvent_t gpu_start, gpu_stop;
	cudaEventCreate(&gpu_start);
	cudaEventCreate(&gpu_stop);
	cudaEventRecord(gpu_start, 0);


	// set kernel
	dim3 gridSize = 256;
	dim3 blockSize = 256;
	setup_kernel << < gridSize, blockSize >> > (d_state);


	// monti carlo kernel
	monte_carlo_pi_kernel << <gridSize, blockSize >> > (d_state, d_count, m);


	// copy results back to the host
	cudaMemcpy(h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
	cudaEventRecord(gpu_stop, 0);
	cudaEventSynchronize(gpu_stop);
	cudaEventElapsedTime(&gpu_elapsed_time, gpu_start, gpu_stop);
	cudaEventDestroy(gpu_start);
	cudaEventDestroy(gpu_stop);


	// display results and timings for gpu
	pi = *h_count * 4.0 / (n * m);
	std::cout << "Approximate pi calculated on GPU is: " << pi << " and calculation took " << gpu_elapsed_time << std::endl;

	// delete memory
	free(h_count);
	cudaFree(d_count);
	cudaFree(d_state);
}
