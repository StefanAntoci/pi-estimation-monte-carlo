#define INTERVAL 512*512*128
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <random>

using namespace std;
__global__ void count_pi_1(float *dev_randX, float *dev_randY, int *dev_threads_num, int n) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int cont = 0;
	for (int i = tid * 128; i < 128 * (tid + 1); i++) {
		if (dev_randX[i] * dev_randX[i] + dev_randY[i] * dev_randY[i] < 1.0f) {
			cont++;
		}
	}
	dev_threads_num[tid] = cont;
}

int main()
{
  cout << "entry point" << flush << endl;
 clock_t c_start, c_end;
 float* randX;
 float* randY;
 	size_t size = INTERVAL * sizeof(float);
 randX = (float*) malloc (size);
 randY = (float*) malloc (size);

 srand((unsigned)time(NULL));
 
  
	for (int i = 0; i < INTERVAL; i++) {
		randX[i] = float(rand()) / RAND_MAX;
		randY[i] = float(rand()) / RAND_MAX;
	}
 

	float pi;
 

 // set up timing stuff
	float gpu_elapsed_time;
	cudaEvent_t gpu_start, gpu_stop;
	cudaEventCreate(&gpu_start);
	cudaEventCreate(&gpu_stop);
	cudaEventRecord(gpu_start, 0);

	// allocate memory

	float *dev_randX;
	float *dev_randY;
	cudaMalloc((void**)&dev_randX, size);
	cudaMalloc((void**)&dev_randY, size);
  cudaMemcpy(dev_randX, randX, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_randY, randY, size, cudaMemcpyHostToDevice);
 cout << "memory allocation" << flush << endl;
	int *dev_threads_num;
  //block dimensions setup
  int threadsPerBlock = 512;
	int block_num = INTERVAL / (128 * threadsPerBlock);
	cudaMalloc((void**)&dev_threads_num, INTERVAL / 128 * sizeof(int));

 
	count_pi_1 <<<block_num, threadsPerBlock >>> (dev_randX, dev_randY, dev_threads_num,INTERVAL);

	int* threads_num = new int[INTERVAL / 128];
	cudaMemcpy(threads_num, dev_threads_num, INTERVAL / 128 * sizeof(int), cudaMemcpyDeviceToHost);

	int g_count = 0;
	for (int i = 0; i < INTERVAL / 128; i++) {
		g_count += threads_num[i];
	};

	//end cont gpu time
	cudaEventRecord(gpu_stop, 0);
	cudaEventSynchronize(gpu_stop);
	float t_gpu1;
	cudaEventElapsedTime(&t_gpu1, gpu_start, gpu_stop);
	cudaEventDestroy(gpu_start);
	cudaEventDestroy(gpu_stop);

	float g_num = float(g_count) * 4.0 / INTERVAL;
	cout << "GPU_1 Time" << endl;
	cout << g_num << endl;
	cout << "time = " << t_gpu1 << " ms" << endl;

	// delete memory
	cudaFree(dev_randX);
	cudaFree(dev_randY);
 	cudaFree(dev_threads_num);
 
 return 0; 
}