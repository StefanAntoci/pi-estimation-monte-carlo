#define INTERVAL 16777216
#define ITEMS_IN_THREAD 16
#define THREADS_PER_BLOCK 16
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <random>

using namespace std;
__global__ void count_pi_1( float *dev_randX, float *dev_randY, int *dev_threads_num ) {
	int tid =  blockIdx.x * blockDim.x  + threadIdx.x ;
	float result = dev_randX[tid] * dev_randX[tid] + dev_randY[tid] * dev_randY[tid];
  if (result <= 1.0) {
      dev_threads_num[tid] = 1;
  } else {
      dev_threads_num[tid] = 0;
  }
}	

int main()
{ 
  vector<float> randX(INTERVAL);
	vector<float> randY(INTERVAL);
	clock_t c_start, c_end;
	srand((unsigned)time(NULL));
	for (int i = 0; i < INTERVAL; i++) {
		randX[i] = float(rand()) / RAND_MAX;
		randY[i] = float(rand()) / RAND_MAX;
	}
  //cout << "initial array" << endl << endl;
  //for (int i = 0; i < INTERVAL; i++) {
	//	cout << randX[i] << " | ";
	//}

	//start cont gpu time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	//send data to GPU
	size_t size = INTERVAL * sizeof(float);
	float *dev_randX;
	float *dev_randY;
	cudaMalloc((void**)&dev_randX, size);
	cudaMalloc((void**)&dev_randY, size);

	cudaMemcpy(dev_randX, &randX.front(), size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_randY, &randY.front(), size, cudaMemcpyHostToDevice);

	
	int block_num = INTERVAL / THREADS_PER_BLOCK ;
	int *dev_threads_num;
	cudaMalloc((void**)&dev_threads_num,  INTERVAL  * sizeof(int) );
 
 	float *test;
	cudaMalloc((void**)&test,  INTERVAL  * sizeof(float) );

	count_pi_1 <<<block_num, THREADS_PER_BLOCK >>> ( dev_randX, dev_randY, dev_threads_num );

	int* threads_num = (int*)malloc( INTERVAL * sizeof(int) );
	cudaMemcpy(threads_num, dev_threads_num, INTERVAL * sizeof(int), cudaMemcpyDeviceToHost);

	float* test_host = (float*)malloc( INTERVAL * sizeof(float) );
	cudaMemcpy(test_host, test, INTERVAL * sizeof(float), cudaMemcpyDeviceToHost);

	int g_count = 0;
	for (int i = 0; i < INTERVAL ; i++) {
		g_count += threads_num[i];
	};

	//end cont gpu time
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float t_gpu1;
	cudaEventElapsedTime(&t_gpu1, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	float g_num = float(g_count) * 4.0 / INTERVAL;
	cout << "GPU_1 Time" << endl;
	cout << g_num << endl;
	cout << "time = " << t_gpu1 << " ms" << endl;
}