#include "my_gemm.cuh"
#include <iostream>

__global__ void init_mat(int* out, int size) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < size) {
		out[index] = index;
	}
}

__global__ void my_gemm(int* m1, int* m2, int h1, int w1, int h2, int w2, int* out) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < h1 * w2) {
		int h = index / w2;
		int w = index % w2;
		int out_element = 0;
		for (int i = 0; i < w1; ++i) {
			out_element += m1[h * w1 + i] * m2[w + i * w2];
		}
		out[w2 * h + w] = out_element;
	}
}

void my_gemm() {
	int h1 = 8;
	int w1 = 4;
	int h2 = w1;
	int w2 = 8;
	int* m1 = nullptr;
	int* m2 = nullptr;
	int* ret = nullptr;
	cudaMallocManaged(&m1, h1 * w1 * sizeof(int));
	cudaMallocManaged(&m2, h2 * w2 * sizeof(int));
	cudaMallocManaged(&ret, h1 * w2 * sizeof(int));
	cudaMemPrefetchAsync(m1, h1 * w1 * sizeof(int), 0);
	cudaMemPrefetchAsync(m2, h2 * w2 * sizeof(int), 0);
	cudaMemPrefetchAsync(ret, h1 * w2 * sizeof(int), 0);
	init_mat<<<4, 1024>>>(m1, h1 * w1);
	init_mat<<<4, 1024>>>(m2, h2 * w2);
	my_gemm <<<4, 1024 >>>(m1, m2, h1, w1, h2, w2, ret);
	cudaMemPrefetchAsync(m1, h1 * w1 * sizeof(int), cudaCpuDeviceId);
	cudaMemPrefetchAsync(m2, h2 * w2 * sizeof(int), cudaCpuDeviceId);
	cudaMemPrefetchAsync(ret, h1 * w2 * sizeof(int), cudaCpuDeviceId);
	cudaDeviceSynchronize();
	std::cout << " m1 :  " << std::endl;
	for (int i = 0; i < h1 * w1; ++i) {
		std::cout << m1[i] << "  ";
		if ((i + 1) % w1 == 0)
			std::cout << std::endl;
	}	
	std::cout << " m2 :  " << std::endl;
	for (int i = 0; i < h2 * w2; ++i) {
		std::cout << m2[i] << "  ";
		if ((i + 1) % w2 == 0)
			std::cout << std::endl;
	}
	std::cout << " ret :  " << std::endl;
	for (int i = 0; i < h1 * w2; ++i) {
		std::cout << ret[i] << "  ";
		if ((i+1) % w2 == 0) 
			std::cout << std::endl;
	}
	cudaFree(m1);
	cudaFree(m2);
	cudaFree(ret);
}


