#include "learn_stream.cuh"
#include <iostream>

void learn_stream() 
{
	cudaStream_t stream1;  // 如果使用异步的内存拷贝，host memory需要固定内存（通过cudaHostAlloc分配） 
	cudaStreamCreate(&stream1);
	cudaStream_t stream2;
	cudaStreamCreate(&stream2);
	int* data = nullptr;
	int grid_size = 32;
	int block_size = 512;
	int data_size = grid_size * block_size;
	cudaMallocManaged(&data, data_size * sizeof(int));
	cudaMemPrefetchAsync(data, data_size * sizeof(int), 0);
	for (size_t i = 0; i < 2; i++)
	{
		int offset = data_size / 2;
		kernal_add << <grid_size, block_size >> > (data + i * offset, offset);
	}
	cudaMemPrefetchAsync(data, data_size * sizeof(int), cudaCpuDeviceId);
	cudaDeviceSynchronize();
	std::cout << "learn_stream ret: " << std::endl;
	for (int i = 0; i < 16; ++i) {
		std::cout << data[i];
		if (i + 1 < 16) {
			std::cout << ", ";
		}
		else {
			std::cout << std::endl;
		}
	}
	cudaFree(data);
}

__global__ void kernal_add(int* out, int size) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < size)
		out[index] += 1;
}
