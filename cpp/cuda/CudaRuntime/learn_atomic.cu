#include "learn_atomic.cuh"
#include <iostream>


void learn_atomic() {
	int* ret_array = nullptr;
	cudaError_t ret = cudaMalloc(&ret_array, sizeof(int));
	if (ret != cudaSuccess || ret_array == nullptr) {
		std::cerr << "Alloc device memory fail ! " << cudaGetErrorString(ret) << std::endl;
		return;
	}
	learn_atomic <<<16, 512 >>> (ret_array);
	int* host_ret_array = nullptr;
	ret = cudaMallocHost(&host_ret_array, sizeof(int));
	if (ret != cudaSuccess || host_ret_array == nullptr) {
		std::cerr << "Alloc host memory fail ! " << cudaGetErrorString(ret) << std::endl;
		return;
	}
	ret = cudaMemcpy(host_ret_array, ret_array, sizeof(int), cudaMemcpyDeviceToHost);
	if (ret != cudaSuccess) {
		std::cerr << "Memcpy fail ! " << cudaGetErrorString(ret) << std::endl;
		return;
	}
	std::cout << "learn_atomic ret: " << host_ret_array[0];

	cudaFreeHost(host_ret_array);
	cudaFree(ret_array);
	
}


__global__ void learn_atomic(int* out) {
	atomicAdd(out, 1);
}