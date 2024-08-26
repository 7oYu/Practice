#include "learn_shuffle.cuh"
#include <iostream>

void learn_shuffle() {
	int* ret_array = nullptr;
	size_t ret_size = 8 * 64;
	cudaError_t ret = cudaMalloc(&ret_array, ret_size * sizeof(int));
	if (ret != cudaSuccess || ret_array == nullptr) {
		std::cerr << "Alloc device memory fail ! " << cudaGetErrorString(ret) << std::endl;
		return;
	}
	learn_shuffle<<<8, 64 >>> (ret_array);
	int* host_ret_array = nullptr;
	ret = cudaMallocHost(&host_ret_array, ret_size * sizeof(int));
	if (ret != cudaSuccess || host_ret_array == nullptr) {
		std::cerr << "Alloc host memory fail ! " << cudaGetErrorString(ret) << std::endl;
		return;
	}
	ret = cudaMemcpy(host_ret_array, ret_array, ret_size * sizeof(int), cudaMemcpyDeviceToHost);
	if (ret != cudaSuccess) {
		std::cerr << "Memcpy fail ! " << cudaGetErrorString(ret) << std::endl;
		return;
	}

	std::cout << "learn_shared ret: " << std::endl;
	for (int i = 0; i < ret_size; ++i) {
		std::cout << host_ret_array[i];
		if (i + 1 < ret_size) {
			std::cout << ", ";
		}
		else {
			std::cout << std::endl;
		}
	}

	cudaFreeHost(host_ret_array);
	cudaFree(ret_array);
}

__global__ void learn_shuffle(int* out) {
	int warp_id = threadIdx.x / warpSize;
	int lane_id = threadIdx.x % warpSize;
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	out[index] = __shfl_sync(0xFFFFFFFF, index, 6, 0);  // 把lane id 为6的线程的index广播给warp内的其他线程
}