#include <iostream>
#include "learn_idx.cuh"

void learn_idx() {
	int* ret_array = nullptr;
	size_t ret_size = 16;
	cudaError_t ret = cudaMallocManaged(&ret_array, ret_size * sizeof(int));
	if (ret != cudaSuccess || ret_array == nullptr) {
		std::cerr << "Alloc memory fail ! " << cudaGetErrorString(ret) << std::endl;
		return;
	}
	dim3 grid_size(ret_size, 1, 1);
	dim3 block_size(2, 1, 1);
	// cudaMallocManaged �����ͳһ�ڴ治��ֱ�ӷ����豸�ڴ棬 cudaMemPrefetchAsync������ǰ�����豸�ڴ�
	cudaMemPrefetchAsync(ret_array, ret_size * sizeof(int), 0); 
	learn_idx<<<grid_size, block_size >>>(ret_array, ret_size);
	// ���豸�ڴ��ϵ�����ͬ���������ڴ�
	cudaMemPrefetchAsync(ret_array, ret_size * sizeof(int), cudaCpuDeviceId);
	cudaDeviceSynchronize();
	std::cout << "learn_idx ret: " << std::endl;
	for (int i = 0; i < ret_size; ++i) {
		std::cout << ret_array[i];
		if (i + 1 < ret_size) {
			std::cout << ", ";
		} else {
			std::cout << std::endl;
		}
	}
	
	cudaFree(ret_array);
}

__global__ void learn_idx(int* ret_array, int size) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < size)
		ret_array[index] = index;  // �޸��Թ۲� blockDim blockIdx threadIdx
}