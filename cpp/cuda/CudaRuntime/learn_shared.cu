#include <iostream>
#include <vector>
#include "learn_shared.cuh"

#define RADIUS (3)
#define BLOCK_SIZE (32)
#define GRID_SIZE (4)


void learn_shared() {
	int* in_array = nullptr;
	size_t array_size = BLOCK_SIZE * GRID_SIZE;
	cudaError_t ret = cudaMalloc(&in_array, array_size * sizeof(int));
	if (ret != cudaSuccess || in_array == nullptr) {
		std::cerr << "Alloc device memory fail ! " << cudaGetErrorString(ret) << std::endl;
		return;
	}
	int* out_array = nullptr;
	ret = cudaMalloc(&out_array, array_size * sizeof(int));
	if (ret != cudaSuccess || out_array == nullptr) {
		std::cerr << "Alloc device memory fail ! " << cudaGetErrorString(ret) << std::endl;
		return;
	}
	std::vector<int> host_array(array_size, 1);
	ret = cudaMemcpy(in_array, host_array.data(), array_size * sizeof(int), cudaMemcpyHostToDevice);
	if (ret != cudaSuccess) {
		std::cerr << "Memcpy fail ! " << cudaGetErrorString(ret) << std::endl;
		return;
	}
	dim3 grid_size(GRID_SIZE, 1, 1);
	dim3 block_size(BLOCK_SIZE, 1, 1);
	learn_shared<<<grid_size, block_size>>>(in_array, out_array);

	ret = cudaMemcpy(host_array.data(), out_array, array_size * sizeof(int), cudaMemcpyDeviceToHost);
	if (ret != cudaSuccess) {
		std::cerr << "Memcpy fail ! " << cudaGetErrorString(ret) << std::endl;
		return;
	}

	std::cout << "learn_shared ret: " << std::endl;
	for (int i = 0; i < array_size; ++i) {
		std::cout << host_array[i];
		if (i + 1 < array_size) {
			std::cout << ", ";
		}
		else {
			std::cout << std::endl;
		}
	}

	cudaFree(in_array);
	cudaFree(out_array);
}

__global__ void learn_shared(int* in, int* out) {
	__shared__ int temp[BLOCK_SIZE + 2 * RADIUS];  // �����ڴ棬��ͬһblock�ڹ�����ȫ���ڴ棨cudaMalloc����ģ����죬һ�����ڿ����߳�ͨ�ź͹�����Ϣ
	int global_index = threadIdx.x + blockDim.x * blockIdx.x;
	int shared_index = threadIdx.x + RADIUS;
	temp[shared_index] = in[global_index];
	__syncthreads();  // ����ͬ��ͬһ�����ڵ������̣߳���ֹ�����ڴ��ʼ�����ǰ��ʼ�����������
	int ret = 0;
	for (int offset = -RADIUS; offset <= RADIUS; ++offset) {
		ret += temp[shared_index + offset];
	}
	out[global_index] = ret;
}