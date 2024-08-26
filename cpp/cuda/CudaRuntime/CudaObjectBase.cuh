#pragma once

#include <device_launch_parameters.h>
#include <cuda_runtime.h>

class CudaObjectBase 
{
public: 
	void* operator new (size_t len) {
		void* ptr = nullptr;
		cudaMallocManaged(&ptr, len);
		cudaDeviceSynchronize();
		return ptr;
	}

	void operator delete(void* ptr) {
		cudaDeviceSynchronize();
		cudaFree(ptr);
	}
};
