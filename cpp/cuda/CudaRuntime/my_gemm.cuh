#pragma once
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

void my_gemm();

__global__ void my_gemm(int* out, int size);