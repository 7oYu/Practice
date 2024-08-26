#pragma once
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

void learn_stream();

__global__ void kernal_add(int* out, int size);
