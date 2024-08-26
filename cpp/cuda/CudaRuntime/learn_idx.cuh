#ifndef LEARN_CUDA_INDEX_H
#define LEARN_CUDA_INDEX_H

#include <device_launch_parameters.h>
#include <cuda_runtime.h>

void learn_idx();

__global__ void learn_idx(int* ret_array, int size);

#endif