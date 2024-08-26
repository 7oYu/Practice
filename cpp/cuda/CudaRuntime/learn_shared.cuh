#ifndef LEARN_CUDA_SHARED_H
#define LEARN_CUDA_SHARED_H

#include <device_launch_parameters.h>
#include <cuda_runtime.h>

void learn_shared();

__global__ void learn_shared(int* in, int* out);

#endif