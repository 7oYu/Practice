#ifndef LEARN_CUDA_ATOMIC_H
#define LEARN_CUDA_ATOMIC_H

#include <device_launch_parameters.h>
#include <cuda_runtime.h>

void learn_atomic();

__global__ void learn_atomic(int* out);

#endif