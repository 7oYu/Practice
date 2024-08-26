#ifndef LEARN_CUDA_SHUFFLE_H
#define LEARN_CUDA_SHUFFLE_H

#include <device_launch_parameters.h>
#include <cuda_runtime.h>

void learn_shuffle();

__global__ void learn_shuffle(int* out);

#endif