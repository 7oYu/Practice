#include "learn_cublas.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>

void learn_cublas() {
    cublasHandle_t handle;
    cublasCreate(&handle);
    float h_A[4] = { 1.0f, 2.0f, 3.0f, 4.0f };  // 2x2 矩阵
    float h_x[2] = { 1.0f, 1.0f };               // 长度为 2 的向量
    float h_y[2] = { 0.0f, 0.0f };               // 长度为 2 的向量，初始化为 0
    float alpha = 1.0f;
    float beta = 0.0f;
    float* d_A, * d_x, * d_y;
    cudaMalloc((void**)&d_A, 4 * sizeof(float));
    cudaMalloc((void**)&d_x, 2 * sizeof(float));
    cudaMalloc((void**)&d_y, 2 * sizeof(float));
    cudaMemcpy(d_A, h_A, 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, 2 * sizeof(float), cudaMemcpyHostToDevice);
    cublasSgemv(handle, 
        CUBLAS_OP_N, // 是否对矩阵A进行转置操作
        2,           // A的行数
        2,           // A的列数
        &alpha,      // 标量, 矩阵A与向量x的乘积的系数
        d_A,         // 矩阵A
        2,           // 隔几个元素进行换行（数据在内存中连续存储需要指定几个数据为一行或一列）
        d_x,         // 向量x
        1,           // 向量x中各元素间隔（如果使用x中所有数据指定为1即可）
        &beta,       // beta==0 then y does not have to be a valid input
        d_y,         // y的指针, 保存计算结果
        1            // 向量y中各元素间隔（如果使用x中所有数据指定为1即可）
    );
    cudaMemcpy(h_y, d_y, 2 * sizeof(float), cudaMemcpyDeviceToHost);  // 此处同步，无需sync
    std::cout << "Result: y = [" << h_y[0] << ", " << h_y[1] << "]" << std::endl;
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);
    cublasDestroy(handle);
}
