#include "learn_cublas.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>

void learn_cublas() {
    cublasHandle_t handle;
    cublasCreate(&handle);
    float h_A[4] = { 1.0f, 2.0f, 3.0f, 4.0f };  // 2x2 ����
    float h_x[2] = { 1.0f, 1.0f };               // ����Ϊ 2 ������
    float h_y[2] = { 0.0f, 0.0f };               // ����Ϊ 2 ����������ʼ��Ϊ 0
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
        CUBLAS_OP_N, // �Ƿ�Ծ���A����ת�ò���
        2,           // A������
        2,           // A������
        &alpha,      // ����, ����A������x�ĳ˻���ϵ��
        d_A,         // ����A
        2,           // ������Ԫ�ؽ��л��У��������ڴ��������洢��Ҫָ����������Ϊһ�л�һ�У�
        d_x,         // ����x
        1,           // ����x�и�Ԫ�ؼ�������ʹ��x����������ָ��Ϊ1���ɣ�
        &beta,       // beta==0 then y does not have to be a valid input
        d_y,         // y��ָ��, ���������
        1            // ����y�и�Ԫ�ؼ�������ʹ��x����������ָ��Ϊ1���ɣ�
    );
    cudaMemcpy(h_y, d_y, 2 * sizeof(float), cudaMemcpyDeviceToHost);  // �˴�ͬ��������sync
    std::cout << "Result: y = [" << h_y[0] << ", " << h_y[1] << "]" << std::endl;
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);
    cublasDestroy(handle);
}
