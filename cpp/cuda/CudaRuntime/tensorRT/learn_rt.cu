#include "learn_rt.cuh"
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
using namespace nvinfer1;

__global__ void initializeArray(float* data, float value, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) {
		data[idx] = value;
	}
}


// ref https://blog.csdn.net/weixin_45252450/article/details/123777166
void learn_rt() {
	Logger logger;
	IBuilder* builder = createInferBuilder(logger);
	uint32_t flag = 1 << int(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	INetworkDefinition* network = builder->createNetworkV2(flag);
	ITensor* input = network->addInput("in", DataType::kFLOAT, Dims4(1, 1, 1, 9)); // dim: {N, C, H, W}
	float weight_data[3] = { 1, 1, 1 };
	Weights cov_weights{ DataType::kFLOAT, weight_data, 3 };
	Weights cov_blas{ DataType::kFLOAT, nullptr, 0};
	IConvolutionLayer* cov_layer = network->addConvolution(*input, 1, DimsHW(1, 3), cov_weights, cov_blas);
	cov_layer->setStride(DimsHW{ 1, 1 });
	cov_layer->getOutput(0)->setName("out");
	network->markOutput(*cov_layer->getOutput(0)); // 用于指示网络中的某个张量（Tensor）是最终的输出张量
	IBuilderConfig* config = builder->createBuilderConfig();
	IHostMemory* serialed_network = builder->buildSerializedNetwork(*network, *config);
	IRuntime* runtime = createInferRuntime(logger);
	ICudaEngine* engine = runtime->deserializeCudaEngine(serialed_network->data(), serialed_network->size());
	IExecutionContext* execution = engine->createExecutionContext();
	float* input_ptr = nullptr;
	float* output_ptr = nullptr;
	cudaMallocManaged((void**)&input_ptr, 9 * sizeof(float));
	cudaMallocManaged((void**)&output_ptr, (9 - 2) * sizeof(float));
	// cudaMemset(input_ptr, 1, 9 * sizeof(float)); // cudaMemset只能按字节初始化， 用来初始化float数组有问题
	initializeArray<<<1, 16>>>(input_ptr, 1, 9);
	float* buffer[2] = { input_ptr , output_ptr };
	cudaMemPrefetchAsync(input_ptr, 9 * sizeof(float), cudaMemLocationTypeDevice);
	execution->executeV2((void* const*)buffer);
	cudaMemPrefetchAsync(output_ptr, (9 - 2) * sizeof(float), cudaMemLocationTypeHost);
	cudaDeviceSynchronize();
	std::cout << "input is : \n";
	for (int i = 0; i < 7; ++i) {
		std::cout << input_ptr[0] << " ";
	}
	std::cout << std::endl;
	std::cout << "output is : \n";
	for (int i = 0; i < 7; ++i) {
		std::cout << output_ptr[0] << " ";
	}
	std::cout << std::endl;
	cudaFree(input_ptr);
	cudaFree(output_ptr);
	
}