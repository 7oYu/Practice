#pragma once
#include <NvInfer.h>
#include <iostream>
#include <NvInfer.h>
#include <vector>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include "logger.h"

class Bbox
{
public:
    float x1 = 0;
    float y1 = 0;
    float x2 = 0;
    float y2 = 0;
    float conf = 0.0;
    int type = 0;
};

class InferBase 
{
public:
    nvinfer1::ICudaEngine* loadEngineFromFile(const std::string& fileName, nvinfer1::IRuntime* runtime);
    void saveEngineToFile(nvinfer1::ICudaEngine* engine, const std::string& fileName);
    nvinfer1::ICudaEngine* loadModelAndCreateEngine(const std::string& modelFile, int maxBatchSize, nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config);
};

class YoloDetectInfer : public InferBase
{
public:
    void init(const std::string& model_path, int height = 640, int width = 640, int anchor_size = 25500, int class_num = 85);
    void infer(const std::string& image_path, float threshold = 0.5f);
    ~YoloDetectInfer();
private:
    void processOutput(float* output, std::vector<Bbox>& processed_results, float threshold);
    float iou(const Bbox& b1, const Bbox& b2);
    void show(cv::Mat& image, const std::vector<Bbox>& boxes);
private:
    int height = 0;
    int width = 0;
    int batch = 1;
    int anchor_size = 0;
    int class_num = 0;
    nvinfer1::IBuilder* builder = nullptr;
    nvinfer1::IBuilderConfig* config = nullptr;
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    Logger logger;
};

void deploy_yolo();