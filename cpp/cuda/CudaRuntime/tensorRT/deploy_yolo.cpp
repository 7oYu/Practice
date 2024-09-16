#include "deploy_yolo.h"
#include <iostream>
#include <NvOnnxParser.h>
#include <algorithm>
#include <fstream>

using namespace nvinfer1;

ICudaEngine* InferBase::loadEngineFromFile(const std::string& fileName, IRuntime* runtime) 
{
    std::ifstream engineFile(fileName, std::ios::binary);
    if (!engineFile) {
        std::cout << "Failed to open engine file: " << fileName << std::endl;
        return nullptr;
    }

    engineFile.seekg(0, std::ios::end);
    size_t fileSize = engineFile.tellg();
    engineFile.seekg(0, std::ios::beg);

    std::vector<char> engineData(fileSize);
    engineFile.read(engineData.data(), fileSize);
    engineFile.close();

    ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), fileSize, nullptr);
    if (!engine) {
        std::cerr << "Failed to deserialize engine" << std::endl;
    }
    return engine;
}

void InferBase::saveEngineToFile(ICudaEngine* engine, const std::string& fileName) 
{
    IHostMemory* serializedEngine = engine->serialize(); // 序列化引擎
    std::ofstream engineFile(fileName, std::ios::binary);
    if (!engineFile) {
        std::cerr << "Failed to open file for writing: " << fileName << std::endl;
        return;
    }
    engineFile.write(reinterpret_cast<const char*>(serializedEngine->data()), serializedEngine->size());
    serializedEngine->destroy();
    engineFile.close();
    std::cout << "Engine saved to " << fileName << std::endl;
}

ICudaEngine* InferBase::loadModelAndCreateEngine(const std::string& modelFile, int maxBatchSize, IBuilder* builder, IBuilderConfig* config) {
    INetworkDefinition* network = builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
    auto parser = nvonnxparser::createParser(*network, *builder->getLogger());
    
    if (!parser->parseFromFile(modelFile.c_str(), static_cast<int>(ILogger::Severity::kINFO))) {
        std::cerr << "ERROR: could not parse the model." << std::endl;
        return nullptr;
    }

    builder->setMaxBatchSize(maxBatchSize);
    config->setFlag(BuilderFlag::kTF32);
    return builder->buildEngineWithConfig(*network, *config);
}

float YoloDetectInfer::iou(const Bbox& b1, const Bbox& b2) {
    int x1 = std::max(b1.x1, b2.x1);
    int y1 = std::max(b1.y1, b2.y1);
    int x2 = std::min(b1.x2, b2.x2);
    int y2 = std::min(b1.y2, b2.y2);

    int intersection_width = std::max(0, x2 - x1);
    int intersection_height = std::max(0, y2 - y1);
    float intersection = intersection_width * intersection_height;

    float box1_area = (b1.x2 - b1.x1) * (b1.y2 - b1.y1);
    float box2_area = (b2.x2 - b2.x1) * (b2.y2 - b2.y1);
    float union_area = box1_area + box2_area - intersection;
    return intersection / union_area;
}

void YoloDetectInfer::show(cv::Mat& image, const std::vector<Bbox>& boxes) {
    std::cout << "show results size: "<< boxes.size() << std::endl;
    for (const auto& box : boxes) {
        cv::Scalar color(0, 255, 0);
        cv::rectangle(image, cv::Point(int(box.x1), int(box.y1)), cv::Point(int(box.x2), int(box.y2)), color, 2);
        std::string label = "Conf: " + std::to_string(box.conf);
        cv::putText(image, label, cv::Point(box.x1, box.y1 - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
    }
    cv::imshow("results", image);
    cv::waitKey(0);
}

void deploy_yolo() {
    YoloDetectInfer model;
    model.init("./model/yolov5s.onnx");
    model.infer("./images/zidane.jpg");
}

void YoloDetectInfer::init(const std::string& model_path, int height, int width, int anchor_size, int class_num)
{
    this->height = height;
    this->width = width;
    this->anchor_size = anchor_size;
    this->class_num = class_num;
    builder = createInferBuilder(logger);
    config = builder->createBuilderConfig();
    runtime = nvinfer1::createInferRuntime(logger);
    if (runtime == nullptr) {
        std::cerr << "Failed to create runtime." << std::endl;
        return;
    }
    engine = loadEngineFromFile("yolo.engine", runtime);
    if (engine == nullptr) {
        engine = loadModelAndCreateEngine(model_path, batch, builder, config);
        if (!engine) {
            std::cerr << "Failed to create engine." << std::endl;
            return;
        }
        saveEngineToFile(engine, "yolo.engine");
    }
    context = engine->createExecutionContext();
    if (context == nullptr) {
        std::cerr << "Failed to create context." << std::endl;
    }
}

void YoloDetectInfer::infer(const std::string& image_path, float threshold)
{
    int channels = 3;
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "Failed to open image" << std::endl;
        return;
    }
    cv::Mat input_image;
    cv::resize(img, img, cv::Size(width, height));
    cv::cvtColor(img, input_image, cv::COLOR_BGR2RGB);
    input_image.convertTo(input_image, CV_32FC3, 1.0 / 255.0);
    float* input = nullptr;
    cudaHostAlloc((void**)&input, channels * height * width * sizeof(float), cudaHostAllocDefault);

    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                input[c * height * width + h * width + w] = input_image.at<cv::Vec3f>(h, w)[c];
            }
        }
    }

    float* output = nullptr;
    cudaHostAlloc((void**)&output, anchor_size * class_num * sizeof(float), cudaHostAllocDefault);
    // 绑定模型输入输出层, Netron即可看到对应层的name
    const int inputIndex = engine->getBindingIndex("images"); 
    const int outputIndex = engine->getBindingIndex("output0");

    // 输入输出分别放0和1
    void* buffers[2];
    cudaMalloc(&buffers[inputIndex], channels * height * width * sizeof(float));
    cudaMalloc(&buffers[outputIndex], anchor_size * class_num * sizeof(float));
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 设置输入维度
    Dims inputDims;
    inputDims.nbDims = 4; 
    inputDims.d[0] = batch;
    inputDims.d[1] = channels;
    inputDims.d[2] = height;
    inputDims.d[3] = width;
    if (!context->setBindingDimensions(inputIndex, inputDims)) {
        std::cerr << "Failed to set input dimensions." << std::endl;
        return;
    }

    cudaMemcpyAsync(buffers[inputIndex], input, channels * height * width * sizeof(float), cudaMemcpyHostToDevice, stream);
    context->enqueueV2(buffers, stream, nullptr); // 异步 同步用executv2
    cudaMemcpyAsync(output, buffers[outputIndex], anchor_size * class_num * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    std::vector<Bbox> processed_results;
    processOutput(output, processed_results, threshold);
    show(img, processed_results);

    cudaStreamDestroy(stream);
    cudaFree(buffers[inputIndex]);
    cudaFree(buffers[outputIndex]);
    cudaFreeHost(output);
    cudaFreeHost(input);
}

YoloDetectInfer::~YoloDetectInfer()
{
    if (context != nullptr)
        context->destroy();
    if (engine != nullptr)
        engine->destroy();
    if (runtime != nullptr)
        runtime->destroy();
    if (config != nullptr)
        config->destroy();
    if (builder != nullptr)
        builder->destroy();
}

void YoloDetectInfer::processOutput(float* output, std::vector<Bbox>& processed_results, float threshold)
{
    if (output == nullptr)
        return;

    std::vector<Bbox> results;
    float* ptr = output;
    for (uint32_t i = 0; i < anchor_size; ++i) {
        Bbox box;
        // 边界框坐标（归一化）
        float x_center = output[i * class_num + 0];
        float y_center = output[i * class_num + 1];
        float width = output[i * class_num + 2];
        float height = output[i * class_num + 3];
        box.x1 = x_center - (width / 2.0f);
        box.x2 = x_center + (width / 2.0f);
        box.y1 = y_center - (height / 2.0f);
        box.y2 = y_center + (height / 2.0f);

        box.conf = output[i * class_num + 4];
        box.type = std::distance(ptr + 5, std::max_element(ptr + 5, ptr + class_num));
        ptr += class_num;
        if (box.conf < threshold) {
            continue;
        }
        results.push_back(box);
    }
    std::sort(results.begin(), results.end(), [](const Bbox& l, const Bbox& r) { return l.conf > r.conf; });
    // 记录哪些锚框被抑制
    std::vector<bool> suppressed(results.size(), false);
    for (size_t i = 0; i < results.size(); ++i) {
        if (suppressed[i])
            continue;

        const Bbox& box_i = results[i];
        processed_results.push_back(box_i);
        for (size_t j = i + 1; j < results.size(); ++j) {
            if (suppressed[j])
                continue;
            const Bbox& box_j = results[j];
            if (box_i.type == box_j.type && iou(box_i, box_j) > threshold) {
                // iou大于阈值的被抑制
                suppressed[j] = true;
            }
        }
    }
}
