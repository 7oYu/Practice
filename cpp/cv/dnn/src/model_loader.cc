#include "model_loader.h"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <yaml-cpp/yaml.h>


ModelLoader::ModelLoader(const std::string& model_path)
{
    read_model(model_path);
}

void ModelLoader::read_model(const std::string& model_path)
{
    model_ = cv::dnn::readNetFromONNX(model_path);
    if(model_.empty()) {
        std::cerr << "read model fail" << std::endl;             
        return;
    }
    std::cout << "read model success" << std::endl;        
    model_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    model_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
}

void ModelLoader::predict(const std::string& image_path)
{
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if(image.empty()) {
        std::cerr << "read image fail" << std::endl;             
        return;
    }
    cv::Mat blob = cv::dnn::blobFromImage(square_image(image), 1.0/255.0, cv::Size(512,512), cv::Scalar(), true);
    std::cout << "blob shape: { ";  
    for (int i = 0; i < blob.size.dims(); ++i) {
        std::cout << blob.size[i] << " ";
    }
    std::cout << "}" << std::endl;
    model_.setInput(blob);
    std::vector<cv::Mat> outputs;
    model_.forward(outputs);
    std::cout << "outputs size : " << outputs.size() << std::endl;
    cv::Mat reshape_mat = outputs[0].reshape(1, outputs[0].size[1]);
    std::cout << "reshape_mat shape: { ";  
    for (int i = 0; i < reshape_mat.size.dims(); ++i) {
        std::cout << reshape_mat.size[i] << " ";
    }
    std::cout << "}" << std::endl; 
    for (int ret = 0; ret < reshape_mat.size[1]; ++ret) {
        std::cout << "reshape_mat: { ";  
        for (int ret1 = 0; ret1 < reshape_mat.size[0]; ++ret1) {
            std::cout << reshape_mat.at<float>(ret1, ret) << " ";
            // std::cout << reshape_mat.ptr<float>(ret1, ret) << " ";
        }
        std::cout << "}" << std::endl;
        break;
    }

    cv::Mat t_mat = reshape_mat.t();
    std::cout << "t_mat shape: { ";  
    for (int i = 0; i < t_mat.size.dims(); ++i) {
        std::cout << t_mat.size[i] << " ";
    }
    std::cout << "}" << std::endl; 

    std::vector<cv::Rect> predict_boxes;

    float* data = (float*)(t_mat.data);
    for (int ret = 0; ret < t_mat.size[0]; ++ret) {
        cv::Mat output_class(1, t_mat.size[1] - 4, CV_32FC1, data + 4);
        double max_val = 0;
        cv::Point max_index;
        cv::minMaxLoc(output_class, 0, &max_val, 0, &max_index);
        data += t_mat.size[1];
        if (max_val > 0.7) {
            if (max_index.x > labels_.size()) {
                std::cout << "error index" << std::endl;  
                continue;
            }
            std::cout << "type : " << labels_[max_index.x] << std::endl;
            std::cout << "max_val: " << max_val << std::endl;  
            std::cout << "max_index: " << max_index.x << " " << max_index.y << std::endl;  
            int max_size = std::max({image.cols, image.rows});
            double scale = max_size / 512;
            predict_boxes.push_back(cv::Rect(int((data[0] - 0.5 * data[2])*scale), int((data[1] - 0.5 * data[3])*scale), 
                                                int(data[2]*scale), int(data[3]*scale)));
        }
    }
    for (const auto& rect : predict_boxes) {
        cv::rectangle(image, rect, cv::Scalar(0, 255, 0));
    }
    cv::imshow("ret", image);
    cv::waitKey(0);
}

cv::Mat ModelLoader::square_image(const cv::Mat &image)
{
    int max_size = std::max({image.cols, image.rows});
    cv::Mat ret = cv::Mat::zeros(max_size, max_size, CV_8UC3);
    image.copyTo(ret(cv::Rect(0, 0, image.cols, image.rows)));
    return ret;
}

void ModelLoader::load_labels(const std::string &model_path)
{
    YAML::Node node = YAML::LoadFile(model_path);
    if (node.IsNull()) {
        std::cerr << "load yaml file fail" << std::endl;
        abort();
        return;
    }
    int i = 0;
    for (auto iter = node["names"].begin(); iter != node["names"].end(); ++iter) {
        labels_.insert(std::make_pair(i, iter->second.as<std::string>()));
        ++i;
    }
}
