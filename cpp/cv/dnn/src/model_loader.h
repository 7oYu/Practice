#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <map>

// reference to yolo example https://github.com/ultralytics/ultralytics/blob/main/examples
class ModelLoader
{
public:
    ModelLoader(const std::string& model_path);
    void read_model(const std::string& model_path);
    void predict(const std::string& image_path);
    static cv::Mat square_image(const cv::Mat& image);
    void load_labels(const std::string& model_path);
    cv::dnn::Net model_;
    std::map<int, std::string> labels_;
};
