#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void Scalar_Demo() {
	cv::Mat image1 = cv::Mat::ones(cv::Size(128, 128), CV_8UC3);
	image1 = cv::Scalar(1, 10, 100);
	std::cout << image1.cols<<"  " << image1.rows<<"  "<<image1.channels() << std::endl;
	Mat image2;
	image1.copyTo(image2);
	std::cout << image2.cols<<"  " << image2.rows<<"  "<<image2.channels() << std::endl;
	image2 = Scalar(0, 255, 255);
	imshow("image1", image1);
	imshow("image2", image2);
} 

int main(int argc, char** argv) {
    Scalar_Demo();
	waitKey(0);
	destroyAllWindows();
	return 0;
}
