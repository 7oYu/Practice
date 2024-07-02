#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

void Demo() {
	Mat src = imread("../test.jpg");
	if (src.empty()) {
		printf("imread fail \n");
		return;
	}
	std::vector<cv::Mat> split_ret;
	cv::split(src, split_ret);
	cv::imshow("b", split_ret[0]);
	cv::imshow("g", split_ret[1]);
	cv::imshow("r", split_ret[2]);
	split_ret[0] = 0;
	split_ret[1] = 0;
	cv::Mat merge_ret;
	cv::merge(split_ret, merge_ret);
	cv::imshow("merge_ret", merge_ret);
} 

int main(int argc, char** argv) {
    Demo();
	waitKey(0);
	destroyAllWindows();
	return 0;
}
