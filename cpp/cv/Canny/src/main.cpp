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
	cv::Mat canny_ret;
	cv::Canny(src, canny_ret, 128, 256, 3);
	cv::imshow("canny_ret", canny_ret);
} 

int main(int argc, char** argv) {
    Demo();
	waitKey(0);
	destroyAllWindows();
	return 0;
}
