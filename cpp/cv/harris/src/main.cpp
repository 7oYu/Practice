#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

void Demo() {
	Mat src = imread("../test.jpg");
	Mat gray_src;
	cv::cvtColor(src, gray_src, cv::COLOR_BGR2GRAY);
	if (gray_src.empty()) {
		printf("imread fail \n");
		return;
	}
	Mat output = Mat::zeros(gray_src.size(), CV_32FC1);
	cv::cornerHarris(gray_src, output, 3, 3, 0.04);
	Mat normalized_output;
	cv::normalize(output, normalized_output, 0, 255, NORM_MINMAX, CV_32FC1);
	Mat abs_normalized_output;
	// convertScaleAbs output mat type is uint8
	cv::convertScaleAbs(normalized_output, abs_normalized_output);
	    for (int i = 0; i < abs_normalized_output.rows; i++) {
        for (int j = 0; j < abs_normalized_output.cols; j++) {
            if ((int)abs_normalized_output.at<uchar>(i, j) > 150) {
                circle(src, Point(j, i), 5, Scalar(0, 0, 255), 2, 8, 0);
            }
        }
    }
	imshow("harris ret", src);
} 

int main(int argc, char** argv) {
    Demo();
	waitKey(0);
	destroyAllWindows();
	return 0;
}
