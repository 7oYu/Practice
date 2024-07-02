#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void colorSpace_Demo(Mat &image) {
	Mat hsv,gray;
	cvtColor(image, hsv, COLOR_BGR2HSV);
	cvtColor(image, gray, COLOR_BGR2GRAY);
	imshow("1", hsv);
	imshow("2", gray);
	imwrite("hsv.jpg", hsv);
	imwrite("gray.jpg", gray);
} 

int main(int argc, char** argv) {
	Mat src = imread("../test.jpg");
	if (src.empty()) {
		printf("....\n");
		return -1;
	}

	imshow("输入窗口", src);

    colorSpace_Demo(src);

	waitKey(0);
	destroyAllWindows();
	return 0;
}
