#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void Modify_pixel()
{
	cv::Mat image = cv::Mat::ones(cv::Size(256, 256), CV_8UC3);
	if (image.channels() != 3) {
		std::cout << "error channels"; 
		return;
	}
	static int color = 0;
	for (int row = 0; row < image.rows; ++row)
	{
		uchar* ptr = image.ptr<uchar>(row);
		if (ptr == nullptr) {
			std::cout << "nullptr"; 
			return;
		}
		for (int col = 0; col < image.cols; ++col)
		{
			*ptr = static_cast<uchar>(color);
			++ptr;
			*ptr = static_cast<uchar>(color);
			++ptr;
			*ptr = static_cast<uchar>(256- color);
			++ptr;
			color = (++color)%256;
		}
	}
	imshow("modify", image);
}

int main(int argc, char **argv)
{
	Modify_pixel();
	waitKey(0);
	destroyAllWindows();
	return 0;
}
