#include <opencv2/opencv.hpp>
#include "model_loader.h"

void Demo() {
	ModelLoader model_loader("../model/last.onnx");
	model_loader.load_labels("../cfg/coco128.yaml");
	model_loader.predict("../image/zidane.jpg");
} 

int main(int argc, char** argv) {
    Demo();
	return 0;
}
