#include <string>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

int main(int argc, char* argv[]) {
    const std::string window_name("OpenCV Sample");

    cv::VideoCapture cap(0, cv::CAP_V4L2);

    if ( !cap.isOpened() ) {
        std::cerr << "Can't open capture device" << std::endl;
        return -1;
    }
   
    cv::Mat frame;
    while (1) {
        cap >> frame;
        if ( frame.empty() ) {
            std::cerr << "Fail to capture video" << std::endl;
            break;
        }
        cv::imshow(window_name, frame);
        if ( cv::waitKey(33) >= 0 ) {
            break;
        }
    }

    return 0;
}