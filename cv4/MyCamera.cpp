#include "MyCamera.hpp"

void app::MyCamera::setWidth(int _width) {
    width = _width;
}
void app::MyCamera::setHeight(int _height) {
    height = _height;
}
void app::MyCamera::setSize(int _width, int _height) {
    width = _width;
    height = _height;
}

int app::MyCamera::run(void) {
    /// 仮想カメラの機能を実行する ///

    /// 各種ファイルの読み込み ///

    /// 処理開始 ///

    cv::VideoCapture cap(0, cv::CAP_V4L2);
    if (!cap.isOpened()) {
        return -1;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);

    cv::Mat frame;
    cap >> frame;
    int rows = frame.rows, cols = frame.cols;

    /// 画像ファイルのリサイズ ///


    /// 各クラスの読み込み ///

    std::cout << "起動します" << std::endl;
    prepare = true;
    while (status != 0) {
        cap >> frame;

        /// 出力 ///
        switch (status) {
            break;
        }
        cv::imshow("MyCamera", frame);
        
        if (cv::waitKey(interval) == 0x1b) {
            status = 0;
            break;
        }
    }
    cv::destroyAllWindows();
    std::cout << "終了しました" << std::endl;
}

void app::run(void) {
    MyCamera mycamera(24);
    mycamera.setSize(960, 720);
    mycamera.run();

    return;
}