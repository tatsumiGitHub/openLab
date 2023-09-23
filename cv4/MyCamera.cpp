#include "MyCamera.hpp"

int app::MyCamera::run(void) {
    /// 仮想カメラの機能を実行する ///

    /// 各種ファイルの読み込み ///

    /// 処理開始 ///

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        return -1;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

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
    mycamera.run();

    return;
}