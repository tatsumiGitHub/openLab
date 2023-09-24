#include "MyCamera.h"

void gamma_wrapper(unsigned char *_frame_data, int _size, double _gamma) {
    if (_frame_data == NULL) {
        return;
    }
    for (int i = 0; i < _size; i++) {
        _frame_data[i] = 255 * pow(_frame_data[i] / 255.0, 1 / _gamma);
    }
    return;
}

void gamma_wrapper_openMP(unsigned char *_frame_data, int _size, double _gamma) {
    if (_frame_data == NULL) {
        return;
    }
    #pragma omp parallel for
    for (int i = 0; i < _size; i++) {
        _frame_data[i] = 255 * pow(_frame_data[i] / 255.0, 1 / _gamma);
    }
    return;
}

__global__ void gamma_wrapper_cuda(unsigned char *_frame_data, int _size, double _gamma) {
    if (_frame_data == NULL) {
        return;
    }
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < _size) {
        _frame_data[tid] = 255 * pow(_frame_data[tid] / 255.0, 1 / _gamma);
    }
    return;
}

int run(int _width, int _height, int _fps) {
    /// 仮想カメラの機能を実行する ///

    /// 各種ファイルの読み込み ///

    /// 処理開始 ///

    cv::VideoCapture cap(0, cv::CAP_V4L2);
    if (!cap.isOpened()) {
        return -1;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, _width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, _height);
    int size = _width * _height * 3;
    int interval = 1000 / _fps;
    int blocks = size / 1024, threads = 1024;

    cv::Mat h_frame;

    /// デバイスメモリの確保 ///
    unsigned char *d_frame;
    cudaMalloc( (void**) &d_frame, size);

    /// 各クラスの読み込み ///

    std::cout << "起動します" << std::endl;
    int key_input, status = 0;
    while (status != -1) {
        cap >> h_frame;

        key_input = cv::waitKey(interval);
        switch (key_input) {
        case 0x30:
            status = 0;
            break;
        case 0x1b:
        case 0x31:
        case 0x32:
        case 0x33:
            status = key_input;
            break;
        }

        /// 出力 ///
        switch (status) {
        case 0x31:
            gamma_wrapper(h_frame.data, size, 2);
            break;
        case 0x32:
            gamma_wrapper_openMP(h_frame.data, size, 2);
            break;
        case 0x33:
            cudaMemcpy(d_frame, h_frame.data, size, cudaMemcpyHostToDevice);
		    gamma_wrapper_cuda<<< blocks, threads >>>(d_frame, size, 2);
            cudaMemcpy(h_frame.data, d_frame, size, cudaMemcpyDeviceToHost);
            break;
        }
        cv::imshow("MyCamera", h_frame);
        
        if (status == 0x1b) {
            break;
        }
    }
    cv::destroyAllWindows();
    cudaFree(d_frame);
    std::cout << "終了しました" << std::endl;

    return 1;
}