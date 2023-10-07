#include "MyCamera.h"

void gamma(unsigned char *_frame_data, int _size, double _gamma)
{
    if (_frame_data == NULL)
    {
        return;
    }
    for (int i = 0; i < _size; i++)
    {
        _frame_data[i] = 255 * pow(_frame_data[i] / 255.0, 1 / _gamma);
    }
    return;
}

void gamma_openMP(unsigned char *_frame_data, int _size, double _gamma)
{
    if (_frame_data == NULL)
    {
        return;
    }
#pragma omp parallel for
    for (int i = 0; i < _size; i++)
    {
        _frame_data[i] = 255 * pow(_frame_data[i] / 255.0, 1 / _gamma);
    }
    return;
}

__global__ void gamma_cuda(unsigned char *_frame_data, int _size, double _gamma)
{
    if (_frame_data == NULL)
    {
        return;
    }
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < _size)
    {
        _frame_data[tid] = 255 * pow(_frame_data[tid] / 255.0, 1 / _gamma);
    }
    return;
}

void pca(unsigned char *_frame_data, int _size)
{
    if (_frame_data == NULL)
    {
        return;
    }
    int i;
    double r_ave = 0.0, g_ave = 0.0, b_ave = 0.0;
    for (i = 0; i < _size; i += 3)
    {
        r_ave += _frame_data[i] * 3.0 / _size;
        g_ave += _frame_data[i + 1] * 3.0 / _size;
        b_ave += _frame_data[i + 2] * 3.0 / _size;
    }
    double S_rr = 0.0, S_rg = 0.0, S_rb = 0.0, S_gg = 0.0, S_gb = 0.0, S_bb = 0.0;
    double r_tmp, g_tmp, b_tmp;
    for (i = 0; i < _size; i += 3)
    {
        r_tmp = (_frame_data[i] - r_ave);
        g_tmp = (_frame_data[i + 1] - g_ave);
        b_tmp = (_frame_data[i + 2] - b_ave);
        S_rr += r_tmp * r_tmp * 3 / _size;
        S_rg += r_tmp * g_tmp * 3 / _size;
        S_rb += r_tmp * b_tmp * 3 / _size;
        S_gg += g_tmp * g_tmp * 3 / _size;
        S_gb += g_tmp * b_tmp * 3 / _size;
        S_bb += b_tmp * b_tmp * 3 / _size;
    }
    mat_t mat = {0, 0, NULL};
    m_init(&mat, 3, 3);
    mat.elements[0] = S_rr;
    mat.elements[1] = S_rg;
    mat.elements[2] = S_rb;
    mat.elements[3] = S_rg;
    mat.elements[4] = S_gg;
    mat.elements[5] = S_gb;
    mat.elements[6] = S_rb;
    mat.elements[7] = S_gb;
    mat.elements[8] = S_bb;
    printf("//======== mat ========//\n");
    m_show(&mat);
    m_eig(&mat);
    m_free(&mat);
    return;
}

int run(int _width, int _height, int _fps)
{
    /// 仮想カメラの機能を実行する ///

    /// 各種ファイルの読み込み ///

    /// 処理開始 ///

    cv::VideoCapture cap(0, cv::CAP_V4L2);
    if (!cap.isOpened())
    {
        return -1;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, _width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, _height);
    int size = _width * _height * 3;
    int interval = 1000 / _fps;
    int blocks = size / 1024, threads = 1024;
    long start, end;
    double fps;

    cv::Mat h_frame;

    /// デバイスメモリの確保 ///
    unsigned char *d_frame;
    cudaMalloc((void **)&d_frame, size);

    /// 各クラスの読み込み ///

    std::cout << "起動します" << std::endl;
    int key_input, status = 0;
    while (status != -1)
    {
        cap >> h_frame;

        key_input = cv::waitKey(interval);
        switch (key_input)
        {
        case 0x30:
            status = 0;
            break;
        case 0x1b:
        case 0x31:
        case 0x32:
        case 0x33:
            status = key_input;
            break;
        case 0x34:
            status = key_input;
            break;
        }

        /// 出力 ///
        switch (status)
        {
        case 0x31:
            gamma(h_frame.data, size, 2);
            break;
        case 0x32:
            gamma_openMP(h_frame.data, size, 2);
            break;
        case 0x33:
            cudaMemcpy(d_frame, h_frame.data, size, cudaMemcpyHostToDevice);
            gamma_cuda<<<blocks, threads>>>(d_frame, size, 2);
            cudaMemcpy(h_frame.data, d_frame, size, cudaMemcpyDeviceToHost);
            break;
        case 0x34:
            pca(h_frame.data, size);
            break;
        }
        end = cv::getTickCount();
        fps = 1000 / ((end - start) * 1000 / cv::getTickFrequency());
        start = cv::getTickCount();
        char text[32];
        snprintf(text, 32, "%.3f fps", fps);
        cv::putText(
            h_frame,
            text,
            cv::Point(25, 75),
            cv::FONT_HERSHEY_SIMPLEX,
            2.5,
            cv::Scalar(0, 255, 0),
            3);
        cv::imshow("MyCamera", h_frame);

        if (status == 0x1b)
        {
            break;
        }
    }
    cv::destroyAllWindows();
    cudaFree(d_frame);
    std::cout << "終了しました" << std::endl;

    return 1;
}