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

__global__ void gamma_kernel(unsigned char *_d_frame_data, int _size, double _gamma)
{
    if (_d_frame_data == NULL)
    {
        return;
    }
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < _size)
    {
        _d_frame_data[tid] = 255 * pow(_d_frame_data[tid] / 255.0, 1 / _gamma);
    }
    return;
}

void gamma_kernel_wrapper(unsigned char *_h_frame_data, int _size, double _gamma)
{
    int blocks = _size / 1024, threads = 1024;
    /// デバイスメモリの確保 ///
    unsigned char *d_frame_data;
    cudaMalloc((void **)&d_frame_data, _size);

    cudaMemcpy(d_frame_data, _h_frame_data, _size, cudaMemcpyHostToDevice);
    gamma_kernel<<<blocks, threads>>>(d_frame_data, _size, 2);
    cudaMemcpy(_h_frame_data, d_frame_data, _size, cudaMemcpyDeviceToHost);

    cudaFree(d_frame_data);
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
    printf("r_ave: %.3f, g_ave: %.3f, b_ave: %.3f\n", r_ave, g_ave, b_ave);
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

__global__ void pca_kernel1(double *_d_pca, unsigned char *_d_frame_data, int _size)
{
    if (_d_pca == NULL || _d_frame_data == NULL)
    {
        return;
    }
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < _size)
    {
        _d_pca[tid] = _d_frame_data[tid] * 3.0 / _size;
    }
    return;
}

__global__ void pca_kernel2(double *_d_pca, int _n, int _odd)
{
    if (_d_pca == NULL)
    {
        return;
    }
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < _n)
    {
        _d_pca[tid * 3] += _d_pca[(tid + _n) * 3];
        _d_pca[tid * 3 + 1] += _d_pca[(tid + _n) * 3 + 1];
        _d_pca[tid * 3 + 2] += _d_pca[(tid + _n) * 3 + 2];
        if (tid == 0 && _odd == 1)
        {
            _d_pca[0] += _d_pca[_n * 6];
            _d_pca[1] += _d_pca[_n * 6 + 1];
            _d_pca[2] += _d_pca[_n * 6 + 2];
        }
    }
    return;
}

__global__ void pca_kernel3(double *_d_pca, unsigned char *_d_frame_data, double *_d_ave, int _n)
{
    if (_d_pca == NULL)
    {
        return;
    }
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < _n)
    {
        _d_pca[tid * 3] = (_d_frame_data[tid * 3] - _d_ave[0]) * (_d_frame_data[tid * 3] - _d_ave[0]) / _n;             // rr
        _d_pca[tid * 3 + 1] = (_d_frame_data[tid * 3 + 1] - _d_ave[1]) * (_d_frame_data[tid * 3 + 1] - _d_ave[1]) / _n; // gg
        _d_pca[tid * 3 + 2] = (_d_frame_data[tid * 3 + 2] - _d_ave[2]) * (_d_frame_data[tid * 3 + 2] - _d_ave[2]) / _n; // bb
    }
    return;
}

__global__ void pca_kernel4(double *_d_pca, unsigned char *_d_frame_data, double *_d_ave, int _n)
{
    if (_d_pca == NULL)
    {
        return;
    }
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < _n)
    {
        _d_pca[tid * 3] = (_d_frame_data[tid * 3] - _d_ave[0]) * (_d_frame_data[tid * 3 + 1] - _d_ave[1]) / _n;         // rg
        _d_pca[tid * 3 + 1] = (_d_frame_data[tid * 3] - _d_ave[0]) * (_d_frame_data[tid * 3 + 2] - _d_ave[2]) / _n;     // rb
        _d_pca[tid * 3 + 2] = (_d_frame_data[tid * 3 + 1] - _d_ave[1]) * (_d_frame_data[tid * 3 + 2] - _d_ave[2]) / _n; // gb
    }
    return;
}

void pca_kernel_wrapper(unsigned char *_h_frame_data, int _size)
{
    int tmp, n;
    int blocks = _size / 1024, threads = 1024;
    unsigned char *d_frame_data;
    cudaMalloc((void **)&d_frame_data, _size);

    double *h_ave, *h_pca, *d_pca[2], *d_ave;
    if ((h_ave = (double *)malloc(sizeof(double) * 3)) == NULL || (h_pca = (double *)malloc(sizeof(double) * 6)) == NULL)
    {
        return;
    }
    cudaMalloc((void **)&d_pca[0], sizeof(double) * _size);
    cudaMalloc((void **)&d_pca[1], sizeof(double) * _size);
    cudaMalloc((void **)&d_ave, sizeof(double) * 3);

    cudaMemcpy(d_frame_data, _h_frame_data, _size, cudaMemcpyHostToDevice);
    pca_kernel1<<<blocks, threads>>>(d_pca[0], d_frame_data, _size);
    tmp = _size % 2;
    n = _size / 6;
    while (0 < n)
    {
        pca_kernel2<<<blocks, threads>>>(d_pca[0], n, tmp);
        tmp = n % 2;
        n = n >> 1;
    }
    n = _size / 3;
    cudaMemcpy(d_ave, d_pca[0], sizeof(double) * 3, cudaMemcpyDeviceToDevice);
    pca_kernel3<<<blocks, threads>>>(d_pca[0], d_frame_data, d_ave, n);
    pca_kernel4<<<blocks, threads>>>(d_pca[1], d_frame_data, d_ave, n);
    tmp = _size % 2;
    n = _size / 6;
    while (0 < n)
    {
        pca_kernel2<<<blocks, threads>>>(d_pca[0], n, tmp);
        pca_kernel2<<<blocks, threads>>>(d_pca[1], n, tmp);
        tmp = n % 2;
        n = n >> 1;
    }
    cudaMemcpy(h_ave, d_ave, sizeof(double) * 3, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pca, d_pca[0], sizeof(double) * 3, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pca + 3, d_pca[1], sizeof(double) * 3, cudaMemcpyDeviceToHost);
    mat_t mat = {0, 0, NULL};
    m_init(&mat, 3, 3);
    mat.elements[0] = h_pca[0];
    mat.elements[1] = h_pca[3];
    mat.elements[2] = h_pca[4];
    mat.elements[3] = h_pca[3];
    mat.elements[4] = h_pca[1];
    mat.elements[5] = h_pca[5];
    mat.elements[6] = h_pca[4];
    mat.elements[7] = h_pca[5];
    mat.elements[8] = h_pca[2];
    printf("r_ave: %.3f, g_ave: %.3f, b_ave: %.3f\n", h_ave[0], h_ave[1], h_ave[2]);
    printf("//======== mat ========//\n");
    m_show(&mat);
    m_eig(&mat);
    m_free(&mat);

    free(h_ave);
    cudaFree(d_frame_data);
    cudaFree(d_pca[0]);
    cudaFree(d_pca[1]);
    cudaFree(d_ave);
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
    long start, end;
    double fps;

    cv::Mat frame;

    /// 各クラスの読み込み ///

    std::cout << "起動します" << std::endl;
    int key_input, status = 0;
    while (status != -1)
    {
        cap >> frame;

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
        case 0x35:
            status = key_input;
            break;
        }

        /// 出力 ///
        switch (status)
        {
        case 0x31:
            gamma(frame.data, size, 2);
            break;
        case 0x32:
            gamma_openMP(frame.data, size, 2);
            break;
        case 0x33:
            gamma_kernel_wrapper(frame.data, size, 2);
            break;
        case 0x34:
            pca(frame.data, size);
            break;
        case 0x35:
            pca_kernel_wrapper(frame.data, size);
            break;
        }
        end = cv::getTickCount();
        fps = 1000 / ((end - start) * 1000 / cv::getTickFrequency());
        start = cv::getTickCount();
        char text[32];
        snprintf(text, 32, "%.3f fps", fps);
        cv::putText(
            frame,
            text,
            cv::Point(25, 75),
            cv::FONT_HERSHEY_SIMPLEX,
            2.5,
            cv::Scalar(0, 255, 0),
            3);
        cv::imshow("MyCamera", frame);

        if (status == 0x1b)
        {
            break;
        }
    }
    cv::destroyAllWindows();
    std::cout << "終了しました" << std::endl;

    return 1;
}