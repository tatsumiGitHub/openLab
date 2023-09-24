#include <string>
#include <iostream>
#include <cuda.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

void gamma_wrapper(unsigned char *_frame_data, int _size, double _gamma);
__global__ void gamma_wrapper_cuda(unsigned char *_frame_data, int _size, double _gamma);
int run(int _width, int _height, int _fps);