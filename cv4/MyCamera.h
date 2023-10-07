#include <string>
#include <iostream>
#include <math.h>
#include <omp.h>
#include <cuda.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "matrix.h"

void gamma_wrapper(unsigned char *_frame_data, int _size, double _gamma);
void gamma_wrapper_openMP(unsigned char *_frame_data, int _size, double _gamma);
__global__ void gamma_wrapper_cuda(unsigned char *_frame_data, int _size, double _gamma);
int run(int _width, int _height, int _fps);