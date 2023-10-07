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
__global__ void gamma_kernel(unsigned char *_d_frame_data, int _size, double _gamma);
void gamma_kernel_wrapper(unsigned char *_h_frame_data, int _size, double _gamma);
void pca(unsigned char *_frame_data, int _size);
__global__ void pca_kernel1(double *_d_pca, unsigned char *_d_frame_data, int _size);
__global__ void pca_kernel2(double *_d_pca, int _n, int _odd);
__global__ void pca_kernel3(double *_d_pca, unsigned char *_d_frame_data, double *_d_ave, int _n);
__global__ void pca_kernel4(double *_d_pca, unsigned char *_d_frame_data, double *_d_ave, int _n);
void pca_kernel_wrapper(unsigned char *_h_frame_data, int _size);
int run(int _width, int _height, int _fps);