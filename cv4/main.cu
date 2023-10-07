#include "MyCamera.h"

void test(void);

int main(int argc, char *argv[])
{
    if (1 < argc)
    {
        int n = atoi(argv[1]);
        switch (n)
        {
        case 1:
            run(960, 720, 24);
            break;
        case 2:
            test();
            break;
        default:
            printf("undefind argument: %s\n", argv[1]);
            break;
        }
    }
    else
    {
        printf("no argument: >$ ./main <test| 1 / 2>\n");
    }
    return 0;
}

void test(void)
{
    cv::Mat image = cv::imread("img/lena.jpg");

    int size = image.rows * image.cols * 3;
    pca(image.data, size);

    pca_kernel_wrapper(image.data, size);
}