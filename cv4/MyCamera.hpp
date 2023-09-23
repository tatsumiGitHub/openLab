#include <string>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

namespace app {
    class MyCamera {
    private:
        bool prepare = false;
        int interval;
        int status;
        int width = 640;
        int height = 480;
    public:
        MyCamera(int _fps) {
            interval = 1000 / _fps;
            status = 1;
        };
        void setWidth(int _width);
        void setHeight(int _height);
        void setSize(int _width, int _height);
        int run(void);
    };
    void run(void);
}