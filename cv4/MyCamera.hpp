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
    public:
        MyCamera(int _fps) {
            interval = 1000 / _fps;
            status = 1;
        };
        int run(void);
    };
    void run(void);
}