/******************************************************************************/

// Example : HOG person recognition of image / video / camera
// usage: hog_gpu

// e.g.  ./hog_detection --src-is-video true --src ./video.mp4
// --hit-threshold 1.0 --gr-threshold 3

/*
Usage: hog_gpu\n"
--src <path> # it's image file by default
[--src-is-video <true/false>] # says to interpretate src as video
[--src-is-camera <TRUE/false>] # says to interpretate src as camera (DEFAULT)
[--make-gray <true/false>] # convert image to gray one or not
[--resize-src <true/false>] # do resize of the source image or not
[--width <int>] # resized image width
[--height <int>] # resized image height
[--hit-threshold <double>] # classifying plane distance threshold (0.0 usually)
[--scale <double>] # HOG window scale factor
[--nlevels <int>] # max number of HOG window scales
[--win-width <int>] # width of the window (48 or 64)
[--win-stride-width <int>] # distance by OX axis between neighbour wins
[--win-stride-height <int>] # distance by OY axis between neighbour wins
[--gr-threshold <int>] # merging similar rects constant
[--gamma-correct <int>] # do gamma correction or not
[--write-video <bool>] # write video or not
[--dst-video <path>] # output video path
[--dst-video-fps <double>] # output video fps

*/

// by default uses camera (ID 0) for input

/******************************************************************************/

// Acknowledgement: entirely based on the OpenCV 2.3.1 provided example

// (very) Minor modifications : Toby Breckon, toby.breckon@cranfield.ac.uk
// Copyright (c) 2012 School of Engineering, Cranfield University
// License : LGPL - http://www.gnu.org/licenses/lgpl.html

// bug fixes: Machizaud, Antoine a.machizaud@cranfield.ac.uk, 2012

/******************************************************************************/

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include "opencv2/gpu/gpu.hpp"              // specifid opencv2 headers
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

/******************************************************************************/

// class def. to read/store command line arguments

class Args
{
public:
    Args();
    static Args read(int argc, char** argv);

    string src;
    bool src_is_video;
    bool src_is_camera;
    int camera_id;

    bool write_video;
    string dst_video;
    double dst_video_fps;

    bool make_gray;

    bool resize_src;
    int width, height;

    double scale;
    int nlevels;
    int gr_threshold;

    double hit_threshold;
    bool hit_threshold_auto;

    int win_width;
    int win_stride_width, win_stride_height;

    bool gamma_corr;
};

/******************************************************************************/

// class def. of main applictaion itself

class App
{
public:
    App(const Args& s);
    void run();

    void handleKey(char key);

    void hogWorkBegin();
    void hogWorkEnd();
    string hogWorkFps() const;

    void workBegin();
    void workEnd();
    string workFps() const;

    string message() const;

private:
    App operator=(App&);

    Args args;
    bool running;

    bool use_gpu;
    bool make_gray;
    double scale;
    int gr_threshold;
    int nlevels;
    double hit_threshold;
    bool gamma_corr;

    int64 hog_work_begin;
    double hog_work_fps;

    int64 work_begin;
    double work_fps;
};

/******************************************************************************/

// good old main()


int main(int argc, char** argv)
{
    try
    {
        cout << "Histogram of Oriented Gradients descriptor and detector sample.\n";
        cout << "\nUsage: hog_gpu\n"
                << "  --src <path> # it's image file by default\n"
                << "  [--src-is-video <true/false>] # says to interpretate src as video\n"
                << "  [--src-is-camera <TRUE/false>] # says to interpretate src as camera (DEFAULT)\n"
                << "  [--make-gray <true/false>] # convert image to gray one or not\n"
                << "  [--resize-src <true/false>] # do resize of the source image or not\n"
                << "  [--width <int>] # resized image width\n"
                << "  [--height <int>] # resized image height\n"
                << "  [--hit-threshold <double>] # classifying plane distance threshold (0.0 usually)\n"
                << "  [--scale <double>] # HOG window scale factor\n"
                << "  [--nlevels <int>] # max number of HOG window scales\n"
                << "  [--win-width <int>] # width of the window (48 or 64)\n"
                << "  [--win-stride-width <int>] # distance by OX axis between neighbour wins\n"
                << "  [--win-stride-height <int>] # distance by OY axis between neighbour wins\n"
                << "  [--gr-threshold <int>] # merging similar rects constant\n"
                << "  [--gamma-correct <int>] # do gamma correction or not\n"
                << "  [--write-video <bool>] # write video or not\n"
                << "  [--dst-video <path>] # output video path\n"
                << "  [--dst-video-fps <double>] # output video fps\n\n\n";

        App app(Args::read(argc, argv));
        app.run();
    }
    catch (const Exception& e) { return cout << "error: "  << e.what() << endl, 1; }
    catch (const exception& e) { return cout << "error: "  << e.what() << endl, 1; }
    catch(...) { return cout << "unknown exception" << endl, 1; }
    return 0;
}


/******************************************************************************/

// Args() class implementation


// defaults

Args::Args()
{
    src_is_video = false;
    src_is_camera = true;   // by default look for a camera
    camera_id = 1;

    write_video = false;    // no video output by default
    dst_video_fps = 25.;    // assume 25 fps

    make_gray = false;

    resize_src = false;
    width = 640;
    height = 480;

    // these can be adjusted with reference to the
    // HOG paper/notes + OpenCV manual

    scale = 1.05;
    nlevels = 64;
    gr_threshold = 8;
    hit_threshold = 0;
    hit_threshold_auto = true;

    win_width = 32;
    win_stride_width = 8;
    win_stride_height = 8;

    gamma_corr = true;     // do use gamma correction pre-processing
}

// read command line arguments (nice example of how to do this in C++)

Args Args::read(int argc, char** argv)
{
    Args args;
    for (int i = 1; i < argc - 1; i += 2)
    {
        string key = argv[i];
        string val = argv[i + 1];
        if (key == "--src") args.src = val;
        else if (key == "--src-is-video") args.src_is_video = (val == "true");
        else if (key == "--src-is-camera") args.src_is_camera = (val == "true");
        else if (key == "--camera-id") args.camera_id = atoi(val.c_str());
        else if (key == "--make-gray") args.make_gray = (val == "true");
        else if (key == "--resize-src") args.resize_src = (val == "true");
        else if (key == "--width") args.width = atoi(val.c_str());
        else if (key == "--height") args.height = atoi(val.c_str());
        else if (key == "--hit-threshold")
        {
            args.hit_threshold = atof(val.c_str());
            args.hit_threshold_auto = false;
        }
        else if (key == "--scale") args.scale = atof(val.c_str());
        else if (key == "--nlevels") args.nlevels = atoi(val.c_str());
        else if (key == "--win-width") args.win_width = atoi(val.c_str());
        else if (key == "--win-stride-width") args.win_stride_width = atoi(val.c_str());
        else if (key == "--win-stride-height") args.win_stride_height = atoi(val.c_str());
        else if (key == "--gr-threshold") args.gr_threshold = atoi(val.c_str());
        else if (key == "--gamma-correct") args.gamma_corr = (val == "true");
        else if (key == "--write-video") args.write_video = (val == "true");
        else if (key == "--dst-video") args.dst_video = val;
        else if (key == "--dst-video-fps") args.dst_video_fps= atof(val.c_str());
        else throw runtime_error((string("unknown key: ") + key));
    }

    // check we are still using a camera (i.e. not a video and supplied source
    // filename for an image/video is also empty)

    args.src_is_camera = (!(args.src_is_video) && (args.src.empty()));

    return args;
}

/******************************************************************************/

// App() class implementation

//constructor
App::App(const Args& s)
{

    // set up all the agruments and output controls etc.

    args = s;
    cout << "\nControls:\n"
         << "\tESC - exit\n"
         << "\tm - change mode GPU <-> CPU\n"
         << "\tg - convert image to gray or not\n"
         << "\t1/q - increase/decrease HOG scale\n"
         << "\t2/w - increase/decrease levels count\n"
         << "\t3/e - increase/decrease HOG group threshold\n"
         << "\t4/r - increase/decrease hit threshold\n"
         << endl;

    use_gpu = false;    // by default start with GPU
    make_gray = args.make_gray;
    scale = args.scale;
    gr_threshold = args.gr_threshold;
    nlevels = args.nlevels;

    if (args.hit_threshold_auto)
        args.hit_threshold = args.win_width == 48 ? 1.4 : 0.;
    hit_threshold = args.hit_threshold;

    gamma_corr = args.gamma_corr;

    if (args.win_width != 64 && args.win_width != 48)
        args.win_width = 64;

    cout << "Scale: " << scale << endl;
    if (args.resize_src)
        cout << "Resized source: (" << args.width << ", " << args.height << ")\n";
    cout << "Group threshold: " << gr_threshold << endl;
    cout << "Levels number: " << nlevels << endl;
    cout << "Win width: " << args.win_width << endl;
    cout << "Win stride: (" << args.win_stride_width << ", " << args.win_stride_height << ")\n";
    cout << "Hit threshold: " << hit_threshold << endl;
    cout << "Gamma correction: " << gamma_corr << endl;
    cout << endl;
}

//run the application
void App::run()
{
    running = true;
    cv::VideoWriter video_writer;

    Size win_size(args.win_width, args.win_width * 2); //(64, 128) or (48, 96)
    Size win_stride(args.win_stride_width, args.win_stride_height);

    // check for presence of suitable GPU

    bool GPUfound = (cv::gpu::getCudaEnabledDeviceCount() > 0);

    // Create HOG descriptors and detectors here (using people default)

    vector<float> detector;

    // create both GPU and CPU HOG detectors

    cv::gpu::HOGDescriptor *gpu_hog = NULL; // use a pointer here to avoid problems when a GPU is not available

    if (GPUfound)
    {
           cout << "\n\nSuitable GPU detected - using GPU HOG as DEFAULT\n\n";
           gpu_hog = new cv::gpu::HOGDescriptor(win_size, Size(16, 16), Size(8, 8), Size(8, 8), 9,
                                   cv::gpu::HOGDescriptor::DEFAULT_WIN_SIGMA, 0.2, gamma_corr,
                                   cv::gpu::HOGDescriptor::DEFAULT_NLEVELS);
           use_gpu = true;

           if (win_size == Size(64, 128))
           {
                //detector = cv::gpu::HOGDescriptor::getPeopleDetector64x128();
           } else {
                //detector = cv::gpu::HOGDescriptor::getPeopleDetector48x96();
            }
            //gpu_hog->setSVMDetector(detector);
    } else {
        cout << "\n\nNo Suitable GPU detected - using CPU HOG as DEFAULT\n\n";
        detector = cv::HOGDescriptor::getDefaultPeopleDetector();
        win_size = Size(64, 128); // fix window size to CPU default

    }

    // set non-GPU (i.e. CPU) HOG detector

    cv::HOGDescriptor cpu_hog(win_size, Size(16, 16), Size(8, 8), Size(8, 8), 9, 1, -1,
                              HOGDescriptor::L2Hys, 0.2, gamma_corr, cv::HOGDescriptor::DEFAULT_NLEVELS);
    cpu_hog.setSVMDetector(detector);

    // define window

    namedWindow("HOG Person Detection", CV_WINDOW_NORMAL);

    // the "big loop" over images from camera/video

    while (running)
    {
        VideoCapture vc;
        Mat frame;

        // get an image source

        if (args.src_is_video)
        {
            vc.open(args.src.c_str());
            if (!vc.isOpened())
                throw runtime_error(string("can't open video file: " + args.src));
            vc >> frame;
        }
        else if (args.src_is_camera)
        {
            vc.open(args.camera_id);
            if (!vc.isOpened())
                throw runtime_error(string("can't open camera ID: " + args.src));
            vc >> frame;
        }
        else
        {
            frame = imread(args.src);
            if (frame.empty())
                throw runtime_error(string("can't open image file: " + args.src));
        }

        Mat img_aux, img, img_to_show;
        gpu::GpuMat gpu_img;

        // Iterate over all frames
        while (running && !frame.empty())
        {
            workBegin();

            // Change format of the image
            if (make_gray) cvtColor(frame, img_aux, CV_BGR2GRAY);
            else if (use_gpu && GPUfound) cvtColor(frame, img_aux, CV_BGR2BGRA);
            else frame.copyTo(img_aux);

            // Resize image
            if (args.resize_src) resize(img_aux, img, Size(args.width, args.height));
            else img = img_aux;
            img_to_show = img;

            if (GPUfound) {gpu_hog->nlevels = nlevels;}
            cpu_hog.nlevels = nlevels;

            vector<Rect> found;

            // Perform HOG classification

            hogWorkBegin(); // start timer
            if (use_gpu && GPUfound)
            {
                gpu_img.upload(img);
                //gpu_hog->detectMultiScale(gpu_img, found, hit_threshold, win_stride,
                //                         Size(0, 0), scale, gr_threshold);
            }
            else cpu_hog.detectMultiScale(img, found, hit_threshold, win_stride,
                                          Size(0, 0), scale, gr_threshold);
            hogWorkEnd(); // stop timer

            // Draw positive detection sub-windows (boxes)

            for (size_t i = 0; i < found.size(); i++)
            {
                Rect r = found[i];
                rectangle(img_to_show, r.tl(), r.br(), CV_RGB(0, 255, 0), 3);
            }

            // draw text of performance / mode etc.

            if (use_gpu && GPUfound)
                putText(img_to_show, "Mode: GPU", Point(5, 25), FONT_HERSHEY_SIMPLEX, .5, Scalar(255, 100, 0), 2);
            else
                putText(img_to_show, "Mode: CPU", Point(5, 25), FONT_HERSHEY_SIMPLEX, .5, Scalar(255, 100, 0), 2);

            putText(img_to_show, "FPS (HOG only): " + hogWorkFps(), Point(5, 65), FONT_HERSHEY_SIMPLEX, .5, Scalar(255, 100, 0), 2);
            putText(img_to_show, "FPS (total): " + workFps(), Point(5, 105), FONT_HERSHEY_SIMPLEX, .5, Scalar(255, 100, 0), 2);
            imshow("HOG Person Detection", img_to_show);

            // get next frame

            if (args.src_is_video || args.src_is_camera) vc >> frame;

            workEnd(); // end timer

            // output to video file if requested to do so

            if (args.write_video)
            {
                if (!video_writer.isOpened())
                {
                    video_writer.open(args.dst_video, CV_FOURCC('x','v','i','d'), args.dst_video_fps,
                                      img_to_show.size(), true);
                    if (!video_writer.isOpened())
                        throw std::runtime_error("can't create video writer");
                }

                if (make_gray) cvtColor(img_to_show, img, CV_GRAY2BGR);
                else cvtColor(img_to_show, img, CV_BGRA2BGR);

                video_writer << img;
            }

            // get user keyboard input with waitKey and handle input

            handleKey((char)waitKey(3));
        }
    }
}

// handle key input
void App::handleKey(char key)
{
    switch (key)
    {
    case 27:
    case 'x':
    case 'X':
        running = false;
        break;
    case 'm':
    case 'M':
        use_gpu = !use_gpu;
        cout << "Switched to " << (use_gpu ? "GPU (if detected)" : "CPU") << " mode\n";
        break;
    case 'g':
    case 'G':
        make_gray = !make_gray;
        cout << "Convert image to gray: " << (make_gray ? "YES" : "NO") << endl;
        break;
    case '1':
        scale *= 1.05;
        cout << "Scale: " << scale << endl;
        break;
    case 'q':
    case 'Q':
        scale /= 1.05;
        cout << "Scale: " << scale << endl;
        break;
    case '2':
        nlevels++;
        cout << "Levels number: " << nlevels << endl;
        break;
    case 'w':
    case 'W':
        nlevels = max(nlevels - 1, 1);
        cout << "Levels number: " << nlevels << endl;
        break;
    case '3':
        gr_threshold++;
        cout << "Group threshold: " << gr_threshold << endl;
        break;
    case 'e':
    case 'E':
        gr_threshold = max(0, gr_threshold - 1);
        cout << "Group threshold: " << gr_threshold << endl;
        break;
    case '4':
        hit_threshold+=0.25;
        cout << "Hit threshold: " << hit_threshold << endl;
        break;
    case 'r':
    case 'R':
        hit_threshold = max(0.0, hit_threshold - 0.25);
        cout << "Hit threshold: " << hit_threshold << endl;
        break;
    case 'c':
    case 'C':
        gamma_corr = !gamma_corr;
        cout << "Gamma correction: " << gamma_corr << endl;
        break;
    }
}

// timers
inline void App::hogWorkBegin() { hog_work_begin = getTickCount(); }

inline void App::hogWorkEnd()
{
    int64 delta = getTickCount() - hog_work_begin;
    double freq = getTickFrequency();
    hog_work_fps = freq / delta;
}

// calculate performance of in FPS
inline string App::hogWorkFps() const
{
    stringstream ss;
    ss << hog_work_fps;
    return ss.str();
}

// timers
inline void App::workBegin() { work_begin = getTickCount(); }

inline void App::workEnd()
{
    int64 delta = getTickCount() - work_begin;
    double freq = getTickFrequency();
    work_fps = freq / delta;
}

inline string App::workFps() const
{
    stringstream ss;
    ss << work_fps;
    return ss.str();
}

/******************************************************************************/

