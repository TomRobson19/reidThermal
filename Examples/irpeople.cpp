// #include "../../../cranfield/opencv/record_output/opencv_record_output.h"

// Example : run people detection on thermal-band image / video / camera
// usage: prog {<image_name> | <video_name>}

// Author : Toby Breckon, toby.breckon@durham.ac.uk
// version : OpenCV C++ 2.x / version 0.3

// Based technique outlined in [Breckon et al, 2013]
//

// Copyright (c) 2011 School of Engineering, Cranfield University
// Copyright (c) 2014 School of Engineering and Computing Sciences, Durham University

#include <opencv2/opencv.hpp>   		// open cv general include file
#include <opencv2/ml.hpp>
#include <iostream>		// standard C++ I/O

using namespace std;
using namespace cv; // OpenCV API is in the C++ "cv" namespace
using namespace ml;

/******************************************************************************/

#define CAMERA_ID_TO_USE 0
#define CASCADE_TO_USE "classifiers/people_thermal_23_07_casALL16x32_stump_sym_24_n4.xml"
#define SVM_TO_USE  "classifiers/peopleir_lap.svm"

/******************************************************************************/

int main( int argc, char** argv )
{

    Mat img, gray, roi, resized, resized2, test, test2;	// image objects
    VideoCapture cap;                       // capture object
    vector<Rect> objects;                   // bounding boxes of detected objects

    const string windowName = "People Detection"; // window name

    bool keepProcessing = true;	// loop control flag
    char  key;	                // user input

    // if command line arguments are provided try to read image/video_name
    // otherwise default to capture from attached H/W camera

    if(
        ( argc == 2 && (!(img = imread( argv[1], CV_LOAD_IMAGE_COLOR)).empty()))||
        ( argc == 2 && (cap.open(argv[1]) == true )) ||
        ( argc != 2 && (cap.open(CAMERA_ID_TO_USE) == true))
    )
    {
        // create window object (use flag=0 to allow resize, 1 to auto fix size)

        namedWindow(windowName, 0);

        // load the fast-reject trained haar cascade classifier (stage 1)
        // (the orginal work only used the first 16 stages but this is
        // not directly accessable in the new CascadeClassifier() objects)

        CascadeClassifier cascade = CascadeClassifier(CASCADE_TO_USE);
        //cascade.count = 16;

        if (cascade.empty())
        {
            std::cout << "ERROR: Could not load classifier cascade" << std::endl;
            exit(1);
        }

        // load the confirmation trained SVM classifier (stage 2)

        Ptr<SVM> svm = Algorithm::load<SVM>(SVM_TO_USE);

        if (svm->empty())
        {
            std::cout << "ERROR: Could not load SVM" << std::endl;
            exit(1);
        }
        // start main loop

        while (keepProcessing)
        {

            double time1 = (double)getTickCount(); // record current time

            // if capture object in use (i.e. video/camera)
            // get image from capture object

            if (cap.isOpened())
            {

                cap >> img;
                if(img.empty())
                {
                    if (argc == 2)
                    {
                        std::cerr << "End of video file reached" << std::endl;
                    }
                    else
                    {
                        std::cerr << "ERROR: cannot get next fram from camera"
                                  << std::endl;
                    }
                    exit(0);
                }

            }
            else
            {

                // if not a capture object set event delay to zero so it waits
                // indefinitely (as single image file, no need to loop)

            }

            // convert input image to grayscale (single channel)

            cvtColor(img, gray, CV_BGR2GRAY );

            // run the haar cascade detection
            // with parameters scale=1.1, neighbours = 4 and with Canny pruning
            // turned on with minimum detection scale 64x32 pixels

            objects.clear();
            cascade.detectMultiScale(gray, objects, 1.1, 4, CV_HAAR_DO_CANNY_PRUNING, cvSize(64, 32));

            // draw a red rectange in the image where the objects are detected

            for( vector<Rect>::const_iterator r = objects.begin(); r != objects.end(); r++)
            {
                roi = gray(*r);

                resize(roi, resized, Size(16,32), 0, 0, CV_INTER_CUBIC);

                // do a whole load of messing around so we can use the
                // old CvLaplace() call as in the original work

                IplImage tmp1, tmp2;
                resized2 = resized.clone();
                tmp1 = IplImage(resized);
                tmp2 = IplImage(resized2);

                cvLaplace(&tmp1, &tmp2, 3);

                // now normalize both, add them, then re-normalize

                resized.convertTo(test, CV_32F);
                resized2.convertTo(test2, CV_32F);

                normalize(test, test, 0.0, 1.0, NORM_MINMAX);
                normalize(test2, test2, 0.0, 1.0, NORM_MINMAX);

                test = test + test2;

                normalize(test, test, 0.0, 1.0, NORM_MINMAX);

                // reshape to single row for SVM prediction

                test = test.reshape(0, 1);
                resize(resized2, resized2, Size(), 8, 8, CV_INTER_CUBIC);
                imshow("Cascade -> SVM Input (upscaled)", resized2);

                if (svm->predict(test) == 1)
                {
                    rectangle(img, *r, Scalar(0,0,255), 2, 8, 0);

                    // This SVM was trained sometime ago and distance to the hyper-plane
                    // seems not to work

                    // std::cout << "SVM: " << (float) svm.predict(test, true) << std::endl;
                }
            }

            // display image in window

            imshow(windowName, img);

            // calculate time taken in ms
            double timetaken = (((double)getTickCount() - time1) / getTickFrequency()) * 1000.0;

            // start event processing loop (very important,in fact essential for GUI)
            // 40 ms roughly equates to 1000ms/25fps = 4ms per frame
            // but here adjust this delay for the time taken in processing the image

            key = waitKey(max(2, (40 - (int) timetaken)));

            if (key == 'x')
            {

                // if user presses "x" then exit

                std::cout << "Keyboard exit requested : exiting now - bye!"
                          << std::endl;
                keepProcessing = false;
            }
        }

        // the camera will be deinitialized automatically in VideoCapture destructor

        // all OK : main returns 0

        return 0;
    }

    // not OK : main returns -1

    return -1;
}
/******************************************************************************/
