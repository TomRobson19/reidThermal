// Example : background / foreground separation of video / camera
// usage: prog {<video_name>}

// Author : Toby Breckon, toby.breckon@cranfield.ac.uk

// Copyright (c) 2012 School of Engineering, Cranfield University
// License : LGPL - http://www.gnu.org/licenses/lgpl.html
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <stdexcept>

using namespace cv;
using namespace std;

/******************************************************************************/

int main( int argc, char** argv )
{

  Mat img, fg, fg_msk, bg;	// image objects
  VideoCapture cap;     // capture object

  const string windowName = "Live Image"; // window name
  const string windowNameF = "Foreground"; // window name
  const string windowNameB = "Background"; // window name

  bool keepProcessing = true;	// loop control flag
  unsigned char  key;			// user input
  int  EVENT_LOOP_DELAY = 40;	// delay for GUI window
                                // 40 ms equates to 1000ms/25fps = 40ms per frame

  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;
  int width = 40;
  int height = 100;
  int learning = 1000;
  int padding = 5; // pad extracted objects by 5%

  // if command line arguments are provided try to read image/video_name
  // otherwise default to capture from attached H/W camera

    if(( argc == 2 && (cap.open(argv[1]) == true )) ||
	  ( argc != 2 && (cap.open(0) == true)))
    {
      // create window object (use flag=0 to allow resize, 1 to auto fix size)

      namedWindow(windowName, 0);
      namedWindow(windowNameF, 0);
      namedWindow(windowNameB, 0);

      createTrackbar("width", windowName, &width, 700);
      createTrackbar("height", windowName, &height, 700);
      createTrackbar("1 / learning", windowName, &learning, 5000);
      createTrackbar("padding n%", windowName, &padding, 100);

      // create background / foreground Mixture of Gaussian (MoG) model

      Ptr<BackgroundSubtractorMOG2> MoG = createBackgroundSubtractorMOG2();

	  // start main loop

	  while (keepProcessing) {

		  // if capture object in use (i.e. video/camera)
		  // get image from capture object

		  if (cap.isOpened()) {

			  cap >> img;
			  if(img.empty()){
				if (argc == 2){
					std::cerr << "End of video file reached" << std::endl;
				} else {
					std::cerr << "ERROR: cannot get next fram from camera"
						      << std::endl;
				}
				exit(0);
			  }

		  }	else {

			  // if not a capture object set event delay to zero so it waits
			  // indefinitely (as single image file, no need to loop)

			  EVENT_LOOP_DELAY = 0;
		  }

		  // update background model and get background/foreground

		  MoG->apply(img, fg_msk, double (1.0 / learning));
      MoG->getBackgroundImage(bg);

          // perform erosion

          erode(fg_msk, fg_msk, Mat());

          // perform morphological closing

          dilate(fg_msk, fg_msk, Mat());
          erode(fg_msk, fg_msk, Mat());

          // extract portion of img using foregorund mask (colour bit)

          fg = Scalar::all(0);
          img.copyTo(fg, fg_msk);

          // get connected components from the foreground

          findContours(fg_msk, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

          // iterate through all the top-level contours,
          // and get bounding rectangles for them (if larger than given value)

          for(int idx = 0; idx >=0; idx = hierarchy[idx][0])
          {
            Rect r = boundingRect(contours[idx]);

            // adjust bounding rectangle to be padding% larger
            // around the object

            r.x = max(0, r.x - (int) (padding/100.0 * (double) r.width));
            r.y = max(0, r.y - (int) (padding/100.0 * (double) r.height));

            r.width = min(img.cols - 1, (r.width + 2 * (int) (padding/100.0 * (double) r.width)));
            r.height = min(img.rows - 1, (r.height + 2 * (int) (padding/100.0 * (double) r.height)));

            // draw rectangle if greater than width/height constraints and if
            // also still inside image

            if ((r.width >= width) && (r.height >= height) &&
                (r.x + r.width < img.cols) && (r.y + r.height < img.rows))
            {

                // JIWAN - we don't want to draw red boxes on the version
                // that goes to the classifer - remember to check for this! :o)

                rectangle(img, r, Scalar(0,0,255), 2, 8, 0);

                // JIWAN - example of how extract region r of image
                // for sending to classifier - here we just display it

                imshow ("Extracted Region", img(r));
            }
          }

		  // display image in window

		  imshow(windowName, img);
          imshow(windowNameF, fg);
          if (!bg.empty())
          {
            imshow(windowNameB, bg);
          }

		  // start event processing loop (very important,in fact essential for GUI)
	      // 40 ms roughly equates to 1000ms/25fps = 4ms per frame

		  key = waitKey(EVENT_LOOP_DELAY);

		  if (key == 'x'){

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
