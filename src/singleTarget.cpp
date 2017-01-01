#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <stdexcept>
#include <stdio.h>

using namespace cv;
using namespace std;
using namespace ml;

//#include "person.hpp"

#define CASCADE_TO_USE "classifiers/people_thermal_23_07_casALL16x32_stump_sym_24_n4.xml"
//#define SVM_TO_USE "classifiers/peopleir_lap.svm"

cv::KalmanFilter KF;
cv::Mat_<float> measurement(6,1); 
// cv::Mat_<float> state(6, 1); //(x, y, Vx, Vy, h, w)

//initialise kalman when first encounter object
int initialised = 0;

//enable velocity 
int timeSteps = 0;
int lastSeen = 0;

// void drawCross(Mat img, Point2f center, Scalar color, int d )
// {
//   line(img, Point2f( center.x - d, center.y - d ), Point2f( center.x + d, center.y + d ), color, 1, LINE_AA, 0);
//   line( img, Point2f( center.x + d, center.y - d ), Point2f( center.x - d, center.y + d ), color, 1, LINE_AA, 0);
// }                                                        

void initKalman(float x, float y, float w, float h)
{
  KF.init(6, 6, 0);  //position(x,y) velocity(x,y) rectangle(h,w)

  measurement(0) = x;
  measurement(1) = y;
  measurement(2) = 0.0;
  measurement(3) = 0.0;
  measurement(4) = w;
  measurement(5) = h;

  KF.statePre.at<float>(0, 0) = x;
  KF.statePre.at<float>(1, 0) = y;
  KF.statePre.at<float>(2, 0) = 0.0;
  KF.statePre.at<float>(3, 0) = 0.0;
  KF.statePre.at<float>(4, 0) = w;
  KF.statePre.at<float>(5, 0) = h;

  KF.statePost.at<float>(0, 0) = x;
  KF.statePost.at<float>(1, 0) = y; 
  KF.statePost.at<float>(2, 0) = 0.0;
  KF.statePost.at<float>(3, 0) = 0.0;
  KF.statePost.at<float>(4, 0) = w;
  KF.statePost.at<float>(5, 0) = h;

  KF.transitionMatrix = (Mat_<float>(6, 6) << 1, 0, 1, 0, 0, 0,
                                              0, 1, 0, 1, 0, 0,
                                              0, 0, 1, 0, 0, 0,
                                              0, 0, 0, 1, 0, 0,
                                              0, 0, 0, 0, 1, 0,
                                              0, 0, 0, 0, 0, 1);
  setIdentity(KF.measurementMatrix);

  setIdentity(KF.processNoiseCov, Scalar::all(0.03)); //adjust this for faster convergence - but higher noise
  //small floating point errors present

  setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));
  setIdentity(KF.errorCovPost, Scalar::all(0.1));

  lastSeen = timeSteps;
}

Point2f kalmanCorrect(float x, float y, int timeSteps, float w, float h)
{
  float currentX = measurement(0);
  float currentY = measurement(1);

  int timeGap = timeSteps-lastSeen;

  if(timeGap == 0) //come up with a better way to do this, for now deals with multiple detections in the same timestep
  {
    timeGap = 1;
  }

  measurement(0) = x;
  measurement(1) = y;
  measurement(2) = (float) ((x - currentX)/timeGap);
  measurement(3) = (float) ((y - currentY)/timeGap);
  measurement(4) = w;
  measurement(5) = h;

  cout << "measurement" << measurement << '\n';

  Mat estimated = KF.correct(measurement);

  cout << "corrected" << estimated << '\n';

  Point2f statePt(estimated.at<float>(0),estimated.at<float>(1));

  lastSeen = timeSteps;
  return statePt;
}

Rect kalmanPredict() 
{
  Mat prediction = KF.predict();

  cout << "prediction" << prediction << '\n';

  Point2f topLeft(prediction.at<float>(0)-prediction.at<float>(4)/2,prediction.at<float>(1)+prediction.at<float>(5)/2);

  Point2f bottomRight(prediction.at<float>(0)+prediction.at<float>(4)/2,prediction.at<float>(1)-prediction.at<float>(5)/2);

  Rect kalmanRect(topLeft, bottomRight);

  KF.statePre.copyTo(KF.statePost);
  KF.errorCovPre.copyTo(KF.errorCovPost);

  return kalmanRect;
}

int main( int argc, char** argv )
{

  Mat img, fg, fg_msk, bg;	// image objects
  VideoCapture cap;     // capture object

  const string windowName = "Live Image"; // window name
  //const string windowNameF = "Foreground"; // window name
  //const string windowNameB = "Background"; // window name

  bool keepProcessing = true;	// loop control flag
  unsigned char  key;			// user input
  int  EVENT_LOOP_DELAY = 40;	// delay for GUI window, 40 ms equates to 1000ms/25fps = 40ms per frame

  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;
  int width = 40;
  int height = 100;
  int learning = 1000;
  int padding = 40; 

  // if command line arguments are provided try to read image/video_name
  // otherwise default to capture from attached H/W camera

  if(( argc == 2 && (cap.open(argv[1]) == true )) ||
  ( argc != 2 && (cap.open(0) == true)))
  {
    // create window object (use flag=0 to allow resize, 1 to auto fix size)

    namedWindow(windowName, 1);
    //namedWindow(windowNameF, 0);
    //namedWindow(windowNameB, 0);

    createTrackbar("width", windowName, &width, 700);
    createTrackbar("height", windowName, &height, 700);
    createTrackbar("1 / learning", windowName, &learning, 5000);
    createTrackbar("padding n%", windowName, &padding, 100);

    // create background / foreground Mixture of Gaussian (MoG) model

    Ptr<BackgroundSubtractorMOG2> MoG = createBackgroundSubtractorMOG2(500,25,false);

    HOGDescriptor hog;
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
    //namedWindow("people detector", 1);

    CascadeClassifier cascade = CascadeClassifier(CASCADE_TO_USE);

    // start main loop

	  while (keepProcessing)
    {
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
  					std::cerr << "ERROR: cannot get next frame from camera" << std::endl;
  				}
  				exit(0);
			  }

		  }
      else
      {
			  // if not a capture object set event delay to zero so it waits
			  // indefinitely (as single image file, no need to loop)
			  EVENT_LOOP_DELAY = 0;
		  }

		  // update background model and get background/foreground

		  MoG->apply(img, fg_msk, (double) (1.0 / learning));
		  MoG->getBackgroundImage(bg);

      // perform erosion - removes boundaries of foreground object

      erode(fg_msk, fg_msk, Mat(),Point(),1);

      // perform morphological closing

      dilate(fg_msk, fg_msk, Mat(),Point(),5);
      erode(fg_msk, fg_msk, Mat(),Point(),1);

      // extract portion of img using foreground mask (colour bit)

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
          vector<Rect> found, found_filtered;

          Mat roi = img(r);

          int method = 1; //0 for Hog, 1 for cascade

          if (method == 0)
          {
            //changing last parameter helps deal with multiple rectangles per person
            hog.detectMultiScale(roi, found, 0, Size(8,8), Size(32,32), 1.05, 5);
          }
          else 
          {
            cascade.detectMultiScale(roi, found, 1.1, 4, CV_HAAR_DO_CANNY_PRUNING, cvSize(64,32));
          }
          for(size_t i = 0; i < found.size(); i++ )
          {
            Rect rec = found[i];

            rec.x += r.x;
            rec.y += r.y;

            size_t j;
            // Do not add small detections inside a bigger detection.
            for ( j = 0; j < found.size(); j++ )
              if ( j != i && (rec & found[j]) == rec )
                  break;

            if ( j == found.size() )
              found_filtered.push_back(rec);
          }

          for (size_t i = 0; i < found_filtered.size(); i++)
          {
            Rect rec = found_filtered[i];

            // The HOG detector returns slightly larger rectangles than the real objects,
            // so we slightly shrink the rectangles to get a nicer output.
            rec.x += rec.width*0.1;
            rec.width = rec.width*0.8;
            rec.y += rec.height*0.1;
            rec.height = rec.height*0.8;
            //rectangle(img, rec.tl(), rec.br(), cv::Scalar(0,255,0), 3);

            Point2f center = Point2f(float(rec.x + rec.width/2.0), float(rec.y + rec.height/2.0));

            //for rectangle, expand state vector to 4 dimensions,store top left corner(2D) or center, width and height, maybe also velocity

            if (initialised == 0)
            {
              initKalman(center.x,center.y,rec.width,rec.height);
              initialised = 1;
            }
            else
            {
              Point2f s = kalmanCorrect(center.x, center.y, timeSteps, rec.width, rec.height);

              Rect p = kalmanPredict();

              //drawCross(img, p, Scalar(255,0,0), 5);
              rectangle(img, p.tl(), p.br(), cv::Scalar(255,0,0), 3);

              // cout << "center" << center << '\n';  
              // cout << "correct" << s << '\n';  
              // cout << "predict" << p << '\n'; 
            }             
          } 
          //imshow("people detector", img);

          // draws calculated rectangle onto image

          rectangle(img, r, Scalar(0,0,255), 2, 8, 0);

          // displays extracted region

          //imshow ("Extracted Region", img(r));
        }
      }
		  // display image in window
		  imshow(windowName, img);
      //imshow(windowNameF, fg);
      //if (!bg.empty())
      //{
        //imshow(windowNameB, bg);
      //}

		  key = waitKey(EVENT_LOOP_DELAY);

		  if (key == 'x')
      {
	   		// if user presses "x" then exit
		  	std::cout << "Keyboard exit requested : exiting now - bye!"
			  		  << std::endl;
   			keepProcessing = false;
		  }
      timeSteps += 1;
	  }

	  // the camera will be deinitialized automatically in VideoCapture destructor
      // all OK : main returns 0
      return 0;
    }
    // not OK : main returns -1
    return -1;
}
/******************************************************************************/