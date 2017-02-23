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

#include "person.hpp"

#define CASCADE_TO_USE "classifiers/people_thermal_23_07_casALL16x32_stump_sym_24_n4.xml"

//enable velocity 
int timeSteps = 0;

std::vector<Person> activeTargets;
std::vector<Person> inactiveTargets;                                         

int main( int argc, char** argv )
{

  Mat img, outputImage, fg_msk, bg;	// image objects
  VideoCapture cap;     // capture object

  const string windowName = "Live Image"; // window name

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

    // create background / foreground Mixture of Gaussian (MoG) model
    Ptr<BackgroundSubtractorMOG2> MoG = createBackgroundSubtractorMOG2(500,25,false);

    HOGDescriptor hog;
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

    CascadeClassifier cascade = CascadeClassifier(CASCADE_TO_USE);

    // start main loop

	  while (keepProcessing)
    {
      int64 timeStart = getTickCount();
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
        outputImage = img.clone();

        cvtColor(img, img, CV_BGR2GRAY);
		  }
      else
      {
			  // if not a capture object set event delay to zero so it waits
			  // indefinitely (as single image file, no need to loop)
			  EVENT_LOOP_DELAY = 0;
		  }

		  // update background model and get background/foreground

		  MoG->apply(img, fg_msk, (double)(1.0/learning));
		  MoG->getBackgroundImage(bg);

      // perform erosion - removes boundaries of foreground object

      erode(fg_msk, fg_msk, Mat(),Point(),1);

      // perform morphological closing

      dilate(fg_msk, fg_msk, Mat(),Point(),5);
      erode(fg_msk, fg_msk, Mat(),Point(),1);

      // extract portion of img using foreground mask (colour bit)

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

            // The HOG/Cascade detector returns slightly larger rectangles than the real objects,
            // so we slightly shrink the rectangles to get a nicer output.
            rec.x += rec.width*0.1;
            rec.width = rec.width*0.8;
            rec.y += rec.height*0.1;
            rec.height = rec.height*0.8;
            // rectangle(img, rec.tl(), rec.br(), cv::Scalar(0,255,0), 3);

            Point2f center = Point2f(float(rec.x + rec.width/2.0), float(rec.y + rec.height/2.0));

/////////////////////////////////////////////////////////////////////////////////////////////////////Histogram
            
            Mat regionOfInterest;

            Mat regionOfInterestOriginal = img(rec);

            Mat regionOfInterestForeground = fg_msk(rec);

            bitwise_and(regionOfInterestOriginal, regionOfInterestForeground, regionOfInterest);

            MatND hist;
            int histSize = 16;    // bin size - need to determine which pixel threshold to use
            float range[] = {0,255};
            const float *ranges[] = {range};
            int channels[] = {0, 1};

            calcHist(&regionOfInterest, 1, channels, Mat(), hist, 1, &histSize, ranges, true, false);

            normalize(hist, hist, 1, 0, NORM_L2, -1, Mat());

            // cout << hist << endl;
            // cout << endl;
            // cout << endl;

/////////////////////////////////////////////////////////////////////////////////////////////////////HuMoments

            vector<vector<Point> > contoursHu;
            vector<Vec4i> hierarchyHu;

            findContours(regionOfInterest, contoursHu, hierarchyHu, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

            double largestSize;
            int largestContour;

            for(int i = 0; i< contoursHu.size(); i++)
            {
              double size = contoursHu[i].size();

              if(size>largestSize)
              {
                largestSize=size;
                largestContour=i;
              }
            }

            Moments contourMoments;
            double huMoments[7];

            contourMoments = moments(contoursHu[largestContour]);

            HuMoments(contourMoments, huMoments);

            // for (int i=0; i<7; i++)
            // {
            //   cout << huMoments[i] << endl;
            // }
            // cout << endl;
            // cout << endl;

/////////////////////////////////////////////////////////////////////////////////////////////////////HOGDescriptor

            //copy regionOfInterest and resize to 64x128 (same size as in compute call)

            Mat clone = regionOfInterest.clone();

            Mat resized;

            resize(clone, resized, Size(64,128), CV_INTER_LINEAR);

            imshow("resized",resized);

            cv::HOGDescriptor descriptor;

            vector<float> descriptorsValues;

            descriptor.compute(resized, descriptorsValues);

            // for (int i=0; i<descriptorsValues.size(); i++)
            // {
            //   cout << descriptorsValues[i] << endl;
            // }

            // cout << endl;
            // cout << endl;

/////////////////////////////////////////////////////////////////////////////////////////////////////
            
            int allocated = 0;

            if(activeTargets.size() == 0 and inactiveTargets.size() == 0) //if first target found
            {
              Person person(0, center.x, center.y, timeSteps, rec.width, rec.height);

              person.kalmanCorrect(center.x, center.y, timeSteps, rec.width, rec.height);
              
              Rect p = person.kalmanPredict();

              rectangle(outputImage, p.tl(), p.br(), cv::Scalar(255,0,0), 3);

              char str[200];
              sprintf(str,"Person %d",person.getIdentifier());

              putText(outputImage, str, center, FONT_HERSHEY_SIMPLEX,1,(0,0,0));

              activeTargets.push_back(person);
              allocated = 1;
            }
            
            if(allocated == 0) //check if it is similar enough to a currently active target
            {
              for(int a=0; a<activeTargets.size(); a++)
              {
                Point2f lastPosition = activeTargets[a].getLastPosition();

                if(fabs(center.x-lastPosition.x)<100 and fabs(center.y-lastPosition.y)<100) 
                //if close enough to last postion,it is that person
                //will change this when features are implemented
                {
                  activeTargets[a].kalmanCorrect(center.x, center.y, timeSteps, rec.width, rec.height);

                  Rect p = activeTargets[a].kalmanPredict();

                  rectangle(outputImage, p.tl(), p.br(), cv::Scalar(255,0,0), 3);

                  char str[200];
                  sprintf(str,"Person %d",activeTargets[a].getIdentifier());

                  putText(outputImage, str, center, FONT_HERSHEY_SIMPLEX,1,(0,0,0));

                  allocated = 1;
                  break;
                }

                if(timeSteps-activeTargets[a].getLastSeen() > 100 and allocated == 0)//if hasn't been seen for long enough, make inactive
                {
                  inactiveTargets.push_back(activeTargets[a]);
                  activeTargets.erase(activeTargets.begin()+a);
                }
              }
            }

            if(allocated == 0) //now check if it is similar to an inactive target
            {
              for(int b=0; b<inactiveTargets.size(); b++)
              {                
                Point2f lastPosition = inactiveTargets[b].getLastPosition();

                if(fabs(center.x-lastPosition.x)<100 and fabs(center.y-lastPosition.y)<100) 
                //if close enough to last postion,it is that person
                //will change this when features are implemented
                {
                  inactiveTargets[b].kalmanCorrect(center.x, center.y, timeSteps, rec.width, rec.height);

                  Rect p = inactiveTargets[b].kalmanPredict();

                  rectangle(outputImage, p.tl(), p.br(), cv::Scalar(255,0,0), 3);

                  char str[200];
                  sprintf(str,"Person %d",inactiveTargets[b].getIdentifier());

                  putText(outputImage, str, center, FONT_HERSHEY_SIMPLEX,1,(0,0,0));

                  activeTargets.push_back(inactiveTargets[b]);
                  inactiveTargets.erase(inactiveTargets.begin()+b);

                  allocated = 1;
                  break;
                }
              } 
            }
            if(allocated == 0) //if not close enough to any existing targets, create a new target
            {
              int identifier = activeTargets.size()+inactiveTargets.size();
              Person person(identifier, center.x, center.y, timeSteps, rec.width, rec.height);

              person.kalmanCorrect(center.x, center.y, timeSteps, rec.width, rec.height);
              
              Rect p = person.kalmanPredict();

              rectangle(outputImage, p.tl(), p.br(), cv::Scalar(255,0,0), 3);

              char str[200];
              sprintf(str,"Person %d",person.getIdentifier());

              putText(outputImage, str, center, FONT_HERSHEY_SIMPLEX,1,(0,0,0));

              activeTargets.push_back(person);
              allocated = 1;
            }
          }
          rectangle(outputImage, r, Scalar(0,0,255), 2, 8, 0);
        }
      }
		  // display image in window
		  imshow(windowName, outputImage);

      key = waitKey((int) std::max(2.0, EVENT_LOOP_DELAY - (((getTickCount() - timeStart) / getTickFrequency())*1000)));

      // key = waitKey(EVENT_LOOP_DELAY);

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