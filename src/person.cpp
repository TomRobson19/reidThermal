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

class Person()
{
public:
	int lastSeen = 0;
	void initKalman(float x, float y, float w, float h, int timeSteps)
	{
	  // Instantate Kalman Filter with
	  // 4 dynamic parameters and 2 measurement parameters,
	  // where my measurement is: 2D location of object,
	  // and dynamic is: 2D location and 2D velocity.
	  KF.init(6, 6, 0);

	  //position(x,y) velocity(x,y) rectangle(h,w)

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

	  //setIdentity(KF.transitionMatrix); 
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
	  setIdentity(KF.errorCovPost, Scalar::all(.1));

	  lastSeen = timeSteps;
	}

	Point2f kalmanCorrect(float x, float y, int timeSteps, float w, float h, int timeSteps)
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

	  //cout << "measurement" << measurement << '\n';

	  Mat estimated = KF.correct(measurement);

	  //cout << "estimated" << estimated << '\n';

	  Point2f statePt(estimated.at<float>(0),estimated.at<float>(1));

	  lastSeen = timeSteps;
	  return statePt;
	}

	Point2f kalmanPredict() 
	{
	  Mat prediction = KF.predict();

	  //cout << "prediction" << prediction << '\n';

	  Point2f predictPt(prediction.at<float>(0),prediction.at<float>(1));

	  KF.statePre.copyTo(KF.statePost);
	  KF.errorCovPre.copyTo(KF.errorCovPost);

	  return predictPt;
	}


private:
	cv::KalmanFilter KF;
	cv::Mat_<float> measurement(6,1); 	
}
