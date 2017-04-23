/*
Run like this : 
./main -d=1 -f=1
*/

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
#include <thread>
#include <X11/Xlib.h>

#include <opencv2/ximgproc.hpp>

using namespace cv;
using namespace std;
using namespace ml;
using namespace cv::ximgproc;

#include "person.hpp"

#define CASCADE_TO_USE "classifiers/people_thermal_23_07_casALL16x32_stump_sym_24_n4.xml"

vector<Person> targets;
pthread_mutex_t myLock;

static const char* keys =
    ("{h help       | | Help Menu}"
     "{d dataset    | | Dataset - 1, 2, 3}"
     "{t testing 		| | 1 - yes, 2 - no}");

bool matIsEqual(const cv::Mat mat1, const cv::Mat mat2){
	// treat two empty mat as identical as well
	if (mat1.empty() && mat2.empty()) {
	  return true;
	}
	// if dimensionality of two mat is not identical, these two mat is not identical
	if (mat1.cols != mat2.cols || mat1.rows != mat2.rows || mat1.dims != mat2.dims) {
	  return false;
	}
	cv::Mat diff;
	cv::compare(mat1, mat2, diff, cv::CMP_NE);
	int nz = cv::countNonZero(diff);
	if(nz==0) 
  { 
    return true; 
  } 
  else 
  { 
    return false; 
  } 
}
 
int runOnSingleCamera(String file, int cameraID, int multipleCameras) 
{
	VideoWriter video(file+"results.avi",CV_FOURCC('M','J','P','G'),10, Size(640,480),true);

	int timeSteps = 0;

  Mat img, previousImg, outputImage, foreground;	// image objects
  VideoCapture cap;

  bool keepProcessing = true;	// loop control flag
  unsigned char  key;			// user input
  int  EVENT_LOOP_DELAY = 40;	// delay for GUI window, 40 ms equates to 1000ms/25fps = 40ms per frame

  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;
  int width = 40;
  int height = 100;
  int learning = 1000;
  int padding = 40; 

  //vectors for flow
  vector<Mat> previousROIs;
	vector<Point2f> centersOfROIs;

	bool specialCase = false;

  // if command line arguments are provided try to read image/video_name
  // otherwise default to capture from attached H/W camera
  if((cap.open(file) == true))
  {
		// create background / foreground Mixture of Gaussian (MoG) model
		Ptr<BackgroundSubtractorMOG2> MoG = createBackgroundSubtractorMOG2(500,25,false);

		CascadeClassifier cascade = CascadeClassifier(CASCADE_TO_USE);

    Ptr<SuperpixelSEEDS> seeds;

		// start main loop
	  while(keepProcessing)
		{
		  int64 timeStart = getTickCount();
			if (cap.isOpened())
		  {
				cap >> img;
				if(img.empty())
				{
					std::cerr << cameraID << " End of video file reached" << std::endl;
					keepProcessing = false;
					break;
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

		  if(matIsEqual(img,previousImg) == false)
		  {
		  	previousImg = img;
			  // update background model and get background/foreground
			  MoG->apply(img, foreground, (double)(1.0/learning));

			  // perform erosion - removes boundaries of foreground object
			  erode(foreground, foreground, Mat(),Point(),1);

			  // perform morphological closing
			  dilate(foreground, foreground, Mat(),Point(),5);
			  erode(foreground, foreground, Mat(),Point(),1);

			  // get connected components from the foreground
			  findContours(foreground, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

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
					if ((r.width >= width) && (r.height >= height) && (r.x + r.width < img.cols) && (r.y + r.height < img.rows))
					{
					  vector<Rect> found, found_filtered;

					  Mat roi = img(r);

				  	if (cameraID == 3)
				  	{
				  		cascade.detectMultiScale(roi, found, 1.1, 4, CV_HAAR_DO_CANNY_PRUNING, cvSize(32,64));
				  	}
				  	else
				  	{
				  		cascade.detectMultiScale(roi, found, 1.1, 4, CV_HAAR_DO_CANNY_PRUNING, cvSize(96,160));
				  	}

					  for(size_t i = 0; i < found.size(); i++ )
					  {
							Rect rec = found[i];

							rec.x += r.x;
							rec.y += r.y;

							size_t j;
							// Do not add small detections inside a bigger detection.
							for ( j = 0; j < found.size(); j++ )
							{
							  if ( j != i && (rec & found[j]) == rec )
							  {
								  break;
							  }
							}

							if (j == found.size())
							{
							  found_filtered.push_back(rec);
							}
					  }
					  for (size_t i = 0; i < found_filtered.size(); i++)
					  {
							Rect rec = found_filtered[i];

							Point2f center = Point2f(float(rec.x + rec.width/2.0), float(rec.y + rec.height/2.0));

							bool useKalmanRectangle = false;
							int targetID, closestX = 100, closestY = 100;

							for(int iterator = 0; i < targets.size(); i++)
							{
								Point2f lastPosition = targets[iterator].getLastPosition();
								int xDistance = fabs(center.x-lastPosition.x);
								int yDistance = fabs(center.y-lastPosition.y);

								Rect p = targets[iterator].kalmanPredict();

								if((targets[iterator].getCurrentCamera() == cameraID) && (timeSteps - targets[iterator].getLastSeen() < 10) \
								   && (xDistance<50 && yDistance<50) && (xDistance+yDistance < closestX+closestY) && (p.width*p.height - rec.width*rec.height > 2000))
								{
									targetID = iterator;
									closestX = xDistance;
									closestY = yDistance;
									
									useKalmanRectangle = true;
								}
							}

							Mat regionOfInterest, regionOfInterestOriginal, regionOfInterestForeground;

							if(useKalmanRectangle == true)
							{
								Rect p = targets[targetID].kalmanPredict();

								try
								{
									regionOfInterestOriginal = img(p);

									regionOfInterestForeground  = foreground(p);
								}
								catch (exception& e)
								{
									regionOfInterestOriginal = img(rec);

									regionOfInterestForeground  = foreground(rec);
								}
								
							}
							else
							{
								regionOfInterestOriginal = img(rec);

								regionOfInterestForeground  = foreground(rec);
							}

							bitwise_and(regionOfInterestOriginal, regionOfInterestForeground, regionOfInterest);

							Mat clone = regionOfInterest.clone();

							resize(clone, regionOfInterest, Size(64,128), CV_INTER_CUBIC);

							Mat hist;

							Mat feature;

							double classificationThreshold, learningThreshold;

							bool classify = true;

							classificationThreshold = 6;
							learningThreshold = 4;

						  int histSize = 32;    // bin size - need to determine which pixel threshold to use
						  float range[] = {0,255};
						  const float *ranges[] = {range};
						  int channels[] = {0, 1};

						  calcHist(&regionOfInterest, 1, channels, Mat(), hist, 1, &histSize, ranges, true, false);

						  hist.convertTo(hist, CV_64F);

						  normalize(hist, hist, 1, 0, NORM_L1, -1, Mat());

						  feature.push_back(hist);

							classify = false;

							if(previousROIs.size() == 0)
							{
								previousROIs.push_back(regionOfInterest);
								centersOfROIs.push_back(center);
							}
							else
							{
								Mat previousROI;
								bool hasPrevious = false;
								for(int i = 0; i<centersOfROIs.size(); i++)
								{
									if(fabs(center.x-centersOfROIs[i].x)<100 and fabs(center.y-centersOfROIs[i].y)<100)
									{
										previousROI = previousROIs[i];
										hasPrevious = true;
									}
								}
								if(hasPrevious == true)
								{
									classify = true;
								}
								else
								{
									previousROIs.push_back(regionOfInterest);
									centersOfROIs.push_back(center);
								}
							}
							//classifier
							if(classify == true)
							{
								feature=feature.t();


								if(multipleCameras == 1)
								{
									//LOCK
									pthread_mutex_lock(&myLock);
								}

								if(targets.size() == 0) //if first target found
								{
								  Person person(0, center.x, center.y, timeSteps, rec.width, rec.height);

								  person.kalmanCorrect(center.x, center.y, timeSteps, rec.width, rec.height);

								  Rect p = person.kalmanPredict();

								  person.updateFeatures(feature);

								  person.setCurrentCamera(cameraID);

								  rectangle(outputImage, p.tl(), p.br(), person.getColour(), 3);

								  char str[200];
								  sprintf(str,"Person %d",person.getIdentifier());

								  putText(outputImage, str, center, FONT_HERSHEY_SIMPLEX,1,(0,0,0));

								  targets.push_back(person);
								  specialCase = true;
								}
								else if(specialCase == true)
								{
									targets[0].kalmanCorrect(center.x, center.y, timeSteps, rec.width, rec.height);

								  Rect p = targets[0].kalmanPredict();

						  		targets[0].updateFeatures(feature);

						  		targets[0].setCurrentCamera(cameraID);

								  rectangle(outputImage, p.tl(), p.br(), targets[0].getColour(), 3);

								  char str[200];
								  sprintf(str,"Person %d",targets[0].getIdentifier());

								  putText(outputImage, str, center, FONT_HERSHEY_SIMPLEX,1,(0,0,0));

								  specialCase = false;
								}
								else
								{
									vector<double> mDistances;

									Mat benchmarkCovar;
									
									for(int i = 0; i<targets.size(); i++)
									{
										Mat covar, mean;
										Mat data = targets[i].getFeatures();
										
										calcCovarMatrix(data,covar,mean,CV_COVAR_NORMAL|CV_COVAR_ROWS);

										if(i == 0)
										{
											benchmarkCovar = covar.clone();
										}

										double mDistance;

										if(data.rows == 1)
										{
											Mat invCovar;

											invert(benchmarkCovar,invCovar,DECOMP_SVD);

											mDistance = Mahalanobis(feature,mean,invCovar);

											//cout << "Target " << i << " Mahalanobis error from current image = " << mDistance << endl;
										}
										else
										{
											Mat invCovar;

											invert(covar,invCovar,DECOMP_SVD);

											mDistance = Mahalanobis(feature,mean,invCovar);

											//cout << "Target " << i << " Mahalanobis error from current image = " << mDistance << endl;
										}
										mDistances.push_back(mDistance);
									}

									Mat dists = Mat(mDistances);
									
									//cout << endl << endl << endl << endl << endl << endl << endl;
									//cout << dists << endl;

		    					double lowestDist = 0.0;
		    					int identifier = 0;

		    					double min, max;
									Point min_loc, max_loc;
									minMaxLoc(dists, &min, &max, &min_loc, &max_loc);

									lowestDist = min;
									identifier = min_loc.y;

		    					if(lowestDist <= classificationThreshold)
		    					{
		    						if(lowestDist <= learningThreshold)
		    						{
		    							targets[identifier].updateFeatures(feature);
		    							//decide if I should also put Kalman stuff in here
		    						}

		    						targets[identifier].kalmanCorrect(center.x, center.y, timeSteps, rec.width, rec.height);

									  Rect p = targets[identifier].kalmanPredict();

							  		targets[identifier].setCurrentCamera(cameraID);

									  rectangle(outputImage, p.tl(), p.br(), targets[identifier].getColour(), 3);

									  char str[200];
									  sprintf(str,"Person %d",targets[identifier].getIdentifier());

									  putText(outputImage, str, center, FONT_HERSHEY_SIMPLEX,1,(0,0,0));
		    					}
		    					else
		    					{
		    						int identifier = targets.size();
									  Person person(identifier, center.x, center.y, timeSteps, rec.width, rec.height);

									  person.kalmanCorrect(center.x, center.y, timeSteps, rec.width, rec.height);
									  
									  Rect p = person.kalmanPredict();

							  		person.updateFeatures(feature);

							  		person.setCurrentCamera(cameraID);

									  rectangle(outputImage, p.tl(), p.br(), person.getColour(), 3);

									  char str[200];
									  sprintf(str,"Person %d",person.getIdentifier());

									  putText(outputImage, str, center, FONT_HERSHEY_SIMPLEX,1,(0,0,0));

									  targets.push_back(person);
		    					}
		    					if(multipleCameras == 0)
		    					{
										if (lowestDist > classificationThreshold)
										{
											imshow("error",regionOfInterest);
											//waitKey(1000000000);
										}
										else
										{
											imshow("roi", regionOfInterest);
											//waitKey(1000000);
										}
									}
				    		}
				    		if(multipleCameras == 1)
				    		{
				    			//UNLOCK
			    				pthread_mutex_unlock(&myLock);
						  	}
			    		}
					  }
					  if (multipleCameras == 0)
					  {
					  	//rectangle(outputImage, r, Scalar(0,0,255), 2, 8, 0);
					  }
					}
			  }
			}
			String cameras[4] = {"Alpha", "Beta", "Gamma", "Delta"};

			putText(outputImage, cameras[cameraID], Point2f(20,50), FONT_HERSHEY_SIMPLEX,1,(0,0,0));
		  // display image in window
		  if(multipleCameras == 1)
		  {
		  	video.write(outputImage);
		  	key = waitKey(1000);
		  }
		  else 
		  {
		  	imshow(file, outputImage);
		  	key = waitKey((int) std::max(2.0, EVENT_LOOP_DELAY - (((getTickCount() - timeStart) / getTickFrequency())*1000)));
		  }

		  if (key == 'x')
	  	{
				// if user presses "x" then exit
				std::cout << "Keyboard exit requested : exiting now - bye!" << std::endl;
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

int postProcessing(String alphaFile, String betaFile, String gammaFile, String deltaFile, String directory)
{
	VideoWriter video(directory+"/fullResults.avi",CV_FOURCC('M','J','P','G'),10, Size(1280,960),true);
	unsigned char key;

	Mat imgAlpha, imgBeta, imgGamma, imgDelta;
	VideoCapture capAlpha, capBeta, capGamma, capDelta;

	capAlpha.open(alphaFile+"results.avi");
	capBeta.open(betaFile+"results.avi");
	capGamma.open(gammaFile+"results.avi");
	capDelta.open(deltaFile+"results.avi");

	while(1)
	{
		capAlpha >> imgAlpha;
		capBeta >> imgBeta;
		capGamma >> imgGamma;
		capDelta >> imgDelta;

		if(imgAlpha.empty() || imgBeta.empty() || imgGamma.empty() || imgDelta.empty())
		{
			std::cerr << "End of video file reached" << std::endl;
			exit(0);
		}

    Mat out = Mat(960, 1280, CV_8UC3);

    Mat roiAlpha = out(Rect(0, 0, 640, 480));
    Mat roiBeta = out(Rect(640, 0, 640, 480));
    Mat roiGamma = out(Rect(0, 480, 640, 480));
    Mat roiDelta = out(Rect(640, 480, 640, 480));

    imgAlpha.copyTo(roiAlpha);
    imgBeta.copyTo(roiBeta);
    imgGamma.copyTo(roiGamma);
    imgDelta.copyTo(roiDelta);

		//imshow("output",out);
		video.write(out);

		key = waitKey(1);
	}
}

int main(int argc,char** argv)
{
	XInitThreads();
	CommandLineParser cmd(argc,argv,keys);
  if (cmd.has("help")) 
  {
    cmd.printMessage();
    return 0;
  }

  int datasetToUse = cmd.get<int>("dataset");
  int testing = cmd.get<int>("testing");

  String directory = "data/Dataset" + to_string(datasetToUse);

  String alphaFile = directory + "/alphaInput.webm";
  String betaFile = directory + "/betaInput.webm";
  String gammaFile = directory + "/gammaInput.webm";
  String deltaFile = directory + "/deltaInput.webm";

  if(testing == 1)
  {
	  runOnSingleCamera(alphaFile, 0, 0); 
	  runOnSingleCamera(betaFile, 1, 0); 
	  runOnSingleCamera(gammaFile, 2, 0); 
	  runOnSingleCamera(deltaFile, 3, 0); 
  }

  else
  {
	  if (pthread_mutex_init(&myLock, NULL) != 0)
	  {
	    printf("\n mutex init failed\n");
	    return 1;
	  }

	  cout << "processing stage" << endl;

	  std::thread t1(runOnSingleCamera, alphaFile, 0, 1);
	  std::thread t2(runOnSingleCamera, betaFile, 1, 1);
	  std::thread t3(runOnSingleCamera, gammaFile, 2, 1);
	  std::thread t4(runOnSingleCamera, deltaFile, 3, 1);
	  t1.join();
	  t2.join();
	  t3.join();
	  t4.join();
	  pthread_mutex_destroy(&myLock);

	  cout << "post processing stage" << endl;

	  postProcessing(alphaFile, betaFile, gammaFile, deltaFile, directory);
	}

  return 0;
}