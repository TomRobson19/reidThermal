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
 
int runOnSingleCamera(String file, int cameraID, int multipleCameras) 
{
	VideoWriter video(file+"results.avi",CV_FOURCC('M','J','P','G'),10, Size(640,480),true);

	int timeSteps = 0;

  Mat img, outputImage, foreground;	// image objects
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
					std::cerr << "End of video file reached" << std::endl;
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
		  MoG->apply(img, foreground, (double)(1.0/learning));

/////////////////////////////////////////////////////////////////////////////////SUPERPIXELS
		  int useSuperpixels = 0;
		  
		  if(useSuperpixels == 1)
			{
			  Mat seedMask, labels, result;

			  result = img.clone();

			  int width = img.size().width;
	    	int height = img.size().height;

		    seeds = createSuperpixelSEEDS(width, height, 1, 2000, 10, 2, 5, true);

	  	  seeds->iterate(img, 10);

	    	seeds->getLabels(labels);

	    	vector<int> counter(seeds->getNumberOfSuperpixels(),0);
	    	vector<int> numberOfPixelsPerSuperpixel(seeds->getNumberOfSuperpixels(),0);

	    	vector<bool> useSuperpixel(seeds->getNumberOfSuperpixels(),false);

	    	for(int i = 0; i<foreground.rows; i++)
	    	{
	    		for(int j = 0; j<foreground.cols; j++)
	    		{
	    			numberOfPixelsPerSuperpixel[labels.at<int>(i,j)] += 1;
	    			if(foreground.at<unsigned char>(i,j)==255)
	    			{
	    				counter[labels.at<int>(i,j)] += 1;
	    			}
	    		}
	    	}

	    	for(int i = 0; i<counter.size(); i++)
	    	{
	    		if(counter[i]/numberOfPixelsPerSuperpixel[i] > 0.0001)
	    		{
	    			useSuperpixel[i] = true;
	    		}
	    	}

	    	for(int i = 0; i<foreground.rows; i++)
	    	{
	    		for(int j = 0; j<foreground.cols; j++)
	    		{
	    			if(useSuperpixel[labels.at<int>(i,j)] == true)
	    			{
	    				foreground.at<unsigned char>(i,j) = 255;
	    			}
	    			else
	    			{
	    				foreground.at<unsigned char>(i,j) = 0;
	    			}
	    		}
	    	}
			}
/////////////////////////////////////////////////////////////////////////////////
			else
			{
			  // perform erosion - removes boundaries of foreground object
			  erode(foreground, foreground, Mat(),Point(),1);

			  // perform morphological closing
			  dilate(foreground, foreground, Mat(),Point(),5);
			  erode(foreground, foreground, Mat(),Point(),1);
			}

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

						// The HOG/Cascade detector returns slightly larger rectangles than the real objects,
						// so we slightly shrink the rectangles to get a nicer output.
						rec.x += rec.width*0.1;
						rec.width = rec.width*0.8;
						rec.y += rec.height*0.1;
						rec.height = rec.height*0.8;
						// rectangle(img, rec.tl(), rec.br(), cv::Scalar(0,255,0), 3);

						Point2f center = Point2f(float(rec.x + rec.width/2.0), float(rec.y + rec.height/2.0));

						Mat regionOfInterest;

						Mat regionOfInterestOriginal = img(rec);
						//Mat regionOfInterestOriginal = img(r);

						Mat regionOfInterestForeground  = foreground(rec);
						//Mat regionOfInterestForeground = foreground(r);

						bitwise_and(regionOfInterestOriginal, regionOfInterestForeground, regionOfInterest);

						Mat clone = regionOfInterest.clone();

						resize(clone, regionOfInterest, Size(64,128), CV_INTER_CUBIC);



						double huMoments[7];
						vector<double> hu(7);
						Mat hist;
						Mat histFlow;
						vector<float> descriptorsValues;

						Mat feature;

						double classificationThreshold;

						bool classify = true;

						classificationThreshold = 6;

					  


					  int histSize = 32;    // bin size - need to determine which pixel threshold to use
					  float range[] = {0,255};
					  const float *ranges[] = {range};
					  int channels[] = {0, 1};

					  calcHist(&regionOfInterest, 1, channels, Mat(), hist, 1, &histSize, ranges, true, false);

					  hist.convertTo(hist, CV_64F);

					  feature.push_back(hist);


						int sizes[] = { 8, 8, 3 };
						Mat correlogram(3, sizes, CV_32S, cv::Scalar(0));

						Mat newCorrelogram;

						int xIntensity, yIntensity;

						for(int i = 0; i<regionOfInterest.rows; i++)
						{
							for(int j = 0; j<regionOfInterest.cols; j++)
							{
								xIntensity = floor(regionOfInterest.at<unsigned char>(i,j)/32);

								for(int k = i; k<regionOfInterest.rows; k++)
								{
									for(int l = j; l<regionOfInterest.cols; l++)
									{
										if((k == i && l > j) || k > i)
										{
											yIntensity = floor(regionOfInterest.at<unsigned char>(k,l)/32);
										
											double distance = (norm(Point(i,j)-Point(k,l)));
											correlogram.at<int>(xIntensity,yIntensity,floor(distance/50)) += 1;
										}
									}
								}
							}
						}
						for(int i = 0; i<8; i++)
						{
							for(int j = 0; j<8; j++)
							{
								for(int k = 0; k<3; k++)
								{
									newCorrelogram.push_back(correlogram.at<int>(i,j,k));
								}
							}
						}

						newCorrelogram.convertTo(newCorrelogram, CV_64F);

						feature.push_back(newCorrelogram);


												
						Mat opticalFlow;
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
								calcOpticalFlowFarneback(previousROI, regionOfInterest, opticalFlow, 0.5, 3, 15, 3, 5, 1.2, 0);
							}
							else
							{
								previousROIs.push_back(regionOfInterest);
								centersOfROIs.push_back(center);
							}
						}

						if(classify == true)
						{
							Mat temp;
							transform(opticalFlow, temp, cv::Matx12f(1,1));

							int histFlowSize = 50;    // bin size - need to determine which pixel threshold to use
						  float flowRange[] = {-25,25};
						  const float *flowRanges[] = {flowRange};
						  int flowChannels[] = {0, 1};

						  calcHist(&temp, 1, flowChannels, Mat(), histFlow, 1, &histFlowSize, flowRanges, true, false);
						  
						  histFlow.convertTo(histFlow, CV_64F);

						  cout << histFlow << endl;

						  feature.push_back(histFlow);
						}

						//classifier
						if(classify == true)
						{
							feature=feature.t();

							normalize(feature, feature, 1, 0, NORM_L1, -1, Mat());
							//cout << "New Feature" << endl << feature << endl;

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

							  rectangle(outputImage, p.tl(), p.br(), cv::Scalar(255,0,0), 3);

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

							  rectangle(outputImage, p.tl(), p.br(), cv::Scalar(255,0,0), 3);

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

										cout << "Target " << i << " Mahalanobis error from current image = " << mDistance << endl;
									}
									else
									{
										Mat invCovar;

										invert(covar,invCovar,DECOMP_SVD);

										mDistance = Mahalanobis(feature,mean,invCovar);

										cout << "Target " << i << " Mahalanobis error from current image = " << mDistance << endl;
									}
									mDistances.push_back(mDistance);
								}

								Mat dists = Mat(mDistances);
								
								cout << endl << endl << endl << endl << endl << endl << endl;
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
	    						targets[identifier].kalmanCorrect(center.x, center.y, timeSteps, rec.width, rec.height);

								  Rect p = targets[identifier].kalmanPredict();

						  		targets[identifier].updateFeatures(feature);

						  		targets[identifier].setCurrentCamera(cameraID);

								  rectangle(outputImage, p.tl(), p.br(), cv::Scalar(255,0,0), 3);

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

								  rectangle(outputImage, p.tl(), p.br(), cv::Scalar(255,0,0), 3);

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
				  rectangle(outputImage, r, Scalar(0,0,255), 2, 8, 0);
				}
		  }

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
	  //runOnSingleCamera(alphaFile, 0, 0); 
	  runOnSingleCamera(betaFile, 1, 0); 
	  //runOnSingleCamera(gammaFile, 2, 0); 
	  //runOnSingleCamera(deltaFile, 3, 0); 
  }
  //use this to run multithreaded - need to remove all imshow and named window calls, and uncomment all lock stuff and videowriter

  else
  {
	  if (pthread_mutex_init(&myLock, NULL) != 0)
	  {
	    printf("\n mutex init failed\n");
	    return 1;
	  }

	  std::thread t1(runOnSingleCamera, alphaFile, 0, 1);
	  std::thread t2(runOnSingleCamera, betaFile, 1, 1);
	  std::thread t3(runOnSingleCamera, gammaFile, 2, 1);
	  std::thread t4(runOnSingleCamera, deltaFile, 3, 1);
	  t1.join();
	  t2.join();
	  t3.join();
	  t4.join();
	  pthread_mutex_destroy(&myLock);
	}

  return 0;
}