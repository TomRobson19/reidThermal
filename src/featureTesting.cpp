/*
Run like this : 
./main -d=1 -f=1 -c=1
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
//pthread_mutex_t myLock;

static const char* keys =
    ("{h help       | | Help Menu}"
     "{d dataset    | | Dataset - 1, 2, 3}"
     "{f feature    | | 1 - Hu, 2 - Hist, 3 - HOG, 4 - Correlogram, 5 - Flow}"
     "{c classifier | | 1 - HOG, 2 - Haar}");
 
int runOnSingleCamera(String file, int featureToUse, int classifier, int cameraID) 
{
	//VideoWriter video(file+"results.avi",CV_FOURCC('M','J','P','G'),10, Size(640,480),true);

	int timeSteps = 0;

  string windowName = file; // window name

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

  // if command line arguments are provided try to read image/video_name
  // otherwise default to capture from attached H/W camera
  if((cap.open(file) == true))
  {
		// create window object (use flag=0 to allow resize, 1 to auto fix size)
		namedWindow(windowName, 1);

		// create background / foreground Mixture of Gaussian (MoG) model
		Ptr<BackgroundSubtractorMOG2> MoG = createBackgroundSubtractorMOG2(500,25,false);

		HOGDescriptor hog;
		hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

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

		  //imshow("old foreground", foreground);

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
		  //imshow("foreground", foreground);

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

				  Mat roi = outputImage(r);

				  if (classifier == 1)
				  {
						//changing last parameter helps deal with multiple rectangles per person
						if (cameraID == 3)
						{
							//this doesn't work
							hog.detectMultiScale(roi, found, 0, Size(2,2), Size(16,16), 1.05, 5);
						}
						else
						{
							hog.detectMultiScale(roi, found, 0, Size(16,16), Size(64,64), 1.05, 5);
						}
				  }
				  else 
				  {
				  	if (cameraID == 3)
				  	{
				  		cascade.detectMultiScale(roi, found, 1.1, 4, CV_HAAR_DO_CANNY_PRUNING, cvSize(32,64));
				  	}
				  	else
				  	{
				  		cascade.detectMultiScale(roi, found, 1.1, 4, CV_HAAR_DO_CANNY_PRUNING, cvSize(64,128));
				  	}
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

						imshow("roi", regionOfInterest);

						double huMoments[7];
						vector<double> hu(7);
						Mat hist;
						vector<float> descriptorsValues;

						Mat feature;

						bool classify = true;

						if(featureToUse == 1) //HuMoments
						{
						  vector<vector<Point> > contoursHu;
						  vector<Vec4i> hierarchyHu;

						  findContours(regionOfInterest, contoursHu, hierarchyHu, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

						  double largestSize,size;
						  int largestContour;

						  for(int i = 0; i < contoursHu.size(); i++)
						  {
								size = contoursHu[i].size();

								if(size > largestSize)
								{
								  largestSize = size;
								  largestContour = i;
								}
						  }
						  Moments contourMoments;

						  contourMoments = moments(contoursHu[largestContour]);

						  HuMoments(contourMoments, huMoments);

						  hu.assign(huMoments,huMoments+7);

              feature = Mat(hu);
              feature = feature.t();
						}
						else if(featureToUse == 2) //HistogramOfIntensities
						{
						  int histSize = 16;    // bin size - need to determine which pixel threshold to use
						  float range[] = {0,255};
						  const float *ranges[] = {range};
						  int channels[] = {0, 1};

						  calcHist(&regionOfInterest, 1, channels, Mat(), hist, 1, &histSize, ranges, true, false);

						  feature = hist.clone();
						  feature = feature.t();
						}

						else if(featureToUse == 3) //HOG
						{
						  //play with these parameters to change HOG size 
						  cv::HOGDescriptor descriptor(Size(64, 128), Size(16, 16), Size(16, 16), Size(16, 16), 4, -1, 0.2, true, 64);

						  descriptor.compute(regionOfInterest, descriptorsValues);

						  feature = Mat(descriptorsValues);
						  feature = feature.t();
						}

						else if(featureToUse == 4) //Correlogram
						{					
							Mat distanceSum(8,8,CV_64F);
							Mat correlogram(8,8,CV_64F);
							Mat occurances(8,8,CV_8U);

							int xIntensity, yIntensity;

							for(int i = 0; i<regionOfInterest.rows; i++)
							{
								for(int j = 0; j<regionOfInterest.cols; j++)
								{
									xIntensity = floor(regionOfInterest.at<unsigned char>(i,j)/32);

									for(int k = i; k<regionOfInterest.rows; k++)
									{
										for(int l = 0; l<regionOfInterest.cols; l++)
										{
											if((k == i && l > j) || k > i)
											{
												yIntensity = floor(regionOfInterest.at<unsigned char>(k,l)/32);
											
												distanceSum.at<double>(xIntensity,yIntensity) += (norm(Point(i,j)-Point(k,l)));
												distanceSum.at<double>(yIntensity,xIntensity) += (norm(Point(k,l)-Point(i,j)));
												
												occurances.at<unsigned char>(xIntensity,yIntensity) += 1;
												occurances.at<unsigned char>(yIntensity,xIntensity) += 1;
											}
										}
									}
								}
							}
							//average it out
							for(int i = 0; i<distanceSum.rows; i++)
							{
								for(int j = 0; j<distanceSum.cols; j++)
								{
									if(occurances.at<unsigned char>(i,j) > 0 and distanceSum.at<double>(i,j) > 0.0)
									{
										correlogram.at<double>(i,j) = distanceSum.at<double>(i,j)/occurances.at<unsigned char>(i,j);
									}
									else 
									{
										correlogram.at<double>(i,j) = 0;
									}
								}
							}
							feature = correlogram.reshape(1,1);
						}
						else if(featureToUse == 5) //Flow
						{
							//this is going to require multiple frames, so need some way of storing previous frames
							//Needs to be a Histogram of Flow
							//OpenCV has many flow functions, still don't know which one to use
							//Based on StackOverflow answers, think it will be PyrLK

							classify = false;
							Mat opticalFlow;

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
								Mat temp2;
								for(int i = 8; i<regionOfInterest.rows; i+=8)
								{
									for(int j = 8; j< regionOfInterest.cols; j+=8)
									{
										temp.push_back(opticalFlow.at<Point2f>(i,j));
									}
								}
								transform(temp, temp2, cv::Matx12f(1,1));

								bool useHistogram = false;

								if(useHistogram == true)
								{
									int histSize = 60;    // bin size - need to determine which pixel threshold to use
								  float range[] = {-30,30};
								  const float *ranges[] = {range};
								  int channels[] = {0, 1};

								  calcHist(&temp2, 1, channels, Mat(), hist, 1, &histSize, ranges, true, false);
								  feature = hist.clone();
								  feature = feature.t();
								}
							  else
							  {
							  	feature = temp2.reshape(1,1);
								}
							}
						}
						if(classify == true)
						{
							feature.convertTo(feature, CV_64F);

							normalize(feature, feature, 1, 0, NORM_L1, -1, Mat());
							//cout << "New Feature" << endl << feature << endl;

							//LOCK
							//pthread_mutex_lock(&myLock);

							//classify first target
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
							}
							else
							{
								vector<double> mDistances;
								bool singleEntry = false;

								for(int i = 0; i<targets.size(); i++)
								{
									if(targets[i].getFeatures().rows == 1)
									{
										singleEntry = true;
									}
								}

								for(int i = 0; i<targets.size(); i++)
								{
									Mat covar, mean;
									Mat data = targets[0].getFeatures();
									
									calcCovarMatrix(data,covar,mean,CV_COVAR_NORMAL|CV_COVAR_ROWS);

									//cout << covar << endl;

									// cout << i << " data" << endl << data << endl;
									// cout << i << " Covar" << endl << covar << endl;
									// cout << i << " mean" << endl << mean << endl;

									double mDistance;

									if(singleEntry == false)
									{
										Mat invCovar;

										invert(covar,invCovar,DECOMP_SVD);

										mDistance = Mahalanobis(feature,mean,invCovar);

										//cout << i << " Mahalanobis Distance" << endl << mDistance << endl;
										if(i==0)
										{
											cout << mDistance << endl;
										}
									}
									else
									{
										mDistance = norm(feature,mean,NORM_L1);

										//cout << i << " Norm Distance" << endl << mDistance << endl;
										//cout << "norm" << endl;
									}
									mDistances.push_back(mDistance);
								}

								Mat test = Mat(mDistances); 
								//cout << "Distances" << endl << test << endl;

								double sum = 0.0;
								for(int i = 0; i<mDistances.size(); i++)
								{
									sum += mDistances[i];
								}
								for(int i = 0; i<mDistances.size(); i++)
								{
									mDistances[i] = sum/mDistances[i];
								}

								normalize(mDistances,mDistances,1,0,NORM_L1,-1,Mat());

								Mat probabilities = Mat(mDistances);

								//cout << "Probabilities" << endl << probabilities << endl;
													
								//special case to classify second target
		    				if(targets.size() == 1)
		    				{
		    					if(fabs(center.x-targets[0].getLastPosition().x)<100 and fabs(center.y-targets[0].getLastPosition().y)<100
		    					   and cameraID == targets[0].getCurrentCamera() and timeSteps - targets[0].getLastSeen() < 100)
		    					{
		    						targets[0].kalmanCorrect(center.x, center.y, timeSteps, rec.width, rec.height);

									  Rect p = targets[0].kalmanPredict();

							  		targets[0].updateFeatures(feature);

							  		targets[0].setCurrentCamera(cameraID);

									  rectangle(outputImage, p.tl(), p.br(), cv::Scalar(255,0,0), 3);

									  char str[200];
									  sprintf(str,"Person %d",targets[0].getIdentifier());

									  putText(outputImage, str, center, FONT_HERSHEY_SIMPLEX,1,(0,0,0));
		    					}
		    					else
		    					{
		    						cout << "////////////////////////////////" << endl;
		    					 	Person person(1, center.x, center.y, timeSteps, rec.width, rec.height);

									  // person.kalmanCorrect(center.x, center.y, timeSteps, rec.width, rec.height);
									  
									  // Rect p = person.kalmanPredict();

							  		// person.updateFeatures(feature);

							  		// person.setCurrentCamera(cameraID);

									  // rectangle(outputImage, p.tl(), p.br(), cv::Scalar(255,0,0), 3);

									  // char str[200];
									  // sprintf(str,"Person %d",person.getIdentifier());

									  // putText(outputImage, str, center, FONT_HERSHEY_SIMPLEX,1,(0,0,0));

									   targets.push_back(person);
		    					}
		    				}

		    				else
		    				{
		    			// 		double greatestProbability = 0.0;
		    			// 		int identifier = 0;

		    			// 		double min, max;
									// Point min_loc, max_loc;
									// minMaxLoc(probabilities, &min, &max, &min_loc, &max_loc);

									// greatestProbability = max;
									// identifier = max_loc.y;

									// //cout << greatestProbability << " at " << identifier << endl;

									// //cout << (1.2/targets.size()) << endl;

									// //change this value
		    			// 		if(greatestProbability >= (1.2/targets.size()))
		    			// 		{
		    			// 			targets[identifier].kalmanCorrect(center.x, center.y, timeSteps, rec.width, rec.height);

									//   Rect p = targets[identifier].kalmanPredict();

							  // 		targets[identifier].updateFeatures(feature);

							  // 		targets[identifier].setCurrentCamera(cameraID);

									//   rectangle(outputImage, p.tl(), p.br(), cv::Scalar(255,0,0), 3);

									//   char str[200];
									//   sprintf(str,"Person %d",targets[identifier].getIdentifier());

									//   putText(outputImage, str, center, FONT_HERSHEY_SIMPLEX,1,(0,0,0));
		    			// 		}
		    			// 		else
		    			// 		{
		    			// 			int identifier = targets.size();
									//   Person person(identifier, center.x, center.y, timeSteps, rec.width, rec.height);

									//   person.kalmanCorrect(center.x, center.y, timeSteps, rec.width, rec.height);
									  
									//   Rect p = person.kalmanPredict();

							  // 		person.updateFeatures(feature);

							  // 		person.setCurrentCamera(cameraID);

									//   rectangle(outputImage, p.tl(), p.br(), cv::Scalar(255,0,0), 3);

									//   char str[200];
									//   sprintf(str,"Person %d",person.getIdentifier());

									//   putText(outputImage, str, center, FONT_HERSHEY_SIMPLEX,1,(0,0,0));

									//   targets.push_back(person);
		    			// 		}
		    				}
		    			}
		    		}
		    		//UNLOCK
	    			//pthread_mutex_unlock(&myLock);
				  }
				  rectangle(outputImage, r, Scalar(0,0,255), 2, 8, 0);
				}
		  }
		  // display image in window
		  imshow(windowName, outputImage);
		  //video.write(outputImage);

	  key = waitKey((int) std::max(2.0, EVENT_LOOP_DELAY - (((getTickCount() - timeStart) / getTickFrequency())*1000)));

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
  int featureToUse = cmd.get<int>("feature");
  int classifier = cmd.get<int>("classifier");

  String directory = "data/Dataset" + to_string(datasetToUse);

  //other option
  // vector<String> filenames;

  // glob(directory,filenames);

  // #pragma omp parallel for
  // for(size_t i = 0; i < filenames.size(); i++)
  // {
  //     runOnSingleCamera(filenames[i], featureToUse, classifier, i);
  // }

  String alphaFile = directory + "/alphaInput.webm";
  String betaFile = directory + "/betaInput.webm";
  String gammaFile = directory + "/gammaInput.webm";
  String deltaFile = directory + "/deltaInput.webm";

  //runOnSingleCamera(alphaFile, featureToUse, classifier, 0); 
  runOnSingleCamera(betaFile, featureToUse, classifier, 1); 
  //runOnSingleCamera(gammaFile, featureToUse, classifier, 2); 
  //runOnSingleCamera(deltaFile, featureToUse, classifier, 3); 

  //use this to run multithreaded - need to remove all imshow and named window calls, and uncomment all lock stuff and videowriter

  // if (pthread_mutex_init(&myLock, NULL) != 0)
  // {
  //   printf("\n mutex init failed\n");
  //   return 1;
  // }

  // std::thread t1(runOnSingleCamera, alphaFile, featureToUse, classifier,0);
  // std::thread t2(runOnSingleCamera, betaFile, featureToUse, classifier,1);
  // std::thread t3(runOnSingleCamera, gammaFile, featureToUse, classifier,2);
  // std::thread t4(runOnSingleCamera, deltaFile, featureToUse, classifier,3);
  // t1.join();
  // t2.join();
  // t3.join();
  // t4.join();
  // pthread_mutex_destroy(&myLock);

  return 0;
}