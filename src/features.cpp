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

MatND histogramOfIntensities(Mat roi)
{
	MatND hist;
	int histSize = 256;    // bin size
	float range[] = { 0, 255 };
	const float *ranges[] = { range };
	int channels[] = {0, 1};

	calcHist(&roi, 1, channels, Mat(), hist, 1, &histSize, ranges, true, false);

	// cout << hist << endl;
	// cout << endl;
	// cout << endl;
	return hist;
}

double * calculateHuMoments(vector<Point> contours)
{
    Moments contourMoments;
    double huMoments[7];

    contourMoments = moments(contours);

    HuMoments(contourMoments, huMoments); 

    // for (int i=0; i<7; i++)
    // {
    //   cout << huMoments[i] << endl;
    // }
    // cout << endl;
    // cout << endl;
    return huMoments;
}