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
#include "moghog.hpp"


'''
Need to ensure that these target vectors are threadsafe if Im going to handle the 
different cameras with multithreading, as there may be access conflicts
'''
std::vector<Person> activeTargets;
std::vector<Person> inactiveTargets;
std::vector<string> videos;

int timeSteps = 0;

int main(int argc, char** argv)
{
	for (int i=1; i<argc; i++)
	{
		videos.push_back(argv[i]);
		cout << argv[i] << '\n';
	}	



	//experiment with threading for running this

	return 1;
}