Instructions

Requirements - OpenCV 3, CMake
Run the following commands:

cmake CMakeLists.txt
make

This creates the "testing" and "main" executables.



"testing" runs the code used for feature experimentation, returning Mahalanobis distances from person 0 on the command line

./testing -d=1 -f=1 -c=1

-d = dataset - possible values are 1,2,3

-f = feature - 1 - HuMoments, 2 - HistogramOfIntensities, 3 - HistogramofOrientedGradients, 4 - CorrelogramVariant, 5 - CorrelogramOriginal, 6 - Flow, 7 - HistogramOfFlow, 10 - Combination of 2,5,6

-c = person classifier/detector - 1 - HOG, 2 - Haar



"main" runs the full Re-Identification System

./main -d=1 -t=1

-d = dataset - possible values are 1,2,3

-t = test mode - 1 - on, runs one camera at a time and shows live using imshow, 2 - off, runs on multiple cameras simultaneously, creates video files of results per camera and a file of results concanenated together. (existing files are overwritten when code is run)