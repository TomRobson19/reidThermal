class Person()
{
public:
	int lastSeen;
	void initKalman(float x, float y, float w, float h, int timeSteps);

	Point2f kalmanCorrect(float x, float y, int timeSteps, float w, float h, int timeSteps);

	Point2f kalmanPredict();

private:
	cv::KalmanFilter KF;
	cv::Mat_<float> measurement(6,1); 	
}