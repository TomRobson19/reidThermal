#include "common.h"

using namespace std;
using namespace cv;

BgmModule::BgmModule(std::shared_ptr<Data> _data){
  data = _data;
  lastFrameTime = getTickCount(); //start bgm frame grabbing timer.

  //initialise MOG background model with black image, to see when the model is learnt.
  data->LogMsg(MSG,"Background model initialisation...");
  mogModel = createBackgroundSubtractorMOG2(data->setup.mogHistory,data->setup.mogVarThre,data->setup.bMogShadow);
  data->LogMsg(MSG,"Background model initialised!");
}

void BgmModule::ResetBgModel(){
  mogModel = createBackgroundSubtractorMOG2(data->setup.mogHistory,data->setup.mogVarThre,data->setup.bMogShadow);
}

void BgmModule::Loop(){
  while(data->status.keepProcessing){
    //sleep for a bit before starts waiting for a new frame.
    int how_much_sleep = FpsToInterval(data->setup.inputFps) - MsTimeSinceLast(lastFrameTime);
    tmpFrameIt = data->frameIt;
    this_thread::sleep_for(chrono::milliseconds(std::max(2,how_much_sleep)));
    tmpFrameIt = data->frameIt;

    if(tmpFrameIt > frameIt){
      frameIt = tmpFrameIt;
      lastFrameTime = getTickCount(); //start bgm frame grabbing timer
      cFrame = data->CopyFrame(frameIt);
      tmpImg = cFrame.inImg;
      resize(tmpImg, img, Size(), 0.5, 0.5, CV_INTER_AREA);

      //process images
      Update();
      data->WriteBgMaskFrame(frameIt, morphmogFGmaskUpsized);
      if(data->status.resetBgModel){
        data->LogMsg(MSG,"Reseting Background Model!");
        ResetBgModel();
        data->status.resetBgModel = false;
      }
    }
  }
}

void BgmModule::Update(){
  mogModel->apply(img, mogFGmask,(double)(data->setup.mogLearning/10000.0));
  mogModel->getBackgroundImage(mogBG);
  //morphology
  erode(mogFGmask, morphmogFGmask, Mat(), Point(), data->setup.morphEroSize);
  dilate(morphmogFGmask, morphmogFGmask, Mat(), Point(), data->setup.morphDilSize);
  data->status.bganomaly = DetectAnomaliesAndFillMask(morphmogFGmask);

  //up size
  resize(morphmogFGmask, morphmogFGmaskUpsized, Size(), 2, 2, CV_INTER_LINEAR);

}

bool BgmModule::DetectAnomaliesAndFillMask(Mat& bgimg){

  float numPix = bgimg.cols * bgimg.rows;
  int white = 0;

  if (bgimg.type()==0) {
    MatConstIterator_<uchar> mIt;
    MatConstIterator_<uchar> itEnd = bgimg.end<uchar>();
    for ( mIt = bgimg.begin<uchar>(); mIt != itEnd; ++mIt){
      if( (*mIt) == 255)
        white++;
    }
  }
  else if (bgimg.type()==16) {
    MatConstIterator_<Vec3b> mIt;
    MatConstIterator_<Vec3b> itEnd = bgimg.end<Vec3b>();
    for ( mIt = bgimg.begin<Vec3b>(); mIt != itEnd; ++mIt){
      if( (*mIt)[0] == 255 || (*mIt)[1] == 255 ||(*mIt)[2] == 255)
        white++;
    }
  }

  if (100*((float)white/numPix) > data->setup.bgmodelTamperThre){
    bgimg.setTo(Scalar::all(255));
    return true; //anomalies found return true
  }

  return false;
}
