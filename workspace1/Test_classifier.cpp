/**
 * @file objectDetection.cpp
 * @author A. Huaman ( based in the classic facedetect.cpp in samples/c )
 * @brief A simplified version of facedetect.cpp, show how to load a cascade classifier and how to find objects (Face + eyes) in a video stream
 */
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay( Mat frame );

/** Global variables */
//-- Note, either copy these two files from opencv/data/haarscascades to your current folder, or change these locations
//String speed_marker_name = "../Training_35/trained_classifiers/banana_classifier.xml";
String speed_marker_name = "../Training_35/classifier/cascade.xml";
CascadeClassifier speed_marker_cascade;
string window_name = "Capture - speed_marker";
RNG rng(12345);

/**
 * @function main
 */
int main( void )
{ 
  //frameSource = FrameSource::video(videoFile);
  //VideoCapture capture("/home/work/Documents/Workspace/workspace1/Capture_9.mp4");  
  VideoCapture capture("/media/3637-3366/videos/Capture 9 (8-8-2014 10-53 AM).mp4");
  if (!capture.isOpened()){
      cout << "file not opened" << endl;
      return -1;
  }
  Mat frame;
  //cout << capture.get(CV_CAP_PROP_FRAME_WIDTH) << endl;

  //-- 1. Load the cascades
  if( !speed_marker_cascade.load( speed_marker_name ) ){ printf("--(!)Error loading\n"); return -1; };

  //-- 2. Read the video stream
  //capture.open( -1 );

  if( capture.isOpened() )
  { 
    //frame = imread("/home/work/Documents/Workspace/Training_35/positive_images/Capture_5_Sec_1.034368.jpg",1);
    //frame = imread("Positive35.jpg",1);

    for(;;)
    {
      capture >> frame;


      if( !frame.empty() )
       {  detectAndDisplay( frame );}
      else
       { printf(" --(!)  No captured frame -- Break!"); //break;
       }

      //waitKey(5000);
      //if( (char)c == 'c' ) { break; }
        if( waitKey (30) >= 0){}
    }
  }
  else
  {
      cout<< "Video not found"<<endl;
  }

  return 0;
}

/**
 * @function detectAndDisplay
 */
void detectAndDisplay( Mat frame )
{
   std::vector<Rect> speed_marker;
   Mat frame_gray;

   cvtColor( frame, frame_gray, COLOR_BGR2GRAY );

   equalizeHist( frame_gray, frame_gray );

   //-- Detect speed_marker
   speed_marker_cascade.detectMultiScale( frame_gray, speed_marker, 1.05,10, CV_HAAR_SCALE_IMAGE,Size(150,150));
                   cout << "I am here" <<endl;
   for( size_t i = 0; i < speed_marker.size(); i++ )
    {
      Point center( speed_marker[i].x + speed_marker[i].width/2, speed_marker[i].y + speed_marker[i].height/2 );
      ellipse( frame, center, Size( speed_marker[i].width/2, speed_marker[i].height/2), 0, 0, 360, Scalar( 255, 0, 255 ), 2, 8, 0 );

      Mat faceROI = frame_gray( speed_marker[i] );

    }
   //-- Show what you got


   imshow( window_name, frame );

}
