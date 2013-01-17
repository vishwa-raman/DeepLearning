// Globals.h
// File that contains the set of all global constants and typedefs. We define a 
// class with static members for each constant that will be used at application
// scope

#ifndef __GLOBALS_H
#define __GLOBALS_H

#include <string>

// The standard OpenCV headers
#include <cv.h>
#include <highgui.h>

// The OMP stuff
#include <omp.h>

// fftw3 stuff
#include <fftw3.h>

using namespace std;

class Globals {
 public:
  Globals() { }
  ~Globals() { }

  static int imgWidth;                    // default image width that we handle
  static int imgHeight;                   // image height
  static int roiWidth;                    // roi width
  static int roiHeight;                   // roi height
  static int maxDistance;                 // max pixel distance for normalization
  static int maxAngle;                    // max angle for normalization
  static int maxArea;                     // max area of the L, R and N triangle
  static int binWidth;                    // the bin width for binning annotations
  static int gaussianWidth;               // width of the gaussian
  static int psrWidth;                    // width of window to compute PSR
  static int nPastLocations;              // number of past locations for smoothing
  static int noseDrop;                    // approx. drop below the eyes for the nose

  static int smallBufferSize;             // small stack buffer size
  static int midBufferSize;               // mid stack buffer size
  static int largeBufferSize;             // large stack buffer size
  static int nSequenceLength;             // the length of frame sequences we need

  static unsigned numZones;               // number of zones

  static double learningRate;             // the learning rate for online filters
  static double initialGaussianScale;     // the gaussian scale for the face filter

  // the window function is computed as image width times the x scale and
  // the image height times the y scale
  static double windowXScale;             // the X scale factor for the window function
  static double windowYScale;             // the Y scale factor for the window function

  static string annotationsFileName;      // the name of the annotations file
  static string modelNamePrefix;          // prefix for SVM model names
  static string faceFilter;               // name of the face filter
  static string leftEyeFilter;            // name of left eye filter
  static string rightEyeFilter;           // name of right eye filter
  static string noseFilter;               // name of nose filter
  static string paramsFileName;           // name of parameters file
  static string configFileName;           // name of the config file

  static void setRoiSize(CvSize size) {
    roiWidth = size.width;
    roiHeight = size.height;
  }
};

#endif
