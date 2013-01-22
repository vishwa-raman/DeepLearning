#ifndef __PREPROCESS_H
#define __PREPROCESS_H

// Preprocess.h
// This file contains the definition of the Preprocess class. It generates
// data in the IDX file format, similar to the MNIST dataset

#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <float.h>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>
#include <stdlib.h>
#include <bitset>
#include <map>

// The standard OpenCV headers
#include <cv.h>
#include <highgui.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "Globals.h"
#include "Annotations.h"

using namespace std;
using namespace cv;

#include "Annotations.h"

// the ROI extraction function pointer type
typedef IplImage* (*roiFnT)(IplImage*, FrameAnnotation&, 
			    CvPoint&, Annotations::Tag xmlTag);

// an image and location pair type
typedef pair<IplImage*, CvPoint> ImgLocPairT;

class Preprocess {
 protected:
  string outputFileName;           // the name of the file for preprocessed data
  CvSize imgSize;                  // the size of the input image
  int length;                      // length of the image array
  roiFnT roiFunction;              // an optional function to get image ROI
  ofstream dataFile;               // the output file handle for data
  ofstream labelFile;              // the output file handle for labels
  unsigned int nSamples;           // the number of samples written
  double scaleFactor;              // the scale factor for the final image data
  bool inBinaryFormat;             // flag to generate pixels in {0, 1} instead
                                   // of in [0, 1]
  double binaryThreshold;          // the threshold for choosing 0 or 1 in binary format
  bool useBins;                    // whether or not to bin frames based on sector

  // filters used to pick images that match specific car status or intersection type
  map<int, bool> statusFilter;          
  map<int, bool> intersectionFilter;

  // map used to associate an image filename with its frame annotation data
  vector<pair<string, FrameAnnotation*> > fileToAnnotations;

  // map used to associate an image filename with its frame annotations data for
  // additional validation and test data
  vector<pair<string, FrameAnnotation*> > fileToAnnotationsForValidation;
  vector<pair<string, FrameAnnotation*> > fileToAnnotationsForTest;

 public:
  // Constructor used to construct a filter
  Preprocess(string output, CvSize size, double scale, CvPoint& windowCenter, 
	     map<int, bool>& statusFilter,
	     map<int, bool>& intersectionFilter,
	     roiFnT roiFunction = 0, bool useBins = false,
	     bool inBinaryFormat = false, double binaryThreshold = 0.5);
  Preprocess(CvSize size, double scale, CvPoint& windowCenter, 
	     roiFnT roiFunction = 0);

  virtual ~Preprocess();

  // main methods
  virtual void addTrainingSet(string trainingDirectory);
  virtual void addTestSet(string annotationsFileName, 
			  double validationFraction, double testFraction);
  virtual void setAffineTransforms() {
    doAffineTransforms = true;
  }

  // method to generate the output files with percentages of images to
  // be used for training, validation and testing
  virtual void generate(double training, double validation, bool doWriteTests = false);

  // method to generate the output files with percentages of images to
  // be used for training, validation and testing. The difference here
  // is that we have to preserve the sequencing of image frames. We do
  // not randomize frames but instead pick sub-sequences to cover the
  // desired partition
  virtual void generateSequences(double training, double validation);

  // method to preprocess images
  virtual double* preprocessImage(IplImage* image);

  // method to generate an image vector after preprocessing
  virtual IplImage* generateImageVector(IplImage* image);

  CvSize getSize() { return imgSize; }

  // test methods
  static void showImage(string window, IplImage* image) {
    cvNamedWindow((const char*)window.c_str(), CV_WINDOW_NORMAL | CV_WINDOW_AUTOSIZE);
    cvShowImage((const char*)window.c_str(), image);
    cvWaitKey(1);
  }
  void showRealImage(string window, double* data) {
    IplImage* temp = cvCreateImage(imgSize, IPL_DEPTH_64F, 1);
    int step = temp->widthStep;
    double* imageData = (double*)temp->imageData;
    for (int i = 0; i < imgSize.height; i++) {
      for (int j = 0; j < imgSize.width; j++) {
	(*imageData++) = data[i * imgSize.width + j];
      }
      imageData += step /sizeof(double) - imgSize.width;
    }  
    cvNamedWindow((const char*)window.c_str(), CV_WINDOW_NORMAL | CV_WINDOW_AUTOSIZE);
    cvShowImage((const char*)window.c_str(), temp);
    cvWaitKey(1);
    cvReleaseImage(&temp);
  }

 protected:
  void openFiles(string simpleName);
  void closeFiles(int nSamples);
  void generate(int startIndex, int nSamples, string simpleName, bool doWriteTests = false,
		vector<pair<string, FrameAnnotation*> >* additionalPairs = 0);
  void generateSequences(int startIndex, int nChunks, 
			 vector<int>& indices, string simpleName);
  void update(string filename, FrameAnnotation* fa, int& samples);
  vector<ImgLocPairT>& getAffineTransforms(IplImage* image, CvPoint& location);
  void destroyAffineTransforms(vector<ImgLocPairT>& imgLocPairs);
  double* createWindow(CvPoint& location, double xSpread, double ySpread);
  void applyWindow(IplImage* src, double* window, double* dest);

  // if the following is set then for each update operation during
  // filter generation, we take a set of affine transformations of
  // each image and update the filter using the original image and
  // the affine transformations of the image
  bool doAffineTransforms;

  // used to store a set of images after doing small perturbations of the
  // images using affine transformations for filter update
  vector<ImgLocPairT> transformedImages;

  double* window;
  IplImage* realImg;
  IplImage* tempImg;
  double* imageBuffer;

  CvPoint windowCenter;
};

#endif // __PREPROCESS_H
