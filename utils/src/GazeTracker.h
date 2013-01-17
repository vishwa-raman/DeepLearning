// GazeTracker.h
// File that contains the definition of class GazeTracker. This class is used to 
// perform the following operations,
// 1. Learn filters for LOIs
// 2. Apply a filter to an image and get the co-ordinates of the LOI
// 3. Train SVM models for given data
// 4. Apply SVM models to classify gaze zones

#ifndef __GAZETRACKER_H
#define __GAZETRACKER_H

#include <sys/types.h>
#include <dirent.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>

#include "Filter.h"
#include "OnlineFilter.h"
#include "Location.h"
#include "Annotations.h"
#include "Trainer.h"
#include "Classifier.h"

class GazeTracker {
 public:
  static Trainer::KernelType kernelType;  // The SVM Kernel type

  static CvSize roiSize;                  // ROI size

  string outputDirectory;                 // The directory for all generated info

  // Construction and destruction
  GazeTracker(string outputDirectory, bool online);
  virtual ~GazeTracker();

  // create filters
  void addFrameSet(string directory);
  void createFilters();

  // train models for SVM
  void addTrainingSet(string directory);
  void train();

  // get the gaze zone given an image
  int getZone(IplImage* image, double& confidence, FrameAnnotation& fa);

  // get error for a given filter used by the gaze tracker
  double getFilterAccuracy(string trainingDirectory, Annotations::Tag xmlTag,
			   Classifier::ErrorType errorType);
  // get classification error
  pair<double, string> getClassifierAccuracy(string trainingDirectory);

  // show annotations
  void showAnnotations();

  // Function used to compute an image ROI based on partial frame annotations
  // As we recognize LOIs, we use those location to potentially cull the 
  // input image and use a reduced ROI for subsequent recognition
  static IplImage* roiFunction(IplImage* image, FrameAnnotation& fa, 
			       CvPoint& offset, Annotations::Tag xmlTag);

 private:
  bool isOnline;                           // true when using online filters
  string svmPath;

  // the center of the face for classification
  CvPoint faceCenter;

  // The classifier
  Classifier* classifier;

  // The frame sets in the training data for filter generation
  vector<string> frameSetDirectories;

  // The training set directories for SVM model generation
  vector<string> trainingSetDirectories;

  // location extractors
  Location* leftEyeExtractor;
  Location* rightEyeExtractor;
  Location* noseExtractor;

  // create a classifier
  void createClassifier();

  // window center functions
  CvPoint getWindowCenter();
  CvPoint computeWindowCenter(string trainingDirectory = "");
  void updateWindowCenter(string trainingDirectory, 
			  int& minX, int& maxX, int& minY, int& maxY);
  void readConfiguration();
};

#endif
