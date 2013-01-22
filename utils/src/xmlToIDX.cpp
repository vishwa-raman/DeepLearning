// test.cpp
// Code that tests the various pieces of functionality we have for gaze tracking

#include <unistd.h>
#include <ios>
#include <iostream>
#include <fstream>
#include <string>

#include "Preprocess.h"

using namespace std;

// roiFunction
// We support the use of LOIs identified to cull images to smaller regions of interest
// (ROI) for use in locating future LOIs. This function is passed to the constructors
// of the filter and classifier classes. Those classes in turn call this function when
// they need a culled image. The input parameters are the original image, a frame
// annotation object that is annotated with all the LOIs that we have found before this
// function gets called. The offset parameter is an output parameter that contains the
// offset of the ROI within the image. The function returns a culled image object

IplImage* roiFunction(IplImage* image, FrameAnnotation& fa, 
		      CvPoint& offset, Annotations::Tag xmlTag) {
  offset.x = 0;
  offset.y = 0;

  CvPoint& location = fa.getLOI(xmlTag);
  offset.y = location.y - (Globals::roiHeight / 2);
  offset.x = location.x - (Globals::roiWidth / 2);

  // now check if the roi overflows the image boundary. If it does then
  // we move it so that it is contained within the image boundary
  if (offset.x + Globals::roiWidth > Globals::imgWidth)
    offset.x = Globals::imgWidth - Globals::roiWidth;
  if (offset.x < 0)
    offset.x = 0;
  if (offset.y + Globals::roiHeight > Globals::imgHeight)
    offset.y = Globals::imgHeight - Globals::roiHeight;
  if (offset.y < 0)
    offset.y = 0;

  cvSetImageROI(image, cvRect(offset.x, offset.y, Globals::roiWidth, Globals::roiHeight));
  IplImage* roi = cvCreateImage(cvGetSize(image), image->depth, image->nChannels);
  cvCopy(image, roi);
  cvResetImageROI(image);

  return roi;
}

int main(int argc, char** argv) {
  string outputFileName = "";
  vector<string> dataDirs;
  vector<string> testAnnotations;
  double trainingFraction = 0;
  double validationFraction = 0;
  double additionalValidationFraction = 0;
  double additionalTestFraction = 0;
  map<int, bool> statusFilter;
  map<int, bool> intersectionFilter;
  double binaryThreshold = 1;
  bool inBinaryFormat = false;
  bool useBins = false;

  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-o"))
      outputFileName = argv[i + 1];
    else if (!strcmp(argv[i], "-r"))
      trainingFraction = atof(argv[i + 1]);
    else if (!strcmp(argv[i], "-v"))
      validationFraction = atof(argv[i + 1]);
    else if (!strcmp(argv[i], "-d"))
      dataDirs.push_back(argv[i + 1]);
    else if (!strcmp(argv[i], "-f"))
      testAnnotations.push_back(argv[i + 1]);
    else if (!strcmp(argv[i], "-status"))
      statusFilter[atoi(argv[i + 1])] = true;
    else if (!strcmp(argv[i], "-inter"))
      intersectionFilter[atoi(argv[i + 1])] = true;
    else if (!strcmp(argv[i], "-usebins"))
      useBins = true;
    else if (!strcmp(argv[i], "-av"))
      additionalValidationFraction = atof(argv[i + 1]);
    else if (!strcmp(argv[i], "-at"))
      additionalTestFraction = atof(argv[i + 1]);
    else if (!strcmp(argv[i], "-b")) {
      binaryThreshold = atof(argv[i + 1]);
      inBinaryFormat = true;
    } else if (!strcmp(argv[i], "-h")) {
      cout << 
	"Usage: xmlToIDX -o <outputFileName> -r <training_fraction> -v <validation_fraction>" <<
	" -status <statusFilter> -inter <intersectionFilter> [-d <trainingDirectory>]+" << 
	" [-b binaryThreshold] [-usebins] [-f testAnnotationFileName]" <<
	" [-av <validation_fraction_for_additional_test_files>]" <<
	" [-at <test_fraction_for_additional_test_files>] [-h for usage]" << endl;
      return 0;
    }
  }

  if (outputFileName == "") {
    cout << 
      "Usage: xmlToIDX -o <outputFileName> -r <training_fraction> -v <validation_fraction>" <<
      " -status <statusFilter> -inter <intersectionFilter> [-d <trainingDirectory>]+" << 
      " [-b binaryThreshold] [-usebins] [-f testAnnotationFileName]" <<
      " [-av <validation_fraction_for_additional_test_files>]" <<
      " [-at <test_fraction_for_additional_test_files>] [-h for usage]" << endl;
    return -1;
  }

  CvPoint center;
  center.x = Globals::roiWidth / 2;
  center.y = Globals::roiHeight / 2;

  CvSize size;
  size.width = Globals::roiWidth;
  size.height = Globals::roiHeight;

  double scale = 0.3;

  try {
    Preprocess preprocess(outputFileName, size, scale, center,
			  statusFilter, intersectionFilter,
			  roiFunction, useBins, inBinaryFormat, binaryThreshold);

    for (unsigned int i = 0; i < dataDirs.size(); i++)
      preprocess.addTrainingSet(dataDirs[i]);
    for (unsigned int i = 0; i < testAnnotations.size(); i++)
      preprocess.addTestSet(testAnnotations[i], additionalValidationFraction,
			    additionalTestFraction);

    if (useBins)
      preprocess.generate(trainingFraction, validationFraction);
    else
      preprocess.generateSequences(trainingFraction, validationFraction);
  } catch (string err) {
    cout << err << endl;
  }

  return 0;
}
