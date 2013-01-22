// test.cpp
// Code that tests the various pieces of functionality we have for gaze tracking

#include <unistd.h>
#include <ios>
#include <iostream>
#include <fstream>
#include <string>

#include "Preprocess.h"

using namespace std;

vector<IplImage*> g_weights;
vector<IplImage*> g_biases;
vector<IplImage*> g_results;

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

void loadModel(string modelFileName) {
  ifstream ifs;
  string line;
  int layer = 1;

  ifs.open(modelFileName.c_str());
  while (ifs.good()) {
    getline(ifs, line);
    if (line == "") break;

    char* str = (char*)line.c_str();
    if (strchr(str, ',')) {
      int rows, cols;
      sscanf(str, "%d,%d", &rows, &cols);
      cout << "Layer " << layer << ". (" << rows << ", " << cols << ")." << endl;
      CvSize size;
      size.height = rows;
      size.width = cols;
      IplImage* W = cvCreateImage(size, IPL_DEPTH_64F, 1);
      double* imageData = (double*)W->imageData;
      int step = W->widthStep;
      for (int i = 0; i < size.height; i++) {
	getline(ifs, line);
	str = (char*)line.c_str();
	char* token = strtok(str, ",");
	for (int j = 0; j < size.width; j++) {
	  (*imageData++) = atof(token);
	  token = strtok(NULL, ",");
	}
	imageData += step / sizeof(double) - size.width;
      }
      g_weights.push_back(W);

      size.height = 1;
      size.width = cols;
      IplImage* result = cvCreateImage(size, IPL_DEPTH_64F, 1);
      g_results.push_back(result);
    } else {
      int elems;
      sscanf(str, "%d", &elems);
      cout << "Layer " << layer << ". Bias length (" << elems << ")." << endl;

      CvSize size;
      size.height = 1;
      size.width = elems;
      IplImage* b = cvCreateImage(size, IPL_DEPTH_64F, 1);
      double* imageData = (double*)b->imageData;
      int step = b->widthStep;
      for (int i = 0; i < size.height; i++) {
	getline(ifs, line);
	str = (char*)line.c_str();
	char* token = strtok(str, ",");
	for (int j = 0; j < size.width; j++) {
	  (*imageData++) = atof(token);
	  token = strtok(NULL, ",");
	}
	imageData += step / sizeof(double) - size.width;
      }
      g_biases.push_back(b);
      layer++;
    }
  }
}

int classify(IplImage* image) {
  IplImage* input = image;
  int nLayers = g_weights.size();
  for (int i = 0; i < nLayers; i++) {
    IplImage* W = g_weights[i];
    IplImage* b = g_biases[i];
    IplImage* result = g_results[i];

    cvMatMulAdd(input, W, b, result);
    CvSize size = cvGetSize(result);
    double* imageData = (double*)result->imageData;
    for (int j = 0; j < size.width; j++) {
      if (i == nLayers - 1)
	*imageData = exp(*imageData);
      else
	*imageData = tanh(*imageData);
      imageData++;
    }
    if (i == nLayers - 1) {
      CvScalar sum = cvSum(result);
      double scale = 1.0 / sum.val[0];
      cvConvertScale(result, result, scale, 0.0);
    }
    input = result;
  }

  double min, max;
  CvPoint minIndex, maxIndex;
  cvMinMaxLoc(input, &min, &max, &minIndex, &maxIndex);
  //  cout << "Min = " << min << ", Max = " << max << endl;
  //  cout << "Min index = (" << minIndex.x << ", " << minIndex.y << ")" << endl;
  //  cout << "Max index = (" << maxIndex.x << ", " << maxIndex.y << ")" << endl;
  return maxIndex.x + 1;
}

int main(int argc, char** argv) {
  string modelsFileName = "";
  string imageFileName = "";
  string imageDirectory = "";
  int x = Globals::imgWidth / 2;
  int y = Globals::imgHeight / 2;

  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-i"))
      imageFileName = argv[i + 1];
    else if (!strcmp(argv[i], "-d"))
      imageDirectory = argv[i + 1];
    else if (!strcmp(argv[i], "-m"))
      modelsFileName = argv[i + 1];
    else if (!strcmp(argv[i], "-c")) {
      char* str = argv[i + 1];
      char* token = strtok(str, "(),");
      if (token)
	x = atoi(token);
      token = strtok(NULL, "(),");
      if (token)
	y = atoi(token);
    }
    else if (!strcmp(argv[i], "-h")) {
      cout << 
	"Usage: classify -i <imageFileName> -m <modelsFileName>" << endl;
      return 0;
    }
  }

  if (modelsFileName == "") {
    cout << 
      "Usage: classify -i <imageFileName> -m <modelsFileName>" << endl;
    return -1;
  }

  CvPoint center;
  center.x = x;
  center.y = y;

  CvSize size;
  size.width = Globals::roiWidth;
  size.height = Globals::roiHeight;

  double scale = 0.3;

  try {
    loadModel(modelsFileName);

    if (imageFileName != "") {
      Preprocess preprocess(size, scale, center, roiFunction);
      IplImage* image = cvLoadImage(imageFileName.c_str());
      IplImage* imageVector = preprocess.generateImageVector(image);

      cout << "Sector " << classify(imageVector) << endl;
      cvReleaseImage(&image);
      cvReleaseImage(&imageVector);
    } else if (imageDirectory != "") {
      int counts[5][6];
      for (int i = 0; i < 5; i++)
	for (int j = 0; j < 6; j++)
	  counts[i][j] = 0;

      string annotationsFileName = imageDirectory + "/annotations.xml";
      Annotations annotations;
      annotations.readAnnotations(annotationsFileName);
      CvPoint& center = annotations.getCenter();

      Preprocess preprocess(size, scale, center, roiFunction);
      vector<FrameAnnotation*>& frameAnnotations = annotations.getFrameAnnotations();
      for (unsigned int i = 0; i < frameAnnotations.size(); i++) {
	FrameAnnotation* fa = frameAnnotations[i];
	fa->setFace(center);

	int expectedZone = fa->getSector();
	counts[expectedZone - 1][5]++;

	// compose filename and update map
	char buffer[256];
	sprintf(buffer, "frame_%d.png", fa->getFrameNumber());
	string simpleName = buffer;
	string fileName = imageDirectory + "/" + simpleName;
	IplImage* image = cvLoadImage(fileName.c_str());
	IplImage* imageVector = preprocess.generateImageVector(image);

	int zone = classify(imageVector);
	if (expectedZone == zone)
	  counts[zone - 1][zone - 1]++;
	else
	  counts[expectedZone - 1][zone - 1]++;

	cvReleaseImage(&image);
	cvReleaseImage(&imageVector);
      }
      cout << "Errors by class" << endl;
      for (int i = 0; i < 5; i++) {
	for (int j = 0; j < 6; j++)
	  cout << counts[i][j] << "\t";
	cout << endl;
      }
    }
  } catch (string err) {
    cout << err << endl;
  }

  return 0;
}
