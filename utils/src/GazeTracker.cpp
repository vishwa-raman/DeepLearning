// GazeTracker.cpp
// File that contains the definition of the methods of class GazeTracker. It provides 
// all the functionality needed for in car gaze tracking. It uses the Filter, Location,
// Trainer and Classifier classes underneath

#include "GazeTracker.h"

#ifdef SINGLETHREADED
#define fftw_init_threads() ;
#define fftw_plan_with_nthreads(a) ;
#define fftw_cleanup_threads() ;
#endif

// static member initialization
CvSize GazeTracker::roiSize;
Trainer::KernelType GazeTracker::kernelType = Trainer::Polynomial;

// Class construction and destruction

GazeTracker::GazeTracker(string outputDir, bool online) {
  isOnline = online;

  // fftw3 initialization to use openmp. These functions should be called
  // once at application scope before any other fftw functions are called
  fftw_init_threads();
  fftw_plan_with_nthreads(1);
  
  // check if the output directory exists, or else bail
  DIR* dir;
  dir = opendir(outputDir.c_str());
  if (dir == NULL) {
    string err = "GazeTracker::GazeTracker. The directory " + outputDir + 
      " does not exist. Bailing out.";
    throw (err);
  }
  closedir(dir);

  // compute full path names
  char fullPath[PATH_MAX + 1];
  outputDirectory = realpath((const char*)outputDir.c_str(), fullPath);

  char* path = getenv("SVM_PATH");
  if (!path) {
    string err = "GazeTracker::GazeTracker. The SVM_PATH environment variable is not set";
    throw (err);
  }
  svmPath = path;
  classifier = 0;

  roiSize.width = Globals::roiWidth;
  roiSize.height = Globals::roiHeight;

  // we cannot initialize these extractors at this point because
  // we don't at this point what this object is going to be used
  // for. It could be for creating offline filters or it could
  // be for online tracking
  leftEyeExtractor = rightEyeExtractor = noseExtractor = 0;

  faceCenter.x = faceCenter.y = 0;

  // now read the config file and update state specific to the current
  // classification task
  readConfiguration();
}

GazeTracker::~GazeTracker() {
  if (classifier) delete classifier;

  delete leftEyeExtractor;
  delete rightEyeExtractor;
  delete noseExtractor;

  // final cleanup of all fftw thread data
  fftw_cleanup_threads();
}

// addFrameSet
// Method used to add frame sets in the training data. These sets are the
// directories containing training samples. Each directory is expected to
// contain an annotations file with annotated LOIs

void GazeTracker::addFrameSet(string directory) {
  // compute full path names
  char fullPath[PATH_MAX + 1];
  string framesDirectory = realpath((const char*)directory.c_str(), fullPath);

  frameSetDirectories.push_back(framesDirectory);
}

// getWindowCenter
// Method used to compute the center of the window we apply to a given
// frame during LOI extraction

CvPoint GazeTracker::getWindowCenter() {
  CvPoint windowCenter;

  windowCenter.x = Globals::roiWidth / 2;
  windowCenter.y = Globals::roiHeight / 2;
  
  return windowCenter;
}

// updateWindowCenter
// Method used to update the min and max co-ordinates of the window center
// by walking through all face annotations

void GazeTracker::updateWindowCenter(string trainingDirectory,
				     int& minX, int& maxX, int& minY, int& maxY) {
  Annotations annotations;

  // first capture the mapping from file names to locations of interest
  string locationsFileName = trainingDirectory + "/" + 
    Globals::annotationsFileName;
  annotations.readAnnotations(locationsFileName);

  // now get the set of all annotations
  vector<FrameAnnotation*>& frameAnnotations = annotations.getFrameAnnotations();
  for (unsigned int i = 0; i < frameAnnotations.size(); i++) {
    FrameAnnotation* fa = frameAnnotations[i];
    CvPoint& faceLocation = fa->getLOI(Annotations::Face);
    if (minX > faceLocation.x)
      minX = faceLocation.x;
    if (maxX < faceLocation.x)
      maxX = faceLocation.x;
    
    if (minY > faceLocation.y)
      minY = faceLocation.y;
    if (maxY < faceLocation.y)
      maxY = faceLocation.y;
  }
}

// computeWindowCenter
// Method used to compute a window center for face annotations. This method 
// reverts to getWindowCenter for all annotations when not in training mode.
// For in filter training mode, for all annotations other than the face, we
// simply return the result of getWindowCenter, but for the face we use the
// midpoint of extremal x and y co-ordinates

CvPoint GazeTracker::computeWindowCenter(string trainingDirectory) {
  if (trainingDirectory == "" && 
      !frameSetDirectories.size() && !trainingSetDirectories.size())
    return getWindowCenter();

  // iterate over all frameset directories and pick face annotations
  // to compute the average of the extremal face locations
  int minX = INT_MAX;
  int maxX = INT_MIN;
  int minY = INT_MAX;
  int maxY = INT_MIN;

  if (trainingDirectory == "") {
    for (unsigned int i = 0; i < frameSetDirectories.size(); i++)
      updateWindowCenter(frameSetDirectories[i], minX, maxX, minY, maxY);
    for (unsigned int i = 0; i < trainingSetDirectories.size(); i++)
      updateWindowCenter(trainingSetDirectories[i], minX, maxX, minY, maxY);
  } else {
      updateWindowCenter(trainingDirectory, minX, maxX, minY, maxY);
  }

  // now take the average of the min and max co-ordinate values
  CvPoint windowCenter;

  windowCenter.x = (minX + maxX) / 2;
  windowCenter.y = (minY + maxY) / 2;

  cout << "Face windowCenter = " << windowCenter.x << ", " << windowCenter.y << endl;

  return windowCenter;
}

// createFilters
// Method used to create filters. This method will iterate over all the
// training frames directories, add them to each filter we want to create,
// create those filters and save them

void GazeTracker::createFilters() {
  // get the center of the window function
  CvPoint windowCenter = getWindowCenter();

  // create filters
  Filter* leftEyeFilter = new Filter(outputDirectory, Annotations::LeftEye, roiSize,
				     Globals::gaussianWidth /* gaussian spread */, 
				     windowCenter, roiFunction);
  Filter* rightEyeFilter = new Filter(outputDirectory, Annotations::RightEye, roiSize,
				      Globals::gaussianWidth /* gaussian spread */, 
				      windowCenter, roiFunction);
  Filter* noseFilter = new Filter(outputDirectory, Annotations::Nose, roiSize,
				  Globals::gaussianWidth /* gaussian spread */, 
				  windowCenter, roiFunction);

  //leftEyeFilter->setAffineTransforms();
  //rightEyeFilter->setAffineTransforms();
  //noseFilter->setAffineTransforms();

  for (unsigned int i = 0; i < frameSetDirectories.size(); i++) {
    cout << "Adding left eye annotations..." << endl;
    leftEyeFilter->addTrainingSet(frameSetDirectories[i]);

    cout << "Adding right eye annotations..." << endl;
    rightEyeFilter->addTrainingSet(frameSetDirectories[i]);

    cout << "Adding nose annotations..." << endl;
    noseFilter->addTrainingSet(frameSetDirectories[i]);
  }

  #pragma omp parallel sections num_threads(4) 
  {
    #pragma omp section
    {
      cout << "Creating left eye filter..." << endl;
      leftEyeFilter->create();
      leftEyeFilter->save();
    }
  
    #pragma omp section
    {
      cout << "Creating right eye filter..." << endl;
      rightEyeFilter->create();
      rightEyeFilter->save();
    }

    #pragma omp section
    {
      cout << "Creating nose filter..." << endl;
      noseFilter->create();
      noseFilter->save();  
    }
  }

  delete leftEyeFilter;
  delete rightEyeFilter;
  delete noseFilter;
}

// addTrainingSet
// Method used to add directories with training data for SVM models. Each directory is 
// expected to contain an annotations file with annotated LOIs

void GazeTracker::addTrainingSet(string directory) {
  // compute full path names
  char fullPath[PATH_MAX + 1];
  string trainingDirectory = realpath((const char*)directory.c_str(), fullPath);

  trainingSetDirectories.push_back(trainingDirectory);
}

// train
// Method used to train the SVM classifier. The trainer expects to find an
// annotations file in the output directory that contains all the annotations that
// were applied during filter training. All these annotations are used to create
// models using the SVM Light trainer

void GazeTracker::train() {
  if (!leftEyeExtractor) {
    // get the center of the window we want to apply
    CvPoint windowCenter = getWindowCenter();
    leftEyeExtractor = new Location(outputDirectory, Annotations::LeftEye,
				    windowCenter);
    rightEyeExtractor = new Location(outputDirectory, Annotations::RightEye,
				     windowCenter);
    noseExtractor = new Location(outputDirectory, Annotations::Nose,
				 windowCenter);
  }
  
  Trainer trainer(outputDirectory, kernelType, 
		  leftEyeExtractor, rightEyeExtractor, noseExtractor,
		  roiFunction, svmPath);

  // add training sets
  for (unsigned int i = 0; i < trainingSetDirectories.size(); i++)
    trainer.addTrainingSet(trainingSetDirectories[i]);

  // generate models
  trainer.generate();
}

// getZone
// Method used to get the gaze zone given an image. If the classifier object is as
// yet not created, we first create it here. If the online flag is set, then 
// we create online filters and then initialize the location extractors with those
// filters, else we use the offline filters that are expected to have been
// generated before this function is called

int GazeTracker::getZone(IplImage* image, double& confidence, FrameAnnotation& fa) {
  if (!classifier)
    createClassifier();

  fa.setFace(faceCenter);
  return classifier->getZone(image, confidence, fa);
}

// getFilterAccuracy
// Method used to compute the error for a filter identified by xml tag for the 
// annotations in a given directory

double GazeTracker::getFilterAccuracy(string trainingDirectory, Annotations::Tag xmlTag,
				      Classifier::ErrorType errorType) {
  if (!classifier)
    createClassifier();

  return classifier->getFilterError(trainingDirectory, xmlTag, errorType);
}

// getClassifierAccuracy
// Method to get the classifier accuracy

pair<double, string> GazeTracker::getClassifierAccuracy(string trainingDirectory) {
  if (!classifier)
    createClassifier();

  return classifier->getError(trainingDirectory);
}

// createClassifier
// Method used to create a classifier object

void GazeTracker::createClassifier() {
  // create location extractors
  if (isOnline) {
    // create new online filters and use them as filters in the location
    // extractors for all subsequent images

    CvPoint windowCenter = getWindowCenter();

    // left eye filter and extractor
    Filter* filter = new OnlineFilter(outputDirectory, Annotations::LeftEye, roiSize,
				      Globals::gaussianWidth /* gaussian spread */,
				      Globals::learningRate, windowCenter);
    leftEyeExtractor = new Location(filter);

    // right eye filter and extractor
    filter = new OnlineFilter(outputDirectory, Annotations::RightEye, roiSize,
			      Globals::gaussianWidth /* gaussian spread */,
			      Globals::learningRate, windowCenter);
    rightEyeExtractor = new Location(filter);

    // nose filter and extractor
    filter = new OnlineFilter(outputDirectory, Annotations::Nose, roiSize,
			      Globals::gaussianWidth /* gaussian spread */,
			      Globals::learningRate, windowCenter);
    noseExtractor = new Location(filter);
  } else {
    if (!leftEyeExtractor) {
      CvPoint windowCenter = getWindowCenter();
      leftEyeExtractor = new Location(outputDirectory, Annotations::LeftEye,
				      windowCenter);
      rightEyeExtractor = new Location(outputDirectory, Annotations::RightEye,
				       windowCenter);
      noseExtractor = new Location(outputDirectory, Annotations::Nose,
				   windowCenter);
    }
  }

  // now create the classifier
  classifier = new Classifier(outputDirectory, kernelType, 
			      leftEyeExtractor, rightEyeExtractor,
			      noseExtractor, roiFunction);
}

// showAnnotations
// Method used to show annotations in the training set

void GazeTracker::showAnnotations() {
  if (!frameSetDirectories.size())
    return;

  string wName = "Annotations";
  cvNamedWindow((const char*)wName.c_str(), CV_WINDOW_NORMAL | CV_WINDOW_AUTOSIZE);

  // initialize font and add text
  CvFont font;
  cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 1.0, 1.0, 0, 1, CV_AA);

  for (unsigned int i = 0; i < frameSetDirectories.size(); i++) {
    Annotations annotations;

    // first capture the mapping from file names to locations of interest
    string locationsFileName = frameSetDirectories[i] + "/" + 
      Globals::annotationsFileName;
    annotations.readAnnotations(locationsFileName);

    string framesDirectory = annotations.getFramesDirectory();
    string prefix = framesDirectory + "/frame_";

    // now get the set of all annotations
    vector<FrameAnnotation*>& frameAnnotations = annotations.getFrameAnnotations();
    for (unsigned int j = 0; j < frameAnnotations.size(); j++) {
      FrameAnnotation* fa = frameAnnotations[j];

      char buffer[256];
      sprintf(buffer, "%d.png", fa->getFrameNumber());
      string filename = prefix + buffer;
      IplImage* image = cvLoadImage((const char*)filename.c_str());

      CvPoint& faceLocation = fa->getLOI(Annotations::Face);
      if (!faceLocation.x && !faceLocation.y)
	continue;

      cvCircle(image, fa->getLOI(Annotations::LeftEye), 5, cvScalar(0, 255, 255, 0), 2, 8, 0);
      cvCircle(image, fa->getLOI(Annotations::RightEye), 5, cvScalar(255, 255, 0, 0), 2, 8, 0);
      cvCircle(image, fa->getLOI(Annotations::Nose), 5, cvScalar(255, 0, 255, 0), 2, 8, 0);

      sprintf(buffer, "%d", fa->getZone());
      cvPutText(image, buffer, cvPoint(580, 440), &font, cvScalar(255, 255, 255, 0));

      cvShowImage((const char*)wName.c_str(), image);
      char c = cvWaitKey();
      if (c != 'c') {
	cvReleaseImage(&image);
	break;
      }

      cvReleaseImage(&image);
    }
  }
}

// readConfiguration
// Function used to read the config.xml file in the models directory. The file
// contains the configuration that is specific to the location of the driver 
// with respect to the camera and will grow to other pieces of information
// eventually

void GazeTracker::readConfiguration() {
  string fileName = outputDirectory + '/' + Globals::configFileName;

  ifstream file;

  file.open((const char*)fileName.c_str());
  if (file.good()) {
    string line;

    getline(file, line); // ignore the first line
    while (!file.eof()) {
      getline(file, line);
      if (line.find("center") != string::npos) {
	getline(file, line);
	const char* token = strtok((char*)line.c_str(), "<>/x");
	if (!token) {
	  string err = "GazeTracker::readConfiguration. Malformed config file for center";
	  throw (err);
	}
	faceCenter.x = atoi(token);
	getline(file, line);
	token = strtok((char*)line.c_str(), "<>/y");
	if (!token) {
	  string err = "GazeTracker::readConfiguration. Malformed config file for center";
	  throw (err);
	}
	faceCenter.y = atoi(token);
      }
    }
  }
}

// roiFunction
// We support the use of LOIs identified to cull images to smaller regions of interest
// (ROI) for use in locating future LOIs. This function is passed to the constructors
// of the filter and classifier classes. Those classes in turn call this function when
// they need a culled image. The input parameters are the original image, a frame
// annotation object that is annotated with all the LOIs that we have found before this
// function gets called. The offset parameter is an output parameter that contains the
// offset of the ROI within the image. The function returns a culled image object

IplImage* GazeTracker::roiFunction(IplImage* image, FrameAnnotation& fa, 
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
