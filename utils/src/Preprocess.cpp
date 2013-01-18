// Preprocess.cpp
// This file contains the implementation of class Preprocess.
// This class is used to generate training and test sets from directories of
// labelled or unlabelled images.

#include "Preprocess.h"

// union to enable accessing bytes of an integer
typedef union {
  unsigned int i;
  unsigned char u[4];
} IntBytesT;

// Class construction and destruction

Preprocess::Preprocess(string output, CvSize size, double scale, CvPoint& center, 
		       map<int, bool>& sFilter, map<int, bool>& iFilter,
		       roiFnT roiFn, bool bins,
		       bool inBinFmt, double binThreshold) {
  outputFileName = output;
  useBins = bins;
  roiFunction = roiFn;
  inBinaryFormat = inBinFmt;
  binaryThreshold = binThreshold;
  imgSize.height = size.height;
  imgSize.width = size.width;
  scaleFactor = scale;

  map<int, bool>::iterator it;
  for (it = sFilter.begin(); it != sFilter.end(); it++) {
    int key = (*it).first;
    bool value = (*it).second;
    statusFilter[key] = value;
  }
  for (it = iFilter.begin(); it != iFilter.end(); it++) {
    int key = (*it).first;
    bool value = (*it).second;
    intersectionFilter[key] = value;
  }

  nSamples = 0;
  doAffineTransforms = false;

  length = imgSize.height * imgSize.width;

  // allocate real complex vectors for use during filter creation or update
  realImg = cvCreateImage(imgSize, IPL_DEPTH_64F, 1);
  tempImg = cvCreateImage(imgSize, IPL_DEPTH_64F, 1);
  imageBuffer = (double*)fftw_malloc(sizeof(double) * length);

  windowCenter.x = center.x;
  windowCenter.y = center.y;

  //  cvNamedWindow("window", CV_WINDOW_NORMAL | CV_WINDOW_AUTOSIZE);
}

Preprocess::Preprocess(CvSize size, double scale, CvPoint& center, 
		       roiFnT roiFn) {
  roiFunction = roiFn;
  imgSize.height = size.height;
  imgSize.width = size.width;
  scaleFactor = scale;

  nSamples = 0;
  doAffineTransforms = false;

  length = imgSize.height * imgSize.width;

  // allocate real complex vectors for use during filter creation or update
  realImg = cvCreateImage(imgSize, IPL_DEPTH_64F, 1);
  tempImg = cvCreateImage(imgSize, IPL_DEPTH_64F, 1);
  imageBuffer = (double*)fftw_malloc(sizeof(double) * length);

  windowCenter.x = center.x;
  windowCenter.y = center.y;

  //  cvNamedWindow("window", CV_WINDOW_NORMAL | CV_WINDOW_AUTOSIZE);
}

Preprocess::~Preprocess() {
  cvReleaseImage(&realImg);
  cvReleaseImage(&tempImg);
  fftw_free(imageBuffer);

  for (unsigned int i = 0; i < fileToAnnotations.size(); i++)
    delete fileToAnnotations[i].second;
  for (unsigned int i = 0; i < fileToAnnotationsForValidation.size(); i++)
    delete fileToAnnotationsForValidation[i].second;
  for (unsigned int i = 0; i < fileToAnnotationsForTest.size(); i++)
    delete fileToAnnotationsForTest[i].second;
}

// addTrainingSet
// Method used to add directories that contains images. For training data,
// the directory is also expected to contains labels for each image.

void Preprocess::addTrainingSet(string trainingDirectory) {
  Annotations annotations;

  // first capture the mapping from file names to locations of interest
  string locationsFileName = trainingDirectory + "/" + Globals::annotationsFileName;
  annotations.readAnnotations(locationsFileName);

  // trim ends
  annotations.trimEnds(Annotations::Center, 20 /* nTrim */);
  annotations.trimEnds(Annotations::DriverWindow, 10 /* nTrim */);
  annotations.trimEnds(Annotations::LeftOfCenter, 10 /* nTrim */);
  annotations.trimEnds(Annotations::RightOfCenter, 10 /* nTrim */);
  annotations.trimEnds(Annotations::PassengerWindow, 10 /* nTrim */);

  // if we want to pick the same number of frames from each sector, then
  // we create bins. This will pick as many frames for each sector as the
  // smallest number of frames we have across all sectors
  if (useBins)
    annotations.createBins();

  // get the frames directory
  string framesDirectory = annotations.getFramesDirectory();

  // get the window center
  CvPoint& center = annotations.getCenter();

  // collect the number of transitions from each sector to other sectors
  unsigned int transitions[Globals::numZones][Globals::numZones];
  for (unsigned int i = 0; i < Globals::numZones; i++)
    for (unsigned int j = 0; j < Globals::numZones; j++)
      transitions[i][j] = 0;

  FrameAnnotation* prev = 0;

  // now get the set of all annotations
  vector<FrameAnnotation*>& frameAnnotations = annotations.getFrameAnnotations();
  for (unsigned int i = 0; i < frameAnnotations.size(); i++) {
    FrameAnnotation* fa = frameAnnotations[i];

    //  CvPoint faceCenter = fa->getFace();
    //  if (!faceCenter.x && !faceCenter.y)
    fa->setFace(center);

    // collect transition counts
    if (prev) {
      unsigned int prevSector = prev->getSector();
      unsigned int sector = fa->getSector();

      if (prevSector > 0 && prevSector <= 5 && sector > 0 && sector <= 5)
	transitions[prevSector - 1][sector - 1]++;
    }

    // now check if we have status and/or intersection filters. If yes, then
    // pick only those frames that match the filter
    if (statusFilter.size() && statusFilter.find(fa->getStatus()) == statusFilter.end())
      continue;
    if (intersectionFilter.size() && 
	intersectionFilter.find(fa->getIntersection()) == intersectionFilter.end())
      continue;

    // compose filename and update map
    char buffer[256];
    sprintf(buffer, "frame_%d.png", fa->getFrameNumber());
    string simpleName = buffer;
    string fileName = framesDirectory + "/" + simpleName;
    fileToAnnotations.push_back(make_pair(fileName, new FrameAnnotation(*fa)));

    prev = fa;
  }

  // report transition counts
  cout << "Transitions" << endl;
  for (unsigned int i = 0; i < Globals::numZones; i++) {
    for (unsigned int j = 0; j < Globals::numZones; j++)
      cout << transitions[i][j] << "\t";
    cout << endl;
  }
}

// addTestSet
// Method used to add annotation files for validation and test. The method
// takes as input an annotation file name and the fraction of the images that
// should be used for validation in that set. The remainder are treated as 
// part of the test set.

void Preprocess::addTestSet(string annotationsFileName, 
			    double validationFraction, double testFraction) {
  Annotations annotations;

  // first capture the mapping from file names to locations of interest
  annotations.readAnnotations(annotationsFileName);

  // get the frames directory
  string framesDirectory = annotations.getFramesDirectory();

  // get the window center
  CvPoint& center = annotations.getCenter();

  // now get the set of all annotations
  vector<FrameAnnotation*>& frameAnnotations = annotations.getFrameAnnotations();
  random_shuffle(frameAnnotations.begin(), frameAnnotations.end());

  unsigned int nValidationLength = validationFraction * frameAnnotations.size();
  unsigned int nTestLength = testFraction * frameAnnotations.size();

  for (unsigned int i = 0; i < frameAnnotations.size(); i++) {
    FrameAnnotation* fa = frameAnnotations[i];

    fa->setFace(center);

    // compose filename and update map
    char buffer[256];
    sprintf(buffer, "frame_%d.png", fa->getFrameNumber());
    string simpleName = buffer;
    string fileName = framesDirectory + "/" + simpleName;
    if (i < nValidationLength)
      fileToAnnotationsForValidation.push_back(make_pair(fileName, new FrameAnnotation(*fa)));
    else if (i < nValidationLength + nTestLength)
      fileToAnnotationsForTest.push_back(make_pair(fileName, new FrameAnnotation(*fa)));
  }
}

// The following method is used to open a data file and a label file given
// as input a simpleName. It generates the preambles in these files based
// on the MNIST data format

void Preprocess::openFiles(string simpleName) {
  // now open the data file and labels file and create their respective preambles
  string dataFileName = "data-" + simpleName;
  dataFile.open(dataFileName.c_str(), ofstream::binary);
  if (!dataFile.good()) {
    string err = "Preprocess::generate. Cannot open " + dataFileName + 
      " for write";
    throw (err);
  }
  // The magic number 0x00000803 is used for data, where the 0x08 
  // is for unsigned byte valued data and the 0x03 is for the number
  // of dimensions
  IntBytesT ib;
  ib.i = 8;
  ib.i <<= 16;
  ib.i |= 3;
  dataFile.write((char*)&(ib.u), 4);
  ib.i = 0;
  dataFile.write((char*)&(ib.u), 4); // for the number of samples
  ib.i = (unsigned int)imgSize.width * scaleFactor;
  dataFile.write((char*)&(ib.u), 4);
  ib.i = (unsigned int)imgSize.height * scaleFactor;
  dataFile.write((char*)&(ib.u), 4);

  string labelFileName = "label-" + simpleName;
  labelFile.open(labelFileName.c_str(), ofstream::binary);
  if (!labelFile.good()) {
    string err = "Preprocess::generate. Cannot open " + labelFileName + 
      " for write";
    throw (err);
  }
  // The magic number 0x00000803 is used for data, where the 0x08 
  // is for unsigned byte valued data and the 0x03 is for the number
  // of dimensions
  ib.i = 8;
  ib.i <<= 16;
  ib.i |= 1;
  labelFile.write((char*)&(ib.u), 4);
  ib.i = 0;
  labelFile.write((char*)&(ib.u), 4); // for the number of samples
}

// The following method closes the data file and the label file. It re-writes
// the number of samples contained in these files first before closing. The
// number of samples is not available at the time the files are created, as
// we may choose to do affine transforms increasing the number of images 
// written over and above the number of images we get through the annotation
// files

void Preprocess::closeFiles(int nSamples) {
  // now write out the total number of samples that were written to file
  // and close both the data and label files
  IntBytesT ib;
  ib.i = nSamples;

  dataFile.seekp(4, ios_base::beg);
  dataFile.write((char*)&(ib.u), 4);
  dataFile.close();

  labelFile.seekp(4, ios_base::beg);
  labelFile.write((char*)&(ib.u), 4);
  labelFile.close();
}

// Method to do the actual image data and sector writing into data and label
// files. We use a randomized access into the set of all files accumulated
// to generate the number of samples required based on a user specified 
// percentage

void Preprocess::generate(int startIndex, int nSamples, string simpleName,
			  bool doWrite,
			  vector<pair<string, FrameAnnotation*> >* additionalPairs) {
  // open data and label files
  openFiles(simpleName);

  int nSectorFrames[Globals::numZones];
  for (unsigned int i = 0; i < Globals::numZones; i++)
    nSectorFrames[i] = 0;

  // if we want to write out test annotations then open a file for that
  // purpose
  ofstream annotationsFile;
  string dirName;
  if (doWrite) {
    // get absolute path of the input directory
    char fullPath[PATH_MAX + 1];
    string fullPathName = realpath("./", fullPath);

    string fileName = fullPathName + "/annotations.xml";
    dirName = string(fullPath) + "/_files";
    annotationsFile.open(fileName.c_str());
    mkdir(dirName.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    if (annotationsFile.good()) {
      annotationsFile << "<?xml version=\"1.0\"?>" << endl;
      annotationsFile << "<annotations dir=\"" + dirName + "\">" << endl;
    }
  }

  // write content
  int samples = 0;
  for (int i = startIndex; i < startIndex + nSamples; i++) {
    string fileName = fileToAnnotations[i].first;
    FrameAnnotation* fa = fileToAnnotations[i].second;

    int sector = fa->getSector();
    if (sector < Annotations::DriverWindow || sector > Annotations::PassengerWindow)
      continue;

    if (doWrite) {
      annotationsFile << "  <frame>" << endl;
      annotationsFile << "    <frameNumber>" << samples + 1 << 
	"</frameNumber>" << endl;
      annotationsFile << "    <face>" << fa->getFace().y << "," << 
	fa->getFace().x << "</face>" << endl;
      annotationsFile << "    <zone>" << fa->getSector() << "</zone>" << endl;
      annotationsFile << "    <status>" << fa->getStatus() << "</status>" << endl;
      annotationsFile << "    <intersection>" << fa->getIntersection() << 
	"</intersection>" << endl;
      annotationsFile << "  </frame>" << endl;

      char buffer[256];
      sprintf(buffer, "/frame_%d.png", samples + 1);
      string destFileName = dirName + string(buffer);
      string command = "/bin/cp " + fileName + " " + destFileName;
      static_cast<void>(system(command.c_str()));
    }

    nSectorFrames[fa->getSector() - 1]++;

    // update files with image and label data
    update(fileName, fa, samples);
  }
  // now add any frames from the additional pairs if any
  if (additionalPairs) {
    for (unsigned int i = 0; i < additionalPairs->size(); i++) {
      string fileName = (*additionalPairs)[i].first;
      FrameAnnotation* fa = (*additionalPairs)[i].second;

      int sector = fa->getSector();
      if (sector < Annotations::DriverWindow || sector > Annotations::PassengerWindow)
	continue;

      if (doWrite) {
	annotationsFile << "  <frame>" << endl;
	annotationsFile << "    <frameNumber>" << samples + 1 << 
	  "</frameNumber>" << endl;
	annotationsFile << "    <face>" << fa->getFace().y << "," << 
	  fa->getFace().x << "</face>" << endl;
	annotationsFile << "    <zone>" << fa->getSector() << "</zone>" << endl;
	annotationsFile << "    <status>" << fa->getStatus() << "</status>" << endl;
	annotationsFile << "    <intersection>" << fa->getIntersection() << 
	  "</intersection>" << endl;
	annotationsFile << "  </frame>" << endl;
	
	char buffer[256];
	sprintf(buffer, "/frame_%d.png", samples + 1);
	string destFileName = dirName + string(buffer);
	string command = "/bin/cp " + fileName + " " + destFileName;
	static_cast<void>(system(command.c_str()));
      }

      nSectorFrames[fa->getSector() - 1]++;
      
      // update files with image and label data
      update(fileName, fa, samples);
    }
  }

  if (doWrite) {
    annotationsFile << "</annotations>" << endl;
    annotationsFile.close();
  }

  // close data and label files
  closeFiles(samples);

  // info
  cout << "For " << simpleName << " generated the following frames by sector" << endl;
  cout << "[" << nSectorFrames[0] << ", " << nSectorFrames[1] << ", " <<
    nSectorFrames[2] << ", " << nSectorFrames[3] << ", " << nSectorFrames[4] << "]" <<
    endl;
}

// Method that generates three sets of files; training, validation and test.
// The user specified training and validation percentages are used with the
// remaining images being treated as test images

void Preprocess::generate(double training, double validation, bool doWriteTests) {
  // first shuffle the vector of image frames
  random_shuffle(fileToAnnotations.begin(), fileToAnnotations.end());

  // compute training, validation, and testing required samples
  int len = fileToAnnotations.size();
  int nTrainingSamples = len * training;
  int nValidationSamples = len * validation;
  int nTestingSamples = len - (nTrainingSamples + nValidationSamples);

  // generate training files
  string simpleName = "train-" + outputFileName;
  generate(0 /* startIndex */, nTrainingSamples, simpleName);

  // generate validation files
  simpleName = "valid-" + outputFileName;
  generate(nTrainingSamples, nValidationSamples, simpleName,
	   false /* doWrite */, &fileToAnnotationsForValidation);

  // we don't want affine transforms for validation and test sets
  doAffineTransforms = false;

  // the remainder of the images are test images
  simpleName = "test-" + outputFileName;
  generate(nTrainingSamples + nValidationSamples, nTestingSamples, simpleName,
	   doWriteTests, &fileToAnnotationsForTest);
}

// Method to do the actual image data and sector writing into data and label
// files. We pick Globals::nSequenceLength chunks based on the chunk indices vector,
// the starting chunk index and the number of chunks desired

void Preprocess::generateSequences(int startIndex, int nChunks, 
				   vector<int>& indices, string simpleName) {
  // open data and label files
  openFiles(simpleName);

  int nSectorFrames[Globals::numZones];
  for (unsigned int i = 0; i < Globals::numZones; i++)
    nSectorFrames[i] = 0;

  // if we want to write out test annotations then open a file for that
  // purpose
  ofstream annotationsFile;
  string dirName;

  // collect the number of transitions from each sector to other sectors
  unsigned int transitions[Globals::numZones][Globals::numZones];
  for (unsigned int i = 0; i < Globals::numZones; i++)
    for (unsigned int j = 0; j < Globals::numZones; j++)
      transitions[i][j] = 0;

  FrameAnnotation* prev = 0;

  // write content
  int samples = 0;
  for (int i = startIndex; i < startIndex + nChunks; i++) 
    for (int j = 0; j < Globals::nSequenceLength; j++) {
      int offset = indices[i] * Globals::nSequenceLength + j;
      string fileName = fileToAnnotations[offset].first;
      FrameAnnotation* fa = fileToAnnotations[offset].second;

      int sector = fa->getSector();
      if (sector < Annotations::DriverWindow || sector > Annotations::PassengerWindow)
	continue;

      // collect transition counts
      if (prev) {
	unsigned int prevSector = prev->getSector();

	if (prevSector > 0 && prevSector <= 5)
	  transitions[prevSector - 1][sector - 1]++;
      }

      nSectorFrames[fa->getSector() - 1]++;

      // update files with image and label data
      update(fileName, fa, samples);

      prev = fa;
    }

  // close data and label files
  closeFiles(samples);

  // report transition counts
  cout << "Transitions" << endl;
  for (unsigned int i = 0; i < Globals::numZones; i++) {
    for (unsigned int j = 0; j < Globals::numZones; j++)
      cout << transitions[i][j] << "\t";
    cout << endl;
  }

  // info
  cout << "For " << simpleName << " generated the following frames by sector" << endl;
  cout << "[" << nSectorFrames[0] << ", " << nSectorFrames[1] << ", " <<
    nSectorFrames[2] << ", " << nSectorFrames[3] << ", " << nSectorFrames[4] << "]" <<
    endl;
}

// Method that generates three sets of files; training, validation and test.
// The user specified training and validation percentages are used with the
// remaining images being treated as test images. This method, unlike the one
// above, will preserve frame sequences and will not randomize

void Preprocess::generateSequences(double training, double validation) {
  // compute training, validation, and testing required samples
  int len = fileToAnnotations.size();
  int nTrainingSamples = len * training;
  int nValidationSamples = len * validation;
  int nTestingSamples = len - (nTrainingSamples + nValidationSamples);

  int chunks = len / Globals::nSequenceLength;
  vector<int> indices;
  for (int i = 0; i < chunks; i++)
    indices.push_back(i);
  random_shuffle(indices.begin(), indices.end());

  int nValidationChunks = nValidationSamples / Globals::nSequenceLength;
  int nTestChunks = nTestingSamples / Globals::nSequenceLength;

  // pick the first nValidationChunks from indices for validation samples
  cout << "Validation frames" << endl;
  string simpleName = "valid-" + outputFileName;
  generateSequences(0 /* startIndex */, nValidationChunks, indices, simpleName);

  // generate test files
  cout << "Test frames" << endl;
  simpleName = "test-" + outputFileName;
  generateSequences(nValidationChunks, nTestChunks, indices, simpleName);

  // generate training files
  // since we want to preserve as much sequentiality in the data as possible,
  // we sort the remaining chunk indices and then use them to generate
  // training data. This way, contiguous chunks will contribute to increasing
  // sequential correlation
  cout << "Train frames" << endl;
  sort(indices.begin() + nValidationChunks + nTestChunks, indices.end());
  simpleName = "train-" + outputFileName;
  generateSequences(nValidationChunks + nTestChunks, 
		    chunks - (nValidationChunks + nTestChunks), 
		    indices, simpleName);
}

// update
// Method used to update terms used to create a filter from an image file
// and a location of interest. This method is called by the method addTrainingSet
// for each image file in the test set and a location of interest for that
// image

void Preprocess::update(string filename, FrameAnnotation* fa, int& samples) {
  IplImage* image = cvLoadImage(filename.c_str());
  if (!image) {
    string err = "Preprocess::update. Cannot load file " + filename + ".";
    throw (err);
  }

  // generate affine transforms if requested
  vector<ImgLocPairT>& imgLocPairs = getAffineTransforms(image, fa->getFace());
  
  for (unsigned int i = 0; i < imgLocPairs.size(); i++) {
    image = imgLocPairs[i].first;
    CvPoint& location = imgLocPairs[i].second;

    CvPoint offset;
    offset.x = offset.y = 0;
    fa->setFace(location);
    if (roiFunction) {
      IplImage* roi = roiFunction(image, *fa, offset, Annotations::Face);
      image = roi;
    }

    // compute size and length of the image data
    CvSize size = cvGetSize(image);

    // check consistency
    if (imgSize.height != size.height || imgSize.width != size.width) {
      char buffer[32];
      sprintf(buffer, "(%d, %d).", imgSize.height, imgSize.width);
      string err = "Preprocess::update. Inconsistent image sizes. Expecting" + string(buffer);
      throw (err);
    }

    // preprocess
    double* preImage = preprocessImage(image);
    IplImage* processedImage = cvCreateImage(imgSize, IPL_DEPTH_8U, 1);
    int step = processedImage->widthStep;
    unsigned char* imageData = (unsigned char*)processedImage->imageData;
    for (int i = 0; i < imgSize.height; i++) {
      for (int j = 0; j < imgSize.width; j++) {
	if (preImage[i * imgSize.width + j] > 1)
	  cout << "(" << i << ", " << j << ") = " << preImage[i * imgSize.width + j] << endl;
	double d = preImage[i * imgSize.width + j];
	d = (inBinaryFormat)? ((d >= binaryThreshold)? 255 : 0) : d * 255;
	unsigned char c = (unsigned char)d;
	(*imageData++) = c;
      }
      imageData += step / sizeof(unsigned char) - imgSize.width;
    }
    //  cvNamedWindow("grayScaleImage", CV_WINDOW_NORMAL | CV_WINDOW_AUTOSIZE);
    //  cvShowImage("grayScaleImage", processedImage);
    //  cvWaitKey(1);

    CvSize scaledSize;
    scaledSize.width = (unsigned int)imgSize.width * scaleFactor;
    scaledSize.height = (unsigned int)imgSize.height * scaleFactor;

    IplImage* scaledImage = cvCreateImage(scaledSize, processedImage->depth, 
					  processedImage->nChannels);
    cvResize(processedImage, scaledImage);
    //    cvNamedWindow("scaledImage", CV_WINDOW_NORMAL | CV_WINDOW_AUTOSIZE);
    //    cvShowImage("scaledImage", scaledImage);
    //    cvWaitKey(1);
    cvReleaseImage(&processedImage);

    samples++;
    unsigned char buffer[16];
    buffer[0] = (unsigned char)fa->getSector();
    labelFile.write((char*)buffer, 1);
    /*
    // write out other bits of annotated information
    unsigned char status = (unsigned char)fa->getStatus();
    bitset<8> statusBits(status);
    for (unsigned int i = 0; i < statusBits.size(); i++) {
      if (statusBits[i])
	buffer[i] = 255;
      else
	buffer[i] = 0;
    }
    dataFile.write((char*)buffer, 8);

    unsigned char intersection = (unsigned char)fa->getIntersection();
    bitset<8> intxBits(intersection);
    for (unsigned int i = 0; i < intxBits.size(); i++) {
      if (intxBits[i])
	buffer[i] = 255;
      else
	buffer[i] = 0;
    }
    dataFile.write((char*)buffer, 8);
    */
    imageData = (unsigned char*)scaledImage->imageData;
    for (int i = 0; i < scaledSize.height; i++) {
      for (int j = 0; j < scaledSize.width; j++) {
	buffer[0] = (*imageData++);
	dataFile.write((char*)buffer, 1);
      }
      imageData += step / sizeof(unsigned char) - scaledSize.width;
    }
    cvReleaseImage(&scaledImage);

    if (roiFunction)
      cvReleaseImage(&image);
  }

  destroyAffineTransforms(imgLocPairs);
}

// generateImageVector
// Method used to create an image vector for classification. We carve out an ROI
// using the ROI function, preprocess the ROI image, scale the preprocessed image
// and return it as an array of doubles

IplImage* Preprocess::generateImageVector(IplImage* image) {
  FrameAnnotation fa;
  CvPoint offset;
  offset.x = offset.y = 0;
  fa.setFace(windowCenter);
  if (roiFunction) {
    IplImage* roi = roiFunction(image, fa, offset, Annotations::Face);
    image = roi;
  }

  //  cvNamedWindow("Image", CV_WINDOW_NORMAL | CV_WINDOW_AUTOSIZE);
  //  cvShowImage("Image", image);

  // compute size and length of the image data
  CvSize size = cvGetSize(image);

  // check consistency
  if (imgSize.height != size.height || imgSize.width != size.width) {
    char buffer[32];
    sprintf(buffer, "(%d, %d).", imgSize.height, imgSize.width);
    string err = "Preprocess::update. Inconsistent image sizes. Expecting" + string(buffer);
    throw (err);
  }

  // preprocess
  double* preImage = preprocessImage(image);
  IplImage* processedImage = cvCreateImage(imgSize, IPL_DEPTH_64F, 1);
  int step = processedImage->widthStep;
  double* imageData = (double*)processedImage->imageData;
  for (int i = 0; i < imgSize.height; i++) {
    for (int j = 0; j < imgSize.width; j++) {
      if (preImage[i * imgSize.width + j] > 1)
	cout << "(" << i << ", " << j << ") = " << preImage[i * imgSize.width + j] << endl;
      double d = preImage[i * imgSize.width + j];
      unsigned char c = (unsigned char)(d * 255);
      (*imageData++) = ((double)c) / 255.0;
    }
    imageData += step / sizeof(double) - imgSize.width;
  }

  CvSize scaledSize;
  scaledSize.width = (unsigned int)imgSize.width * scaleFactor;
  scaledSize.height = (unsigned int)imgSize.height * scaleFactor;

  IplImage* scaledImage = cvCreateImage(scaledSize, processedImage->depth, 
					processedImage->nChannels);
  cvResize(processedImage, scaledImage);
  //  cvNamedWindow("scaledImage", CV_WINDOW_NORMAL | CV_WINDOW_AUTOSIZE);
  //  cvShowImage("scaledImage", scaledImage);
  //  cvWaitKey(1);
  cvReleaseImage(&processedImage);

  size.height = 1;
  size.width = scaledSize.width * scaledSize.height;
  IplImage* result = cvCreateImage(size, IPL_DEPTH_64F, 1);
  double* resultData = (double*)result->imageData;
  imageData = (double*)scaledImage->imageData;
  step = scaledImage->widthStep;
  for (int i = 0; i < scaledSize.height; i++) {
    for (int j = 0; j < scaledSize.width; j++) {
      (*resultData++) = (*imageData++);
    }
    imageData += step / sizeof(double) - scaledSize.width;
  }

  if (roiFunction)
    cvReleaseImage(&image);
  cvReleaseImage(&scaledImage);

  return result;
}

// preprocessImage
// Method that preprocesses an image that has already been loaded for a
// subsequent application of the filter. The method returns a preprocessed
// image which can then be used on a subsequent call to apply or update.

double* Preprocess::preprocessImage(IplImage* inputImg) {
  // we take the complex image and preprocess it here
  if (!inputImg) {
    string err = "Preprocess::preprocessImage. Call setImage with a valid image.";
    throw (err);
  }

  bool releaseImage = false;
  IplImage* image = 0;

  // First check if the image is in grayscale. If not, we first convert it
  // into grayscale. The input image is replaced with its grayscale version
  if ((inputImg->nChannels != 1 && strcmp(inputImg->colorModel, "GRAY"))) {
    image = cvCreateImage(imgSize, IPL_DEPTH_8U, 1);
    cvCvtColor(inputImg, image, CV_BGR2GRAY);
    releaseImage = true;
  } else
    image = inputImg;

  // now do histogram equalization
  cvEqualizeHist(image, image);

  // edge detection
  cvCanny(image, image, 120, 200, 3);

  // We follow preprocessing steps here as outlined in Bolme 2009
  // First populate a real image from the grayscale image
  double scale = 1.0 / 255.0;
  cvConvertScale(image, realImg, scale, 0.0);
  /*
  // compute image inversion
  int step = realImg->widthStep;
  double* imageData = (double*)realImg->imageData;
  for (int i = 0; i < imgSize.height; i++) {
    for (int j = 0; j < imgSize.width; j++) {
      *(imageData) = 1 - *(imageData);
      imageData++;
    }
    imageData += step / sizeof(double) - imgSize.width;
  }
  */
  // suppress DC
  cvDCT(realImg, tempImg, CV_DXT_FORWARD);
  int step = tempImg->widthStep;
  double* imageData = (double*)tempImg->imageData;
  for (int i = 0; i < imgSize.height; i++) {
    for (int j = 0; j < imgSize.width; j++) {
      double sigmoid = (1 / (1 + (exp(-(i * imgSize.width + j)))));
      *(imageData) = *(imageData) * sigmoid;
      imageData++;
    }
    imageData += step / sizeof(double) - imgSize.width;
  }
  cvSet2D(tempImg, 0, 0, cvScalar(0));
  cvDCT(tempImg, realImg, CV_DXT_INVERSE);

  double min, max;
  cvMinMaxLoc(realImg, &min, &max, NULL, NULL);
  if (min < 0)
    cvAddS(realImg, cvScalar(-min), realImg, NULL);
  else
    cvAddS(realImg, cvScalar(min), realImg, NULL);

  cvMinMaxLoc(realImg, &min, &max, NULL, NULL);
  scale = 1.0 / max;
  cvConvertScale(realImg, realImg, scale, 0);

  double* destImageData = imageBuffer;
  double* srcImageData = (double*)realImg->imageData;
  for (int i = 0; i < imgSize.height; i++) {
    for (int j = 0; j < imgSize.width; j++) {
      (*destImageData) = (*srcImageData);
      srcImageData++; destImageData++;
    }
    srcImageData += step / sizeof(double) - imgSize.width;
  }
  //  showRealImage("preprocessedImage", imageBuffer);

  if (releaseImage)
    cvReleaseImage(&image);

  return imageBuffer;
}

// getAffineTransforms
// Method to generate a set of affine transformations of a given image. The
// image is rotated and translated to perturb the LOI around the given location.
// The method returns a vector of images that have been perturbed with small
// affine transforms

vector<ImgLocPairT>& Preprocess::getAffineTransforms(IplImage* image, CvPoint& location) {
  // first check if affine transformations are needed, if not then simply
  // push the input images to the vector of transformed images and return
  transformedImages.push_back(make_pair(image, location));
  if (!doAffineTransforms)
    return transformedImages;

  // Setup unchanging data sets used for transformations
  // for rotation
  Mat imageMat(image);
  Point2f center(imageMat.cols / 2.0F, imageMat.rows / 2.0F);

  CvSize size = cvGetSize(image);

  // for translation
  Mat translationMat = getRotationMatrix2D(center, 0, 1.0);
  translationMat.at<double>(0, 0) = 1;
  translationMat.at<double>(0, 1) = 0;
  translationMat.at<double>(1, 0) = 0;
  translationMat.at<double>(1, 1) = 1;

  // perform a set of translations of each rotated image
  Mat src(image);
  for (double xdist = -20; xdist <= 20; xdist += 10) {
    if (xdist == 0) continue;

    translationMat.at<double>(0, 2) = xdist;
    translationMat.at<double>(1, 2) = 0;
    IplImage* translatedImage = cvCloneImage(image);
    Mat dest(translatedImage);
    warpAffine(src, dest, translationMat, src.size());

    CvPoint translatedLocation;
    translatedLocation.x = location.x + xdist;
    translatedLocation.y = location.y;

    // check if the translated location is out of bounds with respect
    // to the image window. Do not add those images to the set
    if (translatedLocation.x < 0 || translatedLocation.x > size.width ||
	translatedLocation.y < 0 || translatedLocation.y > size.height) {
      cvReleaseImage(&translatedImage);
      continue;
    }
      
    pair<IplImage*, CvPoint> p = make_pair(translatedImage, translatedLocation);
    transformedImages.push_back(p);
  }

  return transformedImages;
}

// destroyAffineTransforms
// Method to destroy the images generated using affine transforms

void Preprocess::destroyAffineTransforms(vector<ImgLocPairT>& imgLocPairs) {
  for (unsigned int i = 0; i < imgLocPairs.size(); i++)
    cvReleaseImage(&(imgLocPairs[i].first));
  imgLocPairs.clear();
}
