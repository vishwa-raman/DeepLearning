// Annotations.cpp
// This file contains the implementation of class Annotations.
// This class is used to read an annotations XML doc and provide an
// iterator to the frame annotations.

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <list>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>

#include "Annotations.h"

// Construction and desctruction

Annotations::Annotations() {
  framesDirectory = "";
  center.x = Globals::imgWidth / 2;
  center.y = Globals::imgHeight / 2;
  useBins = false;
}

Annotations::~Annotations()
{
  // delete annotations
  for (unsigned int i = 0; i < frameAnnotations.size(); i++)
    delete frameAnnotations[i];
}

// getData
// This function takes as input a string and returns an annotation tag 
// corresponding to the annotation. It fills the CvPoint object with
// the data read from that line

Annotations::Tag Annotations::getData(string str, CvPoint& point) {
  const char* token = strtok((char*)str.c_str(), " <>");
  if (token) {
    if (!strcmp(token, "/frame"))
      return EndFrame;
    else if (!strcmp(token, "annotations")) {
      token = strtok(NULL, " <>\"");
      if (token && !strncmp(token, "dir=", 4)) {
	token = strtok(NULL, " <>\"");
	if (!token) {
	  string err = "Annotations::getData. Malformed annotations.xml. No directory name.";
	  throw err;
	}
	framesDirectory = token;
      }
      token = strtok(NULL, " <>\"");
      if (token && !strncmp(token, "center=", 7)) {
	token = strtok(NULL, " <>\"");
	if (!token) {
	  string err = "Annotations::getData. Malformed annotations.xml. No center.";
	  throw err;
	}
	char* chP = (char*)strchr(token, ',');
	if (!chP) {
	  string err = "Annotations::getData. Malformed annotations.xml. No center.";
	  throw err;
	}
	*chP = 0;
	chP++;
	center.x = atoi(token);
	center.y = atoi(chP);
      }
      return Root;
    } else if (!strcmp(token, "frame"))
      return Frame;
    else if (!strcmp(token, "frameNumber")) {
      token = strtok(NULL, " <>");
      point.x = (token)? atoi(token) : 0;
      return FrameNumber;
    } else if (!strcmp(token, "zone")) {
      token = strtok(NULL, " <>");
      point.x = (token)? atoi(token) : 0;
      /*
      if (point.x == 2) 
	point.x = 1;
      else if (point.x == 3)
	point.x = 2;
      else if (point.x == 4 || point.x == 5)
      point.x = 3;
      */
      return Orientation;
    } else if (!strcmp(token, "status")) {
      token = strtok(NULL, " <>");
      point.x = (token)? atoi(token) : 0;
      return CarStatus;
    } else if (!strcmp(token, "intersection")) {
      token = strtok(NULL, " <>");
      point.x = (token)? atoi(token) : 0;
      return IntersectionType;
    } else if (token[0] != '/') {
      string tag = token;

      token = strtok(NULL, " <>");
      const char* field = strtok((char*)token, ",");
      point.y = (field)? atoi(field) : 0;
      field = strtok(NULL, ",");
      point.x = (field)? atoi(field) : 0;

      if (tag == "face") {
 	return Face;
      }
    }
  }
  return Ignore;
}

// trimEnds
// The following method is called to trim the ends of sections of frames that
// are labelled by sector. The number of frames to trim at either end is
// specified through parameter nTrim. We do this to remove badly labelled
// frames at the ends of sections. Since the labelling is contiguous, we 
// are seeing wrongly labelled images at the ends of each section of frames
// by sector. This is particularly acute for sector 3 of which we have 
// an enormous number

void Annotations::trimEnds(int sectorToTrim, int nTrim) {
  vector<FrameAnnotation*> fas;
  int prevSector = Annotations::UnknownSector;
  unsigned int i = 0;
  while (i < frameAnnotations.size()) {
    FrameAnnotation* fa = frameAnnotations[i];

    int sector = fa->getSector();
    if (prevSector != sector && sector == sectorToTrim) {
      unsigned int start = i;
      unsigned int end = i;
      for (unsigned int j = i; j < frameAnnotations.size(); j++) {
	fa = frameAnnotations[j];
	if (fa->getSector() != sectorToTrim) {
	  end = j - 1;
	  break;
	}
      }
      // we now have the indices for the range of frames with 
      // sector equal to sectorToTrim
      for (unsigned int j = start + nTrim; j < end - nTrim; j++) {
	fa = frameAnnotations[j];
	fas.push_back(new FrameAnnotation(fa));
      }
      i = end;
    } else {
      fas.push_back(new FrameAnnotation(fa));
    }
    i++;
    prevSector = sector;
  }
  // now delete and copy frames into frameAnnotations
  for (unsigned int i = 0; i < frameAnnotations.size(); i++)
    delete frameAnnotations[i];
  frameAnnotations.clear();
  for (unsigned int i = 0; i < fas.size(); i++)
    frameAnnotations.push_back(fas[i]);
}

// readAnnotations
// The following method reads an XML file and populates the annotations vector

void Annotations::readAnnotations(string& filename) {
  ifstream file;

  file.open((const char*)filename.c_str());
  if (file.good()) {
    string line;
    int nFrame = 0;
    int sector = UnknownSector;
    int status = UnknownStatus;
    int intersection = UnknownIntersection;

    CvPoint temp, face;
    face.x = face.y = 0;

    getline(file, line); // ignore the first line

    while (!file.eof()) {
      getline(file, line);
      Tag tag = getData(line, temp);
      switch (tag) {
      case FrameNumber: {
	nFrame = temp.x;
	break;
      }
      case Orientation: {
	sector = temp.x;
	break;
      }
      case CarStatus: {
	status = temp.x;
	break;
      }
      case IntersectionType: {
	intersection = temp.x;
	break;
      }
      case Face: {
	face.x = temp.x;
	face.y = temp.y;
	break;
      }
      case EndFrame: {
	if (face.x && face.y) {
	  FrameAnnotation* annotation = 
	    new FrameAnnotation(nFrame, face, sector, status, intersection);
	  frameAnnotations.push_back(annotation); 
	  face.x = face.y = 0;
	} else {
	  FrameAnnotation* annotation = 
	    new FrameAnnotation(nFrame, center /* face */, sector, status, intersection);
	  frameAnnotations.push_back(annotation); 
	}
	break;
      }
      default: {
	continue;
      }
      }
    }
  }
}

// createBins
// The following method is used to create bins for the five gaze sectors, with
// the same number of frames per sector. Since data is typically disproportionately
// skewed towards straigh-ahead, the models are over-trained for that sector

void Annotations::createBins() {
  // first set the useBins flag so that we return the binned annotations on
  // subsequent calls for annotations
  useBins = true;

  int nBins = Globals::numZones;

  // create a counter for each bin and reset it
  int count[nBins];
  for (int i = 0; i < nBins; i++) {
    vector<FrameAnnotation*>* bin = new vector<FrameAnnotation*>();
    bins.push_back(bin);
    count[i] = 0;
  }

  // now iterate over all annotations and place them in their bin based
  // on their zone
  for (unsigned int i = 0; i < frameAnnotations.size(); i++) {
    FrameAnnotation* fa = frameAnnotations[i];
    int index = fa->getSector() - 1;
    if (index < 0 || index > 4)
      continue;

    bins[index]->push_back(fa);
    count[index]++;
  }

  // Get the smallest bin size
  int sampleSize = INT_MAX;
  for (int i = 0; i < nBins; i++)
    if (sampleSize > count[i] && count[i])
      sampleSize = count[i];

  cout << "Creating " << sampleSize << " frames for each sector" << endl;

  // We now create a set of sampleSize * nBins frame annotations in the unif
  // vector
  for (int i = 0; i < nBins; i++) {
    // shuffle images in each bin before picking the first sampleSize images
    random_shuffle(bins[i]->begin(), bins[i]->end());

    // for now pick the first sampleSize elements from each bin
    for (int j = 0; j < sampleSize; j++) {
      if (bins[i]->size()) {
	unif.push_back(bins[i]->back());
	bins[i]->pop_back();
      }
    }
    // Now that we are done with bin i, destroy it
    delete bins[i];
  }

  // final shuffle of the collection of all images
  random_shuffle(unif.begin(), unif.end());

  cout << "Pulled " << unif.size() << " frames from the dataset" << endl;
}
