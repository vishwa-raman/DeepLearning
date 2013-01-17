#ifndef __ANNOTATIONS_H
#define __ANNOTATIONS_H

// Annotations.h
// This file contains the definition of class Annotations. It provides an interface
// to read and access the frames stored in an annotations file.

#include <string>
#include <vector>

#include "Globals.h"

// openCV stuff
#include <cv.h>
#include <highgui.h>

using namespace std;

// forward declaration
class FrameAnnotation;

class Annotations {
 private:
  // the directory containing the frames for a given annotations.xml file
  string framesDirectory;

  // the center of the face in sector 3 (straight ahead)
  CvPoint center;

  // use bins flag
  bool useBins;

  // the set of all annotations
  vector<FrameAnnotation*> frameAnnotations;

  // bins for annotations. We want to only return annotations that are uniformly
  // distributed across the image regions within which the LOIs occur. This 
  // ensures that the training is unbiased
  vector<vector<FrameAnnotation*>* > bins;

  // the set of annotations after binning. We will have min(bin sizes) * number of bins
  // annotations that will be placed in this vector
  vector<FrameAnnotation*> unif;

 public:  
  enum Tag {
    Root,
    Frame,
    FrameNumber,
    Face,
    Orientation,
    CarStatus,
    IntersectionType,
    EndFrame,
    Ignore
  };
  enum Sector {
    DriverWindow = 1,
    LeftOfCenter,
    Center,
    RightOfCenter,
    PassengerWindow,
    CoPilot,
    OverLeftShoulder,
    OverRightShoulder,
    UnknownSector = 9
  };
  enum Status {
    stationaryIgnore = 1,
    stationaryParked,
    stationaryAtIntersection,
    movingAtIntersection,
    movingInCarPark,
    movingOnStreet,
    UnknownStatus = 7
  };
  enum Intersection {
    FourWay = 1,
    TJunction,
    CarParkExit,
    UnknownIntersection = 4
  };

  Annotations();
  ~Annotations();

  void readAnnotations(string& filename);
  void trimEnds(int sector, int nTrim);
  void createBins();
  CvPoint& getCenter() { return center; }
  string getFramesDirectory() { return framesDirectory; }

  vector<FrameAnnotation*>& getFrameAnnotations() { 
    return useBins? unif : frameAnnotations;
  }

 private:
  Annotations::Tag getData(string str, CvPoint& point);
};

// Frame annotations class

class FrameAnnotation {
 private:
  int nFrame;
  CvPoint face;
  int sector;
  int status;
  int intersection;

 public:
  FrameAnnotation() {
    nFrame = 0;
    face.x = face.y = 0;
    sector = Annotations::UnknownSector;
    status = Annotations::UnknownStatus;
    intersection = Annotations::UnknownIntersection;
  }
  FrameAnnotation(FrameAnnotation& fa) {
    nFrame = fa.nFrame;
    face.x = fa.face.x;
    face.y = fa.face.y;
    sector = fa.sector;
    status = fa.status;
    intersection = fa.intersection;
  }
  FrameAnnotation(int frame, CvPoint& f, int s, int st, int intx) { 
    nFrame = frame;
    face.x = f.x; face.y = f.y;
    sector = s;
    status = st;
    intersection = intx;
  }
  FrameAnnotation(FrameAnnotation* fa) {
    nFrame = fa->getFrameNumber();
    setFace(fa->getFace());
    sector = fa->getSector();
    status = fa->getStatus();
    intersection = fa->getIntersection();
  }

  int getFrameNumber() { return nFrame; }
  CvPoint& getFace() { return face; }
  int getSector() { return sector; }
  int getStatus() { return status; }
  int getIntersection() { return intersection; }

  CvPoint& getLOI(Annotations::Tag tag) {
    switch (tag) {
    case Annotations::Face:
      return getFace();
    default: {
      string err = "FrameAnnotation::getLOI. Unknown tag.";
      throw(err);
    }
    }
  }

  // set functions
  void setFace(CvPoint& point) {
    face.x = point.x;
    face.y = point.y;
  }

  // print functions
  void print() {
    cout << "Frame: (" << nFrame << ") "; 
    cout << "Face: (" << face.x << ", " << face.y << ") ";
    cout << "Sector: (" << sector << ") "; 
    cout << "Status: (" << status << ") "; 
    cout << "Intersection: (" << intersection << ")" << endl; 
  }
};

#endif
