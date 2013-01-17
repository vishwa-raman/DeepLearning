// Globals.cpp
// File that defines all the constants used across all modules

#include "Globals.h"

int Globals::imgWidth = 640;
int Globals::imgHeight = 480;
int Globals::roiWidth = 500;
int Globals::roiHeight = 300;
int Globals::maxDistance = 100;
int Globals::maxAngle = 180;
int Globals::maxArea = 200;
int Globals::binWidth = 10;
int Globals::gaussianWidth = 21;
int Globals::psrWidth = 30;
int Globals::nPastLocations = 5;
int Globals::noseDrop = 70;

int Globals::smallBufferSize = 32;
int Globals::midBufferSize = 256;
int Globals::largeBufferSize = 1024;
int Globals::nSequenceLength = 600;

unsigned Globals::numZones = 3;

double Globals::learningRate = 0.125;
double Globals::initialGaussianScale = 0.5;
double Globals::windowXScale = 30;
double Globals::windowYScale = 25;

string Globals::annotationsFileName = "annotations.xml";
string Globals::modelNamePrefix = "zone_";
string Globals::faceFilter = "MOSSE_Face";
string Globals::leftEyeFilter = "MOSSE_LeftEye";
string Globals::rightEyeFilter = "MOSSE_RightEye";
string Globals::noseFilter = "MOSSE_Nose";

string Globals::paramsFileName = "parameters.xml";
string Globals::configFileName = "config.xml";
