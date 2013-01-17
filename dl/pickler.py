"""
pickler.py
This file implements a pickler for the image data. The input files are
expected to be in the MNIST dataset format. The output is a set of files
in the Theano data format.

Please refer to the MNIST documentation for the input format. 
The output format consists of a numpy array of tuples, the first is the
training data, the second is the validation data and the third is the
test data. Each tuple consists of an array of image vectors, where each
vector is a floating point pixel value for image pixels stored in 
row-major, and an array of labels for each image vector.

Since cPickle has limits on the sizes of the arrays that it can process
in memory, we generate chunks of the arrays in files named as follows,

<prefix>_train_%d.gz for the training data
<prefix>_valid_%d.gz for the validation data
<prefix>_test_%d.gz for the test data

The prefix we use is typically gaze_data_pkl.

We set an arbitrary limit of 8000 datapoints and labels per file.
"""
__docformat__ = 'restructedtext en'

import datetime, shutil, glob, sys, os, thread
import time, math, re, unicodedata, struct
import cPickle, gzip
import numpy

global trainDataFile
global trainLabelFile
global validDataFile
global validLabelFile
global testDataFile
global testLabelFile

def readMagic(fileHandle):
    fileHandle.seek(0)
    return struct.unpack('i', fileHandle.read(4))

def readLength(fileHandle):
    fileHandle.seek(4)
    return struct.unpack('i', fileHandle.read(4))

def readWidth(fileHandle):
    fileHandle.seek(8)
    return struct.unpack('i', fileHandle.read(4))

def readHeight(fileHandle):
    fileHandle.seek(12)
    return struct.unpack('i', fileHandle.read(4))

def readByte(fileHandle):
    return struct.unpack('B', fileHandle.read(1))

def loadData(dataFile, labelFile):
    """
    method used to load the input file that is expected to be in the MNIST
    data format
    """
    dataFileHandle = open(dataFile, 'rb')
    labelFileHandle = open(labelFile, 'rb')
    length = readLength(dataFileHandle)[0]
    width = readWidth(dataFileHandle)[0]
    height = readHeight(dataFileHandle)[0]

    data = numpy.zeros((length, width * height), numpy.float32)
    labels = numpy.zeros((length), numpy.int64)
    
    dataFileHandle.seek(16)
    labelFileHandle.seek(8)
    labelFmt = ''
    for i in range(length):
        labelFmt += 'B'
    labelBytes = numpy.int64(struct.unpack(labelFmt, labelFileHandle.read(length)))
    dataFmt = ''
    for i in range(width * height):
        dataFmt += 'B'
    for i in range(length):
        dataBytes = numpy.float32(struct.unpack(dataFmt,\
                                  dataFileHandle.read(width * height)))
        data[i] = numpy.divide(dataBytes, 255.0)
        labels[i] = labelBytes[i] - 1
#        print(data[i])
#        print(labels[i])
        sys.stdout.write('.')
    sys.stdout.write('\n')
    dataFileHandle.close()
    labelFileHandle.close()
    return (data, labels)

def getLists(directory, suffix):
    """
    helper method used within the Theano code to load data from MNIST
    to Theano format, when the entire data can be encapsulated in a
    single file
    """
    dataFileName = directory + '/data-train-' + suffix
    labelFileName = directory + '/label-train-' + suffix
    trainList = loadData(dataFileName, labelFileName)

    dataFileName = directory + '/data-valid-' + suffix
    labelFileName = directory + '/label-valid-' + suffix
    validList = loadData(dataFileName, labelFileName)

    dataFileName = directory + '/data-test-' + suffix
    labelFileName = directory + '/label-test-' + suffix
    testList = loadData(dataFileName, labelFileName)

    return (trainList, validList, testList)

def getPickledLists(directory, prefix, nFiles):
    """
    method used to load data from the MNIST data format to Theano data
    format when just the training data is chunked into multiple files,
    but the validation and test is in a single file
    """
    for i in xrange(nFiles):
        dataFileName = directory + '/' + prefix + ('_train_%d.gz'%i)
        f = gzip.open(dataFileName, 'rb')
        trainList = cPickle.load(f)
        x, y = trainList
        if (i == 0):
            data_x = x
            data_y = y
        else:
            data_x = numpy.concatenate((data_x, x))
            data_y = numpy.concatenate((data_y, y))
        f.close()

    trainList = (data_x, data_y)

    f = gzip.open(directory + '/' + prefix + '.gz')
    validList, testList = cPickle.load(f)
    f.close()

    return (trainList, validList, testList)

def getPickledList(directory, prefix, nFiles):
    """
    method that is used to load a particular set of data files,
    either training, validation or test. This method expects as
    inputs the directory where the files are stored, the prefix
    to use and the number of files that contain the chunked
    data sets.
    """
    for i in xrange(nFiles):
        dataFileName = directory + '/' + prefix + ('%d.gz'%i)
        f = gzip.open(dataFileName, 'rb')
        dataList = cPickle.load(f)
        x, y = dataList
        if (i == 0):
            data_x = x
            data_y = y
        else:
            data_x = numpy.concatenate((data_x, x))
            data_y = numpy.concatenate((data_y, y))
        f.close()

    return (data_x, data_y)

def pickleMeThis(outFile):
    """
    method used to pickle data in the Theano format. Since the pickler
    has limits on data size, we chunk the data arrays into 8000 
    (arbitrary) elements per chunk and store them as pickled files.
    """
    global trainDataFile
    global trainLabelFile
    global validDataFile
    global validLabelFile
    global testDataFile
    global testLabelFile

    trainList = loadData(trainDataFile, trainLabelFile)
    validList = loadData(validDataFile, validLabelFile)
    testList = loadData(testDataFile, testLabelFile)
 
    # checking data that was loaded
    print("length of trainList[0] = %d"%len(trainList[0]))
    print("length of trainList[1] = %d"%len(trainList[1]))
    print("length of trainList[0][1] = %d"%len(trainList[0][1]))
    print("label for trainList[0][1] = %d"%(trainList[1][1]))

    print("length of validList[0] = %d"%len(validList[0]))
    print("length of validList[1] = %d"%len(validList[1]))
    print("length of validList[0][1] = %d"%len(validList[0][1]))
    print("label for validList[0][1] = %d"%(validList[1][1]))

    print("length of testList[0] = %d"%len(testList[0]))
    print("length of testList[1] = %d"%len(testList[1]))
    print("length of testList[0][1] = %d"%len(testList[0][1]))
    print("label for testList[0][1] = %d"%(testList[1][1]))

    index = 0
    data_x, data_y = trainList
    for i in range(0, len(trainList[0]), 8000):
        trainFileHandle = gzip.open(outFile + ('_train_%d.gz'%index), 'wb')
	lb = i
	rb = min(lb + 8000, len(trainList[0]))
	fragment = (data_x[lb:rb], data_y[lb:rb])
        cPickle.dump(fragment, trainFileHandle)
        trainFileHandle.close()
	index = index + 1

    index = 0
    data_x, data_y = validList
    for i in range(0, len(validList[0]), 8000):
        validFileHandle = gzip.open(outFile + ('_valid_%d.gz'%index), 'wb')
	lb = i
	rb = min(lb + 8000, len(validList[0]))
	fragment = (data_x[lb:rb], data_y[lb:rb])
        cPickle.dump(fragment, validFileHandle)
        validFileHandle.close()
	index = index + 1

    index = 0
    data_x, data_y = testList
    for i in range(0, len(testList[0]), 8000):
        testFileHandle = gzip.open(outFile + ('_test_%d.gz'%index), 'wb')
	lb = i
	rb = min(lb + 8000, len(testList[0]))
	fragment = (data_x[lb:rb], data_y[lb:rb])
        cPickle.dump(fragment, testFileHandle)
        testFileHandle.close()
	index = index + 1

if __name__ == "__main__":
    global trainDataFile
    global trainLabelFile
    global validDataFile
    global validLabelFile
    global testDataFile
    global testLabelFile

    trainDataFile = sys.argv[1]
    trainLabelFile = sys.argv[2]
    validDataFile = sys.argv[3]
    validLabelFile = sys.argv[4]
    testDataFile = sys.argv[5]
    testLabelFile = sys.argv[6]
    outFile = sys.argv[7]

    pickleMeThis(outFile)

