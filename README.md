DeepLearning
============

Code to build MLP models for outdoor head orientation tracking

The following is the set of directories with a description of what is
in each.

1. dl
The directory with the python files to train and test an MLP.

The file genmlp.py is based on the mlp.py that is part of the Theano
documentation. It is more general purpose in that one can configure
a network of arbitrary depth and number of nodes per layer. It also
implements a sliding window for training that enables one to train
data sets of arbitrary size on limited GPU memory.

The file logistic_sgd.py comes with a nice reporting function that
builds a matrix with classification results on the test set where we
show the number of correctly classified frames and the distribution
of the incorrectly classified frames across all classes.

The file pickler.py has a number of helper methods that can be used to
build files with the data that conform to the Theano input format. 
The file takes as input files in the MNIST IDX format. It can be used 
to chunk data sets into multiple sets of files, one for training, one
for validation, and the last for test.

2. utils
The directory with C++ code that can be used to generate datasets in
the MNIST IDX format from labeled data. The labels correspond to a
partition of the space in front of a driver in a car, with the 
following values,

1. Driver window
2. Left of center
3. Straight ahead
4. Right of center
5. Passenger window

Given a video of the driver, an annotation file for that video has the
following format,

	<?xml version="1.0"?>
	<annotations dir="/media/CESAR-EXT02/VCode/CESAR_May-Fri-11-11-00-50-2012" center="350,200">
	  <frame>
	    <frameNumber>1</frameNumber>
	    <face>0,0</face>
	    <zone>9</zone>
	    <status>1</status>
	    <intersection>4</intersection>
	  </frame>
	  <frame>
	    <frameNumber>2</frameNumber>
	    <face>0,0</face>
	    <zone>9</zone>
	    <status>1</status>
	    <intersection>4</intersection>
	  </frame>
	  ...
	  ...
	</annotations>

where, the directory is expected to contain frames from the video with the
following filenames "frame_<frameNumber>.png". Each video frame is a
640x480 image file with the zone indicating the class, the status indicating
the car status, and the intersection indicating the type of intersection.
For the purposes of building the data sets, we only use the zone information
at this point. The center is expected to be the rough center of the location
of the face in each frame.

The pre-processing that is done on the images is as follows,

1. A Region of Interest (ROI) of configurable size (Globals.cpp) is picked
around the image center.
2. A histogram equalization followed by edge detection is performed.
3. A DC suppression using a sigmoid is then done.
4. A gaussian window function is applied around the center.
5. The image is scaled and a vector generated from the image matrix in
row-major.

Building everything

Do a make mode=opt in utils/src. This builds everything and places them
in an install directory under DeepLearning.

Running data generation

To generate data sets, use the following commands,

/home/vishwa/work/dbn/train_utils/install/bin/xmlToIDX -o ubyte -r 0.80 -v 0.05 -usebins 
-d /media/CESAR-EXT02/VCode/CESAR_May-Fri-25-17-05-43-2012 -d /media/CESAR-EXT02/VCode/CE
SAR_May-Fri-25-14-55-42-2012 -d /media/CESAR-EXT02/VCode/CESAR_May-Fri-25-11-10-26-2012 -
d /media/CESAR-EXT02/VCode/CESAR_May-Fri-11-11-00-50-2012

python pickler.py data-train-ubyte label-train-ubyte data-valid-ubyte label-valid-ubyte d
ata-test-ubyte label-test-ubyte gaze_data.pkl

The first command builds the IDX format data sets. The second converts
them into a numpy array of tuples, with each tuple being an array of data
points and an array of labels. We have one tuple for the training data, one
for validation, and one for test.

The options to xmlToIDX are as follows,

-o is the suffix to use for all generated files

-r is the training fraction in the interval [0, 1)

-v is the validation fraction in the interval (0, 1)

-usebins is used to bin the data based on their labels. We generate as many
data points as argmin_{l \in labels} |D_l|, where D_l is the set of data 
points with label l; in other workds we pick as many data points as the
cardinality of the smallest set of data points across all labels. This is to
prevent our network from being biased to class label 3, which is straight
ahead. A large fraction of the frames have the driver facing straight ahead
which causes an enormous bias during training without binning.

-d a directory of images for training. An annotation file called 
annotations.xml is expected to be present in each such directory.

The second command builds the tuples of numpy arrays as required by the
Theano based trainer. This one takes as input the training, validation,
and test data and label files with the prefix to use for the generated
file names.

Training and classification

Training and classification can be done using genmlp.py. The following
command will train a network and generate a report with the validation
error rate, test error rate, and the distribution of the numbers of
frames across all classes together with the expected number of frames 
per class.

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python /home/vishwa/work/DeepLearning/dl/genmlp.py -d wide_trimmed.3 -p gaze_data.pkl -nt 3 -nv 1 -ns 1 -l 1500,300 -o 5 -i 15750 -gen wide_trimmed.3/mlpmodel.txt

The options are,

-d the directory that contains the data sets

-p the file name prefix for the files names that hold the data sets

-nt the number of training files

-nv the number of validation files

-ns the number of test files

-l the configuration of the hidden layers in the network, with as many
hidden layers as the number of comma separated elements with the size of 
each hidden layer being the elements

-o the number of labels

-i the input dimension of the data

-gen to generate the trained model for use outside Theano. This is as a text
file. We also generate a pickled file called params.gz in the training data
set directory that contains the numpy weights and biases of all hidden layers
and the final logistic layer.

For questions please send mail to: vishwa.raman@west.cmu.edu

Thanks for looking.

