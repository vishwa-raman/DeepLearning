#!/bin/sh

/home/vishwa/work/dbn/train_utils/install/bin/xmlToIDX -o ubyte -r 0.80 -v 0.05 -usebins -d /media/CESAR-EXT02/VCode/CESAR_May-Fri-25-17-05-43-2012 -d /media/CESAR-EXT02/VCode/CESAR_May-Fri-25-14-55-42-2012 -d /media/CESAR-EXT02/VCode/CESAR_May-Fri-25-11-10-26-2012 -d /media/CESAR-EXT02/VCode/CESAR_May-Fri-11-11-00-50-2012

python pickler.py data-train-ubyte label-train-ubyte data-valid-ubyte label-valid-ubyte data-test-ubyte label-test-ubyte gaze_data.pkl

