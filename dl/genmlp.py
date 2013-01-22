"""
mlpseq.py
This file contains an implementation for the training of a configurable
multi-layer perceptron. It is a generalized implementation of mlp.py
from the theano DeepLearningTutorials. The specific enhancements include,

1. Command line configurable network
2. Ability to generate the model as a pickled file and as a text file
3. A sliding window implementation to handle large data sets given any
   GPU to selectively load data into available GPU memory
4. A reporting infrastructure that shows the expected vs. classified
   classes for all test data points that tracks not only how many data
   points were misclassified but their distribution over classes

The training proceeds in the following manner,

1. The first hidden layer is paired with a logistic layer and the
   parameters are trained
2. For all subsequent hidden layers, the following training steps 
   are followed,
   a. Drop the parameters of the logistic layer, but retain the 
      parameter values of all hidden layers trained so far
   b. Add the next hidden layer and the logistic layer
   c. Train the parameters of the newly added hidden layer and the 
      logistic layer
3. A final pass that includes all parameters is optional and is
   being done in the main function

The model is the values of all weights and biases from the first to the
last hidden layer and the logistic regressor.

References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 5

"""
__docformat__ = 'restructedtext en'


import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from pickler import getLists, getPickledLists, getPickledList

from logistic_sgd import LogisticRegression, load_data

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                              layer
        """
        self.input = input

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W%d'%n_out, borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b%d'%n_out, borrow=True)

        self.W = W
        self.b = b

        self.lin_output = T.dot(input, self.W) + self.b
        self.output = (self.lin_output if activation is None
                       else activation(self.lin_output))
        # parameters of the model
        self.params = [self.W, self.b]

class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``SigmoidalLayer`` class)  while the
    top layer is a softmax layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, input, n_in, n_out, layers, weights, biases,
                 includeAllParams = False, activation = T.tanh):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type layers: numpy.array
        :param layers: the number of layers and the number of hidden units 
        per layer

        :type weights: numpy.array of layer weights
        :param weights: the weights for each layer and the logistic layer
        (any or all elements may be None)

        :type biases: numpy.array of layer biases
        :param biases: the biases for each layer and the logistic layer
        (any or all elements may be None)

        :type includeAllParams: boolean
        :param includeAllParams: flag used to indicate that we want to 
        include all parameters during training as opposed to just the
        top hidden layer and the logistic layer
        """

        print 'building MLP'

        # initialize hidden layers
        self.hiddenLayers = []

        # build hidden layers
        for i in range(len(layers)):
            if (i == 0):
                print 'Layer %d. n_in = %d, n_out = %d'%(i + 1, n_in, layers[i])
                self.hiddenLayers.append(HiddenLayer(rng=rng, input=input,
                                                     n_in=n_in, n_out=layers[i],
                                                     activation=activation,
                                                     W=weights[i], b=biases[i]))
            else:
                print 'Layer %d. n_in = %d, n_out = %d'%(i + 1, layers[i - 1], layers[i])
                self.hiddenLayers.append(HiddenLayer(rng=rng, input=self.hiddenLayers[i - 1].output,
                                                     n_in=layers[i - 1], n_out=layers[i],
                                                     activation=activation,
                                                     W=weights[i], b=biases[i]))

        lastHiddenLayer = self.hiddenLayers[-1]

        # The logistic regression layer gets as input the hidden units
        # of the last hidden layer
        self.logRegressionLayer = LogisticRegression(
            input=lastHiddenLayer.output,
            n_in=layers[-1],
            n_out=n_out,
	    W=weights[-1], b=biases[-1])

        # The MLP implemented here is called Progressive MLP as it learns parameters
        # of hidden layers one at a time, re-using the results for all prior 
        # layers from previous learning iterations

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        if (includeAllParams == True):
            self.L1 = sum([abs(self.hiddenLayers[i].W.sum()) for i in xrange(len(layers))]) + \
                abs(self.logRegressionLayer.W).sum()
        else:
            self.L1 = abs(lastHiddenLayer.W).sum() + abs(self.logRegressionLayer.W).sum()

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        if (includeAllParams == True):
            self.L2_sqr = sum([(self.hiddenLayers[i].W ** 2).sum() for i in xrange(len(layers))]) + \
                (self.logRegressionLayer.W ** 2).sum()
        else:
            self.L2_sqr = (lastHiddenLayer.W ** 2).sum() + (self.logRegressionLayer.W ** 2).sum()

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood

        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors

        # the error report function gives more detailed info on how well we
        # did on the training data. It generates for each class the number of
        # data points that were correctly classified, the number of data points
        # that were expected to be classified as belonging to the class, and
        # the numbers of data points that were incorrectly classifier with their
        # distribution over all the unexpected classes
        self.errorReport = self.logRegressionLayer.errorReport

        # the parameters of the model are the parameters of the final two layers it is
        # made out of, that is the last hidden layer that was added followed by the
        # logistic layer
        self.params = self.logRegressionLayer.params
        if (includeAllParams == True):
            for i in range(len(layers)):
                self.params = self.params + self.hiddenLayers[i].params
        else:
            self.params = self.params + lastHiddenLayer.params

class ProgressiveMLP(object):
    """Training class for MLP

    The class implements methods to train an MLP, test the models generated 
    against the held-out (validation) set and apply the model on the test
    data when there is an improvement in error rates over the validation set.
    The class implements a sliding window over the training data to ensure
    that at any given point in time, we only have as much data in GPU memory
    as the determined by the capacity of the GPU memory.

    The class also provides a classify method that can be used to classify
    a given data set using parameters learned from the training cycle. This
    can be used to target the model to different test sets.
    """
    
    def __init__(self, n_in, n_out, layers, weights=None, biases=None, nBatchSize = 20,
                 nWindowSize = 64000, includeAllParams = False):
        """Initialize the parameters for the multilayer perceptron trainer

        :type n_in: integer
        :param n_in: the number of inputs

        :type n_out: integer
        :param n_out: the number of outputs (classes)

        :type layers: numpy.array
        :param layers: the number of layers and the number of hidden units 
        per layer

        :type weights: numpy.array of layer weights
        :param weights: the weights for each layer and the logistic layer
        (any or all elements may be None)

        :type biases: numpy.array of layer biases
        :param biases: the biases for each layer and the logistic layer
        (any or all elements may be None)

        :type nBatchSize: integer
        :param nBatchSize: the size of each minibatch in data points

        :type nWindowSize: integer
        :param nWindowSize: the size of the sliding window over the training
        data. This can be picked based on the size of the GPU memory
        """
        
        self.classifier = None
        self.nSharedLen = nWindowSize
        self.batchSize = nBatchSize
	self.datasets = None
        self.n_in = n_in

        # allocate symbolic variables for the data
        self.index = T.lscalar('index')  # index to minibatch
        self.x = T.matrix('x')           # the data is presented as rasterized images
        self.y = T.ivector('y')          # the labels are presented as 1D vector of
                                         # [int] labels

        rng = numpy.random.RandomState(1234)

        # construct the MLP class
        self.classifier = MLP(rng=rng, 
                              input = self.x, 
                              n_in = n_in,
                              n_out = n_out,
                              layers = layers,
                              weights = weights, 
                              biases = biases,
                              includeAllParams = includeAllParams)

    def initializeSharedData(self, data_xy, length, borrow=True):
        """
        setup shared data for use on the GPU

        We allocate a numpy array that is as large as length and make it shared.
        All subsequent computations on the GPU use the sharedData_x and
        sharedData_y arrays. The length is configurable and should be so chosen
        that we can load as many elements in the available GPU memory
        """
        data_x, data_y = data_xy
        sharedData_x = theano.shared(numpy.asarray(data_x[:length],
                                                   dtype=theano.config.floatX),
                                     borrow=borrow)
        sharedData_y = theano.shared(numpy.asarray(data_y[:length],
                                                   dtype=theano.config.floatX),
                                     borrow=borrow)
        return sharedData_x, sharedData_y

    def getNumberOfSplitBatches(self):
        """
        Given a size of the arrays that are stored in the GPU memory, this
        method returns the number of batches that can be accomodated in 
        that size
        """
        return self.nSharedLen / self.batchSize

    def getWindowData(self, data_xy, miniBatchIndex):
        """
        method used to return a chunk of data from the data_xy that is as big
        as the size of our sliding window, based on a miniBatchIndex. The
        miniBatchIndex will range over all the minibatches in data_xy. 
        Given a miniBatchIndex, we determine which data chunk contains
        that miniBatchIndex and return that chunk from data_xy
        """
        data_x, data_y = data_xy
        index = miniBatchIndex / self.getNumberOfSplitBatches()
#        print '    Returning data_xy[%d].'%(index)
        return data_x[index], data_y[index]

    def splitList(self, data_xy):
        """
        method used to split data_xy into chunks, where each chunk is as big
        as the window size (self.nSharedLen)
        """
        data_x, data_y = data_xy

        print 'in split. %d %d %d'%(len(data_x), len(data_y), len(data_x[0]))
        split_x = numpy.split(data_x, range(0, len(data_x), self.nSharedLen))
        split_y = numpy.split(data_y, range(0, len(data_y), self.nSharedLen))

        return split_x[1:], split_y[1:]

    def getNumberOfBatches(self, data_xy):
        """
        method used to get the total number of batches in data_xy. The
        method expects an array of chunks, walks each chunk and accumulates
        the number of batches in that chunk
        """
        data_x, data_y = data_xy
        nBatches = 0
        for i in range(0, len(data_x)):
            nBatches = nBatches + len(data_x[i]) / self.batchSize
        return nBatches

    def loadDataSets(self, datasets, datasetFileName, datasetDirectory, prefix,
		     nTrainFiles, nValidFiles, nTestFiles):
        """
        method to load the data sets. The data sets are expected to either be in
        a single pickled file that contains training, validation, and test sets
        as an array of tuples, where each tuple is two arrays, one for the data
        and the other for the labels. Please refer to the MNIST data format for
        more on this. We follow the same format here.

        :type datasets: numpy.array of tuples
        :param datasets: an array of tuples one each for training, validation and test

        :type datasetFileName: string
        :param datasetFileName: the name of a pickled file that contains all the data
        May be None in which case we expect that the datasetDirectory points to 
        the location of the data.

        :type datasetDirectory: string
        :param datasetDirectory: location of the data

        :type prefix: string
        :param prefix: the filename prefix to use for the data files. The filenames
        are composed using the prefix and one of _train_, _valid_, or _test_ 
        followed by an index ranging from 0 to the number of files for each.
        The number of training files, validation files and test files are also
        passed as additional arguments to this method.

        NOTE: We propose both mechanisms to load data as for smaller data sets
        the pickler can generate a single pickled file, but for large data sets
        we need to chunk the data up and pickle each chunk separately. This is
        due to a limitation in cPickle that cannot handle very large files

        """
	if datasets is None:
            # Load the dataset
            if (datasetFileName is not None):
                f = gzip.open(datasetFileName, 'rb')
                trainSet, validSet, testSet = cPickle.load(f)
                f.close()
            else:
	        trainSet = getPickledList(datasetDirectory, prefix + '_train_', nTrainFiles)
	        validSet = getPickledList(datasetDirectory, prefix + '_valid_', nValidFiles)
	        testSet = getPickledList(datasetDirectory, prefix + '_test_', nTestFiles)
            self.datasets = (trainSet, validSet, testSet)
	else:
	    self.datasets = datasets

    def classify(self, learningRate = 0.01, L1_reg = 0.00, L2_reg = 0.0001, n_epochs = 1000,
                 datasetFileName = None,
                 datasetDirectory = None,
                 prefix = 'gaze_data.pkl', 
		 nTrainFiles = 1,
                 nValidFiles = 1, 
                 nTestFiles = 1, 
                 datasets = None,
                 batchSize = 20):
        """
        method used to classify a given test set against a model that is expected
        to have been loaded. The method is akin to a subset of the training method
        in that it simply computes the validation loss, test loss, and reports
        the test error, for a given model
        """

	if self.datasets is None:
	    self.loadDataSets(datasets, datasetFileName, datasetDirectory, prefix,
			      nTrainFiles, nValidFiles, nTestFiles)

        self.batchSize = batchSize

        validSet = self.datasets[1]
        testSet = self.datasets[2]

        validSet_x, validSet_y = self.initializeSharedData(self.datasets[1],len(validSet[0]))
        testSet_x, testSet_y = self.initializeSharedData(self.datasets[2], len(testSet[0]))

        # compute number of minibatches for validation and testing
        nValidBatches = len(validSet[0]) / self.batchSize
        nTestBatches = len(testSet[0]) / self.batchSize

        print nValidBatches
        print nTestBatches

        # compiling a Theano function that computes the mistakes that are made
        # by the model on a minibatch
        test_model = theano.function(inputs=[self.index],
            outputs=self.classifier.errors(self.y),
            givens={
                self.x: testSet_x[self.index * batchSize:(self.index + 1) * batchSize],
                self.y: T.cast(testSet_y[self.index * batchSize:(self.index + 1) * batchSize],
                               'int32')})

        validate_model = theano.function(inputs=[self.index],
            outputs=self.classifier.errors(self.y),
            givens={
                self.x: validSet_x[self.index * batchSize:(self.index + 1) * batchSize],
                self.y: T.cast(validSet_y[self.index * batchSize:(self.index + 1) * batchSize],
                               'int32')})

        # error reporting function that computes the overall rate of misclassification
        # by class
        error_model = theano.function(inputs=[self.index],
            outputs=self.classifier.errorReport(self.y, batchSize),
            givens={
                self.x: testSet_x[self.index * batchSize:(self.index + 1) * batchSize], 
                self.y: T.cast(testSet_y[self.index * batchSize:(self.index + 1) * batchSize],
                               'int32')})

        validationLosses = [validate_model(i) for i
                             in xrange(nValidBatches)]
        validationLoss = numpy.mean(validationLosses)

        # test it on the test set
        testLosses = [test_model(i) for i
                       in xrange(nTestBatches)]
        testScore = numpy.mean(testLosses)

        print(('Best validation score of %f %% with test performance %f %%') %
          (validationLoss * 100., testScore * 100.))
        print('Classification errors by class')
        error_mat = [error_model(i) for i in xrange(nTestBatches)]
        class_errors = error_mat[0]
        for i in xrange(len(error_mat) - 1):
            class_errors = numpy.add(class_errors, error_mat[i + 1])
        print class_errors

    def train(self, learningRate = 0.01, L1_reg = 0.00, L2_reg = 0.0001, n_epochs = 1000,
              datasetFileName = None,
              datasetDirectory = None,
              prefix = 'gaze_data.pkl', 
              nTrainFiles = 1, 
              nValidFiles = 1, 
              nTestFiles = 1, 
              datasets = None,
              batchSize = 20):
        """
        method that trains the MLP

        The training data is accessed through the sliding window. For each
        epoch we walk through the training mini batches and compute a cost
        and update the model. Based on the validation frequency, the model
        is checked against the validation set and if there is an 
        improvement at least as much as the improvement threshold, we check
        the model against the test set. Other pieces of code do things
        such as termination based on patience
        """

	if self.datasets is None:
	    self.loadDataSets(datasets, datasetFileName, datasetDirectory, prefix,
			      nTrainFiles, nValidFiles, nTestFiles)

        # compute the size of the window we would like to use
        self.nSharedLen = batchSize * 2000
        self.batchSize = batchSize

        trainSet_x, trainSet_y = self.initializeSharedData(self.datasets[0], self.nSharedLen)

        validSet = self.datasets[1]
        testSet = self.datasets[2]

        validSet_x, validSet_y = self.initializeSharedData(self.datasets[1],len(validSet[0]))
        testSet_x, testSet_y = self.initializeSharedData(self.datasets[2], len(testSet[0]))

        trainSet = self.splitList(self.datasets[0])

        # compute number of minibatches for training, validation and testing
        nTrainBatches = self.getNumberOfBatches(trainSet)
        nSplitTrainBatches = self.getNumberOfSplitBatches()
        nValidBatches = len(validSet[0]) / self.batchSize
        nTestBatches = len(testSet[0]) / self.batchSize

        print nTrainBatches
        print nSplitTrainBatches
        print nValidBatches
        print nTestBatches

        print '... building the model'

        # the cost we minimize during training is the negative log likelihood of
        # the model plus the regularization terms (L1 and L2); cost is expressed
        # here symbolically
        cost = self.classifier.negative_log_likelihood(self.y) \
            + L1_reg * self.classifier.L1 \
            + L2_reg * self.classifier.L2_sqr

        # compiling a Theano function that computes the mistakes that are made
        # by the model on a minibatch
        test_model = theano.function(inputs=[self.index],
            outputs=self.classifier.errors(self.y),
            givens={
                self.x: testSet_x[self.index * batchSize:(self.index + 1) * batchSize],
                self.y: T.cast(testSet_y[self.index * batchSize:(self.index + 1) * batchSize],
                               'int32')})

        validate_model = theano.function(inputs=[self.index],
            outputs=self.classifier.errors(self.y),
            givens={
                self.x: validSet_x[self.index * batchSize:(self.index + 1) * batchSize],
                self.y: T.cast(validSet_y[self.index * batchSize:(self.index + 1) * batchSize],
                               'int32')})

        # error reporting function that computes the overall rate of misclassification
        # by class
        error_model = theano.function(inputs=[self.index],
            outputs=self.classifier.errorReport(self.y, batchSize),
            givens={
                self.x: testSet_x[self.index * batchSize:(self.index + 1) * batchSize], 
                self.y: T.cast(testSet_y[self.index * batchSize:(self.index + 1) * batchSize],
                               'int32')})

        # compute the gradient of cost with respect to theta (sotred in params)
        # the resulting gradients will be stored in a list gparams
        gparams = []
        for param in self.classifier.params:
            gparam = T.grad(cost, param)
            gparams.append(gparam)

        # specify how to update the parameters of the model as a dictionary
        updates = {}
        # given two list the zip A = [ a1,a2,a3,a4] and B = [b1,b2,b3,b4] of
        # same length, zip generates a list C of same size, where each element
        # is a pair formed from the two lists :
        #    C = [ (a1,b1), (a2,b2), (a3,b3) , (a4,b4) ]
        for param, gparam in zip(self.classifier.params, gparams):
            updates[param] = param - learningRate * gparam

        # compiling a Theano function `train_model` that returns the cost, but
        # in the same time updates the parameter of the model based on the rules
        # defined in `updates`
        train_model = theano.function(inputs=[self.index], outputs=cost,
                updates=updates,
                givens={
                self.x: trainSet_x[self.index * batchSize:(self.index + 1) * batchSize],
                self.y: T.cast(trainSet_y[self.index * batchSize:(self.index + 1) * batchSize],
                               'int32')})

        print '... training'

        # early-stopping parameters
        patience = 20000       # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
                               # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                       # considered significant
        validationFrequency = min(nTrainBatches, patience / 2)
                                       # go through this many
                                       # minibatche before checking the network
                                       # on the validation set; in this case we
                                       # check every epoch

        bestParams = None
        bestValidationLoss = numpy.inf
        bestIter = 0
        testScore = 0.
        startTime = time.clock()

        epoch = 0
        doneLooping = False

        while (epoch < n_epochs) and (not doneLooping):
            epoch = epoch + 1
            for minibatchIndex in xrange(nTrainBatches):

                actualMiniBatchIndex = minibatchIndex % nSplitTrainBatches
#                print '    actualMiniBatchIndex = %d. miniBatchIndex = %d'\
#                    %(actualMiniBatchIndex, minibatchIndex)
                if (actualMiniBatchIndex == 0):
                    data_x, data_y = self.getWindowData(trainSet, minibatchIndex)
#                    print '     Update. data_x[0][0] = %f, data_y[0] = %d.'%(data_x[0][0], data_y[0])
                    trainSet_x.set_value(data_x, borrow=True)
                    trainSet_y.set_value(numpy.asarray(data_y,
                                                       dtype=theano.config.floatX),
                                         borrow=True)

                minibatchAvgCost = train_model(actualMiniBatchIndex)
                # iteration number
                iter = epoch * nTrainBatches + minibatchIndex

                if (iter + 1) % validationFrequency == 0:
                    # compute zero-one loss on validation set
                    validationLosses = [validate_model(i) for i
                                         in xrange(nValidBatches)]
                    thisValidationLoss = numpy.mean(validationLosses)

                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                          (epoch, minibatchIndex + 1, nTrainBatches,
                           thisValidationLoss * 100.))

                    # if we got the best validation score until now
                    if thisValidationLoss < bestValidationLoss:
                        #improve patience if loss improvement is good enough
                        if thisValidationLoss < bestValidationLoss *  \
                                improvement_threshold:
                            patience = max(patience, iter * patience_increase)

                        bestValidationLoss = thisValidationLoss
                        bestIter = iter

                        # test it on the test set
                        testLosses = [test_model(i) for i
                                       in xrange(nTestBatches)]
                        testScore = numpy.mean(testLosses)

                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatchIndex + 1, nTrainBatches,
                               testScore * 100.))

                if patience <= iter:
                    doneLooping = True
                    break

        endTime = time.clock()
        print(('Optimization complete. Best validation score of %f %% '
               'obtained at iteration %i, with test performance %f %%') %
              (bestValidationLoss * 100., bestIter, testScore * 100.))
        print('Classification errors by class')
        error_mat = [error_model(i) for i in xrange(nTestBatches)]
        class_errors = error_mat[0]
        for i in xrange(len(error_mat) - 1):
            class_errors = numpy.add(class_errors, error_mat[i + 1])
        print class_errors
        print >> sys.stderr, ('The code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((endTime - startTime) / 60.))

if __name__ == '__main__':
    datasetDirectory = None
    datasetFileName = None
    batchSize = 20
    nTrainFiles = 12
    nValidFiles = 1
    nTestFiles = 1
    layers = None
    inputs = 1000
    outputs = 5
    doClassify = False
    useParamsFromFile = False
    paramsFileName = None
    genModels = False
    modelsFileName = None
    prefix = 'gaze_data.pkl'

    for i in range(len(sys.argv)):
        if sys.argv[i] == '-d':
            datasetDirectory = sys.argv[i + 1]
        elif sys.argv[i] == '-f':
            datasetDirectory = None
            datasetFileName = sys.argv[i + 1]
        elif sys.argv[i] == '-b':
            batchSize = int(sys.argv[i + 1])
        elif sys.argv[i] == '-nt':
            nTrainFiles = int(sys.argv[i + 1])
        elif sys.argv[i] == '-nv':
            nValidFiles = int(sys.argv[i + 1])
        elif sys.argv[i] == '-ns':
            nTestFiles = int(sys.argv[i + 1])
        elif sys.argv[i] == '-p':
            prefix = sys.argv[i + 1]
        elif sys.argv[i] == '-l':
            l = sys.argv[i + 1]
            li = l.split(',')
            layers = numpy.array(li, dtype=numpy.int64)
        elif sys.argv[i] == '-o':
            outputs = int(sys.argv[i + 1])
        elif sys.argv[i] == '-i':
            inputs = int(sys.argv[i + 1])
        elif sys.argv[i] == '-classify':
            doClassify = True
        elif sys.argv[i] == '-gen':
            genModels = True
            modelsFileName = sys.argv[i + 1]
        elif sys.argv[i] == '-useparams':
            useParamsFromFile = True
            paramsFileName = sys.argv[i + 1]
        elif sys.argv[i] == '-h':
            print('Usage: mlp.py (-d datasetDir | -f datasetFileName) [-p prefix] [-b batchSize]' +
                  '[-nt nTrainFiles] [-nv nValidFiles] [-ns nTestFiles]' +
                  '[-i inputLength] [-o numClasses] [-gen modelFileName]'+
                  '[-l [nLayer1Size, nLayer2Size, ...]] [-classify] [-useparams paramsFileName]' +
                  '[-h help]')
            sys.exit()

    if (doClassify == False):
        if (paramsFileName is not None):
            print 'loading parameters from ' + paramsFileName
            paramsFileHandle = gzip.open(paramsFileName, 'rb')
            params = cPickle.load(paramsFileHandle)
            weights, biases = params
        else:
            weights = []
            biases = []
            # + 1 for the logistic layer
            for i in range(len(layers) + 1):
                weights.append(None)
                biases.append(None)

        # initialize datasets
        datasets = None

        l = []
        for i in xrange(len(layers)):
            W = []
            b = []
            l.append(layers[i])

            for j in xrange(i):
                W.append(weights[j])
                b.append(biases[j])
            # One for the final hidden layer and another for the logistic layer
            W.extend([None, None])
            b.extend([None, None])

            mlp = ProgressiveMLP(n_in = inputs, n_out = outputs, layers = l,
                                 weights = W, biases = b)
            if (datasets is not None):
                mlp.train(datasets = datasets)
            else:
                mlp.train(
                    datasetDirectory = datasetDirectory,
                    datasetFileName = datasetFileName,
                    prefix = prefix,
                    batchSize = batchSize,
                    nTrainFiles = nTrainFiles,
                    nValidFiles = nValidFiles,
                    nTestFiles = nTestFiles)

            weights = []
            biases = []
            for i in range(len(l)):
                weights.append(mlp.classifier.hiddenLayers[i].W)
                biases.append(mlp.classifier.hiddenLayers[i].b)
            datasets = mlp.datasets

        # final pass where we include all parameters
        weights.append(mlp.classifier.logRegressionLayer.W)
        biases.append(mlp.classifier.logRegressionLayer.b)
        mlp = ProgressiveMLP(n_in = inputs, n_out = outputs, layers = l,
                             weights = weights, biases = biases, 
                             includeAllParams = True)
        mlp.train(datasets = datasets)

        # if we want the models written out then
        if (genModels == True):
            # first write out the models as numpy arrays using the pickler
            paramsFileName = datasetDirectory + '/params.gz'
            weights = []
            biases = []
            for i in xrange(len(mlp.classifier.hiddenLayers)):
                weights.append(mlp.classifier.hiddenLayers[i].W)
                biases.append(mlp.classifier.hiddenLayers[i].b)
            weights.append(mlp.classifier.logRegressionLayer.W)
            biases.append(mlp.classifier.logRegressionLayer.b)
        
            params = (weights, biases)
            paramsFileHandle = gzip.open(paramsFileName, 'wb')
            cPickle.dump(params, paramsFileHandle)

            # now write out the models as a text file for loading outside Python
            mfp = open(modelsFileName, 'wb')
            for i in xrange(len(mlp.classifier.hiddenLayers) + 1):
                W = weights[i].get_value()
                b = biases[i].get_value()
                mfp.write('%d,%d\n'%(W.shape[0], W.shape[1]))
                for j in xrange(W.shape[0]):
                    for k in xrange(W.shape[1]):
                        mfp.write('%f'%W[j][k])
                        mfp.write(',')
                    mfp.write('\n')
                mfp.write('%d\n'%b.shape[0])
                for j in xrange(b.shape[0]):
                    mfp.write('%f'%b[j])
                    mfp.write(',')
                mfp.write('\n')
            mfp.close()
    else:
        # classify mode
        if (paramsFileName == ''):
            print 'Parameters file is required for sequential MLP'
            sys.exit()

	print 'loading parameters from ' + paramsFileName
        paramsFileHandle = gzip.open(paramsFileName, 'rb')
        params = cPickle.load(paramsFileHandle)
        weights, biases = params

	mlp = ProgressiveMLP(n_in=inputs, n_out=outputs, layers=layers,
			     weights=weights, biases=biases)
        mlp.classify(
            datasetDirectory=datasetDirectory,
            datasetFileName=datasetFileName,
            prefix=prefix,
            batchSize=batchSize,
            nTrainFiles=nTrainFiles,
            nValidFiles=nValidFiles,
            nTestFiles=nTestFiles)

