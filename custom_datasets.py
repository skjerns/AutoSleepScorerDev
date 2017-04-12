import numpy as np
import random
import chainer.datasets as datasets

#####
## Base classes

class Data(object):

    def __init__(self):

        self.nbatches = len(self.X) // self.batch_size

        self.step = 0

        self.nexamples = self.X.shape[0]

        # print 'Constructing labelled dataset; batch size: {0}; n batches: {1}'.format(self.batch_size, self.nbatches)
        # print 'Input data: {0} data points x {1} inputs'.format(self.nexamples,self.X.shape[1])
        # print 'Output data: {0} data points x {1} outputs'.format(self.nexamples, self.T.shape[1])

    def __iter__(self):
        return self  # simplest iterator creation

    def next(self):
        pass

    def reset(self):
        self.step = 0

class StaticData(Data):
    """
    Data class for static data consisting of independent data points
    """

    def __init__(self, X, T, batch_size=32):
        """

        :param X: ndatapoints x ninputs input data
        :param T: ndatapoints [x noutputs] target data
        :param batch_size: number of trials per batch

        """

        self.X = X
        self.T = T

        self.batch_size = batch_size

        self.perm = np.random.permutation(np.arange(len(self.X)))

        super(StaticData, self).__init__()

    def next(self):
        """

        :return: x: list of 1D arrays representing examples in the current minibatch
        """

        if self.step == self.nbatches:
            self.step = 0
            raise StopIteration

        x = [self.X[self.perm[(seq * self.nbatches + self.step) % len(self.X)]] for seq in xrange(self.batch_size)]
        t = [self.T[self.perm[(seq * self.nbatches + self.step) % len(self.T)]] for seq in xrange(self.batch_size)]

        self.step += 1

        return x, t

class DynamicData(Data):
    """
       Data class for dynamic data consisting of temporally ordered data points
    """

    def __init__(self, X, T, batch_size=32):
        """

        :param X: ntimepoints x ninputs or ntrials x ntimepoints x ninputs input data
        :param T: ntimepoints [x noutputs] or ntrials x ntimepoints [x noutputs] target data
        :param batch_size: number of trials per batch

        NOTE:
        3D data is converted to 2D data. In this case, each of the trials will be processed in batch mode
        The batch size then becomes equal to ntrials since in each batch, all trials are processed at a certain time point

        """

        if np.ndim(X)==3:
            # convert to 2D

            ntrials, ntimepoints, nvariables = X.shape

            self.X = X.reshape([ntrials * ntimepoints, nvariables])
            self.T = T.reshape([ntrials * ntimepoints, T.shape[2]])

            # number of batches must be equal to number of trials
            self.batch_size = ntrials

        else:

            self.X = X
            self.T = T

            self.batch_size = batch_size

        super(DynamicData, self).__init__()

    def next(self):
        """

        :return: x: list of 1D arrays representing examples in the current minibatch
        """

        if self.step == self.nbatches:
            self.step = 0
            raise StopIteration

        x = [self.X[(seq * self.nbatches + self.step) % len(self.X)] for seq in xrange(self.batch_size)]
        t = [self.T[(seq * self.nbatches + self.step) % len(self.T)] for seq in xrange(self.batch_size)]

        self.step += 1

        return x, t


class DynamicLasagna(Data):
    """
       Data class for dynamic data consisting of batches x 
    """

    def __init__(self, X, T, batch_size=32):
        """

        :param X: ntimepoints x ninputs or ntrials x ntimepoints x ninputs input data
        :param T: ntimepoints [x noutputs] or ntrials x ntimepoints [x noutputs] target data
        :param batch_size: number of trials per batch

        NOTE:
        3D data is converted to 2D data. In this case, each of the trials will be processed in batch mode
        The batch size then becomes equal to ntrials since in each batch, all trials are processed at a certain time point

        """

        if np.ndim(X)==3:
            # convert to 2D

            ntrials, ntimepoints, nvariables = X.shape

            self.X = X.reshape([ntrials * ntimepoints, nvariables])
            self.T = T.reshape([ntrials * ntimepoints, T.shape[2]])

            # number of batches must be equal to number of trials
            self.batch_size = ntrials

        else:

            self.X = X
            self.T = T

            self.batch_size = batch_size

        super(DynamicData, self).__init__()

    def next(self):
        """

        :return: x: list of 1D arrays representing examples in the current minibatch
        """

        if self.step == self.nbatches:
            self.step = 0
            raise StopIteration

        x = [self.X[(seq * self.nbatches + self.step) % len(self.X)] for seq in xrange(self.batch_size)]
        t = [self.T[(seq * self.nbatches + self.step) % len(self.T)] for seq in xrange(self.batch_size)]

        self.step += 1

        return x, t

#####
## Toy datasets

class StaticDataClassification(StaticData):
    """
    Toy dataset for static classification data
    """

    def __init__(self, batch_size=32):

        X = [[random.random(), random.random()] for _ in xrange(1000)]
        T = [0 if sum(i) < 1.0 else 1 for i in X]
        X = np.array(X, 'float32')
        T = np.array(T, 'int32')

        self.nin = X.shape[1]
        self.nout = np.max(T) + 1

        super(StaticDataClassification, self).__init__(X, T, batch_size)


class StaticDataRegression(StaticData):
    """
    Toy dataset for static regression data
    """

    def __init__(self, batch_size=32):

        X = [[random.random(), random.random()] for _ in xrange(1000)]
        T = [[np.sum(i), np.prod(i)] for i in X]
        X = np.array(X, 'float32')
        T = np.array(T, 'float32')

        self.nin = X.shape[1]
        self.nout = T.shape[1]

        super(StaticDataRegression, self).__init__(X, T, batch_size)


class DynamicDataClassification(DynamicData):
    """
    Toy dataset for dynamic classification data in continuous mode
    """

    def __init__(self, batch_size=32):

        X = [[random.random(), random.random()] for _ in xrange(1000)]
        T = [0] + [0 if sum(i) < 1.0 else 1 for i in X][:-1]
        X = np.array(X, 'float32')
        T = np.array(T, 'int32')

        self.nin = X.shape[1]
        self.nout = np.max(T) + 1

        super(DynamicDataClassification, self).__init__(X, T, batch_size)


class DynamicDataRegression(DynamicData):
    """
    Toy dataset for dynamic regression data in continuous mode
    """

    def __init__(self, batch_size=32):

        X = [[random.random(), random.random()] for _ in xrange(1000)]
        T = [[1, 0]] + [[np.sum(i), np.prod(i)] for i in X][:-1]
        X = np.array(X, 'float32')
        T = np.array(T, 'float32')

        self.nin = X.shape[1]
        self.nout = T.shape[1]

        super(DynamicDataRegression, self).__init__(X, T, batch_size)

class DynamicDataRegressionBatch(DynamicData):
    """
    Toy dataset for dynamic regression data in batch mode
    """

    def __init__(self):

        X = [[random.random(), random.random()] for _ in xrange(992)]
        T = [[1, 0]] + [[np.sum(i), np.prod(i)] for i in X][:-1]
        X = np.array(X, 'float32')
        T = np.array(T, 'float32')

        self.nin = X.shape[1]
        self.nout = T.shape[1]

        X = np.reshape(X,[32,31,2])
        T = np.reshape(T,[32,31,2])

        super(DynamicDataRegressionBatch, self).__init__(X, T)

class MNISTData(StaticData):
    """
    Handwritten character dataset; example of handling convolutional input
    """


    def __init__(self, validation=False, convolutional=True, batch_size=32):

        if validation:
            data = datasets.get_mnist()[1]
        else:
            data = datasets.get_mnist()[0]

        X = data._datasets[0].astype('float32')
        T = data._datasets[1].astype('int32')

        if convolutional:
            X = np.reshape(X,np.concatenate([[X.shape[0]], [1], [28, 28]]))
            self.nin = [1, 28, 28]
        else:
            self.nin = X.shape[1]

        self.nout = (np.max(T) + 1)

        super(MNISTData, self).__init__(X, T, batch_size)