import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from chainer import Variable, cuda
import chainer
import scipy.stats as ss
print("loaded costum Analysis.py")
class Analysis(object):

    def __init__(self, model, fname=None, gpu=-1):

        self.fname = fname
        self.model = model
        self.Y = []
        self.T = []
        self.xp = np if gpu == -1 else cuda.cupy


    def predict(self, supervised_data):
        """
        Return predicted Y and T
        :param supervised_data: SupervisedData object
        """
        
        self.model.predictor.reset_state()
        # check if we are in train or test mode (e.g. for dropout)
        self.model.predictor.test = True
        self.model.predictor.train = False
        Y = []
        T = []
        for data in supervised_data:
            x = Variable(self.xp.asarray(data[0]), True)
            Y.append(self.xp.argmax(self.model.predict(x),axis=1))
            T.append(data[1])

        Y = self.xp.squeeze(self.xp.asarray(Y, dtype=self.xp.float32))
        T = self.xp.squeeze(self.xp.asarray(T))
        Y = cuda.to_cpu(Y)
        T = cuda.to_cpu(T)
        Y = Y.ravel(order='F')
        T = T.ravel(order='F')
        self.Y = Y
        self.T = T
        return Y,T
    
    def confusion_matrix(self, Y=[], T=[], plot=True):
        """
        Return overall accuracy, confusion matrix, calculated in batches (much faster).
        
        :param supervised_data: SupervisedData object
        """
        if Y==[]: Y = self.Y
        if T==[]: T = self.T
        # confusion matrix
        regressors = np.unique(T)
        nregressors = len(np.unique(T))
        conf_mat = np.zeros([nregressors, nregressors])
        for i in np.arange(nregressors):
            clf = Y[T==regressors[i]]
            for j in np.arange(nregressors):
                conf_mat[i,j] = np.sum(clf == regressors[j])
            conf_mat[i] = conf_mat[i]/np.sum(conf_mat[i])
            
        if plot:
            plt.figure()
            plt.imshow(conf_mat, cmap='jet', interpolation='nearest')
            plt.xlabel('Predicted class')
            plt.ylabel('True class')
            plt.xticks(np.arange(nregressors))
            plt.yticks(np.arange(nregressors))
            plt.gca().set_xticklabels([str(item) for item in regressors])
            plt.gca().set_yticklabels([str(item) for item in regressors])
            plt.colorbar()
            plt.title('Confusion matrix')
            plt.savefig(self.fname + '_classification_analysis.png')
            plt.show()
        return conf_mat


