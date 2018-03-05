import numpy as np
from sklearn import svm


class MulticlassSVM:

    def __init__(self, mode):
        if mode != 'ovr' and mode != 'ovo' and mode != 'crammer-singer':
            raise ValueError('mode must be ovr or ovo or crammer-singer')
        self.mode = mode
        self.indicator = None
        self.max_indicator = None
        self.ins_w = None

    def fit(self, X, y):
        if self.mode == 'ovr':
            self.fit_ovr(X, y)
        elif self.mode == 'ovo':
            self.fit_ovo(X, y)
        elif self.mode == 'crammer-singer':
            self.fit_cs(X, y)

    def fit_ovr(self, X, y):
        self.labels = np.unique(y)
        self.binary_svm = self.bsvm_ovr_student(X, y)

    def fit_ovo(self, X, y):
        self.labels = np.unique(y)
        self.binary_svm = self.bsvm_ovo_student(X, y)

    def fit_cs(self, X, y):
        self.labels = np.unique(y)
        X_intercept = np.hstack([X, np.ones((len(X), 1))])

        N, d = X_intercept.shape
        K = len(self.labels)

        # instance class indicator matrix:
        # instance's class in one-hot format in the columns
        self.indicator = np.zeros((K, N))
        self.indicator[y, np.arange(N)] = 1

        W = np.zeros((K, d))
        # W = np.random.uniform(low=0.0, high=1.0, size=(K, d))

        n_iter = 2000
        learning_rate = 1e-8
        # for i in range(n_iter):
        for i in range(n_iter):
            print("Step: {:7d}/{:7d} | ".format(i + 1, n_iter), end='')
            W -= learning_rate * self.grad_student(W, X_intercept, y, C=10.0)

        self.W = W

    def init_indicator(self, y):
        """
        Initilize the indicator matrix which indicates the class of the
        correponding instances in one-hot column vectors.

        Arguments:
            y(list): A list of the instances' labels

        Returns:
            None
        """
        N = len(y)
        K = len(self.labels)
        # instance class indicator matrix:
        # instance's class in one-hot format in the columns
        self.indicator = np.zeros((K, N))
        self.indicator[y, np.arange(N)] = 1

    def predict(self, X):
        if self.mode == 'ovr':
            return self.predict_ovr(X)
        elif self.mode == 'ovo':
            return self.predict_ovo(X)
        else:
            return self.predict_cs(X)

    def predict_ovr(self, X):
        scores = self.scores_ovr_student(X)
        return self.labels[np.argmax(scores, axis=1)]

    def predict_ovo(self, X):
        scores = self.scores_ovo_student(X)
        return self.labels[np.argmax(scores, axis=1)]

    def predict_cs(self, X):
        X_intercept = np.hstack([X, np.ones((len(X), 1))])
        return np.argmax(self.W.dot(X_intercept.T), axis=0)

    def collect_data(self, label_1, label_0, X, y):
        """
        Collect the data and labels to fit for binary ovo classification.
        Arguments:
            label_1(int): First label to be collected.
            label_0(int): Second label to be collected.
            X(ndarray): Training data.
            y(list): Training labels.

        Returns:
            data(ndarray): Collected data.
            label: Collected binary labels.
        """
        data, label = list(), list()
        for idx, itr in enumerate(y):
            if itr == label_1 or itr == label_0:
                label.append(1 if itr == label_1 else 0)
                data.append(X[idx, :])
            else:
                continue
        #
        return np.array(data), label

    def bsvm_ovr_student(self, X, y):
        '''
        Train OVR binary classfiers.

        Arguments:
            X, y: training features and labels.

        Returns:
            binary_svm: a dictionary with labels as keys,
                        and binary SVM models as values.
        '''
        binary_svm = dict()
        for itr_label in self.labels:
            # Convert to one-vs-rest binary label
            binary_label = [1 if itr == itr_label else 0 for itr in y]
            # Fit the ovr binary svm with binary label
            clf = svm.LinearSVC(random_state=12345, multi_class='ovr')
            clf.fit(X, binary_label)
            # Save into dictionary
            binary_svm[itr_label] = clf
        #
        return binary_svm

    def bsvm_ovo_student(self, X, y):
        '''
        Train OVO binary classfiers.

        Arguments:
            X, y: training features and labels.

        Returns:
            binary_svm: a dictionary with label pairs as keys,
                        and binary SVM models as values.
        '''
        binary_svm = dict()
        for idx, label_1 in enumerate(self.labels):
            for label_0 in self.labels[idx + 1:]:
                _X, _y = self.collect_data(label_1, label_0, X, y)
                # Fit the ovr binary svm with binary label
                clf = svm.LinearSVC(random_state=12345, multi_class='ovr')
                clf.fit(_X, _y)
                # Save into dictionary
                binary_svm[(label_1, label_0)] = clf
        #
        return binary_svm

    def scores_ovr_student(self, X):
        '''
        Compute class scores for OVR.

        Arguments:
            X: Features to predict.

        Returns:
            scores: a numpy ndarray with scores.
        '''
        scores = list()
        for itr_label in self.labels:
            clf = self.binary_svm[itr_label]
            pred = clf.decision_function(X)
            scores.append(pred)
        # Transpose the score matrix to (instance, label)
        scores = np.array(scores).T
        return scores

    def scores_ovo_student(self, X):
        '''
        Compute class scores for OVO.

        Arguments:
            X: Features to predict.

        Returns:
            scores: a numpy ndarray with scores.
        '''
        n_instance = X.shape[0]
        scores = np.zeros((n_instance, len(self.labels)))
        for idx, label_1 in enumerate(self.labels):
            for label_0 in self.labels[idx + 1:]:
                clf = self.binary_svm[(label_1, label_0)]
                pred = clf.predict(X)
                #
                for idx, itr in enumerate(pred):
                    # Voting
                    if itr == 1:
                        scores[idx, label_1] += 1
                    else:
                        scores[idx, label_0] += 1
        #
        return scores

    def loss_student(self, W, X, y, C=1.0):
        '''
        Compute loss function given W, X, y.

        For exact definitions, please check the MP document.

        Arugments:
            W: Weights. Numpy array of shape (K, d)
            X: Features. Numpy array of shape (N, d)
            y: Labels. Numpy array of shape N
            C: Penalty constant. Will always be 1 in the MP.

        Returns:
            The value of loss function given W, X and y.
        '''
        self.labels = np.unique(y)
        K = len(self.labels)
        N, d = X.shape
        if self.indicator is None:
            self.init_indicator(y)
        # (1) Sum of class weights' 2-norm
        # w_norm = (W * W).sum(axis=1)
        val_norm = 0.5 * sum((W * W).sum(axis=1))

        # (2) First summation
        # delta (instance's class in one-hot format in the columns)
        delta = self.indicator
        # Summation of max values across all instance w.r.t all classes
        # matrix of all instance
        mtx_ins = 1 - delta + W.dot(X.T)

        # Save maximum class index
        max_idx = mtx_ins.argmax(axis=0)
        # Indicator for the class of the maximum value
        # to be used in calculating gradient
        self.max_indicator = np.zeros((K, N))
        self.max_indicator[max_idx, np.arange(N)] = 1

        # find maximum
        ins_max = mtx_ins.max(axis=0)
        val_max = C * sum(ins_max)

        # (3) Second summation
        # instance weight:
        # each column correspond to the weight of that the instance's class
        if self.ins_w is None:
            self.ins_w = W[y, :]
        # Instance inner product:
        # inner product of the instance and its corresponding class weight
        ins_prod = np.einsum('ij,ij->i', self.ins_w, X)
        val_prod = C * sum(ins_prod)

        # Calculate the loss
        loss = val_norm + val_max - val_prod
        return loss

    def grad_student(self, W, X, y, C=1.0):
        '''
        Compute gradient function w.r.t. W given W, X, y.

        For exact definitions, please check the MP document.

        Arugments:
            W: Weights. Numpy array of shape (K, d)
            X: Features. Numpy array of shape (N, d)
            y: Labels. Numpy array of shape N
            C: Penalty constant. Will always be 1 in the MP.

        Returns:
            The graident of loss function w.r.t. W,
            in a numpy array of shape (K, d).
        '''
        self.labels = np.unique(y)
        K = len(self.labels)
        N, d = X.shape
        #
        loss = self.loss_student(W, X, y, C)
        print("Loss: {:8.5f}".format(loss))

        # Gradient
        max_indicator = self.max_indicator
        val_max = max_indicator.dot(X)

        # Instance's class filter (K, d)
        # used to sum up instances of the same class with matrix operation
        ins_filter = self.indicator
        # Gradient of instance inner product:
        prod_grad = ins_filter.dot(X)

        # Calculate gradient
        grad = W + C * (val_max - prod_grad)
        return grad
