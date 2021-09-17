############################
#   COSC522 - Functions    #
#       Matthew Roach      #
#        000609145         #
#      Sep 13, 2021        #
############################

# Collection of functions used in COSC522 Machine Learning

import numpy as np


def gaussian(x, mu, sig):
    a = 1/(sig*np.sqrt(2*np.pi))
    b = np.exp(-0.5*np.power(((x-mu)/sig),2))
    return a * b


def accuracy_score(y, y_model):
    """ Return accuracy score.
    Both overall and class-wise
    """
    assert len(y) == len(y_model)

    classes = np.unique(y)
    # class_n = len(classes)  # number of different classes
    correct_all = y == y_model  # all correctly classified samples

    overall = np.sum(correct_all) / len(y)
    cw = []  # this list stores class-wise accuracy

    for j in classes:
        cw_columns = np.zeros((len(y), 2))
        for i in range(len(y)):
            if y[i] == j:
                cw_columns[i, 0] = 1
                if y_model[i] == j:
                    cw_columns[i, 1] = 1
                else:
                    cw_columns[i, 1] = 0
            else:
                cw_columns[i, 0] = 0
        cw.append(np.sum(cw_columns[:, 1]) / np.sum(cw_columns[:, 0]))

    return cw, overall


def load_data(f):
    """ Assume data format:
    feature1 feature 2 ... label(integers)
    """

    # process training data
    data = np.genfromtxt(f)
    # return all feature columns except last
    x = data[:, :-1]
    y = data[:, -1].astype(int)

    return x, y


class MPP:
    """Maximum Posterior Probability
    COSC522 Machine Learning
    Supervised parametric learning of datasets assuming
    gaussian distribution. Three cases of discriminant function.
    """

    def __init__(self, case, prior):
        self.case_ = case
        self.prior_ = prior
        self.covariances_, self.means_, self.pw_ = {}, {}, {}
        self.classes_, self.num_class_ = {}, {}
        self.covsum_, self.covavg_, self.varavg_ = {}, {}, []

    def train(self, train, class_tr):
        self.covsum_ = None
        self.classes_ = np.unique(class_tr)
        self.num_class_ = len(self.classes_)

        k = 0
        for i in self.classes_:
            array = train[class_tr == i]
            self.means_[i] = np.mean(array, axis=0)
            self.covariances_[i] = np.cov(np.transpose(array))
            self.pw_[i] = self.prior_[k]
            k = k + 1

            if self.covsum_ is None:
                self.covsum_ = self.covariances_[i]
            else:
                self.covsum_ += self.covariances_[i]

        self.covavg_ = self.covsum_ / self.num_class_

        self.varavg_ = np.sum(np.diagonal(self.covavg_)) / self.num_class_

    def test(self, test):
        res = []
        disc = np.zeros(self.num_class_)
        samples_test, _ = test.shape

        for i in range(samples_test):
            for j in self.classes_:
                if self.case_ == 1:
                    p1 = np.dot(test[i]-self.means_[j], test[i]-self.means_[j])
                    disc[j] = -p1 / (2 * self.varavg_) + np.log(self.pw_[j])
                elif self.case_ == 2:
                    p1 = np.matmul(np.transpose(self.means_[j]), np.transpose(np.linalg.inv(self.covavg_)))
                    p2 = np.matmul(np.transpose(self.means_[j]), np.linalg.inv(self.covavg_))
                    disc[j] = np.matmul(p1, test[i]) - (1/2)*np.matmul(p2, self.means_[j]) + np.log(self.pw_[j])
                elif self.case_ == 3:
                    p1 = np.matmul(np.transpose(test[i]), np.linalg.inv(self.covariances_[j]))
                    p2 = np.matmul(np.transpose(self.means_[j]), np.transpose(np.linalg.inv(self.covariances_[j])))
                    p3 = np.matmul(np.transpose(self.means_[j]), np.linalg.inv(self.covariances_[j]))
                    p4 = -(1/2)*np.matmul(p1, test[i]) + np.matmul(p2, test[i])
                    p5 = -(1/2)*np.log(np.linalg.det(self.covariances_[j])) + np.log(self.pw_[j])
                    disc[j] = p4 - (1/2)*np.matmul(p3, self.means_[j]) + p5
            res.append(disc.argmax())

        return res

def normalize(train, test):
    [row_tr, col_tr] = train.shape
    [row_te, col_te] = test.shape
    train_norm = np.zeros((row_tr, col_tr))
    test_norm = np.zeros((row_te, col_te))

    for i in range(0, col_tr):
        mu = np.mean(train[:, i])
        sigma = np.std(train[:, i])
        for j in range(0, row_tr):
            train_norm[j][i] = (train[j, i] - mu) / sigma
        for k in range(0, row_te):
            test_norm[k][i] = (test[k, i] - mu) / sigma
    return [train_norm, test_norm]
