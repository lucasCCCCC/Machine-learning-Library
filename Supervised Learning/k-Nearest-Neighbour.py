import collections
import numpy as np


class KNearestNeighbour:

    def __init__(self, K):
        self.K = K

    def euclideanDistance(self, x, y):
        return np.sqrt(np.sum((x - y)**2))

    def fit(self, x, y):
        self.X = x
        self.Y = y

    def predict(self, x):
        label = [self.predictLabel(i) for i in x]
        return np.array(label)

    def predictLabel(self, x):
        euclideanDistance = [self.euclideanDistance(x, X) for X in self.X]
        Ks = np.argsort(euclideanDistance)[:self.K]
        KLabels = [self.Y[i] for i in Ks]

        label = collections.Counter(KLabels).most_common(1)

        return label[0][0]


