# Partial credit to https://www.youtube.com/c/PythonEngineer

import numpy as np


class NaiveBayes:

    def __init__(self, x, y):
        self.samples, self.attributes = x.shape
        self.classes = len(np.unique(y))

    def fit(self, x, y):
        self.mean = {}
        self.variance = {}
        self.prior = {}

        for i in range(self.classes):
            X_c = x[y == i]

            self.mean[str[i]] = np.mean(X_c, axis=0)
            self.variance[str[i]] = np.var(X_c, axis=0)
            self.prior[str[i]] = X_c.shape[0]/self.samples

    def predictClass(self, x):
        probability = np.zeros((self.samples, self.classes))

        for i in range(self.classes):
            prior = self.prior[str[i]]
            prob = self.pdf(x, self.mean[str[i]], self.variance[str[i]])
            probability[:, i] = prob + np.log(prior)

        return np.argmax(probability, 1)

    def pdf(self, x, mean, sigma):
        CONSTANT = -self.attributes/2 * np.log(2*np.pi) - 0.5*np.sum(np.log(sigma + 1e-6))
        PROBABILITY = 0.5*np.sum(np.power(x - mean, 2) / (sigma + 1e-6), 1)
        return (CONSTANT - PROBABILITY)

