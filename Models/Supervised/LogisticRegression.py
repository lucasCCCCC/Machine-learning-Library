import numpy as np
from scipy.special import expit


class LogisticRegression:

    def __init__(self, w0, w1, learningRate, iterations, data):
        self.w0 = w0
        self.w1 = w1
        self.learningRate = learningRate
        self.iterations = iterations
        self.data = data

    def getWeights(self):
        return self.w0, self.w1

    def sigmoid(self, X):
        Zs = self.w1 * X + self.w0
        return 1 / (1 + np.exp(-Zs))

    def crossEntropyLoss(self, X, Y):
        return - np.mean(Y * np.log(self.sigmoid(X)) + (1 - Y) * np.log(1 - self.sigmoid(X)))

    def train(self, X, Y):
        y_val = expit(self.w1 * X + self.w0)

        w0_pd = np.mean(y_val - Y)
        w1_pd = np.mean((y_val - Y) * X)

        self.w0 = self.w0 - self.learningRate * w0_pd
        self.w1 = self.w1 - self.learningRate * w1_pd

    def fit(self):
        for i in range(self.iterations):
            self.train(self.data[:, :1], self.data[:, 1:])

    def getParams(self):
        print("Iteration: ", self.iterations,
              "\nValue of w0: ", self.getWeights()[0],
              "\nValue of w1:", self.getWeights()[1],
              "\nLoss:", self.crossEntropyLoss(self.data[:, :1], self.data[:, 1:]))

    def predict(self, x):
        yPrediction = expit(x * self.getWeights()[1] + self.getWeights()[0])

        if yPrediction > 0.5:
            return 1

        return 0
