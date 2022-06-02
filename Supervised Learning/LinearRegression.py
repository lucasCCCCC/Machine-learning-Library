import numpy as np


class LinearRegression:

    def __init__(self, w0, w1, learningRate, iterations, data):
        self.w0 = w0
        self.w1 = w1
        self.learningRate = learningRate
        self.iterations = iterations
        self.data = data

    def meanSquareError(self, X, Y):
        return (1 / len(X)) * sum([val ** 2 for val in (self.w1 * X + self.w0 - Y)])

    def getWeights(self):
        return self.w0, self.w1

    def train(self, X, Y):
        y_val = self.w1 * X + self.w0

        w0_pd = -(2 / len(X)) * sum(Y - y_val)
        w1_pd = -(2 / len(X)) * sum(X * (Y - y_val))

        self.w0 = self.w0 - self.learningRate * w0_pd
        self.w1 = self.w1 - self.learningRate * w1_pd

    def fit(self):
        for i in range(self.iterations):
            self.train(self.data[:, :1], self.data[:, 1:])

    def getParams(self):
        print("Iteration: ", self.iterations,
              "\nValue of w0: ", self.getWeights()[0],
              "\nValue of w1:", self.getWeights()[1],
              "\nLoss:", self.meanSquareError(self.data[:, :1], self.data[:, 1:]), "\n")

    def predict(self, x):
        yPrediction = x * self.getWeights()[1] + self.getWeights()[0]
        return yPrediction
