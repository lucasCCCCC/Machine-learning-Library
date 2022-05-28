import numpy as np


class LinearRegression:

    def __init__(self, w0, w1, learningRate):
        self.w0 = w0
        self.w1 = w1
        self.learningRate = learningRate

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


test_data = np.array([[1, 1], [2, 2], [3, 4], [4, 4], [5, 5], [5, 6], [6, 5], [7, 7], [7, 6], [8, 8], [9, 7], [10, 11]])

x_train_data = test_data[:, :1]
y_train_data = test_data[:, 1:]

initialW0 = 0
initialW1 = 0
initialLearningRate = 0.01
iterations = 100

model = LinearRegression(initialW0, initialW1, initialLearningRate)

for i in range(iterations):
    model.train(x_train_data, y_train_data)
    print("Iteration: ", i+1,
          "\nValue of w0: ", model.getWeights()[0],
          "\nValue of w1:", model.getWeights()[1],
          "\nCurrent loss:", model.meanSquareError(x_train_data, y_train_data), "\n")
