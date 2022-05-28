import numpy as np
from scipy.special import expit


class LogisticRegression:

    def __init__(self, w0, w1, learningRate):
        self.w0 = w0
        self.w1 = w1
        self.learningRate = learningRate

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


test_data = np.array([[1, 0], [1.5, 0], [2, 0], [2.5, 0], [3, 0], [4, 0], [4.5, 0],
                      [5.5, 0], [6, 1], [7, 1], [7.5, 1], [8, 1], [8.5, 1], [9, 1]])

x_train_data = test_data[:, :1]
y_train_data = test_data[:, 1:]

initialW0 = 0
initialW1 = 0
initialLearningRate = 0.01
iterations = 100

model = LogisticRegression(initialW0, initialW1, initialLearningRate)

for i in range(iterations):
    model.train(x_train_data, y_train_data)
    print("Iteration: ", i+1,
          "\nValue of w0: ", model.getWeights()[0],
          "\nValue of w1:", model.getWeights()[1],
          "\nCurrent loss:", model.crossEntropyLoss(x_train_data, y_train_data), "\n")
