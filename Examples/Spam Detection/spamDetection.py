import ast
from Models.Supervised import NaiveBayes
import numpy as np
import pandas as pd

file = open("vocab.txt", "r")
reviews = file.read()
data = pd.read_csv("Dataset/emailTrainData.csv")
vocab = ast.literal_eval(reviews)

X = np.load("X.npy")
Y = np.load("Y.npy")

predictions = {0: "Not Spam", 1: "Spam"}

model = NaiveBayes(X, Y)
model.fit(X, Y)

X_pred = np.zeros((data.shape[0], len(vocab)))


def checkEmail(current_email):
    for i in range(data.shape[0]):

        review = current_email.split()

        for word in review:
            if word.lower() in vocab:
                X_pred[i, vocab[word.lower()]] += 1

    prediction = model.predictClass(X_pred)

    print("Prediction:", predictions.get(prediction[0]))
