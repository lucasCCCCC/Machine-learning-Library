import pandas as pd
import numpy as np
import ast

data = pd.read_csv("Dataset/emailTrainData.csv")
file = open("vocab.txt", "r")
reviews = file.read()
vocab = ast.literal_eval(reviews)

X = np.zeros((data.shape[0], len(vocab)))
Y = np.zeros((data.shape[0]))

for i in range(data.shape[0]):
    review = data.iloc[i, 0].split()

    for word in review:
        if word.lower() in vocab:
            X[i, vocab[word.lower()]] += 1
            Y[i] = data.iloc[i, 1]


np.save("X.npy", X)
np.save("Y.npy", Y)
