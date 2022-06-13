import pandas as pd
import nltk
from nltk.corpus import words

vocab = {}
data = pd.read_csv("Dataset/emailTrainData.csv")
nltk.download("words")
allWords = set(words.words())


def buildVocab(review):
    index = len(vocab)
    for word in review:
        if word.lower() in allWords and word.lower() not in vocab:
            vocab[word.lower()] = index
            index += 1


for i in range(data.shape[0]):
    currentReview = data.iloc[i, 0].split()

    buildVocab(currentReview)

file = open("vocab.txt", "w")
file.write(str(vocab))
file.close()
