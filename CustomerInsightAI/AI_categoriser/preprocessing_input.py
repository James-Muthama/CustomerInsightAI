import nltk
import numpy as np

from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()


def bag_of_words(sentence, words):
    # Tokenize and stem the input sentence
    tokenized_words = nltk.word_tokenize(sentence)
    stemmed_words = [stemmer.stem(word.lower()) for word in tokenized_words]

    # Create the bag of words
    bag = [0] * len(words)

    # Fill the bag of words
    for i, w in enumerate(words):
        if w in stemmed_words:
            bag[i] = 1

    # Convert the bag to a NumPy array
    return np.array([bag])
