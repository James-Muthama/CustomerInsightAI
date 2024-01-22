import nltk
import numpy as np

from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()


# create a function that takes in input from a user converts it into a numpy array to feed into the model
def bag_of_words(s, words):
    # creates the bag of words
    bag = [0 for _ in range(len(words))]

    # tokenizes every word inputted by the user and stems them
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    # fills the bag of words with 1 if a stemmed word matches a word from the words list
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    # returns a numpy array to be passed in the model
    return np.array(bag)
