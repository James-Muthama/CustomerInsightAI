import nltk
import numpy as np
import pickle
import json
from nltk.stem.lancaster import LancasterStemmer
from sklearn.feature_extraction.text import CountVectorizer

stemmer = LancasterStemmer()

# Opening the intents.json file
with open("intents.json", encoding="utf8") as file:
    data = json.load(file)

# This try and except allow us to avoid preprocessing the data in the .json if it is already saved
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)

except:
    # Creating arrays to store values from the intents.json file
    words = []
    labels = []
    docs_x = []
    docs_y = []

    # Loops through the .json file
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            # Uses the nltk.word_tokenize() to get the syllables of the words in pattern
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)

            # Stores the tokenized words and tags under docs_x and docs_y respectively
            docs_x.append(" ".join(wrds))  # Convert list of words to a sentence
            docs_y.append(intent["tag"])

        # Stores the tags from the .json file
        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    # Converts the words list to lowercase letters and arranges them alphabetically
    words = [stemmer.stem(w.lower()) for w in words if w not in "?"]
    words = sorted(list(set(words)))

    # Sorts the list of tags alphabetically
    labels = sorted(labels)

    # Create an instance of CountVectorizer
    vectorizer = CountVectorizer()

    # Transform the text data into a one-hot encoded matrix
    training = vectorizer.fit_transform(docs_x).toarray()

    # Create empty output list
    output = []

    # Create a list that has zeros for every tag available i.e [0, 0, 0, 0]
    out_empty = [0 for _ in range(len(labels))]

    # Loop through docs_y which contains the tags, with enumerate x will store the
    # index we are on in the list while tag will store the particular tag we are on in the list
    for x, tag in enumerate(docs_y):
        # Copy the out_empty list into output_row
        output_row = out_empty[:]

        # Checks for the specific tag in docs_y that is matched to the tag in x and
        # returns its index in the labels lists and converts that index in the output_row
        # that was full of zero's to a one
        output_row[labels.index(tag)] = 1

        # Append the output_row into the empty output list
        output.append(output_row)

    # Convert output into a numpy array
    output = np.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)
