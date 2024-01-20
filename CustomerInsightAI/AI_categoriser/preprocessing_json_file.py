import nltk
import numpy as np
import pickle
import json
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

# opening the intents.json file
with open("intents.json", encoding="utf8") as file:
    data = json.load(file)

# this try and except allow us to avoid preprocessing the data in the .json if it is already saved
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)

except:
    # creating arrays to store values from the intents.json file
    words = []
    labels = []
    docs_x = []
    docs_y = []

    # loops through the .json file
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            # uses the nltk.word_tokenize() to get the syllables of the words in pattern
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)

            # stores the tokenized words and tags under docs_x and docs_y respectively
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        # stores the tags from the .json file
        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    # converts the words list to lowercase letters and arranges them alphabetically
    words = [stemmer.stem(w.lower()) for w in words if w not in "?"]
    words = sorted(list(set(words)))

    # sorts the list of tags alphabetically
    labels = sorted(labels)

    # create empty training and output lists
    training = []
    output = []

    # create a list that has zeros for every tag available i.e [0,0,0,0]
    out_empty = [0 for _ in range(len(labels))]

    # loop through doc_x which contains the all the patterns from the .json file, with enumerate x will store the
    # index we are on in the list while doc will store the particular pattern we are on in the list
    for x, doc in enumerate(docs_x):
        # create an empty list for the bag of words
        bag = []

        # store the individual words from the patterns as a lowercase letter
        wrds = [stemmer.stem(w.lower()) for w in doc]

        # compare a word in wrds from above to the list of tokenized words in the lists word and store 1 into the list
        # bag if comparison is found else 0
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        # copy the out_empty list into output_row
        output_row = out_empty[:]

        # checks for the specific tag in docs_y that is matched to the pattern in x and returns its index in the labels
        # lists and converts that index in the output_row that was full of zero's to a one
        output_row[labels.index(docs_y[x])] = 1

        # append the bag of words into the empty training list
        training.append(bag)
        # append the output_row into the empty output list
        output.append(output_row)

    # convert training and output into numpy arrays
    training = np.array(training)
    output = np.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)
